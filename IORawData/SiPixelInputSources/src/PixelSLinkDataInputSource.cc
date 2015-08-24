// -*- C++ -*-
//
// Package:    SiPixelInputSources
// Class:      PixelSLinkDataInputSource
// 
/**\class PixelSLinkDataInputSource PixelSLinkDataInputSource.cc IORawData/SiPixelInputSources/src/PixelSLinkDataInputSource.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Freya Blekman
//         Created:  Fri Sep  7 15:46:34 CEST 2007
//
//

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "IORawData/SiPixelInputSources/interface/PixelSLinkDataInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/IOTypes.h"
#include <iostream>

using namespace edm;

// function to get the trigger number from fill words
int PixelSLinkDataInputSource::getEventNumberFromFillWords(const std::vector<uint64_t>& buffer, uint32_t & totword ){
  // buffer validity, should already be pretty clean as this is exactly what goes into the FEDRawDataobject.
  
  // code copied directly from A. Ryd's fill word checker in PixelFEDInterface::PwordSlink64

  int fif2cnt=0;
  int dumcnt=0;
  int gapcnt=0;
  uint32_t gap[9];
  uint32_t dum[9];
  uint32_t word[2]={0,0};
  uint32_t chan=0;
  uint32_t roc=0;

  const uint32_t rocmsk = 0x3e00000;
  const uint32_t chnlmsk = 0xfc000000;
  
  for(int jk=1;jk<9;jk++)gap[jk]=0;
  for(int jk=1;jk<9;jk++)dum[jk]=0;
  
  int fifcnt=1;
  for(size_t kk=0; kk<buffer.size(); ++kk)
    {

      word[0] = (uint32_t) buffer[kk];
      word[1] = (uint32_t) (buffer[kk]>>32);

      for(size_t iw=0; iw<2; iw++)
	{
	  chan= ((word[iw]&chnlmsk)>>26);
	  roc= ((word[iw]&rocmsk)>>21);

	  //count non-error words
	  if(roc<25){
	    if((chan>4)&&(chan<10)&&(fifcnt!=2)) {fif2cnt=0;fifcnt=2;}
	    if((chan>9)&&(chan<14)&&(fifcnt!=3)) {fif2cnt=0;fifcnt=3;}
	    if((chan>13)&&(chan<19)&&(fifcnt!=4)){fif2cnt=0;fifcnt=4;}
	    if((chan>18)&&(chan<23)&&(fifcnt!=5)){fif2cnt=0;fifcnt=5;}
	    if((chan>22)&&(chan<28)&&(fifcnt!=6)){fif2cnt=0;fifcnt=6;}
	    if((chan>27)&&(chan<32)&&(fifcnt!=7)){fif2cnt=0;fifcnt=7;}
	    if((chan>31)&&(fifcnt!=8)){fif2cnt=0;fifcnt=8;} 
	    fif2cnt++;
	  }
	  if(roc==26){gap[fifcnt]=(0x1000+(word[iw]&0xff));gapcnt++;}
	  
	  if((roc==27)&&((fif2cnt+dumcnt)<6)){dumcnt++;dum[fifcnt]=(0x1000+(word[iw]&0xff));}
	  else if((roc==27)&&((fif2cnt+dumcnt)>6)){dumcnt=1;fif2cnt=0;fifcnt++;}
	}

      //word check complete
      if(((fif2cnt+dumcnt)==6)&&(dumcnt>0)) //done with this fifo
	{dumcnt=0;fif2cnt=0;fifcnt++;}
      if((gapcnt>0)&&((dumcnt+fif2cnt)>5))//done with this fifo
	{gapcnt=0;fifcnt++;fif2cnt=0;dumcnt=0;}
      else if((gapcnt>0)&&((dumcnt+fif2cnt)<6)) gapcnt=0;

    }//end of fifo-3 word loop-see what we got!

  int status=0;

  if(gap[1]>0) {totword=(gap[1]&0xff);status=1;}
  else if(gap[2]>0){totword=(gap[2]&0xff);status=1;}
  else if(dum[1]>0){totword=(dum[1]&0xff);status=1;}
  else if(dum[2]>0){totword=(dum[2]&0xff);status=1;}

  if(gap[3]>0) {totword=totword|((gap[3]&0xff)<<8);status=status|0x2;}
  else if(gap[4]>0){totword=totword|((gap[4]&0xff)<<8);status=status|0x2;}
  else if(dum[3]>0){totword=totword|((dum[3]&0xff)<<8);status=status|0x2;}
  else if(dum[4]>0){totword=totword|((dum[4]&0xff)<<8);status=status|0x2;}

  if(gap[5]>0) {totword=totword|((gap[5]&0xff)<<16);status=status|0x4;}
  else if(gap[6]>0){totword=totword|((gap[6]&0xff)<<16);status=status|0x4;}
  else if(dum[5]>0){totword=totword|((dum[5]&0xff)<<16);status=status|0x4;}
  else if(dum[6]>0){totword=totword|((dum[6]&0xff)<<16);status=status|0x4;}

  if(gap[7]>0){totword=totword|((gap[7]&0xff)<<24);status=status|0x8;}
  else if(gap[8]>0){totword=totword|((gap[8]&0xff)<<24);status=status|0x8;}
  else if(dum[7]>0){totword=totword|((dum[7]&0xff)<<24);status=status|0x8;}
  else if(dum[8]>0){totword=totword|((dum[8]&0xff)<<24);status=status|0x8;}
  return(status);

}

// constructor
PixelSLinkDataInputSource::PixelSLinkDataInputSource(const edm::ParameterSet& pset, 
						     const edm::InputSourceDescription& desc) :
  ProducerSourceFromFiles(pset,desc,true),
  m_fedid(pset.getUntrackedParameter<int>("fedid")),
  m_fileindex(0),
  m_runnumber(pset.getUntrackedParameter<int>("runNumber",-1)),
  m_currenteventnumber(0),
  m_currenttriggernumber(0),
  m_eventnumber_shift(0)
{
  produces<FEDRawDataCollection>();

  if (m_fileindex>=fileNames().size()) {
    edm::LogInfo("") << "no more file to read " << std::endl;
    return;// ???
  }
  std::string currentfilename = fileNames()[m_fileindex];
  edm::LogInfo("") << "now examining file "<< currentfilename ;
  m_fileindex++;
  // reading both castor and other ('normal'/dcap) files.
  IOOffset size = -1;
  StorageFactory::getToModify()->enableAccounting(true);
    
  edm::LogInfo("PixelSLinkDataInputSource") << " unsigned long int size = " << sizeof(unsigned long int) <<"\n unsigned long size = " << sizeof(unsigned long)<<"\n unsigned long long size = " << sizeof(unsigned long long) <<  "\n uint32_t size = " << sizeof(uint32_t) << "\n uint64_t size = " << sizeof(uint64_t) << std::endl;

  bool exists = StorageFactory::get() -> check(currentfilename.c_str(), &size);
  
  edm::LogInfo("PixelSLinkDataInputSource") << "file size " << size << std::endl;
  
  if(!exists){
    edm::LogInfo("") << "file " << currentfilename << " cannot be found.";
    return;
  }
  // now open the file stream:
  storage =StorageFactory::get()->open(currentfilename.c_str());
  // (throw if storage is 0)

  // check run number by opening up data file...
  
  Storage & temp_file = *storage;
  //  IOSize n =
  temp_file.read((char*)&m_data,8);
  if((m_data >> 60) != 0x5){ 
    uint32_t runnum = m_data;
    if(m_runnumber!=-1)
      edm::LogInfo("") << "WARNING: observed run number encoded in S-Link dump. Overwriting run number as defined in .cfg file!!! Run number now set to " << runnum << " (was " << m_runnumber << ")";
    m_runnumber=runnum;
  } 
  temp_file.read((char*)&m_data,8);
  m_currenteventnumber = (m_data >> 32)&0x00ffffff ;
}
    
// destructor
PixelSLinkDataInputSource::~PixelSLinkDataInputSource() {


}

bool PixelSLinkDataInputSource::setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time, edm::EventAuxiliary::ExperimentType&) {
  Storage & m_file = *storage;

  // create product (raw data)
  buffers.reset( new FEDRawDataCollection );
    
  //  uint32_t currenteventnumber = (m_data >> 32)&0x00ffffff;
  uint32_t eventnumber =(m_data >> 32)&0x00ffffff ;
  
  do{
    std::vector<uint64_t> buffer;
  
 
  
    uint16_t count=0;
    eventnumber = (m_data >> 32)&0x00ffffff ;
    if(m_currenteventnumber==0)
      m_currenteventnumber=eventnumber;
    edm::LogInfo("PixelSLinkDataInputSource::produce()") << "**** event number = " << eventnumber << " global event number " << m_currenteventnumber << " data " << std::hex << m_data << std::dec << std::endl;
    while ((m_data >> 60) != 0x5){
      //  std::cout << std::hex << m_data << std::dec << std::endl;
      if (count==0){
	edm::LogWarning("") << "DATA CORRUPTION!" ;
	edm::LogWarning("") << "Expected to find header, but read: 0x"
			    << std::hex<<m_data<<std::dec ;
      }
   
      count++;
      int n=m_file.read((char*)&m_data,8);
      edm::LogWarning("") << "next data " << std::hex << m_data << std::dec << std::endl;
    
      if (n!=8) {
	edm::LogInfo("") << "End of input file" ;
	return false;
      }
    }
 

    if (count>0) {
      edm::LogWarning("")<<"Had to read "<<count<<" words before finding header!"<<std::endl;
    }

    if (m_fedid>-1) {
      m_data=(m_data&0xfffffffffff000ffLL)|((m_fedid&0xfff)<<8);
    }

    uint16_t fed_id=(m_data>>8)&0xfff;
    //   std::cout << "fed id = " << fed_id << std::endl;
    buffer.push_back(m_data);
  
    do{
      m_file.read((char*)&m_data,8);
      buffer.push_back(m_data);
    }
    while((m_data >> 60) != 0xa);
    //  std::cout << "read " <<  buffer.size() << " long words" << std::endl;

    std::auto_ptr<FEDRawData> rawData(new FEDRawData(8*buffer.size()));
    //  FEDRawData * rawData = new FEDRawData(8*buffer.size());
    unsigned char* dataptr=rawData->data();

    for (uint16_t i=0;i<buffer.size();i++){
      ((uint64_t *)dataptr)[i]=buffer[i];
    }
    uint32_t thetriggernumber=0;
    int nfillwords = 0;//getEventNumberFromFillWords(buffer,thetriggernumber);

    if(nfillwords>0){
      LogInfo("") << "n fill words = " << nfillwords <<  ", trigger numbers: " << thetriggernumber << "," << m_currenttriggernumber << std::endl;
      m_eventnumber_shift = thetriggernumber - m_currenttriggernumber;
    }
    m_currenttriggernumber = thetriggernumber;
    FEDRawData& fedRawData = buffers->FEDData( fed_id );
    fedRawData=*rawData;
    
    // read the first data member of the next blob to check on event number
    int n =m_file.read((char*)&m_data,8);
    if (n==0) {
      edm::LogInfo("") << "End of input file" ;
    }
    m_currenteventnumber = (m_data >> 32)&0x00ffffff ;
    if(m_currenteventnumber<eventnumber)
      LogError("PixelSLinkDataInputSource") << " error, the previous event number (" << eventnumber << ") is LARGER than the next event number (" << m_currenteventnumber << ")" << std::endl;

  }
  while( eventnumber == m_currenteventnumber);
  
  uint32_t realeventno = synchronizeEvents();
  if(m_runnumber!=0)
    id = edm::EventID(m_runnumber, id.luminosityBlock(), realeventno);
  else
    id = edm::EventID(id.run(), id.luminosityBlock(), realeventno);
  return true;
}

// produce() method. This is the worker method that is called every event.
void PixelSLinkDataInputSource::produce(edm::Event& event) {
  event.put(buffers);
  buffers.reset();  
}

// this function sets the m_globaleventnumber quantity. It uses the m_currenteventnumber and m_currenttriggernumber values as input
uint32_t PixelSLinkDataInputSource::synchronizeEvents(){
  int32_t result= m_currenteventnumber -1;
      
  return(uint32_t) result;
}
