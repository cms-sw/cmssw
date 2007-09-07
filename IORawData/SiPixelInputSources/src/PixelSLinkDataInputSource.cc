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
// $Id$
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
#include <iostream>

using namespace seal;
using namespace edm;
PixelSLinkDataInputSource::PixelSLinkDataInputSource(const edm::ParameterSet& pset, 
							      const edm::InputSourceDescription& desc) :
  ExternalInputSource(pset,desc),
  m_fedid(pset.getUntrackedParameter<int>("fedid")),
  m_fileindex(0)
{
  produces<FEDRawDataCollection>();

  if (m_fileindex>=fileNames().size()) {
    std::cout << "no more file to read " << std::endl;
    return;// ???
  }
  std::string currentfilename = fileNames()[m_fileindex];
  edm::LogInfo("") << "now examining file "<< currentfilename ;
  m_fileindex++;
  // reading both castor and other ('normal'/dcap) files.
  IOOffset size = -1;
  StorageFactory::get()->enableAccounting(true);
    
  bool exists = StorageFactory::get() -> check(currentfilename.c_str(), &size);
  
  std::cout << "file size " << size << std::endl;
  
  if(!exists){
    edm::LogInfo("") << "file " << currentfilename << " cannot be found.";
    return;
  }
  // now open the file stream:
  storage.reset(StorageFactory::get()->open(currentfilename.c_str()));
  // (throw if storage is 0)

}
    
PixelSLinkDataInputSource::~PixelSLinkDataInputSource() {


}

bool PixelSLinkDataInputSource::produce(edm::Event& event) {
  
  seal::Storage & m_file = *storage;

  // create product (raw data)
  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );
    

  unsigned long long data;

  std::vector<unsigned long long> buffer;
  IOSize n = m_file.read((char*)&data,8);
  
  if (n==0) {
    edm::LogInfo("") << "End of input file" ;
    return false;
  }


  unsigned int count=0;
    
  while ((data >> 60) != 0x5){
    if (count==0){
      edm::LogWarning("") << "DATA CORRUPTION!" ;
      edm::LogWarning("") << "Expected to find header, but read: 0x"
			  << std::hex<<data<<std::dec ;
    }
    count++;
    n=m_file.read((char*)&data,8);
    if (n!=8) {
      edm::LogInfo("") << "End of input file" ;
      return false;
    }
  }

  if (count>0) {
    edm::LogWarning("")<<"Had to read "<<count<<" words before finding header!"<<std::endl;
  }

  if (m_fedid>-1) {
    data=(data&0xfffffffffff000ffLL)|((m_fedid&0xfff)<<8);
  }

  unsigned int fed_id=(data>>8)&0xfff;
  
  buffer.push_back(data);
  
  do{
    m_file.read((char*)&data,8);
    buffer.push_back(data);
  }while((data >> 60) != 0xa);

  std::cout << "read " <<  buffer.size() << " long words" << std::endl;

  std::auto_ptr<FEDRawData> rawData(new FEDRawData(8*buffer.size()));
  //  FEDRawData * rawData = new FEDRawData(8*buffer.size());
  unsigned char* dataptr=rawData->data();

  for (unsigned int i=0;i<buffer.size();i++){
    ((unsigned long long *)dataptr)[i]=buffer[i];
  }

  FEDRawData& fedRawData = buffers->FEDData( fed_id );
  fedRawData=*rawData;

  event.put(buffers);
//  StorageAccount::summaryText ();
  return true;

}

