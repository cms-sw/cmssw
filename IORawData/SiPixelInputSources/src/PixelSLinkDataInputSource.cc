// PixelSLinkDataInputSource.cc

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "IORawData/SiPixelInputSources/interface/PixelSLinkDataInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include <fstream>

PixelSLinkDataInputSource::PixelSLinkDataInputSource(const edm::ParameterSet& pset, 
							      const edm::InputSourceDescription& desc) :
  ExternalInputSource(pset,desc),
  m_fileindex(0),
  m_fedid(pset.getUntrackedParameter<int>("fedid"))
{
  bool result = open_file();
  produces<FEDRawDataCollection>();
}
    
bool PixelSLinkDataInputSource::open_file(){
  if(m_fileindex==fileNames().size())
    return false;
  std::string fullname = fileNames()[m_fileindex];
  m_fileindex++;
  if(fullname.find('file:')){
    m_file_name = fullname.substr(5);
  }
  else if(fullname.find('rfio:')){
    edm::LogError("") << "Trying to open file " << fullname <<"... from Castor... (not implemented yet)";
    return false;
  }
  else{
    edm::LogError("") << "Trying to open file " << fullname <<", and do not recognize prefix (only \"rfio:\" and \"file:\" are accepted)";
    return false;
  }
  m_file.open(m_file_name.c_str(),std::ios::in|std::ios::binary);
  if (!m_file.good()) {
    edm::LogError("") << "Error opening file " << fullname ;    
    return false;
  }
  return true;
}
PixelSLinkDataInputSource::~PixelSLinkDataInputSource() {}

bool PixelSLinkDataInputSource::produce(edm::Event& event) {
  
  // create product (raw data)
  std::auto_ptr<FEDRawDataCollection> buffers( new FEDRawDataCollection );
  
  unsigned long long data;

  std::vector<unsigned long long> buffer;

  m_file.read((char*)&data,8);
  
  if (m_file.eof()) {
    edm::LogInfo("") << "End of input file" ;
    return open_file();
  }


  unsigned int count=0;
    
  while ((data >> 60) != 0x5){
    if (count==0){
      edm::LogWarning("") << "DATA CORRUPTION!" ;
      edm::LogWarning("") << "Expected to find header, but read: 0x"
			  << std::hex<<data<<std::dec ;
    }
    count++;
    m_file.read((char*)&data,8);
    if (m_file.eof()) {
      edm::LogInfo("") << "End of input file" ;
      return open_file();
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
  std::auto_ptr<FEDRawData> rawData(new FEDRawData(8*buffer.size()));
  //  FEDRawData * rawData = new FEDRawData(8*buffer.size());
  unsigned char* dataptr=rawData->data();

  for (unsigned int i=0;i<buffer.size();i++){
    ((unsigned long long *)dataptr)[i]=buffer[i];
  }

  FEDRawData& fedRawData = buffers->FEDData( fed_id );
  fedRawData=*rawData;

  event.put(buffers);

  return true;

}
