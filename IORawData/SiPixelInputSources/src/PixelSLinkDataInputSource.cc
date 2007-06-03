// PixelSLinkDataInputSource.cc

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "IORawData/SiPixelInputSources/interface/PixelSLinkDataInputSource.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <fstream>

PixelSLinkDataInputSource::PixelSLinkDataInputSource(const edm::ParameterSet& pset, 
							      const edm::InputSourceDescription& desc) :
  ExternalInputSource(pset,desc) {
  m_file_name = pset.getUntrackedParameter<std::string>("slink_data_file_name");
  m_fedid = pset.getUntrackedParameter<int>("fedid");
  m_file.open(m_file_name.c_str(),std::ios::in|std::ios::binary);
  if (!m_file.good()) {
   edm::LogError("") << "Error opening file" ;    
  }
  produces<FEDRawDataCollection>();
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
    m_file.read((char*)&data,8);
    if (m_file.eof()) {
      edm::LogInfo("") << "End of input file" ;
      return false;
    }

  }

  if (count>0) {
    edm::LogWarning("")<<"Had to read "<<count<<" words before finding header!"<<std::endl;
  }

  if (m_fedid>0) {
    data=(data&0xfffffffffff000ffLL)|((m_fedid&0xfff)<<8);
  }

  unsigned int fed_id=(data>>8)&0xfff;
  
  buffer.push_back(data);
  
  do{
    m_file.read((char*)&data,8);
    buffer.push_back(data);
  }while((data >> 60) != 0xa);
  
  FEDRawData * rawData = new FEDRawData(8*buffer.size());
  unsigned char* dataptr=rawData->data();

  for (unsigned int i=0;i<buffer.size();i++){
    ((unsigned long long *)dataptr)[i]=buffer[i];
  }

  FEDRawData& fedRawData = buffers->FEDData( fed_id );
  fedRawData=*rawData;

  event.put(buffers);

  return true;

}
