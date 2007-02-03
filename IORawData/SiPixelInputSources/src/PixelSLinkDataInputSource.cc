// PixelSLinkDataInputSource.cc

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ExternalInputSource.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "IORawData/SiPixelInputSources/interface/PixelSLinkDataInputSource.h"
#include <fstream>

PixelSLinkDataInputSource::PixelSLinkDataInputSource(const edm::ParameterSet& pset, 
							      const edm::InputSourceDescription& desc) :
  ExternalInputSource(pset,desc) {
  m_file_name = pset.getUntrackedParameter<std::string>("slink_data_file_name");
  m_file.open(m_file_name.c_str(),std::ios::in|std::ios::binary);
  if (!m_file.good()) {
   std::cout << "Error opening file" <<  std::endl;    
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
    std::cout << "End of input file" <<  std::endl;
    return false;
  }
  
  if ((data >> 60) != 0x5){
    std::cout << "DATA CORRUPTION!" <<  std::endl;
    std::cout << "Expected to find header, but read: 0x"
	      <<std::hex<<data<<std::dec <<  std::endl;
    return false;
  }

  std::cout<<"FED header:"<<std::hex<<data<<std::dec <<  std::endl;

  
  unsigned int fed_id=(data>>9)&&0xfff;
  
  std::cout << "FED id="<<fed_id<<std::endl;
  
  buffer.push_back(data);
  
  do{
    m_file.read((char*)&data,8);
    buffer.push_back(data);
  }while((data >> 60) != 0xa);
  
  FEDRawData * rawData = new FEDRawData(8*buffer.size());
  FEDRawData& fedRawData = buffers->FEDData( fed_id );
  fedRawData=*rawData;
  unsigned char* dataptr=fedRawData.data();
  for (unsigned int i=0;i<buffer.size();i++){
    ((unsigned long long *)dataptr)[i]=buffer[i];
  }

  return true;

}
