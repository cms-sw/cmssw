#include "CondTools/SiStrip/plugins/SiStripThresholdReader.h"
#include "CondFormats/DataRecord/interface/SiStripThresholdRcd.h"

using namespace std;
using namespace cms;

SiStripThresholdReader::SiStripThresholdReader( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<uint32_t>("printDebug",1)){}

SiStripThresholdReader::~SiStripThresholdReader(){}

void SiStripThresholdReader::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  edm::ESHandle<SiStripThreshold> SiStripThreshold_;
  iSetup.get<SiStripThresholdRcd>().get(SiStripThreshold_);
  edm::LogInfo("SiStripThresholdReader") << "[SiStripThresholdReader::analyze] End Reading SiStripThreshold" << std::endl;
  
  std::vector<uint32_t> detid;
  SiStripThreshold_->getDetIds(detid);
  edm::LogInfo("Number of detids ")  << detid.size() << std::endl;
  
  if (printdebug_)
    for (size_t id=0;id<detid.size() && id<printdebug_;id++)
      {
	SiStripThreshold::Range range=SiStripThreshold_->getRange(detid[id]);
	
	//int strip=0;
	for(int it=0;it<range.second-range.first;it++){
	  unsigned int value=(*(range.first+it));
	  edm::LogInfo("SiStripThresholdReader")  << "detid: " << detid[id] << " \t"
						<< "firstStrip: " << SiStripThreshold_->decode(value).firstStrip << " \t"
						<< "NumConsecutiveStrip: " << SiStripThreshold_->decode(value).stripRange << " \t"
						<< "lTh: " << SiStripThreshold_->decode(value).lowTh   << " \t" 
						<< "hTh: " << SiStripThreshold_->decode(value).highTh   << " \t" 
						<< "packed integer: " << std::hex << value << std::dec
						<< std::endl; 	    
	} 
      }
}

