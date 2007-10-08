#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CalibTracker/SiStripESProducers/test/testSiStripQualityESProducer.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>

testSiStripQualityESProducer::testSiStripQualityESProducer( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)),
  m_cacheID_(0){}

void testSiStripQualityESProducer::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  unsigned long long cacheID = iSetup.get<SiStripQualityRcd>().cacheIdentifier();

  if (m_cacheID_ == cacheID) 
    return;

  m_cacheID_ = cacheID; 

  edm::ESHandle<SiStripQuality> SiStripQuality_;
  iSetup.get<SiStripQualityRcd>().get(SiStripQuality_);
  edm::LogInfo("testSiStripQualityESProducer") << "[testSiStripQualityESProducer::analyze] End Reading SiStripQuality" << std::endl;
  
  std::vector<uint32_t> detid;
  SiStripQuality_->getDetIds(detid);
  
  if (printdebug_)
    for (size_t id=0;id<detid.size();id++)
      {
	SiStripQuality::Range range=SiStripQuality_->getRange(detid[id]);
	
	for(int it=0;it<range.second-range.first;it++){
	  unsigned int value=(*(range.first+it));
	  std::pair<unsigned short, unsigned short> fs=SiStripQuality_->decode(value);
	  edm::LogInfo("testSiStripQualityESProducer")  << "detid " << detid[id] << " \t"
						 << " firstBadStrip " <<  fs.first << "\t "
						 << " NconsecutiveBadStrips " << fs.second  << "\t "
						 << " packed integer " <<  std::hex << value << std::dec << "\t "
						 << std::endl;
	} 
      }
}
