#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

#include "CalibTracker/SiStripESProducers/test/testSiStripQualityESProducer.h"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <sstream>

testSiStripQualityESProducer::testSiStripQualityESProducer( const edm::ParameterSet& iConfig ):
  printdebug_(iConfig.getUntrackedParameter<bool>("printDebug",false)),
  m_cacheID_(0){}

void testSiStripQualityESProducer::analyze( const edm::Event& e, const edm::EventSetup& iSetup){

  unsigned long long cacheID = iSetup.get<SiStripQualityRcd>().cacheIdentifier();

  std::stringstream ss;

  if (m_cacheID_ == cacheID) 
    return;

  m_cacheID_ = cacheID; 

  edm::ESHandle<SiStripQuality> SiStripQuality_;
  iSetup.get<SiStripQualityRcd>().get(SiStripQuality_);
  edm::LogInfo("testSiStripQualityESProducer") << "[testSiStripQualityESProducer::analyze] End Reading SiStripQuality" << std::endl;
  
  std::vector<uint32_t> detid;
  SiStripQuality_->getDetIds(detid);
  
  if (printdebug_){
    for (size_t id=0;id<detid.size();id++)
      {
	ss.str("");
	ss<< "Full Info";
	SiStripQuality::Range range=SiStripQuality_->getRange(detid[id]);
	
	for(int it=0;it<range.second-range.first;it++){
	  unsigned int value=(*(range.first+it));
	  
	  ss << "\n\tdetid " << detid[id] << " \t"
	     << " firstBadStrip " <<  SiStripQuality_->decode(value).firstStrip << "\t "
	     << " NconsecutiveBadStrips " << SiStripQuality_->decode(value).range  << "\t "
	     << " flag " << SiStripQuality_->decode(value).flag  << "\t "
	     << " packed integer 0x" <<  std::hex << value << std::dec << "\t ";
	}
	
	ss << "\n\nDetBase Info\n\t  IsModuleBad()="<<SiStripQuality_->IsModuleBad(detid[id])
	   << "\t IsFiberBad(n)="<< SiStripQuality_->IsFiberBad(detid[id],0) << " " << SiStripQuality_->IsFiberBad(detid[id],1) << " " << SiStripQuality_->IsFiberBad(detid[id],2)
	   << "\t getBadFibers()=" << SiStripQuality_->getBadFibers(detid[id])
	   << "\t IsApvBad()="<< SiStripQuality_->IsApvBad(detid[id],0) << " " << SiStripQuality_->IsApvBad(detid[id],1) << " " << SiStripQuality_->IsApvBad(detid[id],2) << " " << SiStripQuality_->IsApvBad(detid[id],3) << " " << SiStripQuality_->IsApvBad(detid[id],4) << " " << SiStripQuality_->IsApvBad(detid[id],5)   
	   << "\t getBadApvs()=" << SiStripQuality_->getBadApvs(detid[id]);
	
	edm::LogInfo("testSiStripQualityESProducer") << ss.str();
      }
    ss.str("");
    ss << "Global Info";
    std::vector<SiStripQuality::BadComponent> BC = SiStripQuality_->getBadComponentList();
    ss << "\n\t detid \t IsModuleBad \t BadFibers \t BadApvs";

    for (size_t i=0;i<BC.size();++i)
      ss << "\n\t" << BC[i].detid << "\t " << BC[i].BadModule << "\t "<< BC[i].BadFibers << "\t " << BC[i].BadApvs << "\t ";
    ss<< std::endl;

    edm::LogInfo("testSiStripQualityESProducer") << ss.str();
  }
}
