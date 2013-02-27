//#include <vector>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "TH2F.h"
#include "TProfile.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

#include "DPGAnalysis/SiStripTools/interface/DigiCollectionProfiler.h"

DigiCollectionProfiler::DigiCollectionProfiler():
  m_nevent(0),
  m_tibprof(0),m_tobprof(0),m_tecpprof(0),m_tecmprof(0),
  m_tib2d(0),m_tob2d(0),m_tecp2d(0),m_tecm2d(0),m_maskedmod() { }

DigiCollectionProfiler::DigiCollectionProfiler(TProfile* tibprof,
					       TProfile* tobprof,
					       TProfile* tecpprof,
					       TProfile* tecmprof,
					       TH2F* tib2d,
					       TH2F* tob2d,
					       TH2F* tecp2d,
					       TH2F* tecm2d):
  m_nevent(0),
  m_tibprof(tibprof),m_tobprof(tobprof),m_tecpprof(tecpprof),m_tecmprof(tecmprof),
  m_tib2d(tib2d),m_tob2d(tob2d),m_tecp2d(tecp2d),m_tecm2d(tecm2d),m_maskedmod() { }

void DigiCollectionProfiler::analyze(edm::Handle<edm::DetSetVector<SiStripDigi> > digis) {

  m_nevent++;

  for(edm::DetSetVector<SiStripDigi>::const_iterator mod = digis->begin();mod!=digis->end();mod++) {

    const SiStripDetId detid(mod->detId());

    if(!ismasked(detid)) {
      TProfile* tobefilledprof=0;
      TH2F* tobefilled2d=0;
      if(detid.subDetector()==SiStripDetId::TIB || detid.subDetector()==SiStripDetId::TID) {
	tobefilledprof=m_tibprof;
	tobefilled2d=m_tib2d;
      }
      else if(detid.subDetector()==SiStripDetId::TOB) {
	tobefilledprof=m_tobprof;
	tobefilled2d=m_tob2d;
      }
      else if(detid.subDetector()==SiStripDetId::TEC) {
	const TECDetId tecid(detid);
	if(tecid.side()==1) {
	  tobefilledprof=m_tecpprof;
	  tobefilled2d=m_tecp2d;
	}
	else if(tecid.side()==2) {
	  tobefilledprof=m_tecmprof;
	  tobefilled2d=m_tecm2d;
	}
      }
      
      for(edm::DetSet<SiStripDigi>::const_iterator digi=mod->begin();digi!=mod->end();digi++) {
	
	if(digi->adc()>0) {
	  if(tobefilledprof) tobefilledprof->Fill(digi->strip()%256,digi->adc());
	  if(tobefilled2d) tobefilled2d->Fill(digi->strip()%256,digi->adc());
	}
      }
    }
  }
}

void DigiCollectionProfiler::setMaskedModules(std::vector<unsigned int> maskedmod) {
  
  m_maskedmod = maskedmod;
  sort(m_maskedmod.begin(),m_maskedmod.end());
  
}


int DigiCollectionProfiler::ismasked(const DetId& mod) const {

  int masked=0;
  for(std::vector<unsigned int>::const_iterator it=m_maskedmod.begin();it!=m_maskedmod.end()&&masked==0&&mod>=(*it);it++) {
    if(mod.rawId() == (*it)) masked = 1;
  }
  return masked;

}
