#include "RecoLocalCalo/EcalRecProducers/test/EcalCompactTrigPrimProducerTest.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <iostream>
#include <iomanip>

void EcalCompactTrigPrimProducerTest::analyze(edm::Event const & event, edm::EventSetup const & c){
  edm::Handle<EcalTrigPrimDigiCollection> hTpDigis;
  event.getByLabel(tpDigiColl_, hTpDigis);

  const EcalTrigPrimDigiCollection* trigPrims =  hTpDigis.product();
  EcalTrigPrimDigiCollection trigPrims_ = *trigPrims;
  
  edm::Handle<EcalTrigPrimCompactColl> hTpRecs;
  event.getByLabel(tpRecColl_, hTpRecs);

  const EcalTrigPrimCompactColl* trigPrimRecs =  hTpRecs.product();

  
  int nTps = 0;
  err_ = false;
  for(EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims->begin();
      trigPrim != trigPrims->end(); ++trigPrim){
    
    const EcalTrigTowerDetId& ttId = trigPrim->id();
    
    if(trigPrim->compressedEt()!= trigPrimRecs->compressedEt(ttId)) err("Different compressed Et\n");
    if(trigPrim->fineGrain()!= trigPrimRecs->fineGrain(ttId)) err("Different FGVB\n");
    if(trigPrim->ttFlag()!= trigPrimRecs->ttFlag(ttId)) err("Different compressed TTF\n");
    if(trigPrim->l1aSpike()!= trigPrimRecs->l1aSpike(ttId)) err("Different compressed L1Spike flag\n");
    if(trigPrim->compressedEt()!=0) ++nCompressEt_;
    if(trigPrim->fineGrain()!=0) ++nFineGrain_;
    if(trigPrim->ttFlag()!=0) ++nTTF_;
    if(trigPrim->l1aSpike()!=0) ++nL1aSpike_;
    ++nTps;
  }
  if(nTps!=4032) err("Unexpected number of TPs: ") << nTps << "\n";

  if(!err_) std::cout << "Compact trigger primitive collection is OK.\n";
  
  if(err_){
    std::cout << "Cannot check compact to legacy collection convertion because of previous failure\n";
  } else{
    //test compact to legacy collection convertion:
    EcalTrigPrimDigiCollection col2;
    trigPrimRecs->toEcalTrigPrimDigiCollection(col2);
    if(col2.size()!= trigPrims->size()){
      err("Collection size error!\n");
      err_ = true;
    } else{
      EcalTrigPrimDigiCollection::const_iterator trigPrim2 = col2.begin();
      for(EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims->begin();
	  trigPrim != trigPrims->end() && !err_; ++trigPrim, ++trigPrim2){
	if((trigPrim->sample(trigPrim->sampleOfInterest()).raw())
	   != trigPrim2->sample(0).raw()) err("Trig prim differs: ") << *trigPrim
								     << " (" << std::hex
								     << (trigPrim->sample(trigPrim->sampleOfInterest()).raw())
								     << std::dec << ") "
								     << "  --  " << *trigPrim2
								     << " (" << std::hex
								     << (trigPrim2->sample(trigPrim2->sampleOfInterest()).raw())
								     << std::dec << ") "
								     << "\n"; //err_ = true;
      }
    }
    std::cout << "Validation of compact-to-legacy trigger primitive collection conversion "
	      << (err_ ? "failed" : "succeeded" ) << "\n"; 
  }
}

std::ostream& EcalCompactTrigPrimProducerTest::err(const char* mess){
  err_ = true;
  std::cout << mess;
  return std::cout;
}

EcalCompactTrigPrimProducerTest::~EcalCompactTrigPrimProducerTest(){
  std::cout << "# of non-null compressed Et: " << nCompressEt_ << "\n";
  std::cout << "# of non-null FGVB: " << nFineGrain_ << "\n";
  std::cout << "# of non-null TTF: " << nTTF_ << "\n";
  std::cout << "# of non-null L1ASpike: " << nL1aSpike_ << "\n";    
}

#include "FWCore/Framework/interface/MakerMacros.h"  
DEFINE_FWK_MODULE( EcalCompactTrigPrimProducerTest);

