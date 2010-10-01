#include "RecoLocalCalo/EcalRecProducers/test/EcalCompactTrigPrimProducerTest.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <iostream>

void EcalCompactTrigPrimProducerTest::analyze(edm::Event const & event, edm::EventSetup const & c){
  edm::Handle<EcalTrigPrimDigiCollection> hTpDigis;
  event.getByLabel(tpDigiColl_, hTpDigis);

  const EcalTrigPrimDigiCollection* trigPrims =  hTpDigis.product();
  
  
  edm::Handle<EcalTrigPrimCompactColl> hTpRecs;
  event.getByLabel(tpRecColl_, hTpRecs);

  const EcalTrigPrimCompactColl* trigPrimRecs =  hTpRecs.product();

  
  int nTps = 0;
  for(EcalTrigPrimDigiCollection::const_iterator trigPrim = trigPrims->begin();
      trigPrim != trigPrims->end(); ++trigPrim){

    int ieta = trigPrim->id().ieta();
    int iphi = trigPrim->id().iphi();
    
    if(trigPrim->compressedEt()!= trigPrimRecs->compressedEt(ieta, iphi)) std::cout << "Different compressed Et\n";
    if(trigPrim->fineGrain()!= trigPrimRecs->fineGrain(ieta, iphi)) std::cout << "Different FGVB\n";
    if(trigPrim->ttFlag()!= trigPrimRecs->ttFlag(ieta, iphi)) std::cout << "Different compressed TTF\n";
    if(trigPrim->l1aSpike()!= trigPrimRecs->l1aSpike(ieta, iphi)) std::cout << "Different compressed L1Spike flag\n";
    if(trigPrim->compressedEt()!=0) ++nCompressEt_;
    if(trigPrim->fineGrain()!=0) ++nFineGrain_;
    if(trigPrim->ttFlag()!=0) ++nTTF_;
    if(trigPrim->l1aSpike()!=0) ++nL1aSpike_;
    ++nTps;
  }
  if(nTps!=4032) std::cout << "Unexpected number of TPs: " << nTps << "\n";
  
}

EcalCompactTrigPrimProducerTest::~EcalCompactTrigPrimProducerTest(){
  std::cout << "# of non-null compressed Et: " << nCompressEt_ << "\n";
  std::cout << "# of non-null FGVB: " << nFineGrain_ << "\n";
  std::cout << "# of non-null TTF: " << nTTF_ << "\n";
  std::cout << "# of non-null L1ASpike: " << nL1aSpike_ << "\n";    
}

#include "FWCore/Framework/interface/MakerMacros.h"  
DEFINE_FWK_MODULE( EcalCompactTrigPrimProducerTest);

