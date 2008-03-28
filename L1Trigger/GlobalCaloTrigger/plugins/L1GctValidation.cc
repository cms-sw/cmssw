#include "L1Trigger/GlobalCaloTrigger/plugins/L1GctValidation.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/L1TObjects/interface/L1GctJetEtCalibrationFunction.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1GctJetCalibFunRcd.h"

#include <math.h>

L1GctValidation::L1GctValidation(const edm::ParameterSet& iConfig) :
   m_energy_tag(iConfig.getUntrackedParameter<edm::InputTag>("inputTag",edm::InputTag("gctDigis")))
{
}


L1GctValidation::~L1GctValidation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
L1GctValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

  // Get the scales from the event setup
  ESHandle< L1GctJetEtCalibrationFunction > calibFun ;
  iSetup.get< L1GctJetCalibFunRcd >().get( calibFun ) ; // which record?
  ESHandle< L1CaloEtScale > etScale ;
  iSetup.get< L1JetEtScaleRcd >().get( etScale ) ; // which record?

  double lsbForEt = etScale.product()->linearLsb();
  double lsbForHt = calibFun.product()->getHtScaleLSB();

  // Get the Gct energy sums from the event
  Handle< L1GctEtTotalCollection > sumEtColl ;
  iEvent.getByLabel( m_energy_tag, sumEtColl ) ;
  Handle< L1GctEtHadCollection >   sumHtColl ;
  iEvent.getByLabel( m_energy_tag, sumHtColl ) ;
  Handle< L1GctEtMissCollection >  missEtColl ;
  iEvent.getByLabel( m_energy_tag, missEtColl ) ;

  double etTot = 0.0;
  for (L1GctEtTotalCollection::const_iterator jbx=sumEtColl->begin(); jbx!=sumEtColl->end(); jbx++) {
    if (jbx->bx()==0) { etTot  = static_cast<double>(jbx->et()); }
  } 

  double etHad  = 0.0;
  for (L1GctEtHadCollection::const_iterator jbx=sumHtColl->begin(); jbx!=sumHtColl->end(); jbx++) {
    if (jbx->bx()==0) { etHad  = static_cast<double>(jbx->et()); }
  }

  double etMiss = 0.0;
  double etMAng = 0.0;
  for (L1GctEtMissCollection::const_iterator jbx=missEtColl->begin(); jbx!=missEtColl->end(); jbx++) {
    if (jbx->bx()==0) {
      etMiss = static_cast<double>(jbx->et());
      int phibin = jbx->phi();
      if (phibin>=36) phibin -= 72;
      double etMPhi = static_cast<double>(phibin);

      etMAng = (etMPhi+0.5)*M_PI/36.;
    }
  }

  theSumEtInLsb->Fill(etTot);
  theSumHtInLsb->Fill(etHad);
  theMissEtInLsb->Fill(etMiss);
  theSumEtInGeV->Fill(etTot*lsbForEt);
  theSumHtInGeV->Fill(etHad*lsbForHt);
  theMissEtInGeV->Fill(etMiss*lsbForEt);
  theMissEtAngle->Fill(etMAng);
  theMissEtVector->Fill(etMiss*lsbForEt*cos(etMAng),etMiss*lsbForEt*sin(etMAng));

  // Get jet counts from the event
  Handle< L1GctJetCountsCollection > jetCountColl ;
  iEvent.getByLabel( m_energy_tag, jetCountColl ) ;

  for (L1GctJetCountsCollection::const_iterator jbx=jetCountColl->begin(); jbx!=jetCountColl->end(); jbx++) {
    for (unsigned jc=0; jc<L1GctJetCounts::MAX_TOTAL_COUNTS; jc++) {
      theJetCounts.at(jc)->Fill(jbx->count(jc));
    }

    theHfRing0EtSumPositiveEta->Fill(jbx->hfRing0EtSumPositiveEta());  
    theHfRing0EtSumNegativeEta->Fill(jbx->hfRing0EtSumNegativeEta());  
    theHfRing1EtSumPositiveEta->Fill(jbx->hfRing1EtSumPositiveEta());  
    theHfRing1EtSumNegativeEta->Fill(jbx->hfRing1EtSumNegativeEta());  
    theHfTowerCountPositiveEta->Fill(jbx->hfTowerCountPositiveEta());  
    theHfTowerCountNegativeEta->Fill(jbx->hfTowerCountNegativeEta());  
  }

}

// ------------ method called once each job just before starting event loop  ------------
void 
L1GctValidation::beginJob(const edm::EventSetup&)
{
  edm::Service<TFileService> fs;

  TFileDirectory dir0 = fs->mkdir("L1GctEtSums");

  theSumEtInLsb   = dir0.make<TH1F>("SumEtInLsb",   "Total Et (GCT units)",
                                    128, 0., 2048.); 
  theSumHtInLsb   = dir0.make<TH1F>("SumHtInLsb",   "Total Ht (GCT units)",
                                    128, 0., 2048.); 
  theMissEtInLsb  = dir0.make<TH1F>("MissEtInLsb",  "Missing Et magnitude (GCT units)",
                                    128, 0., 1024.); 
  theSumEtInGeV   = dir0.make<TH1F>("SumEtInGeV",   "Total Et (in GeV)",
                                    100, 0., 1000.); 
  theSumHtInGeV   = dir0.make<TH1F>("SumHtInGeV",   "Total Ht (in GeV)",
                                    100, 0., 1000.); 
  theMissEtInGeV  = dir0.make<TH1F>("MissEtInGeV",  "Missing Et magnitude (in GeV)",
                                    100, 0., 500.); 
  theMissEtAngle  = dir0.make<TH1F>("MissEtAngle",  "Missing Et angle",
                                    72, -M_PI, M_PI);
  theMissEtVector = dir0.make<TH2F>("MissEtVector", "Missing Ex vs Missing Ey",
                                    100, -100., 100., 100, -100., 100.); 

  TFileDirectory dir1 = fs->mkdir("L1GctJetCounts");

  for (unsigned jc=0; jc<L1GctJetCounts::MAX_TOTAL_COUNTS; jc++) {
    std::stringstream ss;
    std::string title;
    std::string header;
    ss << "JetCount#" << jc;
    ss >> title;
    ss << "Jet Count number " << jc;
    if (jc== 6 || jc== 7) { ss << " (Hf tower count)"; }
    if (jc== 8 || jc== 9) { ss << " (Hf Et0 sum MSB)"; }
    if (jc==10 || jc==11) { ss << " (Hf Et1 sum MSB)"; }
    ss >> header;
    theJetCounts.push_back(dir1.make<TH1F>(title.c_str(), header.c_str(), 32, 0., 32.));
  }

  theHfRing0EtSumPositiveEta = dir1.make<TH1F>("HfRing0EtSumPositiveEta", "Hf Inner Ring0 Et eta+",
                                               60, 0., 30.);
  theHfRing0EtSumNegativeEta = dir1.make<TH1F>("HfRing0EtSumNegativeEta", "Hf Inner Ring0 Et eta-",
                                               60, 0., 30.);
  theHfRing1EtSumPositiveEta = dir1.make<TH1F>("HfRing1EtSumPositiveEta", "Hf Inner Ring1 Et eta+",
                                               60, 0., 30.);
  theHfRing1EtSumNegativeEta = dir1.make<TH1F>("HfRing1EtSumNegativeEta", "Hf Inner Ring1 Et eta-",
                                               60, 0., 30.);
  theHfTowerCountPositiveEta = dir1.make<TH1F>("HfTowerCountPositiveEta", "Hf Threshold bits eta+",
                                               20, 0., 20.);
  theHfTowerCountNegativeEta = dir1.make<TH1F>("HfTowerCountNegativeEta", "Hf Threshold bits eta-",
                                               20, 0., 20.);
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1GctValidation::endJob() {
}

DEFINE_ANOTHER_FWK_MODULE(L1GctValidation);

