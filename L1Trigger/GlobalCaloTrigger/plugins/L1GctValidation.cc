#include "L1Trigger/GlobalCaloTrigger/plugins/L1GctValidation.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/DataRecord/interface/L1GctJetFinderParamsRcd.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1HtMissScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1HfRingEtScaleRcd.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <cmath>

L1GctValidation::L1GctValidation(const edm::ParameterSet& iConfig)
    : m_gctinp_tag(iConfig.getUntrackedParameter<edm::InputTag>("rctInputTag", edm::InputTag("rctDigis"))),
      m_energy_tag(iConfig.getUntrackedParameter<edm::InputTag>("gctInputTag", edm::InputTag("gctDigis"))) {}

L1GctValidation::~L1GctValidation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1GctValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Get the scales from the event setup
  ESHandle<L1GctJetFinderParams> jfPars;
  iSetup.get<L1GctJetFinderParamsRcd>().get(jfPars);  // which record?

  double lsbForEt = jfPars.product()->getRgnEtLsbGeV();
  double lsbForHt = jfPars.product()->getHtLsbGeV();

  unsigned httJetThreshold = static_cast<int>(jfPars.product()->getHtJetEtThresholdGeV() / lsbForHt);
  unsigned htmJetThreshold = static_cast<int>(jfPars.product()->getMHtJetEtThresholdGeV() / lsbForHt);

  ESHandle<L1CaloEtScale> htMissScale;
  iSetup.get<L1HtMissScaleRcd>().get(htMissScale);  // which record?
  ESHandle<L1CaloEtScale> hfRingEtScale;
  iSetup.get<L1HfRingEtScaleRcd>().get(hfRingEtScale);  // which record?

  // Get the Gct energy sums from the event
  Handle<L1GctEtTotalCollection> sumEtColl;
  iEvent.getByLabel(m_energy_tag, sumEtColl);
  Handle<L1GctEtHadCollection> sumHtColl;
  iEvent.getByLabel(m_energy_tag, sumHtColl);
  Handle<L1GctEtMissCollection> missEtColl;
  iEvent.getByLabel(m_energy_tag, missEtColl);
  Handle<L1GctHtMissCollection> missHtColl;
  iEvent.getByLabel(m_energy_tag, missHtColl);

  // Get the input calo regions from the event (for checking MEt)
  Handle<L1CaloRegionCollection> inputColl;
  iEvent.getByLabel(m_gctinp_tag, inputColl);

  // Get the internal jet data from the event (for checking Ht)
  Handle<L1GctInternJetDataCollection> internalJetsColl;
  iEvent.getByLabel(m_energy_tag, internalJetsColl);

  double etTot = 0.0;
  for (L1GctEtTotalCollection::const_iterator jbx = sumEtColl->begin(); jbx != sumEtColl->end(); jbx++) {
    if (jbx->bx() == 0) {
      etTot = static_cast<double>(jbx->et());
    }
  }

  double etHad = 0.0;
  for (L1GctEtHadCollection::const_iterator jbx = sumHtColl->begin(); jbx != sumHtColl->end(); jbx++) {
    if (jbx->bx() == 0) {
      etHad = static_cast<double>(jbx->et());
    }
  }

  double etMiss = 0.0;
  double etMAng = 0.0;
  for (L1GctEtMissCollection::const_iterator jbx = missEtColl->begin(); jbx != missEtColl->end(); jbx++) {
    if (jbx->bx() == 0) {
      etMiss = static_cast<double>(jbx->et());
      int phibin = jbx->phi();
      if (phibin >= 36)
        phibin -= 72;
      double etMPhi = static_cast<double>(phibin);

      etMAng = (etMPhi + 0.5) * M_PI / 36.;
    }
  }

  double etTotFromRegions = 0.0;
  double exTotFromRegions = 0.0;
  double eyTotFromRegions = 0.0;
  for (L1CaloRegionCollection::const_iterator jrg = inputColl->begin(); jrg != inputColl->end(); jrg++) {
    if (jrg->bx() == 0) {
      double rgEt = static_cast<double>(jrg->et()) * lsbForEt;
      double rgPhibin = static_cast<double>(jrg->id().iphi());
      double rgPh = (rgPhibin + 0.5) * M_PI / 9.;

      etTotFromRegions += rgEt;
      exTotFromRegions += rgEt * cos(rgPh);
      eyTotFromRegions += rgEt * sin(rgPh);
    }
  }

  double htMissGct = 0.0;
  double htMissAng = 0.0;
  double htMissGeV = 0.0;
  for (L1GctHtMissCollection::const_iterator jbx = missHtColl->begin(); jbx != missHtColl->end(); jbx++) {
    if (jbx->bx() == 0) {
      htMissGct = static_cast<double>(jbx->et());
      htMissGeV = htMissScale->et(jbx->et());
      int phibin = jbx->phi();
      if (phibin >= 9)
        phibin -= 18;
      double htMPhi = static_cast<double>(phibin);
      htMissAng = (htMPhi + 0.5) * M_PI / 9.;
    }
  }

  double htFromJets = 0.0;
  double hxFromJets = 0.0;
  double hyFromJets = 0.0;
  for (L1GctInternJetDataCollection::const_iterator jet = internalJetsColl->begin(); jet != internalJetsColl->end();
       jet++) {
    if (jet->bx() == 0 && !jet->empty()) {
      unsigned jetEtGct = jet->et();
      double jetEt = static_cast<double>(jetEtGct);
      int phibin = jet->regionId().iphi();
      if (phibin >= 9)
        phibin -= 18;
      // The phi bin centres are at 0, 20, 40, ... degrees
      double jetAng = (static_cast<double>(phibin)) * M_PI / 9.;
      if (jetEtGct > httJetThreshold) {
        htFromJets += jetEt;
      }
      if (jetEtGct > htmJetThreshold) {
        hxFromJets += jetEt * cos(jetAng);
        hyFromJets += jetEt * sin(jetAng);
      }
    }
  }

  double dPhiMetMht = deltaPhi(etMAng, htMissAng);

  theSumEtInLsb->Fill(etTot);
  theSumHtInLsb->Fill(etHad);
  theMissEtInLsb->Fill(etMiss);
  theMissHtInLsb->Fill(htMissGct);
  theSumEtInGeV->Fill(etTot * lsbForEt);
  theSumHtInGeV->Fill(etHad * lsbForHt);
  theMissEtInGeV->Fill(etMiss * lsbForEt);
  theMissEtAngle->Fill(etMAng);
  theMissEtVector->Fill(etMiss * lsbForEt * cos(etMAng), etMiss * lsbForEt * sin(etMAng));
  if (htMissGct < 126.5) {
    theMissHtInGeV->Fill(htMissGeV);
    theMissHtAngle->Fill(htMissAng);
    theMissHtVector->Fill(htMissGeV * cos(htMissAng), htMissGeV * sin(htMissAng));
  }

  theSumEtVsInputRegions->Fill(etTot * lsbForEt, etTotFromRegions);
  theMissEtMagVsInputRegions->Fill(etMiss * lsbForEt,
                                   sqrt(exTotFromRegions * exTotFromRegions + eyTotFromRegions * eyTotFromRegions));
  theMissEtAngleVsInputRegions->Fill(etMAng, atan2(-eyTotFromRegions, -exTotFromRegions));
  theMissHtMagVsInputRegions->Fill(htMissGeV,
                                   sqrt(exTotFromRegions * exTotFromRegions + eyTotFromRegions * eyTotFromRegions));

  theMissEtVsMissHt->Fill(etMiss * lsbForEt, htMissGeV);
  theMissEtVsMissHtAngle->Fill(etMAng, htMissAng);
  theDPhiVsMissEt->Fill(dPhiMetMht, etMiss * lsbForEt);
  theDPhiVsMissHt->Fill(dPhiMetMht, htMissGeV);

  theHtVsInternalJetsSum->Fill(etHad * lsbForHt, htFromJets * lsbForHt);
  if (htMissGct < 126.5) {
    theMissHtVsInternalJetsSum->Fill(htMissGeV, sqrt(hxFromJets * hxFromJets + hyFromJets * hyFromJets) * lsbForHt);
    theMissHtPhiVsInternalJetsSum->Fill(htMissAng, atan2(-hyFromJets, -hxFromJets));
    theMissHxVsInternalJetsSum->Fill(htMissGeV * cos(htMissAng), hxFromJets * lsbForHt);
    theMissHyVsInternalJetsSum->Fill(htMissGeV * sin(htMissAng), hyFromJets * lsbForHt);
  }

  // Get minbias trigger quantities from HF
  Handle<L1GctHFRingEtSumsCollection> HFEtSumsColl;
  Handle<L1GctHFBitCountsCollection> HFCountsColl;
  iEvent.getByLabel(m_energy_tag, HFEtSumsColl);
  iEvent.getByLabel(m_energy_tag, HFCountsColl);

  for (L1GctHFRingEtSumsCollection::const_iterator es = HFEtSumsColl->begin(); es != HFEtSumsColl->end(); es++) {
    if (es->bx() == 0) {
      theHfRing0EtSumPositiveEta->Fill(hfRingEtScale->et(es->etSum(0)));
      theHfRing0EtSumNegativeEta->Fill(hfRingEtScale->et(es->etSum(1)));
      theHfRing1EtSumPositiveEta->Fill(hfRingEtScale->et(es->etSum(2)));
      theHfRing1EtSumNegativeEta->Fill(hfRingEtScale->et(es->etSum(3)));
    }
  }

  for (L1GctHFBitCountsCollection::const_iterator bc = HFCountsColl->begin(); bc != HFCountsColl->end(); bc++) {
    if (bc->bx() == 0) {
      theHfRing0CountPositiveEta->Fill(bc->bitCount(0));
      theHfRing0CountNegativeEta->Fill(bc->bitCount(1));
      theHfRing1CountPositiveEta->Fill(bc->bitCount(2));
      theHfRing1CountNegativeEta->Fill(bc->bitCount(3));
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void L1GctValidation::beginJob() {
  edm::Service<TFileService> fs;

  TFileDirectory dir0 = fs->mkdir("L1GctEtSums");

  theSumEtInLsb = dir0.make<TH1F>("SumEtInLsb", "Total Et (GCT units)", 128, 0., 2048.);
  theSumHtInLsb = dir0.make<TH1F>("SumHtInLsb", "Total Ht (GCT units)", 128, 0., 2048.);
  theMissEtInLsb = dir0.make<TH1F>("MissEtInLsb", "Missing Et magnitude (GCT units)", 128, 0., 1024.);
  theMissHtInLsb = dir0.make<TH1F>("MissHtInLsb", "Missing Ht magnitude (GCT units)", 128, 0., 127.);
  theSumEtInGeV = dir0.make<TH1F>("SumEtInGeV", "Total Et (in GeV)", 100, 0., 1000.);
  theSumHtInGeV = dir0.make<TH1F>("SumHtInGeV", "Total Ht (in GeV)", 100, 0., 1000.);
  theMissEtInGeV = dir0.make<TH1F>("MissEtInGeV", "Missing Et magnitude (in GeV)", 100, 0., 500.);
  theMissEtAngle = dir0.make<TH1F>("MissEtAngle", "Missing Et angle", 72, -M_PI, M_PI);
  theMissEtVector = dir0.make<TH2F>("MissEtVector", "Missing Ex vs Missing Ey", 100, -100., 100., 100, -100., 100.);
  theMissHtInGeV = dir0.make<TH1F>("MissHtInGeV", "Missing Ht magnitude (in GeV)", 100, 0., 500.);
  theMissHtAngle = dir0.make<TH1F>("MissHtAngle", "Missing Ht angle", 72, -M_PI, M_PI);
  theMissHtVector = dir0.make<TH2F>("MissHtVector", "Missing Hx vs Missing Hy", 100, -100., 100., 100, -100., 100.);
  theSumEtVsInputRegions =
      dir0.make<TH2F>("SumEtVsInputRegions", "Total Et vs sum of input regions", 100, 0., 1000., 100, 0., 1000.);
  theMissEtMagVsInputRegions = dir0.make<TH2F>(
      "MissEtMagVsInputRegions", "Missing Et magnitude vs sum of input regions", 100, 0., 500., 100, 0., 500.);
  theMissEtAngleVsInputRegions = dir0.make<TH2F>(
      "MissEtAngleVsInputRegions", "Missing Et angle vs sum of input regions", 72, -M_PI, M_PI, 72, -M_PI, M_PI);
  theMissHtMagVsInputRegions = dir0.make<TH2F>(
      "MissHtMagVsInputRegions", "Missing Ht magnitude vs sum of input regions", 100, 0., 500., 100, 0., 500.);
  theMissEtVsMissHt = dir0.make<TH2F>("MissEtVsMissHt", "Missing Et vs Missing Ht", 100, 0., 500., 100, 0., 500.);
  theMissEtVsMissHtAngle = dir0.make<TH2F>(
      "MissEtVsMissHtAngle", "Angle correlation Missing Et vs Missing Ht", 72, -M_PI, M_PI, 72, -M_PI, M_PI);
  theDPhiVsMissEt =
      dir0.make<TH2F>("theDPhiVsMissEt", "Angle difference MET-MHT vs MET magnitude", 72, -M_PI, M_PI, 100, 0., 500.);
  theDPhiVsMissHt =
      dir0.make<TH2F>("theDPhiVsMissHt", "Angle difference MET-MHT vs MHT magnitude", 72, -M_PI, M_PI, 100, 0., 500.);

  theHtVsInternalJetsSum = dir0.make<TH2F>(
      "HtVsInternalJetsSum", "Ht vs scalar sum of jet Et values (in GeV)", 128, 0., 2048., 128, 0., 2048.);
  theMissHtVsInternalJetsSum = dir0.make<TH2F>(
      "MissHtVsInternalJetsSum", "Missing Ht vs vector sum of jet Et values (in GeV)", 128, 0., 512., 128, 0., 512.);
  theMissHtPhiVsInternalJetsSum = dir0.make<TH2F>("MissHtPhiVsInternalJetsSum",
                                                  "Angle correlation Missing Ht vs vector sum of jet Et values",
                                                  72,
                                                  -M_PI,
                                                  M_PI,
                                                  72,
                                                  -M_PI,
                                                  M_PI);
  theMissHxVsInternalJetsSum = dir0.make<TH2F>("MissHxVsInternalJetsSum",
                                               "Missing Ht x component vs sum of jet Et values (in GeV)",
                                               128,
                                               -256.,
                                               256.,
                                               128,
                                               -256.,
                                               256.);
  theMissHyVsInternalJetsSum = dir0.make<TH2F>("MissHyVsInternalJetsSum",
                                               "Missing Ht y component vs sum of jet Et values (in GeV)",
                                               128,
                                               -256.,
                                               256.,
                                               128,
                                               -256.,
                                               256.);

  TFileDirectory dir1 = fs->mkdir("L1GctHfSumsAndJetCounts");

  // Minimum bias triggers from Hf inner rings
  theHfRing0EtSumPositiveEta = dir1.make<TH1F>("HfRing0EtSumPositiveEta", "Hf Inner Ring0 Et eta+", 60, 0., 30.);
  theHfRing0EtSumNegativeEta = dir1.make<TH1F>("HfRing0EtSumNegativeEta", "Hf Inner Ring0 Et eta-", 60, 0., 30.);
  theHfRing1EtSumPositiveEta = dir1.make<TH1F>("HfRing1EtSumPositiveEta", "Hf Inner Ring1 Et eta+", 60, 0., 30.);
  theHfRing1EtSumNegativeEta = dir1.make<TH1F>("HfRing1EtSumNegativeEta", "Hf Inner Ring1 Et eta-", 60, 0., 30.);
  theHfRing0CountPositiveEta = dir1.make<TH1F>("HfRing0CountPositiveEta", "Hf Threshold bits Ring0 eta+", 20, 0., 20.);
  theHfRing0CountNegativeEta = dir1.make<TH1F>("HfRing0CountNegativeEta", "Hf Threshold bits Ring0 eta-", 20, 0., 20.);
  theHfRing1CountPositiveEta = dir1.make<TH1F>("HfRing1CountPositiveEta", "Hf Threshold bits Ring1 eta+", 20, 0., 20.);
  theHfRing1CountNegativeEta = dir1.make<TH1F>("HfRing1CountNegativeEta", "Hf Threshold bits Ring1 eta-", 20, 0., 20.);
}

// ------------ method called once each job just after ending the event loop  ------------
void L1GctValidation::endJob() {}

DEFINE_FWK_MODULE(L1GctValidation);
