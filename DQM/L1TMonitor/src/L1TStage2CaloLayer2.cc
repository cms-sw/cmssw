#include "DQM/L1TMonitor/interface/L1TStage2CaloLayer2.h"

L1TStage2CaloLayer2::L1TStage2CaloLayer2(const edm::ParameterSet & ps) :
  monitorDir_(ps.getUntrackedParameter<std::string>("monitorDir","")),
  stage2CaloLayer2JetToken_(consumes<l1t::JetBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2JetSource"))),
  stage2CaloLayer2EGammaToken_(consumes<l1t::EGammaBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2EGammaSource"))),
  stage2CaloLayer2TauToken_(consumes<l1t::TauBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2TauSource"))),
  stage2CaloLayer2EtSumToken_(consumes<l1t::EtSumBxCollection>(ps.getParameter<edm::InputTag>("stage2CaloLayer2EtSumSource"))),
  verbose_(ps.getUntrackedParameter < bool > ("verbose", false))
{
}

L1TStage2CaloLayer2::~L1TStage2CaloLayer2()
{}

void L1TStage2CaloLayer2::bookHistograms(DQMStore::ConcurrentBooker &booker, edm::Run const&, edm::EventSetup const&, calolayer2dqm::Histograms &histograms) const
{
  
  //central jet
  booker.setCurrentFolder(monitorDir_+"/Central-Jets");

  histograms.stage2CaloLayer2CenJetEtEtaPhi = booker.book2D("CenJetsEtEtaPhi", "CENTRAL JET E_{T} ETA PHI; Jet i#eta; Jet i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2CenJetEta = booker.book1D("CenJetsEta", "CENTRAL JET ETA; Jet i#eta; Counts", 229, -114.5, 114.5);
  histograms.stage2CaloLayer2CenJetPhi = booker.book1D("CenJetsPhi", "CENTRAL JET PHI; Jet i#phi; Counts", 144, -0.5, 143.5);
  histograms.stage2CaloLayer2CenJetRank = booker.book1D("CenJetsRank", "CENTRAL JET E_{T}; Jet iE_{T}; Counts", 2048, -0.5, 2047.5);
  histograms.stage2CaloLayer2CenJetOcc = booker.book2D("CenJetsOcc", "CENTRAL JET OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2CenJetBxOcc = booker.book2D("CenJetsBxOcc", "CENTRAL JET BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 512, -0.5, 2047.5);
  histograms.stage2CaloLayer2CenJetQual = booker.book1D("CenJetsQual", "CENTRAL JET QUALITY; Quality; Counts", 32, -0.5, 31.5);


  //forward jet
  booker.setCurrentFolder(monitorDir_+"/Forward-Jets");

  histograms.stage2CaloLayer2ForJetEtEtaPhi = booker.book2D("ForJetsEtEtaPhi", "FORWARD JET E_{T} ETA PHI; Jet i#eta; Jet i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2ForJetEta = booker.book1D("ForJetsEta", "FORWARD JET ETA; Jet i#eta; Counts", 229, -114.5, 114.5);
  histograms.stage2CaloLayer2ForJetPhi = booker.book1D("ForJetsPhi", "FORWARD JET PHI; Jet i#phi; Counts", 144, -0.5, 143.5);
  histograms.stage2CaloLayer2ForJetRank = booker.book1D("ForJetsRank", "FORWARD JET E_{T}; Jet iE_{T}; Counts", 2048, -0.5, 2047.5);
  histograms.stage2CaloLayer2ForJetOcc = booker.book2D("ForJetsOcc", "FORWARD JET OCCUPANCY; Jet i#eta; Jet i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2ForJetBxOcc = booker.book2D("ForJetsBxOcc", "FORWARD JET BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 512, -0.5, 2047.5);
  histograms.stage2CaloLayer2ForJetQual = booker.book1D("ForJetsQual", "FORWARD JET QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //IsoEG
  booker.setCurrentFolder(monitorDir_+"/Isolated-EG");

  histograms.stage2CaloLayer2EGIso = booker.book1D("EGIso", "EG ISO; Iso Flag; Counts", 4, -0.5, 3.5);
  histograms.stage2CaloLayer2IsoEGEtEtaPhi = booker.book2D("IsoEGsEtEtaPhi", "ISO EG E_{T} ETA PHI; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2IsoEGEta = booker.book1D("IsoEGsEta", "ISO EG ETA; EG i#eta; Counts", 229, -114.5, 114.5);
  histograms.stage2CaloLayer2IsoEGPhi = booker.book1D("IsoEGsPhi", "ISO EG PHI; EG i#phi; Counts", 144, -0.5, 143.5);
  histograms.stage2CaloLayer2IsoEGRank = booker.book1D("IsoEGsRank", "ISO EG E_{T}; EG iE_{T}; Counts", 512, -0.5, 511.5);
  histograms.stage2CaloLayer2IsoEGOcc = booker.book2D("IsoEGsOcc", "ISO EG OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2IsoEGBxOcc = booker.book2D("IsoEGsBxOcc", "ISO EG BX OCCUPANCY; BX; Counts",  5,-2.5, 2.5, 128, -0.5, 511.5);
  histograms.stage2CaloLayer2IsoEGQual = booker.book1D("IsoEGsQual", "ISO EG QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //NonIsoEG
  booker.setCurrentFolder(monitorDir_+"/NonIsolated-EG");

  histograms.stage2CaloLayer2NonIsoEGEtEtaPhi = booker.book2D("NonIsoEGsEtEtaPhi", "NonISO EG E_{T} ETA PHI; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2NonIsoEGEta = booker.book1D("NonIsoEGsEta", "NonISO EG ETA; NonISO EG i#eta; Counts", 229, -114.5, 114.5);
  histograms.stage2CaloLayer2NonIsoEGPhi = booker.book1D("NonIsoEGsPhi", "NonISO EG PHI; NonISO EG i#phi; Counts", 144, -0.5, 143.5);
  histograms.stage2CaloLayer2NonIsoEGRank = booker.book1D("NonIsoEGsRank", "NonISO EG E_{T}; NonISO EG iE_{T}; Counts", 512, -0.5, 511.5);
  histograms.stage2CaloLayer2NonIsoEGOcc = booker.book2D("NonIsoEGsOcc", "NonISO EG OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2NonIsoEGBxOcc = booker.book2D("NonIsoEGsBxOcc", "NonISO EG BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 128, -0.5, 511.5);
  histograms.stage2CaloLayer2NonIsoEGQual = booker.book1D("NonIsoEGsQual", "NonISO EG QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //IsoTau
  booker.setCurrentFolder(monitorDir_+"/Isolated-Tau");

  histograms.stage2CaloLayer2TauIso = booker.book1D("TauIso", "Tau ISO; Iso Flag; Counts", 4, -0.5, 3.5);
  histograms.stage2CaloLayer2IsoTauEtEtaPhi = booker.book2D("IsoTausEtEtaPhi", "ISO Tau E_{T} ETA PHI; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2IsoTauEta = booker.book1D("IsoTausEta", "ISO Tau ETA; ISO Tau i#eta; Counts", 229, -114.5, 114.5);
  histograms.stage2CaloLayer2IsoTauPhi = booker.book1D("IsoTausPhi", "ISO Tau PHI; ISO Tau i#phi; Counts", 144, -0.5, 143.5);
  histograms.stage2CaloLayer2IsoTauRank = booker.book1D("IsoTausRank", "ISO Tau E_{T}; ISO Tau iE_{T}; Counts", 512, -0.5, 511.5);
  histograms.stage2CaloLayer2IsoTauOcc = booker.book2D("IsoTausOcc", "ISO Tau OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2IsoTauBxOcc = booker.book2D("IsoTausBxOcc", "ISO Tau BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 128, -0.5, 511.5);
  histograms.stage2CaloLayer2IsoTauQual = booker.book1D("IsoTausQual", "ISO Tau QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //NonIsoTau
  booker.setCurrentFolder(monitorDir_+"/NonIsolated-Tau");

  histograms.stage2CaloLayer2TauEtEtaPhi = booker.book2D("TausEtEtaPhi", "Tau E_{T} ETA PHI; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2TauEta = booker.book1D("TausEta", "NonISO Tau ETA; Tau i#eta; Counts", 229, -114.5, 114.5);
  histograms.stage2CaloLayer2TauPhi = booker.book1D("TausPhi", "NonISO Tau PHI; Tau i#phi; Counts", 144, -0.5, 143.5);
  histograms.stage2CaloLayer2TauRank = booker.book1D("TausRank", "NonISO Tau E_{T}; Tau iE_{T}; Counts", 512, -0.5, 511.5);
  histograms.stage2CaloLayer2TauOcc = booker.book2D("TausOcc", "NonISO Tau OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  histograms.stage2CaloLayer2TauBxOcc = booker.book2D("TausBxOcc", "NonISO Tau BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 128, -0.5, 511.5);
  histograms.stage2CaloLayer2TauQual = booker.book1D("TausQual", "NonISO Tau QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //EtSums
  booker.setCurrentFolder(monitorDir_+"/Energy-Sums");

  histograms.stage2CaloLayer2EtSumBxOcc = booker.book2D("EtSumBxOcc", "EtSum BX OCCUPANCY; BX; Counts",  5,-2.5, 2.5, 1024, -0.5, 4095.5);
  histograms.stage2CaloLayer2METRank = booker.book1D("METRank", "MET E_{T}; iE_{T}; Counts", 4096, -0.5, 4095.5);
  histograms.stage2CaloLayer2METPhi = booker.book1D("METPhi", "MET Phi; MET i#Phi; Counts", 144, -0.5, 143.5);
  histograms.stage2CaloLayer2ETTRank = booker.book1D("ETTRank", "ETT E_{T}; ETT iE_{T}; Counts", 4096, -0.5, 4095.5);
  histograms.stage2CaloLayer2MHTRank = booker.book1D("MHTRank", "MHT E_{T}; MHT iE_{T}; Counts", 4096, -0.5, 4095.5);
  histograms.stage2CaloLayer2MHTPhi = booker.book1D("MHTPhi", "MHT Phi; MHT i#phi; Counts", 144, -0.5, 143.5);
  histograms.stage2CaloLayer2HTTRank = booker.book1D("HTTRank", "HTT E_{T}; HTT iE_{T}; Counts", 4096, -0.5, 4095.5);
  histograms.stage2CaloLayer2METHFRank = booker.book1D("METHFRank", "METHF E_{T}; METHF iE_{T}; Counts", 4096, -0.5, 4095.5);
  histograms.stage2CaloLayer2METHFPhi = booker.book1D("METHFPhi", "METHF Phi; METHF i#phi; Counts", 144, -0.5, 143.5);
  // histograms.stage2CaloLayer2ETTHFRank = booker.book1D("ETTHFRank", "ETTHF E_{T}", 4096, -0.5, 4095.5);
  histograms.stage2CaloLayer2MHTHFRank = booker.book1D("MHTHFRank", "MHTHF E_{T}; MHTHF iE_{T}; Counts", 4096, -0.5, 4095.5);
  histograms.stage2CaloLayer2MHTHFPhi = booker.book1D("MHTHFPhi", "MHTHF Phi; MHTHF i#phi; Counts", 144, -0.5, 143.5);
  // histograms.stage2CaloLayer2HTTHFRank = booker.book1D("HTTHFRank", "HTTHF E_{T}", 4096, -0.5, 4095.5);
  histograms.stage2CaloLayer2ETTEMRank = booker.book1D("ETTEMRank", "ETTEM E_{T}; ETTEM iE_{T}; Counts", 4096, -0.5, 4095.5);

  histograms.stage2CaloLayer2MinBiasHFP0 = booker.book1D("MinBiasHFP0", "HF Min Bias Sum Threshold 0, Positive #eta; N_{towers}; Events", 16, -0.5, 15.5);
  histograms.stage2CaloLayer2MinBiasHFM0 = booker.book1D("MinBiasHFM0", "HF Min Bias Sum Threshold 1, Nevagive #eta; N_{towers}; Events", 16, -0.5, 15.5);
  histograms.stage2CaloLayer2MinBiasHFP1 = booker.book1D("MinBiasHFP1", "HF Min Bias Sum Threshold 1, Positive #eta; N_{towers}; Events", 16, -0.5, 15.5);
  histograms.stage2CaloLayer2MinBiasHFM1 = booker.book1D("MinBiasHFM1", "HF Min Bias Sum Threshold 1, Negative #eta; N_{towers}; Events", 16, -0.5, 15.5);

  histograms.stage2CaloLayer2TowCount = booker.book1D("TowCount", "Count of Trigger towers above threshold; N_{towers}; Events", 5904, -0.5, 5903.5);
}

void L1TStage2CaloLayer2::dqmAnalyze(const edm::Event & e, const edm::EventSetup & c, const calolayer2dqm::Histograms & histograms) const
{
  if (verbose_) {
    edm::LogInfo("L1TStage2CaloLayer2") << "L1TStage2CaloLayer2: analyze...." << std::endl;
  }

  // analyze Jet
  edm::Handle<l1t::JetBxCollection> Jet;
  e.getByToken(stage2CaloLayer2JetToken_,Jet);

  for(int itBX=Jet->getFirstBX(); itBX<=Jet->getLastBX(); ++itBX){
    for(l1t::JetBxCollection::const_iterator itJet = Jet->begin(itBX); itJet != Jet->end(itBX); ++itJet){
      const bool forward = (itJet->hwEta()>68 || itJet->hwEta()<(-68));
      if(forward){
        histograms.stage2CaloLayer2ForJetBxOcc.fill(itBX, itJet->hwPt());
        if(itBX == 0 ){
          histograms.stage2CaloLayer2ForJetRank.fill(itJet->hwPt());
          if (itJet->hwPt() !=0 ){
            histograms.stage2CaloLayer2ForJetEtEtaPhi.fill(itJet->hwEta(), itJet->hwPhi(), itJet->hwPt());
            histograms.stage2CaloLayer2ForJetOcc.fill(itJet->hwEta(), itJet->hwPhi());
            histograms.stage2CaloLayer2ForJetPhi.fill(itJet->hwPhi());
            histograms.stage2CaloLayer2ForJetEta.fill(itJet->hwEta());
            histograms.stage2CaloLayer2ForJetQual.fill(itJet->hwQual());
          }
        }
      }
      else{
        histograms.stage2CaloLayer2CenJetBxOcc.fill(itBX, itJet->hwPt());
        if(itBX == 0 ){
          histograms.stage2CaloLayer2CenJetRank.fill(itJet->hwPt());
          if (itJet->hwPt() !=0 ){
            histograms.stage2CaloLayer2CenJetEtEtaPhi.fill(itJet->hwEta(), itJet->hwPhi(), itJet->hwPt());
            histograms.stage2CaloLayer2CenJetOcc.fill(itJet->hwEta(), itJet->hwPhi());
            histograms.stage2CaloLayer2CenJetPhi.fill(itJet->hwPhi());
            histograms.stage2CaloLayer2CenJetEta.fill(itJet->hwEta());
            histograms.stage2CaloLayer2CenJetQual.fill(itJet->hwQual());
          }
        }
      }
    }
  }

  //analyze EGamma
  edm::Handle<l1t::EGammaBxCollection> EGamma;
  e.getByToken(stage2CaloLayer2EGammaToken_,EGamma);
  
  for(int itBX=EGamma->getFirstBX(); itBX<=EGamma->getLastBX(); ++itBX){
    for(l1t::EGammaBxCollection::const_iterator itEG = EGamma->begin(itBX); itEG != EGamma->end(itBX); ++itEG){
      bool iso = itEG->hwIso();
      if(iso){
        histograms.stage2CaloLayer2IsoEGBxOcc.fill(itBX, itEG->hwPt());
        if(itBX == 0 ){
          histograms.stage2CaloLayer2IsoEGRank.fill(itEG->hwPt());
          if(itEG->hwPt() !=0 ){
            histograms.stage2CaloLayer2IsoEGEtEtaPhi.fill(itEG->hwEta(), itEG->hwPhi(), itEG->hwPt());
            histograms.stage2CaloLayer2IsoEGOcc.fill(itEG->hwEta(), itEG->hwPhi());
            histograms.stage2CaloLayer2IsoEGPhi.fill(itEG->hwPhi());
            histograms.stage2CaloLayer2IsoEGEta.fill(itEG->hwEta());
            histograms.stage2CaloLayer2IsoEGQual.fill(itEG->hwQual());
            histograms.stage2CaloLayer2EGIso.fill(itEG->hwIso());
          }
        }
      }
      else{
        histograms.stage2CaloLayer2NonIsoEGBxOcc.fill(itBX, itEG->hwPt());
        if(itBX == 0 ){
          histograms.stage2CaloLayer2NonIsoEGRank.fill(itEG->hwPt());
          if(itEG->hwPt() !=0 ){
            histograms.stage2CaloLayer2NonIsoEGEtEtaPhi.fill(itEG->hwEta(), itEG->hwPhi(), itEG->hwPt());
            histograms.stage2CaloLayer2NonIsoEGOcc.fill(itEG->hwEta(), itEG->hwPhi());
            histograms.stage2CaloLayer2NonIsoEGPhi.fill(itEG->hwPhi());
            histograms.stage2CaloLayer2NonIsoEGEta.fill(itEG->hwEta());
            histograms.stage2CaloLayer2NonIsoEGQual.fill(itEG->hwQual());
            histograms.stage2CaloLayer2EGIso.fill(itEG->hwIso());
          }
        }
      }
    }
  }

  //analyze Tau
  edm::Handle<l1t::TauBxCollection> Tau;
  e.getByToken(stage2CaloLayer2TauToken_,Tau);
  for(int itBX=Tau->getFirstBX(); itBX<=Tau->getLastBX(); ++itBX){
    for(l1t::TauBxCollection::const_iterator itTau = Tau->begin(itBX); itTau != Tau->end(itBX); ++itTau){
      bool iso = itTau->hwIso();
      if(iso){
        histograms.stage2CaloLayer2IsoTauBxOcc.fill(itBX, itTau->hwPt());
        if(itBX == 0 ){
          histograms.stage2CaloLayer2IsoTauRank.fill(itTau->hwPt());
          if(itTau->hwPt() !=0 ){
            histograms.stage2CaloLayer2IsoTauEtEtaPhi.fill(itTau->hwEta(), itTau->hwPhi(), itTau->hwPt());
            histograms.stage2CaloLayer2IsoTauOcc.fill(itTau->hwEta(), itTau->hwPhi());
            histograms.stage2CaloLayer2IsoTauPhi.fill(itTau->hwPhi());
            histograms.stage2CaloLayer2IsoTauEta.fill(itTau->hwEta());
            histograms.stage2CaloLayer2IsoTauQual.fill(itTau->hwQual());
            histograms.stage2CaloLayer2TauIso.fill(itTau->hwIso());
          }
        }
      }
      else{
        histograms.stage2CaloLayer2TauBxOcc.fill(itBX, itTau->hwPt());
        if(itBX == 0 ){
          histograms.stage2CaloLayer2TauRank.fill(itTau->hwPt());
          if(itTau->hwPt() !=0 ){
            histograms.stage2CaloLayer2TauEtEtaPhi.fill(itTau->hwEta(), itTau->hwPhi(), itTau->hwPt());
            histograms.stage2CaloLayer2TauOcc.fill(itTau->hwEta(), itTau->hwPhi());
            histograms.stage2CaloLayer2TauPhi.fill(itTau->hwPhi());
            histograms.stage2CaloLayer2TauEta.fill(itTau->hwEta());
            histograms.stage2CaloLayer2TauQual.fill(itTau->hwQual());
            histograms.stage2CaloLayer2TauIso.fill(itTau->hwIso());
          }
        }
      }
    }
  }

  //energy sum
  edm::Handle<l1t::EtSumBxCollection> EtSum;
  e.getByToken(stage2CaloLayer2EtSumToken_,EtSum);
  for(int itBX=EtSum->getFirstBX(); itBX<=EtSum->getLastBX(); ++itBX){
    for(l1t::EtSumBxCollection::const_iterator itEtSum = EtSum->begin(itBX); itEtSum != EtSum->end(itBX); ++itEtSum){
      histograms.stage2CaloLayer2EtSumBxOcc.fill(itBX, itEtSum->hwPt());

      if (itBX==0){
        if(l1t::EtSum::EtSumType::kMissingEt == itEtSum->getType()){;          // MET
          histograms.stage2CaloLayer2METRank.fill(itEtSum->hwPt());
          histograms.stage2CaloLayer2METPhi.fill(itEtSum->hwPhi());
        } else if(l1t::EtSum::EtSumType::kMissingEtHF == itEtSum->getType()){  // METHF
          histograms.stage2CaloLayer2METHFRank.fill(itEtSum->hwPt());
          histograms.stage2CaloLayer2METHFPhi.fill(itEtSum->hwPhi());
        } else if(l1t::EtSum::EtSumType::kTotalEt == itEtSum->getType()){      // ETT
          histograms.stage2CaloLayer2ETTRank.fill(itEtSum->hwPt());
          //} else if(l1t::EtSum::EtSumType::kTotalEtHF == itEtSum->getType()){    // ETTHF
          // histograms.stage2CaloLayer2ETTHFRank.fill(itEtSum->hwPt());
        } else if(l1t::EtSum::EtSumType::kMissingHt == itEtSum->getType()){    // MHT
          histograms.stage2CaloLayer2MHTRank.fill(itEtSum->hwPt());
          histograms.stage2CaloLayer2MHTPhi.fill(itEtSum->hwPhi());
        } else if(l1t::EtSum::EtSumType::kMissingHtHF == itEtSum->getType()){  // MHTHF
          histograms.stage2CaloLayer2MHTHFRank.fill(itEtSum->hwPt());
          histograms.stage2CaloLayer2MHTHFPhi.fill(itEtSum->hwPhi());
        } else if(l1t::EtSum::EtSumType::kMinBiasHFP0 == itEtSum->getType()){  // MBHFP0
          histograms.stage2CaloLayer2MinBiasHFP0.fill(itEtSum->hwPt());
        } else if(l1t::EtSum::EtSumType::kMinBiasHFM0 == itEtSum->getType()){  // MBHFM0
          histograms.stage2CaloLayer2MinBiasHFM0.fill(itEtSum->hwPt());
        } else if(l1t::EtSum::EtSumType::kMinBiasHFP1 == itEtSum->getType()){  // MBHFP1
          histograms.stage2CaloLayer2MinBiasHFP1.fill(itEtSum->hwPt());
        } else if(l1t::EtSum::EtSumType::kMinBiasHFM1 == itEtSum->getType()){  // MBHFM1
          histograms.stage2CaloLayer2MinBiasHFM1.fill(itEtSum->hwPt());
          //} else if(l1t::EtSum::EtSumType::kTotalHtHF == itEtSum->getType()){    // HTTHF
          //stage2CaloLayer2HTTHFRank.fill(itEtSum->hwPt());
        } else if(l1t::EtSum::EtSumType::kTotalEtEm == itEtSum->getType()){    // ETTEM
          histograms.stage2CaloLayer2ETTEMRank.fill(itEtSum->hwPt());
        } else if (l1t::EtSum::EtSumType::kTowerCount == itEtSum->getType()) {
          histograms.stage2CaloLayer2TowCount.fill(itEtSum->hwPt());
        } else{
          histograms.stage2CaloLayer2HTTRank.fill(itEtSum->hwPt());
        }
      }
    }
  }
}

void L1TStage2CaloLayer2::dqmBeginRun(edm::Run const& iRrun, edm::EventSetup const& evSetup, calolayer2dqm::Histograms& histograms) const
{}
