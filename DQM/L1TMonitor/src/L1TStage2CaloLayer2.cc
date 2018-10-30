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

void L1TStage2CaloLayer2::bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&)
{
  
  //central jet
  ibooker.setCurrentFolder(monitorDir_+"/Central-Jets");

  stage2CaloLayer2CenJetEtEtaPhi_ = ibooker.book2D("CenJetsEtEtaPhi", "CENTRAL JET E_{T} ETA PHI; Jet i#eta; Jet i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2CenJetEta_ = ibooker.book1D("CenJetsEta", "CENTRAL JET ETA; Jet i#eta; Counts", 229, -114.5, 114.5);
  stage2CaloLayer2CenJetPhi_ = ibooker.book1D("CenJetsPhi", "CENTRAL JET PHI; Jet i#phi; Counts", 144, -0.5, 143.5);
  stage2CaloLayer2CenJetRank_ = ibooker.book1D("CenJetsRank", "CENTRAL JET E_{T}; Jet iE_{T}; Counts", 2048, -0.5, 2047.5);
  stage2CaloLayer2CenJetOcc_ = ibooker.book2D("CenJetsOcc", "CENTRAL JET OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2CenJetBxOcc_ = ibooker.book2D("CenJetsBxOcc", "CENTRAL JET BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 512, -0.5, 2047.5);
  stage2CaloLayer2CenJetQual_ = ibooker.book1D("CenJetsQual", "CENTRAL JET QUALITY; Quality; Counts", 32, -0.5, 31.5);


  //forward jet
  ibooker.setCurrentFolder(monitorDir_+"/Forward-Jets");

  stage2CaloLayer2ForJetEtEtaPhi_ = ibooker.book2D("ForJetsEtEtaPhi", "FORWARD JET E_{T} ETA PHI; Jet i#eta; Jet i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2ForJetEta_ = ibooker.book1D("ForJetsEta", "FORWARD JET ETA; Jet i#eta; Counts", 229, -114.5, 114.5);
  stage2CaloLayer2ForJetPhi_ = ibooker.book1D("ForJetsPhi", "FORWARD JET PHI; Jet i#phi; Counts", 144, -0.5, 143.5);
  stage2CaloLayer2ForJetRank_ = ibooker.book1D("ForJetsRank", "FORWARD JET E_{T}; Jet iE_{T}; Counts", 2048, -0.5, 2047.5);
  stage2CaloLayer2ForJetOcc_ = ibooker.book2D("ForJetsOcc", "FORWARD JET OCCUPANCY; Jet i#eta; Jet i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2ForJetBxOcc_ = ibooker.book2D("ForJetsBxOcc", "FORWARD JET BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 512, -0.5, 2047.5);
  stage2CaloLayer2ForJetQual_ = ibooker.book1D("ForJetsQual", "FORWARD JET QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //IsoEG
  ibooker.setCurrentFolder(monitorDir_+"/Isolated-EG");

  stage2CaloLayer2EGIso_ = ibooker.book1D("EGIso", "EG ISO; Iso Flag; Counts", 4, -0.5, 3.5);
  stage2CaloLayer2IsoEGEtEtaPhi_ = ibooker.book2D("IsoEGsEtEtaPhi", "ISO EG E_{T} ETA PHI; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2IsoEGEta_ = ibooker.book1D("IsoEGsEta", "ISO EG ETA; EG i#eta; Counts", 229, -114.5, 114.5);
  stage2CaloLayer2IsoEGPhi_ = ibooker.book1D("IsoEGsPhi", "ISO EG PHI; EG i#phi; Counts", 144, -0.5, 143.5);
  stage2CaloLayer2IsoEGRank_ = ibooker.book1D("IsoEGsRank", "ISO EG E_{T}; EG iE_{T}; Counts", 512, -0.5, 511.5);
  stage2CaloLayer2IsoEGOcc_ = ibooker.book2D("IsoEGsOcc", "ISO EG OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2IsoEGBxOcc_ = ibooker.book2D("IsoEGsBxOcc", "ISO EG BX OCCUPANCY; BX; Counts",  5,-2.5, 2.5, 128, -0.5, 511.5);
  stage2CaloLayer2IsoEGQual_ = ibooker.book1D("IsoEGsQual", "ISO EG QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //NonIsoEG
  ibooker.setCurrentFolder(monitorDir_+"/NonIsolated-EG");

  stage2CaloLayer2NonIsoEGEtEtaPhi_ = ibooker.book2D("NonIsoEGsEtEtaPhi", "NonISO EG E_{T} ETA PHI; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2NonIsoEGEta_ = ibooker.book1D("NonIsoEGsEta", "NonISO EG ETA; NonISO EG i#eta; Counts", 229, -114.5, 114.5);
  stage2CaloLayer2NonIsoEGPhi_ = ibooker.book1D("NonIsoEGsPhi", "NonISO EG PHI; NonISO EG i#phi; Counts", 144, -0.5, 143.5);
  stage2CaloLayer2NonIsoEGRank_ = ibooker.book1D("NonIsoEGsRank", "NonISO EG E_{T}; NonISO EG iE_{T}; Counts", 512, -0.5, 511.5);
  stage2CaloLayer2NonIsoEGOcc_ = ibooker.book2D("NonIsoEGsOcc", "NonISO EG OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2NonIsoEGBxOcc_ = ibooker.book2D("NonIsoEGsBxOcc", "NonISO EG BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 128, -0.5, 511.5);
  stage2CaloLayer2NonIsoEGQual_ = ibooker.book1D("NonIsoEGsQual", "NonISO EG QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //IsoTau
  ibooker.setCurrentFolder(monitorDir_+"/Isolated-Tau");

  stage2CaloLayer2TauIso_ = ibooker.book1D("TauIso", "Tau ISO; Iso Flag; Counts", 4, -0.5, 3.5);
  stage2CaloLayer2IsoTauEtEtaPhi_ = ibooker.book2D("IsoTausEtEtaPhi", "ISO Tau E_{T} ETA PHI; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2IsoTauEta_ = ibooker.book1D("IsoTausEta", "ISO Tau ETA; ISO Tau i#eta; Counts", 229, -114.5, 114.5);
  stage2CaloLayer2IsoTauPhi_ = ibooker.book1D("IsoTausPhi", "ISO Tau PHI; ISO Tau i#phi; Counts", 144, -0.5, 143.5);
  stage2CaloLayer2IsoTauRank_ = ibooker.book1D("IsoTausRank", "ISO Tau E_{T}; ISO Tau iE_{T}; Counts", 512, -0.5, 511.5);
  stage2CaloLayer2IsoTauOcc_ = ibooker.book2D("IsoTausOcc", "ISO Tau OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2IsoTauBxOcc_ = ibooker.book2D("IsoTausBxOcc", "ISO Tau BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 128, -0.5, 511.5);
  stage2CaloLayer2IsoTauQual_ = ibooker.book1D("IsoTausQual", "ISO Tau QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //NonIsoTau
  ibooker.setCurrentFolder(monitorDir_+"/NonIsolated-Tau");

  stage2CaloLayer2TauEtEtaPhi_ = ibooker.book2D("TausEtEtaPhi", "Tau E_{T} ETA PHI; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2TauEta_ = ibooker.book1D("TausEta", "NonISO Tau ETA; Tau i#eta; Counts", 229, -114.5, 114.5);
  stage2CaloLayer2TauPhi_ = ibooker.book1D("TausPhi", "NonISO Tau PHI; Tau i#phi; Counts", 144, -0.5, 143.5);
  stage2CaloLayer2TauRank_ = ibooker.book1D("TausRank", "NonISO Tau E_{T}; Tau iE_{T}; Counts", 512, -0.5, 511.5);
  stage2CaloLayer2TauOcc_ = ibooker.book2D("TausOcc", "NonISO Tau OCCUPANCY; i#eta; i#phi", 229, -114.5, 114.5, 144, -0.5, 143.5);
  stage2CaloLayer2TauBxOcc_ = ibooker.book2D("TausBxOcc", "NonISO Tau BX OCCUPANCY; BX; Counts", 5,-2.5, 2.5, 128, -0.5, 511.5);
  stage2CaloLayer2TauQual_ = ibooker.book1D("TausQual", "NonISO Tau QUALITY; Quality; Counts", 32, -0.5, 31.5);

  //EtSums
  ibooker.setCurrentFolder(monitorDir_+"/Energy-Sums");

  stage2CaloLayer2EtSumBxOcc_ = ibooker.book2D("EtSumBxOcc", "EtSum BX OCCUPANCY; BX; Counts",  5,-2.5, 2.5, 1024, -0.5, 4095.5);
  stage2CaloLayer2METRank_ = ibooker.book1D("METRank", "MET E_{T}; iE_{T}; Counts", 4096, -0.5, 4095.5);
  stage2CaloLayer2METPhi_ = ibooker.book1D("METPhi", "MET Phi; MET i#Phi; Counts", 144, -0.5, 143.5);
  stage2CaloLayer2ETTRank_ = ibooker.book1D("ETTRank", "ETT E_{T}; ETT iE_{T}; Counts", 4096, -0.5, 4095.5);
  stage2CaloLayer2MHTRank_ = ibooker.book1D("MHTRank", "MHT E_{T}; MHT iE_{T}; Counts", 4096, -0.5, 4095.5);
  stage2CaloLayer2MHTPhi_ = ibooker.book1D("MHTPhi", "MHT Phi; MHT i#phi; Counts", 144, -0.5, 143.5);
  stage2CaloLayer2HTTRank_ = ibooker.book1D("HTTRank", "HTT E_{T}; HTT iE_{T}; Counts", 4096, -0.5, 4095.5);
  stage2CaloLayer2METHFRank_ = ibooker.book1D("METHFRank", "METHF E_{T}; METHF iE_{T}; Counts", 4096, -0.5, 4095.5);
  stage2CaloLayer2METHFPhi_ = ibooker.book1D("METHFPhi", "METHF Phi; METHF i#phi; Counts", 144, -0.5, 143.5);
  // stage2CaloLayer2ETTHFRank_ = ibooker.book1D("ETTHFRank", "ETTHF E_{T}", 4096, -0.5, 4095.5);
  stage2CaloLayer2MHTHFRank_ = ibooker.book1D("MHTHFRank", "MHTHF E_{T}; MHTHF iE_{T}; Counts", 4096, -0.5, 4095.5);
  stage2CaloLayer2MHTHFPhi_ = ibooker.book1D("MHTHFPhi", "MHTHF Phi; MHTHF i#phi; Counts", 144, -0.5, 143.5);
  // stage2CaloLayer2HTTHFRank_ = ibooker.book1D("HTTHFRank", "HTTHF E_{T}", 4096, -0.5, 4095.5);
  stage2CaloLayer2ETTEMRank_ = ibooker.book1D("ETTEMRank", "ETTEM E_{T}; ETTEM iE_{T}; Counts", 4096, -0.5, 4095.5);
  
  stage2CaloLayer2Asymmetry_ = ibooker.book1D("Asymmetry", "Asymmetry; Assymmetry; Counts", 256, -0.5, 255.5);
  stage2CaloLayer2Centrality_ = ibooker.book1D("Centrality", "Centrality; Centrality; Counts", 9, -1.5, 7.5);
 
  stage2CaloLayer2MinBiasHFP0_ = ibooker.book1D("MinBiasHFP0", "HF Min Bias Sum Threshold 0, Positive #eta; N_{towers}; Events", 16, -0.5, 15.5);
  stage2CaloLayer2MinBiasHFM0_ = ibooker.book1D("MinBiasHFM0", "HF Min Bias Sum Threshold 1, Nevagive #eta; N_{towers}; Events", 16, -0.5, 15.5);
  stage2CaloLayer2MinBiasHFP1_ = ibooker.book1D("MinBiasHFP1", "HF Min Bias Sum Threshold 1, Positive #eta; N_{towers}; Events", 16, -0.5, 15.5);
  stage2CaloLayer2MinBiasHFM1_ = ibooker.book1D("MinBiasHFM1", "HF Min Bias Sum Threshold 1, Negative #eta; N_{towers}; Events", 16, -0.5, 15.5);

  stage2CaloLayer2TowCount_ = ibooker.book1D("TowCount", "Count of Trigger towers above threshold; N_{towers}; Events", 5904, -0.5, 5903.5);

  ibooker.setCurrentFolder(monitorDir_+"/Timing");
  timingStage2CaloLayer2CenJetBxOcc_ = stage2CaloLayer2CenJetBxOcc_;
  timingStage2CaloLayer2ForJetBxOcc_ = stage2CaloLayer2ForJetBxOcc_;
  timingStage2CaloLayer2IsoEGBxOcc_ = stage2CaloLayer2IsoEGBxOcc_;
  timingStage2CaloLayer2NonIsoEGBxOcc_ = stage2CaloLayer2NonIsoEGBxOcc_;
  timingStage2CaloLayer2IsoTauBxOcc_ = stage2CaloLayer2TauBxOcc_;
  timingStage2CaloLayer2TauBxOcc_ = stage2CaloLayer2EtSumBxOcc_;
  timingStage2CaloLayer2EtSumBxOcc_ = stage2CaloLayer2EtSumBxOcc_;
 
  //Shifter
  ibooker.setCurrentFolder(monitorDir_+"/shifter");
  stage2CaloLayer2CenJetEtEtaPhi_shift_ = ibooker.book2D("CenJetsEtEtaPhi_shift", "CENTRAL JET E_{T} ETA PHI; Jet i#eta; Jet i#phi", 68, -68, 68, 72, -0.5, 143.5); 	
  stage2CaloLayer2ForJetEtEtaPhi_shift_ = ibooker.book2D("ForJetsEtEtaPhi_shift", "FORWARD JET E_{T} ETA PHI; Jet i#eta; Jet i#phi", 58, -116, 116, 36, -0.5, 143.5);
  stage2CaloLayer2IsoEGEtEtaPhi_shift_ = ibooker.book2D("IsoEGsEtEtaPhi_shift", "ISO EG E_{T} ETA PHI; i#eta; i#phi", 70, -70, 70, 72, -0.5, 143.5);
  stage2CaloLayer2NonIsoEGEtEtaPhi_shift_ = ibooker.book2D("NonIsoEGsEtEtaPhi_shift", "NonISO EG E_{T} ETA PHI; i#eta; i#phi", 70, -70, 70, 72, -0.5, 143.5);	
  stage2CaloLayer2IsoTauEtEtaPhi_shift_ = ibooker.book2D("IsoTausEtEtaPhi_shift", "ISO Tau E_{T} ETA PHI; i#eta; i#phi", 60, -60, 60, 72, -0.5, 143.5);
  stage2CaloLayer2TauEtEtaPhi_shift_ = ibooker.book2D("TausEtEtaPhi_shift", "Tau E_{T} ETA PHI; i#eta; i#phi", 60, -60, 60, 72, -0.5, 143.5);
  
}

void L1TStage2CaloLayer2::analyze(const edm::Event & e, const edm::EventSetup & c)
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
	stage2CaloLayer2ForJetBxOcc_->Fill(itBX, itJet->hwPt());
	if(itBX == 0 ){
	  stage2CaloLayer2ForJetRank_->Fill(itJet->hwPt());
	  if (itJet->hwPt() !=0 ){
	    stage2CaloLayer2ForJetEtEtaPhi_->Fill(itJet->hwEta(), itJet->hwPhi(), itJet->hwPt());
	    stage2CaloLayer2ForJetEtEtaPhi_shift_->Fill(itJet->hwEta(), itJet->hwPhi(), itJet->hwPt());
	    stage2CaloLayer2ForJetOcc_->Fill(itJet->hwEta(), itJet->hwPhi());
            stage2CaloLayer2ForJetPhi_->Fill(itJet->hwPhi());
            stage2CaloLayer2ForJetEta_->Fill(itJet->hwEta());
            stage2CaloLayer2ForJetQual_->Fill(itJet->hwQual());
	  }
	}
      }
      else{
	stage2CaloLayer2CenJetBxOcc_->Fill(itBX, itJet->hwPt());
	if(itBX == 0 ){
	  stage2CaloLayer2CenJetRank_->Fill(itJet->hwPt());
	  if (itJet->hwPt() !=0 ){
	    stage2CaloLayer2CenJetEtEtaPhi_->Fill(itJet->hwEta(), itJet->hwPhi(), itJet->hwPt());
	    stage2CaloLayer2CenJetEtEtaPhi_shift_->Fill(itJet->hwEta(), itJet->hwPhi(), itJet->hwPt());
	    stage2CaloLayer2CenJetOcc_->Fill(itJet->hwEta(), itJet->hwPhi());
            stage2CaloLayer2CenJetPhi_->Fill(itJet->hwPhi());
            stage2CaloLayer2CenJetEta_->Fill(itJet->hwEta());
            stage2CaloLayer2CenJetQual_->Fill(itJet->hwQual());
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
        stage2CaloLayer2IsoEGBxOcc_->Fill(itBX, itEG->hwPt());
	if(itBX == 0 ){
	  stage2CaloLayer2IsoEGRank_->Fill(itEG->hwPt());
	  if(itEG->hwPt() !=0 ){
	    stage2CaloLayer2IsoEGEtEtaPhi_->Fill(itEG->hwEta(), itEG->hwPhi(), itEG->hwPt());
	    stage2CaloLayer2IsoEGEtEtaPhi_shift_->Fill(itEG->hwEta(), itEG->hwPhi(), itEG->hwPt());
	    stage2CaloLayer2IsoEGOcc_->Fill(itEG->hwEta(), itEG->hwPhi());
            stage2CaloLayer2IsoEGPhi_->Fill(itEG->hwPhi());
            stage2CaloLayer2IsoEGEta_->Fill(itEG->hwEta());
            stage2CaloLayer2IsoEGQual_->Fill(itEG->hwQual());
            stage2CaloLayer2EGIso_->Fill(itEG->hwIso());
	  }
	}
      }
      else{
	stage2CaloLayer2NonIsoEGBxOcc_->Fill(itBX, itEG->hwPt());
	if(itBX == 0 ){
	  stage2CaloLayer2NonIsoEGRank_->Fill(itEG->hwPt());
	  if(itEG->hwPt() !=0 ){
	    stage2CaloLayer2NonIsoEGEtEtaPhi_->Fill(itEG->hwEta(), itEG->hwPhi(), itEG->hwPt());
	    stage2CaloLayer2NonIsoEGEtEtaPhi_shift_->Fill(itEG->hwEta(), itEG->hwPhi(), itEG->hwPt());
            stage2CaloLayer2NonIsoEGOcc_->Fill(itEG->hwEta(), itEG->hwPhi());
            stage2CaloLayer2NonIsoEGPhi_->Fill(itEG->hwPhi());
            stage2CaloLayer2NonIsoEGEta_->Fill(itEG->hwEta());
            stage2CaloLayer2NonIsoEGQual_->Fill(itEG->hwQual());
            stage2CaloLayer2EGIso_->Fill(itEG->hwIso());
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
        stage2CaloLayer2IsoTauBxOcc_->Fill(itBX, itTau->hwPt());
	if(itBX == 0 ){
	  stage2CaloLayer2IsoTauRank_->Fill(itTau->hwPt());
	  if(itTau->hwPt() !=0 ){
	    stage2CaloLayer2IsoTauEtEtaPhi_->Fill(itTau->hwEta(), itTau->hwPhi(), itTau->hwPt());
	    stage2CaloLayer2IsoTauEtEtaPhi_shift_->Fill(itTau->hwEta(), itTau->hwPhi(), itTau->hwPt());
	    stage2CaloLayer2IsoTauOcc_->Fill(itTau->hwEta(), itTau->hwPhi());
            stage2CaloLayer2IsoTauPhi_->Fill(itTau->hwPhi());
            stage2CaloLayer2IsoTauEta_->Fill(itTau->hwEta());
            stage2CaloLayer2IsoTauQual_->Fill(itTau->hwQual());
            stage2CaloLayer2TauIso_->Fill(itTau->hwIso());
	  }
	}
      }
      else{
	stage2CaloLayer2TauBxOcc_->Fill(itBX, itTau->hwPt());
	if(itBX == 0 ){
	  stage2CaloLayer2TauRank_->Fill(itTau->hwPt());
	  if(itTau->hwPt() !=0 ){
	    stage2CaloLayer2TauEtEtaPhi_->Fill(itTau->hwEta(), itTau->hwPhi(), itTau->hwPt());
            stage2CaloLayer2TauEtEtaPhi_shift_->Fill(itTau->hwEta(), itTau->hwPhi(), itTau->hwPt());
	    stage2CaloLayer2TauOcc_->Fill(itTau->hwEta(), itTau->hwPhi());
            stage2CaloLayer2TauPhi_->Fill(itTau->hwPhi());
            stage2CaloLayer2TauEta_->Fill(itTau->hwEta());
            stage2CaloLayer2TauQual_->Fill(itTau->hwQual());
            stage2CaloLayer2TauIso_->Fill(itTau->hwIso());
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
      stage2CaloLayer2EtSumBxOcc_->Fill(itBX, itEtSum->hwPt());

      if (itBX==0){
	if(l1t::EtSum::EtSumType::kMissingEt == itEtSum->getType()){;          // MET
	  stage2CaloLayer2METRank_->Fill(itEtSum->hwPt());
	  stage2CaloLayer2METPhi_->Fill(itEtSum->hwPhi());
	} else if(l1t::EtSum::EtSumType::kMissingEtHF == itEtSum->getType()){  // METHF
	  stage2CaloLayer2METHFRank_->Fill(itEtSum->hwPt());
	  stage2CaloLayer2METHFPhi_->Fill(itEtSum->hwPhi());
	} else if(l1t::EtSum::EtSumType::kTotalEt == itEtSum->getType()){      // ETT
	  stage2CaloLayer2ETTRank_->Fill(itEtSum->hwPt());
	  //} else if(l1t::EtSum::EtSumType::kTotalEtHF == itEtSum->getType()){    // ETTHF
	  // stage2CaloLayer2ETTHFRank_->Fill(itEtSum->hwPt());
	} else if(l1t::EtSum::EtSumType::kMissingHt == itEtSum->getType()){    // MHT
	  stage2CaloLayer2MHTRank_->Fill(itEtSum->hwPt());
          if (itEtSum->hwPt()>0)
		  stage2CaloLayer2MHTPhi_->Fill(itEtSum->hwPhi());
	} else if(l1t::EtSum::EtSumType::kMissingHtHF == itEtSum->getType()){  // MHTHF
	  stage2CaloLayer2MHTHFRank_->Fill(itEtSum->hwPt());
	  if (itEtSum->hwPt()>0)
	          stage2CaloLayer2MHTHFPhi_->Fill(itEtSum->hwPhi());
	} else if(l1t::EtSum::EtSumType::kMinBiasHFP0 == itEtSum->getType()){  // MBHFP0
	  stage2CaloLayer2MinBiasHFP0_->Fill(itEtSum->hwPt());
	} else if(l1t::EtSum::EtSumType::kMinBiasHFM0 == itEtSum->getType()){  // MBHFM0
	  stage2CaloLayer2MinBiasHFM0_->Fill(itEtSum->hwPt());
	} else if(l1t::EtSum::EtSumType::kMinBiasHFP1 == itEtSum->getType()){  // MBHFP1
	  stage2CaloLayer2MinBiasHFP1_->Fill(itEtSum->hwPt());
	} else if(l1t::EtSum::EtSumType::kMinBiasHFM1 == itEtSum->getType()){  // MBHFM1
	  stage2CaloLayer2MinBiasHFM1_->Fill(itEtSum->hwPt());
	  //} else if(l1t::EtSum::EtSumType::kTotalHtHF == itEtSum->getType()){    // HTTHF
	  //stage2CaloLayer2HTTHFRank_->Fill(itEtSum->hwPt());
	} else if(l1t::EtSum::EtSumType::kTotalEtEm == itEtSum->getType()){    // ETTEM
	  stage2CaloLayer2ETTEMRank_->Fill(itEtSum->hwPt());
	} else if (l1t::EtSum::EtSumType::kTowerCount == itEtSum->getType()) {
	  stage2CaloLayer2TowCount_->Fill(itEtSum->hwPt());
	} else if (l1t::EtSum::EtSumType::kTotalHt == itEtSum->getType()){
	  stage2CaloLayer2HTTRank_->Fill(itEtSum->hwPt());
	} else if (l1t::EtSum::EtSumType::kAsymEt == itEtSum->getType()){     //Asym ET
	  stage2CaloLayer2Asymmetry_->Fill(itEtSum->hwPt());
        } else if (l1t::EtSum::EtSumType::kCentrality == itEtSum->getType()){     //Centrality
          if (itEtSum->hwPt()==0) stage2CaloLayer2Centrality_->Fill(-1);
          else {
          	for (int i=0; i<8; i++)
	        	if (((itEtSum->hwPt()>>i)&1)==1) stage2CaloLayer2Centrality_->Fill(i);
	       }
        }
      }
    }
  }
}

void L1TStage2CaloLayer2::dqmBeginRun(edm::Run const& iRrun, edm::EventSetup const& evSetup)
{}

