#include "DQM/Physics/interface/Selection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
		
Selection::Selection(){
      leptonType_ = "muon";
      beamSpot =  reco::TrackBase::Point(0,0,0);
      jets.clear();
      electrons.clear();
      eID.clear();
      muons.clear();
      mets.clear();
      Njets_            = 4;
      JetPtThreshold_   = 30.;
      JetEtaThreshold_  = 2.4;
      JetEHThreshold_  = 0.1;
      JetDeltaRLeptonJetThreshold_ = 0.0;
      applyLeptonJetDeltaRCut_ = false;
      MuonPtThreshold_  = 10.;
      MuonEtaThreshold_ = 2.1;
      MuonD0Cut_    = 0.02;
      Chi2Cut_      = 10;
      NofValidHits_ = 11;
      MuonRelIso_   = 0.1;
      MuonVetoEM_   = 4;
      MuonVetoHad_  = 6; 
      ElectronPtThreshold_  = 20.;
      ElectronEtaThreshold_ = 2.5;
      ElectronD0Cut_        = 0.02;
      ElectronRelIso_       = 0.1;
      METThreshold_        = 0;
      PtThrMuonLoose_          = 0.0;           
      EtaThrMuonLoose_         = 99.9;        
      RelIsoThrMuonLoose_      = 99.9;     
      PtThrElectronLoose_      = 0.0;     
      EtaThrElectronLoose_     = 99.9;    
      RelIsoThrElectronLoose_  = 99.9;

}

//////////////////////////////////////////
/// Set Methods
//////////////////////////////////////////

void Selection::Set(const reco::BeamSpot bspot, const edm::View<reco::CaloJet>& jets_, const edm::View<reco::Muon>& muons_, const edm::View<reco::CaloMET>& mets_){
      beamSpot = bspot.position();
      jets.clear();
      electrons.clear();
      muons.clear();
      mets.clear();
      for(unsigned int i=0;i<jets_.size(); i++) jets.push_back( jets_.at(i));
      for(unsigned int i=0;i<muons_.size();i++) muons.push_back(muons_.at(i));
      for(unsigned int i=0;i<mets_.size(); i++) mets.push_back( mets_.at(i));
}

void Selection::Set(const reco::BeamSpot bspot, const edm::View<reco::CaloJet>& jets_, const edm::View<reco::GsfElectron>& electrons_, const edm::View<reco::CaloMET>& mets_){
      beamSpot = bspot.position();
      jets.clear();
      electrons.clear();
      muons.clear();
      mets.clear();
      for(unsigned int i=0;i<jets_.size();     i++) jets.push_back(jets_.at(i));
      for(unsigned int i=0;i<electrons_.size();i++) electrons.push_back(electrons_.at(i));
      for(unsigned int i=0;i<mets_.size();     i++) mets.push_back(mets_.at(i));
}

void Selection::Set(const reco::BeamSpot bspot, const edm::View<reco::CaloJet>& jets_, const edm::View<reco::Muon>& muons_, const edm::View<reco::GsfElectron>& electrons_, const edm::View<reco::CaloMET>& mets_){
      beamSpot = bspot.position();
      jets.clear();
      electrons.clear();
      muons.clear();
      mets.clear();
      for(unsigned int i=0;i<jets_.size();     i++) jets.push_back(jets_.at(i));
      for(unsigned int i=0;i<muons_.size();    i++) muons.push_back(muons_.at(i));
      for(unsigned int i=0;i<mets_.size();     i++) mets.push_back(mets_.at(i));
      for(unsigned int i=0;i<electrons_.size();i++) electrons.push_back(electrons_.at(i));
}


void Selection::SeteID(const edm::View<reco::GsfElectron>& electrons, const edm::ValueMap<float>& eIDmap){
   eID.clear();
   for(unsigned int idx=0; idx<electrons.size(); idx++){
    eID.push_back(eIDmap[electrons.refAt(idx)] );    
   }
}

//////////////////////////////////////////
/// Copy constructor and assignment method
//////////////////////////////////////////

Selection::Selection(const Selection & s){
      leptonType_ = s.leptonType_;
      beamSpot  = s.beamSpot;
      jets      = s.jets;
      electrons = s.electrons;
      eID       = s.eID;
      muons     = s.muons;
      mets      = s.mets;
      JetPtThreshold_   = s.JetPtThreshold_;
      JetEtaThreshold_  = s.JetEtaThreshold_;
      JetEHThreshold_  = s.JetEHThreshold_;
      JetDeltaRLeptonJetThreshold_ = s.JetDeltaRLeptonJetThreshold_;
      applyLeptonJetDeltaRCut_ = s.applyLeptonJetDeltaRCut_;
      MuonPtThreshold_  = s.MuonPtThreshold_;
      MuonEtaThreshold_ = s.MuonEtaThreshold_;
      MuonD0Cut_        = s.MuonD0Cut_;
      Chi2Cut_          = s.Chi2Cut_;
      NofValidHits_     = s.NofValidHits_;
      MuonRelIso_       = s.MuonRelIso_;
      MuonVetoEM_       = s.MuonVetoEM_;
      MuonVetoHad_      = s.MuonVetoHad_; 
      ElectronPtThreshold_  = s.ElectronPtThreshold_;
      ElectronEtaThreshold_ = s.ElectronEtaThreshold_;
      ElectronD0Cut_        = s.ElectronD0Cut_;
      ElectronRelIso_       = s.ElectronRelIso_;
      METThreshold_        = s.METThreshold_;
      PtThrMuonLoose_          = s.PtThrMuonLoose_;           
      EtaThrMuonLoose_         = s.EtaThrMuonLoose_;        
      RelIsoThrMuonLoose_      = s.RelIsoThrMuonLoose_;     
      PtThrElectronLoose_      = s.PtThrElectronLoose_;     
      EtaThrElectronLoose_     = s.EtaThrElectronLoose_;    
      RelIsoThrElectronLoose_  = s.RelIsoThrElectronLoose_;
}

Selection& Selection::operator=(const Selection & s){
      leptonType_ = s.leptonType_;
      beamSpot  = s.beamSpot;
      jets      = s.jets;
      electrons = s.electrons;
      eID       = s.eID;
      muons     = s.muons;
      mets      = s.mets;
      JetPtThreshold_   = s.JetPtThreshold_;
      JetEtaThreshold_  = s.JetEtaThreshold_;
      JetEHThreshold_  = s.JetEHThreshold_;
      JetDeltaRLeptonJetThreshold_ = s.JetDeltaRLeptonJetThreshold_;
      applyLeptonJetDeltaRCut_ = s.applyLeptonJetDeltaRCut_;
      MuonPtThreshold_  = s.MuonPtThreshold_;
      MuonEtaThreshold_ = s.MuonEtaThreshold_;
      MuonD0Cut_        = s.MuonD0Cut_;
      Chi2Cut_          = s.Chi2Cut_;
      NofValidHits_     = s.NofValidHits_;
      MuonRelIso_       = s.MuonRelIso_;
      MuonVetoEM_       = s.MuonVetoEM_;
      MuonVetoHad_      = s.MuonVetoHad_; 
      ElectronPtThreshold_  = s.ElectronPtThreshold_;
      ElectronEtaThreshold_ = s.ElectronEtaThreshold_;
      ElectronD0Cut_        = s.ElectronD0Cut_;
      ElectronRelIso_       = s.ElectronRelIso_;
      METThreshold_        = s.METThreshold_;
      PtThrMuonLoose_          = s.PtThrMuonLoose_;           
      EtaThrMuonLoose_         = s.EtaThrMuonLoose_;        
      RelIsoThrMuonLoose_      = s.RelIsoThrMuonLoose_;     
      PtThrElectronLoose_      = s.PtThrElectronLoose_;     
      EtaThrElectronLoose_     = s.EtaThrElectronLoose_;    
      RelIsoThrElectronLoose_  = s.RelIsoThrElectronLoose_;
      return *this;
}

/////////////////////////////////////////////
/// Relative Iso methods for muon & electron
///   -> use methods available in reco::objects
/////////////////////////////////////////////

float Selection::RelativeIso (const reco::Muon& muon) const{ return ( (float) (muon.isolationR03().emEt + muon.isolationR03().hadEt + muon.isolationR03().sumPt)/ muon.pt() );}

float Selection::RelativeIso (const reco::GsfElectron& electron) const { return ( (float) (electron.dr03EcalRecHitSumEt() + electron.dr03HcalTowerSumEt() + electron.dr03TkSumPt() )/ electron.pt() );}


float Selection::RelativeIsoCalo (const reco::Muon& muon) const{ 
return ( (float) (muon.pt() / (muon.pt() + muon.isolationR03().emEt + muon.isolationR03().hadEt)));}

float Selection::RelativeIsoTrk (const reco::Muon& muon) const{ 
return ( (float) ( muon.pt() / (muon.pt() + muon.isolationR03().sumPt) ));}

float Selection::RelativeIsoCalo (const reco::GsfElectron& electron) const {
 return ( (float) (electron.pt() / (electron.pt() + electron.dr03EcalRecHitSumEt() + electron.dr03HcalTowerSumEt() ) ));}
 
float Selection::RelativeIsoTrk (const reco::GsfElectron& electron) const { 
return ( (float) ( electron.pt()/( electron.pt() + electron.dr03TkSumPt() ) ) );}



bool Selection::VetoIsoDeposit(const reco::Muon& muon) const { 
 if (muon.isolationR03().emVetoEt < MuonVetoEM_ && muon.isolationR03().hadVetoEt < MuonVetoHad_) return true;
 return false;
}

//////////////////////////////////////////
/// Set Configuration methods
//////////////////////////////////////////

void Selection::SetConfiguration(float PtThrJets, float EtaThrJets, float EHThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float MuonVetoEM, float MuonVetoHad){
      JetPtThreshold_   = PtThrJets;
      JetEtaThreshold_  = EtaThrJets;
      JetEHThreshold_   = EHThrJets;
      MuonPtThreshold_  = PtThrMuons;
      MuonEtaThreshold_ = EtaThrMuons;
      MuonRelIso_       = MuonRelIso;
      MuonVetoEM_       = MuonVetoEM;
      MuonVetoHad_      = MuonVetoHad;
}
      
void Selection::SetConfiguration(float PtThrJets, float EtaThrJets, float EHThrJets, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso){
      JetPtThreshold_       = PtThrJets;
      JetEtaThreshold_      = EtaThrJets;
      JetEHThreshold_      = EHThrJets;
      ElectronPtThreshold_  = PtThrElectrons;
      ElectronEtaThreshold_ = EtaThrElectrons;
      ElectronRelIso_       = ElectronRelIso;
}

void Selection::SetConfiguration(float PtThrJets, float EtaThrJets, float EHThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float MuonVetoEM, float MuonVetoHad, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso){
      JetPtThreshold_   = PtThrJets;
      JetEtaThreshold_  = EtaThrJets;
      JetEHThreshold_  = EHThrJets;
      MuonPtThreshold_  = PtThrMuons;
      MuonEtaThreshold_ = EtaThrMuons;
      MuonRelIso_       = MuonRelIso;
      MuonVetoEM_       = MuonVetoEM;
      MuonVetoHad_      = MuonVetoHad;
      ElectronPtThreshold_  = PtThrElectrons;
      ElectronEtaThreshold_ = EtaThrElectrons;
      ElectronRelIso_       = ElectronRelIso;
}



void Selection::SetConfiguration(float PtThrJets, float EtaThrJets, float EHThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIsoCalo, float MuonRelIsoTrk , float
MuonVetoEM, float MuonVetoHad, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIsoCalo, float ElectronRelIsoTrk){
      JetPtThreshold_           = PtThrJets;
      JetEtaThreshold_          = EtaThrJets;
      JetEHThreshold_           = EHThrJets;
      MuonPtThreshold_          = PtThrMuons;
      MuonEtaThreshold_         = EtaThrMuons;
      MuonRelIsoCalo_           = MuonRelIsoCalo;
      MuonRelIsoTrk_            = MuonRelIsoTrk;
      MuonVetoEM_               = MuonVetoEM;
      MuonVetoHad_              = MuonVetoHad;
      ElectronPtThreshold_      = PtThrElectrons;
      ElectronEtaThreshold_     = EtaThrElectrons;
      ElectronRelIsoCalo_       = ElectronRelIsoCalo;
      ElectronRelIsoTrk_        = ElectronRelIsoTrk;
}






void Selection::SetMuonConfig(float MuonD0Cut, int Chi2Cut, int NofValidHits) { 
	MuonD0Cut_    = MuonD0Cut; 
	Chi2Cut_      = Chi2Cut; 
	NofValidHits_ = NofValidHits;
}

void Selection::SetElectronConfig(float ElectronD0Cut) {
	ElectronD0Cut_ = ElectronD0Cut;
}

void Selection::SetElectronConfig(float ElectronD0Cut, bool vetoEBEETransitionRegion, bool useElectronID) {
	ElectronD0Cut_ = ElectronD0Cut;
	vetoEBEETransitionRegion_ = vetoEBEETransitionRegion;
	useElectronID_ = useElectronID;
}

void Selection::SetMETConfig(float METThreshold) {
	METThreshold_ = METThreshold;
}

void Selection::SetJetConfig(float JetDeltaRLeptonJetThreshold, bool applyLeptonJetDeltaRCut) {
	JetDeltaRLeptonJetThreshold_ = JetDeltaRLeptonJetThreshold;
	applyLeptonJetDeltaRCut_     = applyLeptonJetDeltaRCut;
}

void Selection::SetMuonLooseConfig(float PtThrMuonLoose, float EtaThrMuonLoose, float RelIsoThrMuonLoose) {
	PtThrMuonLoose_     = PtThrMuonLoose;
	EtaThrMuonLoose_    = EtaThrMuonLoose;
	RelIsoThrMuonLoose_ = RelIsoThrMuonLoose;
}

void Selection::SetElectronLooseConfig(float PtThrElectronLoose, float EtaThrElectronLoose, float RelIsoThrElectronLoose) {
	PtThrElectronLoose_     = PtThrElectronLoose;
	EtaThrElectronLoose_    = EtaThrElectronLoose;
	RelIsoThrElectronLoose_ = RelIsoThrElectronLoose;
}

//////////////////////////////////////////
/// Methods that returns a vector of selected objects according to the configuration given
//////////////////////////////////////////

std::vector<reco::CaloJet> Selection::GetSelectedJets() const{
  std::vector<reco::CaloJet> selectedJets;
  
  for(unsigned int i=0;i<jets.size();i++){
    bool passDeltaRCut = true;
    if (applyLeptonJetDeltaRCut_ && "electron" == leptonType_) {
  	std::vector<reco::GsfElectron> nonIsolatedElectrons = this->GetSelectedElectronsNonIso();
	for (std::vector<reco::GsfElectron>::iterator electron = nonIsolatedElectrons.begin(); electron != nonIsolatedElectrons.end(); ++electron) {
	    if (reco::deltaR(jets.at(i), *electron) < JetDeltaRLeptonJetThreshold_) {passDeltaRCut = false;}
        }
    }        
    if (applyLeptonJetDeltaRCut_ && "muon" == leptonType_) {
  	std::vector<reco::Muon> nonIsolatedMuons = this->GetSelectedMuonsNonIso();
	for (std::vector<reco::Muon>::iterator muon = nonIsolatedMuons.begin(); muon != nonIsolatedMuons.end(); ++muon) {
	    if (reco::deltaR(jets.at(i), *muon) < JetDeltaRLeptonJetThreshold_) {passDeltaRCut = false;}
        }
    }
    
    if(fabs(jets[i].eta())<JetEtaThreshold_ && jets[i].pt()>JetPtThreshold_ && jets[i].energyFractionHadronic()>=JetEHThreshold_ && passDeltaRCut) {
    	selectedJets.push_back(jets[i]);
    }	
  }
  std::sort(selectedJets.begin(),selectedJets.end(),HighestPt());
  return selectedJets;
}

std::vector<reco::CaloJet> Selection::GetSelectedJets(float PtThr, float EtaThr, float EHThr) const{
  std::vector<reco::CaloJet> selectedJets;
  for(unsigned int i=0;i<jets.size();i++){
    bool passDeltaRCut = true;
    if (applyLeptonJetDeltaRCut_ && "electron" == leptonType_) {
  	std::vector<reco::GsfElectron> nonIsolatedElectrons = this->GetSelectedElectronsNonIso();
	for (std::vector<reco::GsfElectron>::iterator electron = nonIsolatedElectrons.begin(); electron != nonIsolatedElectrons.end(); ++electron) {
	    if (reco::deltaR(jets.at(i), *electron) < JetDeltaRLeptonJetThreshold_) {passDeltaRCut = false;}
        }
    }        
    if (applyLeptonJetDeltaRCut_ && "muon" == leptonType_) {
  	std::vector<reco::Muon> nonIsolatedMuons = this->GetSelectedMuonsNonIso();
	for (std::vector<reco::Muon>::iterator muon = nonIsolatedMuons.begin(); muon != nonIsolatedMuons.end(); ++muon) {
	    if (reco::deltaR(jets.at(i), *muon) < JetDeltaRLeptonJetThreshold_) {passDeltaRCut = false;}
        }
    }
    
    if(fabs(jets[i].eta())<EtaThr && jets[i].pt()>PtThr && jets[i].energyFractionHadronic()>=EHThr && passDeltaRCut) {
    	selectedJets.push_back(jets[i]);
    }
  }
  std::sort(selectedJets.begin(),selectedJets.end(),HighestPt());
  return selectedJets;
}

std::vector<reco::Muon> Selection::GetSelectedMuons() const{
  std::vector<reco::Muon> selectedMuons;
  for(unsigned int i=0;i<muons.size();i++){
    bool isIsolated = false;
    if( RelativeIso(muons[i]) < MuonRelIso_ && VetoIsoDeposit(muons[i])) isIsolated =  true;
    if( muons[i].isGlobalMuon() && muons[i].globalTrack()->normalizedChi2()< Chi2Cut_ && fabs(muons[i].innerTrack()->dxy(beamSpot))<MuonD0Cut_ && muons[i].innerTrack()->numberOfValidHits()>=NofValidHits_ && fabs(muons[i].eta())<MuonEtaThreshold_ && muons[i].pt()>MuonPtThreshold_ && isIsolated ){
	selectedMuons.push_back(muons[i]);
    }
  }
  std::sort(selectedMuons.begin(),selectedMuons.end(),HighestPt());
  return selectedMuons;
}

// does not take isolation criteria into account
std::vector<reco::Muon> Selection::GetSelectedMuonsNonIso() const{
  std::vector<reco::Muon> selectedMuons;
  for(unsigned int i=0;i<muons.size();i++){
    if( muons[i].isGlobalMuon() && muons[i].globalTrack()->normalizedChi2()< Chi2Cut_ && fabs(muons[i].innerTrack()->dxy(beamSpot))<MuonD0Cut_  && muons[i].innerTrack()->numberOfValidHits()>=NofValidHits_ && fabs(muons[i].eta())< MuonEtaThreshold_ && muons[i].pt()> MuonPtThreshold_ ){
	selectedMuons.push_back(muons[i]);
    }
  }
  std::sort(selectedMuons.begin(),selectedMuons.end(),HighestPt());
  return selectedMuons;
}

std::vector<reco::Muon> Selection::GetSelectedMuons(float PtThr, float EtaThr,float MuonRelIso) const{
  std::vector<reco::Muon> selectedMuons;
  for(unsigned int i=0;i<muons.size();i++){
    bool isIsolated = false;
    if( RelativeIso(muons[i]) < MuonRelIso && VetoIsoDeposit(muons[i])) isIsolated =  true;
    if( muons[i].isGlobalMuon() && muons[i].globalTrack()->normalizedChi2()< Chi2Cut_ && fabs(muons[i].innerTrack()->dxy(beamSpot))<MuonD0Cut_ && muons[i].innerTrack()->numberOfValidHits()>=NofValidHits_ && fabs(muons[i].eta())<EtaThr && muons[i].pt()>PtThr && isIsolated){
	selectedMuons.push_back(muons[i]);
    }
  }
  std::sort(selectedMuons.begin(),selectedMuons.end(),HighestPt());
  return selectedMuons;
}

// does not take isolation criteria into account
std::vector<reco::Muon> Selection::GetSelectedMuonsNonIso(float PtThr, float EtaThr,float MuonRelIso) const{
  std::vector<reco::Muon> selectedMuons;
  for(unsigned int i=0;i<muons.size();i++){
    if(muons[i].isGlobalMuon() && muons[i].globalTrack()->normalizedChi2()< Chi2Cut_ && fabs(muons[i].innerTrack()->dxy(beamSpot))<MuonD0Cut_ && muons[i].innerTrack()->numberOfValidHits()>=NofValidHits_ && fabs(muons[i].eta())<EtaThr && muons[i].pt()>PtThr){
	selectedMuons.push_back(muons[i]);
    }
  }
  std::sort(selectedMuons.begin(),selectedMuons.end(),HighestPt());
  return selectedMuons;
}

// Select loose muons for loose 2nd muon veto.
std::vector<reco::Muon> Selection::GetSelectedMuonsLoose() const{
  std::vector<reco::Muon> selectedMuons;
  for(unsigned int i=0;i<muons.size();i++) {
    if(muons[i].isGlobalMuon() && fabs(muons[i].eta())<EtaThrMuonLoose_ && muons[i].pt()>PtThrMuonLoose_ && RelativeIso(muons[i]) < RelIsoThrMuonLoose_) {
	selectedMuons.push_back(muons[i]);
    }
  }
  std::sort(selectedMuons.begin(),selectedMuons.end(),HighestPt());
  return selectedMuons;
}


std::vector<reco::GsfElectron> Selection::GetSelectedElectrons() const{
  if (electrons.size() != eID.size() && useElectronID_) { 
    throw cms::Exception("EventCorruption") << "Number of electron ID entries does not match number of electrons." << std::endl;
  }
  std::vector<reco::GsfElectron> selectedElectrons;
  for(unsigned int i=0;i<electrons.size();i++){
    bool isIsolated = false;
    bool isInEBEETransitionRegion = false;
    bool passeID = true;
    if (useElectronID_) {passeID = eID.at(i);}
    if (vetoEBEETransitionRegion_ && fabs(electrons[i].eta()) > 1.442 && fabs(electrons[i].eta()) < 1.560) {isInEBEETransitionRegion = true;}
    if(RelativeIso(electrons[i]) < ElectronRelIso_) isIsolated =  true;
    if( fabs(electrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(electrons[i].eta())<ElectronEtaThreshold_ && electrons[i].pt()>ElectronPtThreshold_ && isIsolated && !isInEBEETransitionRegion && passeID){
	selectedElectrons.push_back(electrons[i]);
    }
  }
  std::sort(selectedElectrons.begin(),selectedElectrons.end(),HighestPt());
  return selectedElectrons;
}

// does not take isolation criteria into account
std::vector<reco::GsfElectron> Selection::GetSelectedElectronsNonIso() const{
  if (electrons.size() != eID.size() && useElectronID_) { 
    throw cms::Exception("EventCorruption") << "Number of electron ID entries does not match number of electrons." << std::endl;
  }
  std::vector<reco::GsfElectron> selectedElectrons;
  for(unsigned int i=0;i<electrons.size();i++){
    bool isInEBEETransitionRegion = false;
    bool passeID = true;
    if (useElectronID_) {passeID = eID.at(i);}
    if (vetoEBEETransitionRegion_ && fabs(electrons[i].eta()) > 1.442 && fabs(electrons[i].eta()) < 1.560) {isInEBEETransitionRegion = true;}
    
    if( fabs(electrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(electrons[i].eta())<ElectronEtaThreshold_ && electrons[i].pt()>ElectronPtThreshold_ && !isInEBEETransitionRegion && passeID){
	selectedElectrons.push_back(electrons[i]);
    }
  }
  std::sort(selectedElectrons.begin(),selectedElectrons.end(),HighestPt());
  return selectedElectrons;
}

std::vector<reco::GsfElectron> Selection::GetSelectedElectrons(float PtThr, float EtaThr,float ElectronRelIso) const{
  if (electrons.size() != eID.size() && useElectronID_) { 
    throw cms::Exception("EventCorruption") << "Number of electron ID entries does not match number of electrons." << std::endl;
  }
  std::vector<reco::GsfElectron> selectedElectrons;
  for(unsigned int i=0;i<electrons.size();i++){
    bool isIsolated = false;
    bool isInEBEETransitionRegion = false;
    bool passeID = true;
    if (useElectronID_) {passeID = eID.at(i);}
    if (vetoEBEETransitionRegion_ && fabs(electrons[i].eta()) > 1.442 && fabs(electrons[i].eta()) < 1.560) {isInEBEETransitionRegion = true;}
    if(RelativeIso(electrons[i]) < ElectronRelIso) isIsolated =  true;
    if( fabs(electrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(electrons[i].eta())<EtaThr && electrons[i].pt()>PtThr && isIsolated && !isInEBEETransitionRegion && passeID){
	selectedElectrons.push_back(electrons[i]);
    }
  }
  std::sort(selectedElectrons.begin(),selectedElectrons.end(),HighestPt());
  return selectedElectrons;
}

// does not take isolation criteria into account
std::vector<reco::GsfElectron> Selection::GetSelectedElectronsNonIso(float PtThr, float EtaThr,float ElectronRelIso) const{
  if (electrons.size() != eID.size() && useElectronID_) { 
    throw cms::Exception("EventCorruption") << "Number of electron ID entries does not match number of electrons." << std::endl;
  }
  std::vector<reco::GsfElectron> selectedElectrons;  
  for(unsigned int i=0;i<electrons.size();i++){
    bool isInEBEETransitionRegion = false;
    bool passeID = true;
    if (useElectronID_) {passeID = eID.at(i);}
    if (vetoEBEETransitionRegion_ && fabs(electrons[i].eta()) > 1.442 && fabs(electrons[i].eta()) < 1.560) {isInEBEETransitionRegion = true;}
    if( fabs(electrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(electrons[i].eta())<EtaThr && electrons[i].pt()>PtThr && !isInEBEETransitionRegion && passeID){
	selectedElectrons.push_back(electrons[i]);
    }
  }
  std::sort(selectedElectrons.begin(),selectedElectrons.end(),HighestPt());
  return selectedElectrons;
}

// Get electrons passing loose selection criteria, for use in loose 2nd electron veto
std::vector<reco::GsfElectron> Selection::GetSelectedElectronsLoose() const{
  std::vector<reco::GsfElectron> selectedElectrons;
  for(unsigned int i=0;i<electrons.size();i++){
    if(fabs(electrons[i].eta())<EtaThrElectronLoose_ && electrons[i].pt()>PtThrElectronLoose_ && RelativeIso(electrons[i]) < RelIsoThrElectronLoose_ ){
	selectedElectrons.push_back(electrons[i]);
    }
  }
  std::sort(selectedElectrons.begin(),selectedElectrons.end(),HighestPt());
  return selectedElectrons;
}

//////// MET ////////////////////

std::vector<reco::CaloMET> Selection::GetSelectedMETs() const {
  std::vector<reco::CaloMET> selectedMETs;
  for(unsigned int i=0;i<mets.size();i++){
    if (mets.at(i).et() > METThreshold_) {
      selectedMETs.push_back(mets.at(i));
    }
  }
  return selectedMETs;
}

bool Selection::METpass() const {
	if (mets.size() != 1) {throw cms::Exception("EventCorruption") << "Number of METs is not exactly one." << std::endl;} 
	return mets.at(0).et() > METThreshold_;
}


//////////////////////////////////////////
//////////////////////////////////////////
/// Methods that selected object and re-fill  object members
//////////////////////////////////////////
//////////////////////////////////////////



void Selection::SelectJets(){
  std::vector<reco::CaloJet> tmpjets = jets;
  jets.clear();
  for(unsigned int i=0;i<tmpjets.size();i++){
    if(fabs(tmpjets[i].eta())<JetEtaThreshold_ && tmpjets[i].pt()>JetPtThreshold_ && tmpjets[i].energyFractionHadronic()>=JetEHThreshold_)
    	jets.push_back(tmpjets[i]);
  }  
  std::sort(jets.begin(),jets.end(),HighestPt());
}

void Selection::SelectJets(float PtThr, float EtaThr, float EHThr){
  std::vector<reco::CaloJet> tmpjets = jets;
  jets.clear();
  for(unsigned int i=0;i<tmpjets.size();i++){
    if(fabs(tmpjets[i].eta())<EtaThr && tmpjets[i].pt()>PtThr && tmpjets[i].energyFractionHadronic()>=EHThr)
    	jets.push_back(tmpjets[i]);
  }
  std::sort(jets.begin(),jets.end(),HighestPt());
}



void Selection::SelectJets(const edm::Event& iEvent, const edm::EventSetup& iSetup, const JetCorrector *acorrector){
  std::vector<reco::CaloJet> tmpjets = jets;
  jets.clear();
  for(unsigned int i=0;i<tmpjets.size();i++){
    double corrJES = acorrector->correction((tmpjets)[i], iEvent, iSetup);
    if(fabs(tmpjets[i].eta())<JetEtaThreshold_ && tmpjets[i].pt()*corrJES>JetPtThreshold_ && tmpjets[i].energyFractionHadronic()>=JetEHThreshold_)
    	jets.push_back(tmpjets[i]);
  }  
  std::sort(jets.begin(),jets.end(),HighestPt());
}

void Selection::SelectJets(float PtThr, float EtaThr, float EHThr, const edm::Event& iEvent, const edm::EventSetup& iSetup, const JetCorrector *acorrector){
  std::vector<reco::CaloJet> tmpjets = jets;
  jets.clear();
  for(unsigned int i=0;i<tmpjets.size();i++){
    double corrJES = acorrector->correction((tmpjets)[i], iEvent, iSetup);
    if(fabs(tmpjets[i].eta())<EtaThr && tmpjets[i].pt()*corrJES>PtThr && tmpjets[i].energyFractionHadronic()>=EHThr)
    	jets.push_back(tmpjets[i]);
  }
  std::sort(jets.begin(),jets.end(),HighestPt());
}






//remove unselected muon from vector of muon

void Selection::SelectMuons(){
  std::vector<reco::Muon> tmpmuons = muons;
  muons.clear();
  for(unsigned int i=0;i<tmpmuons.size();i++){
    bool isIsolated = false;
    if( RelativeIso(tmpmuons[i]) < MuonRelIso_ && VetoIsoDeposit(tmpmuons[i])) isIsolated =  true;
    if(tmpmuons[i].isGlobalMuon() && tmpmuons[i].globalTrack()->normalizedChi2 ()< Chi2Cut_ && fabs(tmpmuons[i].innerTrack()->dxy(beamSpot))<MuonD0Cut_ && tmpmuons[i].innerTrack()->numberOfValidHits ()>=NofValidHits_ && fabs(tmpmuons[i].eta())<MuonEtaThreshold_ && tmpmuons[i].pt()>MuonPtThreshold_ &&  isIsolated){
	muons.push_back(tmpmuons[i]);
    }
  }
  std::sort(muons.begin(),muons.end(),HighestPt());
}

//remove unselected muon from vector of muon
//for dilepton channels







void Selection::SelectMuonsDiLeptSimpleSel(){
  std::vector<reco::Muon> tmpmuons = muons;
  muons.clear();
  for(unsigned int i=0;i<tmpmuons.size();i++){
      if(   tmpmuons[i].isGlobalMuon() == 1
         && fabs(tmpmuons[i].eta())   < 2.4
         && tmpmuons[i].pt()          > 20 
       ){
	muons.push_back(tmpmuons[i]);
    }
  }
  std::sort(muons.begin(),muons.end(),HighestPt());
}


void Selection::SelectMuonsDiLeptNonIso(){
  std::vector<reco::Muon> tmpmuons = muons;
  muons.clear();
  for(unsigned int i=0;i<tmpmuons.size();i++){
      if(
           
	    tmpmuons[i].globalTrack()->normalizedChi2 ()< 10 
	 && tmpmuons[i].globalTrack()->hitPattern().numberOfValidMuonHits() >0
	 && fabs(tmpmuons[i].innerTrack()->dxy(beamSpot)) < MuonD0Cut_
	 ){
	muons.push_back(tmpmuons[i]);
    }
  }
  std::sort(muons.begin(),muons.end(),HighestPt());
}


void Selection::SelectMuonsDiLeptIso(){
  std::vector<reco::Muon> tmpmuons = muons;
  muons.clear();
  for(unsigned int i=0;i<tmpmuons.size();i++){
    bool isIsolated = false;
    if( RelativeIso(tmpmuons[i]) < 0.1) isIsolated =  true;
    if(
         isIsolated 
       ){
	muons.push_back(tmpmuons[i]);
    }
  }
  std::sort(muons.begin(),muons.end(),HighestPt());
}















// does not take isolation criteria into account
void Selection::SelectMuonsNonIso(){
  std::vector<reco::Muon> tmpmuons = muons;
  muons.clear();
  for(unsigned int i=0;i<tmpmuons.size();i++){
    if(tmpmuons[i].isGlobalMuon() && tmpmuons[i].globalTrack()->normalizedChi2 ()< Chi2Cut_ && fabs(tmpmuons[i].innerTrack()->dxy(beamSpot))<MuonD0Cut_ && tmpmuons[i].innerTrack()->numberOfValidHits ()>=NofValidHits_ && fabs(tmpmuons[i].eta())< MuonEtaThreshold_ && tmpmuons[i].pt()> MuonPtThreshold_ ){
	muons.push_back(tmpmuons[i]);
    }
  }
  std::sort(muons.begin(),muons.end(),HighestPt());
}

void Selection::SelectMuons(float PtThr, float EtaThr,float MuonRelIso){
  std::vector<reco::Muon> tmpmuons = muons;
  muons.clear();
  for(unsigned int i=0;i<tmpmuons.size();i++){
    bool isIsolated = false;
    if( RelativeIso(tmpmuons[i]) < MuonRelIso && VetoIsoDeposit(tmpmuons[i])) isIsolated =  true;
    if(tmpmuons[i].isGlobalMuon() && tmpmuons[i].globalTrack()->normalizedChi2 ()< Chi2Cut_ && fabs(tmpmuons[i].innerTrack()->dxy(beamSpot))<MuonD0Cut_ && tmpmuons[i].innerTrack()->numberOfValidHits ()>=NofValidHits_ && fabs(tmpmuons[i].eta())<EtaThr && tmpmuons[i].pt()>PtThr && isIsolated){
	muons.push_back(tmpmuons[i]);
    }
  }
  std::sort(muons.begin(),muons.end(),HighestPt());
}


void Selection::SelectMuonsDiLeptIso(float PtThr, float EtaThr,float MuonRelIso){
  std::vector<reco::Muon> tmpmuons = muons;
  muons.clear();
  for(unsigned int i=0;i<tmpmuons.size();i++){
    bool isIsolated = false;
    //if( RelativeIsoCalo(tmpmuons[i]) > MuonRelIsoCalo_ && RelativeIsoTrk(tmpmuons[i]) > MuonRelIsoTrk_ ) isIsolated =  true;
    if( RelativeIso(tmpmuons[i]) < MuonRelIso ) isIsolated =  true;
    if(tmpmuons[i].isGlobalMuon() && tmpmuons[i].globalTrack()->normalizedChi2 ()< Chi2Cut_ &&  tmpmuons[i].innerTrack()->numberOfValidHits ()>=NofValidHits_ && fabs(tmpmuons[i].eta())<EtaThr && tmpmuons[i].pt()>PtThr && isIsolated){
	muons.push_back(tmpmuons[i]);
    }
  }
  std::sort(muons.begin(),muons.end(),HighestPt());
}






// does not take isolation criteria into account
void Selection::SelectMuonsNonIso(float PtThr, float EtaThr,float MuonRelIso){
  std::vector<reco::Muon> tmpmuons = muons;
  muons.clear();
  for(unsigned int i=0;i<tmpmuons.size();i++){
    if(tmpmuons[i].isGlobalMuon() && tmpmuons[i].globalTrack()->normalizedChi2 ()< Chi2Cut_ && fabs(tmpmuons[i].innerTrack()->dxy(beamSpot))<MuonD0Cut_ && tmpmuons[i].innerTrack()->numberOfValidHits ()>=NofValidHits_ && fabs(tmpmuons[i].eta())<EtaThr && tmpmuons[i].pt()>PtThr){
	muons.push_back(tmpmuons[i]);
    }
  }
  std::sort(muons.begin(),muons.end(),HighestPt());
}

void Selection::SelectElectrons(){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  electrons.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
    bool isIsolated = false;
    if(RelativeIso(tmpelectrons[i]) < ElectronRelIso_) isIsolated =  true;
    if( fabs(tmpelectrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(tmpelectrons[i].eta())<ElectronEtaThreshold_ && tmpelectrons[i].pt()>ElectronPtThreshold_ && isIsolated){
	electrons.push_back(tmpelectrons[i]);
    }
  }
  std::sort(electrons.begin(),electrons.end(),HighestPt());
}














//for di-lepton isoaltion
void Selection::SelectElectronsDiLeptIso(){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  electrons.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
    bool isIsolated = false;   
    //if( RelativeIsoCalo(tmpelectrons[i]) > ElectronRelIsoCalo_ && RelativeIsoTrk(tmpelectrons[i]) > ElectronRelIsoTrk_ ) isIsolated =  true;
    if(RelativeIso(tmpelectrons[i]) < ElectronRelIso_) isIsolated =  true;
    if( 
        isIsolated
    ){
	electrons.push_back(tmpelectrons[i]);
    }
  }
  std::sort(electrons.begin(),electrons.end(),HighestPt());
}



void Selection::SelectElectronsDiLeptSimpleeID(){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  std::vector<float >            tmpeID       = eID;
  std::vector<std::pair< reco::GsfElectron, float > > electron_eID;
  std::vector<std::pair< reco::GsfElectron, float > > electron_dxy;
  electrons.clear();
  eID.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
      if( 
           fabs(tmpelectrons[i].eta())  < 2.4 
	&& tmpelectrons[i].pt() > 20 
	
	){
	std::pair< reco::GsfElectron, float > tmpPair(tmpelectrons[i],   tmpeID[i]) ;
	electron_eID.push_back(tmpPair);
    }
  }
  std::sort(electron_eID.begin(),electron_eID.end(),HighestPt());
  for(unsigned int i=0; i<electron_eID.size(); i++){
   electrons.push_back(electron_eID[i].first);
   eID.push_back(electron_eID[i].second);
  }
}



void Selection::SelectElectronsDiLeptNonIsoeID(){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  std::vector<float >            tmpeID       = eID;
  std::vector<std::pair< reco::GsfElectron, float > > electron_eID;
  electrons.clear();
  eID.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){	   
      if(    
           fabs(tmpelectrons[i].gsfTrack()->dxy(beamSpot)) < ElectronD0Cut_  
	    && tmpeID[i] == 1
	){
	std::pair< reco::GsfElectron, float > tmpPair(tmpelectrons[i], tmpeID[i]) ;
	electron_eID.push_back(tmpPair);
    }
  }
  std::sort(electron_eID.begin(),electron_eID.end(),HighestPt());
  for(unsigned int i=0; i<electron_eID.size(); i++){
   electrons.push_back(electron_eID[i].first);
   eID.push_back(electron_eID[i].second);
  }
}




//for di-lepton isoaltion

void Selection::SelectElectronsDiLeptIsoeID(){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  std::vector<float >            tmpeID       = eID;
  std::vector<std::pair< reco::GsfElectron, float > > electron_eID;
  electrons.clear();
  eID.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
    bool isIsolated = false;  
    if(RelativeIso(tmpelectrons[i]) < 0.1) isIsolated =  true; 
    if(   
         isIsolated 
      ){
	std::pair< reco::GsfElectron, float > tmpPair(tmpelectrons[i], tmpeID[i]) ;
	electron_eID.push_back(tmpPair);
    }
  }
  std::sort(electron_eID.begin(),electron_eID.end(),HighestPt());
  for(unsigned int i=0; i<electron_eID.size(); i++){
   electrons.push_back(electron_eID[i].first);
   eID.push_back(electron_eID[i].second);
  }
}















// does not take isolation criteria into account
void Selection::SelectElectronsNonIso(){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  electrons.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
    if( fabs(tmpelectrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(tmpelectrons[i].eta())<ElectronEtaThreshold_
    && tmpelectrons[i].pt()>ElectronPtThreshold_ ){
	electrons.push_back(tmpelectrons[i]);
    }
  }
  std::sort(electrons.begin(),electrons.end(),HighestPt());
}





void Selection::SelectElectrons(float PtThr, float EtaThr,float ElectronRelIso){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  electrons.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
    bool isIsolated = false;
    if(RelativeIso(tmpelectrons[i]) < ElectronRelIso) isIsolated =  true;
    if( fabs(tmpelectrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(tmpelectrons[i].eta())<EtaThr && tmpelectrons[i].pt()>PtThr && isIsolated){
	electrons.push_back(tmpelectrons[i]);
    }
  }
  std::sort(electrons.begin(),electrons.end(),HighestPt());
}





void Selection::SelectElectronsDiLeptIso(float PtThr, float EtaThr,float ElectronRelIso){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  electrons.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
    bool isIsolated = false; 
    //if( RelativeIsoCalo(tmpelectrons[i]) > ElectronRelIsoCalo_ && RelativeIsoTrk(tmpelectrons[i]) > ElectronRelIsoTrk_ ) isIsolated =  true;
    if(RelativeIso(tmpelectrons[i]) < ElectronRelIso) isIsolated =  true;
    if( fabs(tmpelectrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(tmpelectrons[i].eta())<EtaThr && tmpelectrons[i].pt()>PtThr && isIsolated){
	electrons.push_back(tmpelectrons[i]);
    }
  }
  std::sort(electrons.begin(),electrons.end(),HighestPt());
}




void Selection::SelectElectronsDiLeptIsoeID(float PtThr, float EtaThr,float ElectronRelIso){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  std::vector<float >            tmpeID       = eID;
  std::vector<std::pair< reco::GsfElectron, float > > electron_eID;
  electrons.clear();
  eID.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
    bool isIsolated = false;    
    if(RelativeIso(tmpelectrons[i]) < ElectronRelIso) isIsolated =  true; 
    if( RelativeIsoCalo(tmpelectrons[i]) > ElectronRelIsoCalo_ && RelativeIsoTrk(tmpelectrons[i]) > ElectronRelIsoTrk_ ) isIsolated =  true;
    if( fabs(tmpelectrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(tmpelectrons[i].eta())<EtaThr && tmpelectrons[i].pt()>PtThr && isIsolated && tmpeID[i] == 1){
	
	std::pair< reco::GsfElectron, float > tmpPair(tmpelectrons[i], tmpeID[i]) ;
	electron_eID.push_back(tmpPair);
	
	
    }
  }
  
  std::sort(electron_eID.begin(),electron_eID.end(),HighestPt());
  for(unsigned int i=0; i<electron_eID.size(); i++){
   electrons.push_back(electron_eID[i].first);
   eID.push_back(electron_eID[i].second);
  }
  
}



// does not take isolation criteria into account
void Selection::SelectElectronsNonIso(float PtThr, float EtaThr,float ElectronRelIso){
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  electrons.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
    if( fabs(tmpelectrons[i].gsfTrack()->dxy(beamSpot))<ElectronD0Cut_  && fabs(tmpelectrons[i].eta())<EtaThr && tmpelectrons[i].pt()>PtThr){
	electrons.push_back(tmpelectrons[i]);
    }
  }
  std::sort(electrons.begin(),electrons.end(),HighestPt());
}
  
  
  
  
//////////////////////////////////////////
/// Selection for semi-leptonic channel
///  uses electron or muon
///  possibility for a veto on 2nd lepton
//////////////////////////////////////////
 
bool Selection::isSelected (float PtThrJets, float EtaThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso, unsigned int Njets, std::string leptonType, unsigned int Nleptons, bool Veto2ndLepton) const{
  if(leptonType !=std::string("muon") && leptonType != std::string("electron")){
    edm::LogError("InputUnknown") <<"leptonType argument is not valid:"<<leptonType.c_str()<<" instead of 'muon' or 'electron'"<< std::endl; 
  	return false;
  }
  if(Veto2ndLepton){
    if(leptonType == std::string("muon"))
	  if(GetSelectedMuons(PtThrMuons, EtaThrMuons, MuonRelIso).size()==Nleptons && GetSelectedElectrons(PtThrElectrons, EtaThrElectrons, ElectronRelIso).size()==0 && GetSelectedJets(PtThrJets, EtaThrJets, JetEHThreshold_).size()>=Njets) return true;
    if(leptonType == std::string("electron"))
	  if(GetSelectedElectrons(PtThrMuons, EtaThrMuons, MuonRelIso).size()==0 && GetSelectedElectrons(PtThrElectrons, EtaThrElectrons, ElectronRelIso).size()==Nleptons && GetSelectedJets(PtThrJets, EtaThrJets, JetEHThreshold_).size()>=Njets) return true;
  }
  else{
    if(leptonType == std::string("muon"))
	  if(GetSelectedMuons(PtThrMuons, EtaThrMuons, MuonRelIso).size()>=Nleptons && GetSelectedJets(PtThrJets, EtaThrJets, JetEHThreshold_).size()>=Njets) return true;
    if(leptonType == std::string("electron"))
  	  if(GetSelectedElectrons(PtThrElectrons, EtaThrElectrons, ElectronRelIso).size()>=Nleptons && GetSelectedJets(PtThrJets, EtaThrJets, JetEHThreshold_).size()>=Njets) return true;
  }
  return false; 	
}

bool Selection::isSelected(unsigned int Njets, std::string leptonType, unsigned int Nleptons, bool Veto2ndLepton) const{
	return(isSelected (JetPtThreshold_, JetEtaThreshold_, MuonPtThreshold_, MuonEtaThreshold_, MuonRelIso_, ElectronPtThreshold_, ElectronEtaThreshold_, ElectronRelIso_, Njets, leptonType, Nleptons, Veto2ndLepton));
}
 
 
//////////////////////////////////////////
/// Selection for di-leptonic channel (or multi-leptonic channel !! Nleptons>=2)
/// uses electrons and/or muons
//////////////////////////////////////////

bool Selection::isSelected (float PtThrJets, float EtaThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso, unsigned int Njets , unsigned int Nmuons, unsigned int Nelectrons) const{
	if(GetSelectedMuons(PtThrMuons, EtaThrMuons, MuonRelIso).size()>=Nmuons && GetSelectedElectrons(PtThrElectrons, EtaThrElectrons, ElectronRelIso).size()>=Nelectrons && GetSelectedJets(PtThrJets, EtaThrJets, JetEHThreshold_).size()>=Njets) return true;
	return false;
}

bool Selection::isSelected(unsigned int Njets, unsigned int Nmuons, unsigned int Nelectrons) const{
	return(isSelected (JetPtThreshold_, JetEtaThreshold_, MuonPtThreshold_, MuonEtaThreshold_, MuonRelIso_, ElectronPtThreshold_, ElectronEtaThreshold_, ElectronRelIso_, Njets, Nmuons, Nelectrons));
}



bool Selection::isSelectedFromObjects (float PtThrJets, float EtaThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso, unsigned int Njets , unsigned int Nmuons, unsigned int Nelectrons) const{
	if(GetMuons().size()>=Nmuons && GetElectrons().size()>=Nelectrons && GetJets().size()>=Njets) return true;
	return false;
}

bool Selection::isSelectedFromObjects(unsigned int Njets, unsigned int Nmuons, unsigned int Nelectrons) const{
	return(isSelectedFromObjects (JetPtThreshold_, JetEtaThreshold_, MuonPtThreshold_, MuonEtaThreshold_, MuonRelIso_, ElectronPtThreshold_, ElectronEtaThreshold_, ElectronRelIso_, Njets, Nmuons, Nelectrons));
}

///////////////////////////////////////////////////
/// Print a summary of all the cuts currently set
///////////////////////////////////////////////////

void Selection::printSummary() {
	
      edm::LogVerbatim("MainResults") << "Currently set selection criteria:" << std::endl;
      
      edm::LogVerbatim("MainResults")<< "Lepton type selected = " << leptonType_ << std::endl;
      edm::LogVerbatim("MainResults")<< "Njets = " << Njets_ << std::endl;
      edm::LogVerbatim("MainResults")<< "JetPtThreshold = " << JetPtThreshold_ << std::endl;
      edm::LogVerbatim("MainResults")<< "JetEtaThreshold = " << JetEtaThreshold_ << std::endl;
      edm::LogVerbatim("MainResults")<< "JetEHThreshold = " << JetEHThreshold_ << std::endl;
      edm::LogVerbatim("MainResults")<< "MuonPtThreshold = " << MuonPtThreshold_ << std::endl;
      edm::LogVerbatim("MainResults")<< "MuonEtaThreshold = " << MuonEtaThreshold_ << std::endl;
      edm::LogVerbatim("MainResults")<< "MuonD0Cut = " << MuonD0Cut_ << std::endl;
      edm::LogVerbatim("MainResults")<< "Chi2Cut = " << Chi2Cut_ << std::endl;
      edm::LogVerbatim("MainResults")<< "NofValidHits = " << NofValidHits_ << std::endl;
      edm::LogVerbatim("MainResults")<< "MuonRelIso = " << MuonRelIso_ << std::endl;
      edm::LogVerbatim("MainResults")<< "MuonVetoEM = " << MuonVetoEM_ << std::endl;
      edm::LogVerbatim("MainResults")<< "MuonVetoHad = " << MuonVetoHad_ << std::endl; 
      edm::LogVerbatim("MainResults")<< "ElectronPtThreshold = " << ElectronPtThreshold_ << std::endl;
      edm::LogVerbatim("MainResults")<< "ElectronEtaThreshold = " << ElectronEtaThreshold_ << std::endl;
      edm::LogVerbatim("MainResults")<< "ElectronD0Cut = " << ElectronD0Cut_ << std::endl;
      edm::LogVerbatim("MainResults")<< "ElectronRelIso = " << ElectronRelIso_ << std::endl;
      edm::LogVerbatim("MainResults")<< "vetoEBEETransitionRegion = " << vetoEBEETransitionRegion_ << std::endl;
      edm::LogVerbatim("MainResults")<< "METThreshold = " << METThreshold_ << std::endl;
      edm::LogVerbatim("MainResults")<< "beamSpot(x, y, z)  =  (" << beamSpot.x() << " ," << beamSpot.y() << ", " << beamSpot.z() << ")" << std::endl;
 
      edm::LogVerbatim("MainResults")<< "Selection criteria for Loose muons:" << std::endl;          
      edm::LogVerbatim("MainResults")<< "PtThrMuonLoose = " << PtThrMuonLoose_ << std::endl;
      edm::LogVerbatim("MainResults")<< "EtaThrMuonLoose = " << EtaThrMuonLoose_ << std::endl;       
      edm::LogVerbatim("MainResults")<< "RelIsoThrMuonLoose = " << RelIsoThrMuonLoose_ << std::endl;     
      edm::LogVerbatim("MainResults")<< "PtThrElectronLoose = " << PtThrElectronLoose_ << std::endl;    
      edm::LogVerbatim("MainResults")<< "EtaThrElectronLoose = " << EtaThrElectronLoose_ << std::endl;   
      edm::LogVerbatim("MainResults")<< "RelIsoThrElectronLoose_" << RelIsoThrElectronLoose_ << std::endl;
      
}


void Selection::RemoveElecClose2Mu(float deltaREMCut){
   
  std::vector<reco::GsfElectron> tmpelectrons = electrons;
  std::vector<float >            tmpeID       = eID;
  std::vector<std::pair< reco::GsfElectron, float > > electron_eID;
  electrons.clear();
  eID.clear();
  for(unsigned int i=0;i<tmpelectrons.size();i++){
    bool isSepElectron = true;
    for(unsigned int j=0; j<muons.size(); j++){	
    
      double deltaR = pow(  pow(tmpelectrons[i].phi() - muons[j].phi(),2 ) + pow(tmpelectrons[i].eta() - muons[j].eta(),2 ) ,0.5);
      if( deltaR < 0.1 ) isSepElectron = false;
    }
    if(isSepElectron){
      std::pair< reco::GsfElectron, float > tmpPair(tmpelectrons[i], tmpeID[i]) ;
      electron_eID.push_back(tmpPair);
    }  
  }
  
  std::sort(electron_eID.begin(),electron_eID.end(),HighestPt());
  for(unsigned int i=0; i<electron_eID.size(); i++){
   electrons.push_back(electron_eID[i].first);
   eID.push_back(electron_eID[i].second);
  }
  
   
}





void Selection::RemoveJetClose2Muon(float deltaREMCut){


  std::vector<reco::CaloJet> tmpjets;
  tmpjets = jets;
  jets.clear();
  
  for(unsigned int i=0;i<tmpjets.size();i++){
    bool isSepJets = true;
    for(unsigned int j=0; j<muons.size(); j++){	
    
      //double deltaR = pow(  pow(tmpjets[i].phi() - muons[j].phi(),2 ) + pow(tmpjets[i].eta() - muons[j].eta(),2 ) ,0.5);
      double deltaR = reco::deltaR(tmpjets[i].eta(),tmpjets[i].phi(), muons[j].eta(),  muons[j].phi() );
      if( deltaR < deltaREMCut ) isSepJets = false;
    }
    if(isSepJets){
      jets.push_back(tmpjets[i]);
    }  
  }
  std::sort(jets.begin(),jets.end(),HighestPt());

}



void Selection::RemoveJetClose2Electron(float deltaREMCut){


  std::vector<reco::CaloJet> tmpjets;
  tmpjets = jets;
  jets.clear();
  
  for(unsigned int i=0;i<tmpjets.size();i++){
    bool isSepJets = true;
    for(unsigned int j=0; j<electrons.size(); j++){	
    
      //double deltaR = pow(  pow(tmpjets[i].phi() - electrons[j].phi(),2 ) + pow(tmpjets[i].eta() - electrons[j].eta(),2 ) ,0.5);
      double deltaR = reco::deltaR(tmpjets[i].eta(), tmpjets[i].phi(),  electrons[j].eta(), electrons[j].phi());
      if( deltaR < deltaREMCut ) isSepJets = false;
    }
    if(isSepJets){
      jets.push_back(tmpjets[i]);
    }  
  }
  std::sort(jets.begin(),jets.end(),HighestPt());

}










