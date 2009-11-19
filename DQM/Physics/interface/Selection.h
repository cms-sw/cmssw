#ifndef SELECTION_H
#define SELECTION_H


// system include files
#include <memory>
#include <vector>
#include <string>

// DataFormat
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


//jet corrections
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

/**
   \class   Selection Selection.h "DQM/Physics/interface/Selection.h"

   \brief   class to fill monitor histograms for electrons

   This class is tool that can be used by other modules:
   EventFilter, channel-specific modules (LeptonJetsChecker) ...
   For the following objects:
   Electrons, Muons, CaloJets, CaloMets
   it takes a edm::View as input and can return a vector of selected objects (ordered by Pt)
   -> methods: GetSelectedJets, GetSelectedMuons, GetSelectedElectrons  (isolated or not)
   The class is configurable via the methods:
   Set, SetConfiguration, SetMuonConfig, SetElectronConfig
   -> the list of variables dedicated to selection (quality cuts) can be found in the private members section 
   isSelected methods are implemented for semi-leptonic and di-leptonic channels and return booleans
   -> it uses electrons/muons and Veto2ndLepton for semi-leptonic channel
*/

struct HighestPt{
    bool operator()( reco::CaloJet j1, reco::CaloJet j2 ) const{
    	return j1.pt() > j2.pt() ;
    }
    bool operator()( reco::Muon j1, reco::Muon j2 ) const{
    	return j1.pt() > j2.pt() ;
    }
    bool operator()( reco::GsfElectron j1, reco::GsfElectron j2 ) const{
    	return j1.pt() > j2.pt() ;
    }
    bool operator()( std::pair< reco::GsfElectron, float > j1,  std::pair< reco::GsfElectron, float > j2 ) const{
    	return j1.first.pt() > j2.first.pt() ;
    }
};
				

class Selection{
 public:
  
  Selection();
  Selection(const Selection &);
  virtual ~Selection(){};
  Selection& operator=(const Selection &);
  
  
  ////////////////////////
  //Configuration
  ////////////////////////
  
  //Muon+jets
  void Set(const reco::BeamSpot, const edm::View<reco::CaloJet>&, const edm::View<reco::Muon>&, const edm::View<reco::CaloMET>&);
  void SetConfiguration(float PtThrJets, float EtaThrJets, float EHThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float MuonVetoEM, float MuonVetoHad);
  //Electron+jets
  void Set(const reco::BeamSpot, const edm::View<reco::CaloJet>&, const edm::View<reco::GsfElectron>&, const edm::View<reco::CaloMET>&);
  void SetConfiguration(float PtThrJets, float EtaThrJets, float EHThrJets, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso);
      //Di-leptons+jets
  void Set(const reco::BeamSpot, const edm::View<reco::CaloJet>&, const edm::View<reco::Muon>&, const edm::View<reco::GsfElectron>&,const edm::View<reco::CaloMET>&);
  void SetConfiguration(float PtThrJets, float EtaThrJets, float EHThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float MuonVetoEM, float MuonVetoHad, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso);
  void SetConfiguration(float PtThrJets, float EtaThrJets, float EHThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIsoCalo, float MuonRelIsoTrk, float
			MuonVetoEM, float MuonVetoHad, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIsoCalo, float ElectronRelIsoTrk);
  
  void SeteID(const edm::View<reco::GsfElectron>& electrons, const edm::ValueMap<float>& eIDmap);
  
  //muon
  void SetMuonConfig(float MuonD0Cut, int Chi2Cut, int NofValidHits);
  void SetMuonLooseConfig(float PtThrMuonLoose, float EtaThrMuonLoose, float RelIsoThrMuonLoose);
  //electron
  void SetElectronConfig(float ElectronD0Cut);
  void SetElectronConfig(float ElectronD0Cut, bool vetoEBEETransitionRegion, bool useElectronID = false);
  void SetElectronLooseConfig(float PtThrElectronLoose, float EtaThrElectronLoose, float RelIsoThrElectronLoose);
  //MET
  void SetMETConfig(float METThreshold);
  void SetJetConfig(float JetDeltaRLeptonJetThreshold, bool applyLeptonJetDeltaRCut);
  void SetLeptonType(std::string leptonType) {leptonType_ = leptonType;}
  ////////////////////////
  //Summary
  ////////////////////////
  void printSummary();
  
  ////////////////////////
  //Selection of objects
  ////////////////////////
  
  reco::TrackBase::Point getBeamSpot() {return beamSpot;}
  
  //Jets
  std::vector<reco::CaloJet> GetSelectedJets() const;
  std::vector<reco::CaloJet> GetSelectedJets(float PtThr, float EtaThr, float EHThr = 0.1) const;
  
  //NonIso means no isolation required
  
  //Electrons
  std::vector<reco::GsfElectron> GetSelectedElectrons() const;
  std::vector<reco::GsfElectron> GetSelectedElectrons(float PtThr, float EtaThr, float ElectronRelIso) const;
  std::vector<reco::GsfElectron> GetSelectedElectronsNonIso() const;
  std::vector<reco::GsfElectron> GetSelectedElectronsNonIso(float PtThr, float EtaThr, float ElectronRelIso) const;
  std::vector<reco::GsfElectron> GetSelectedElectronsLoose() const;
  //Muons
  std::vector<reco::Muon> GetSelectedMuons() const;
  std::vector<reco::Muon> GetSelectedMuons(float PtThr, float EtaThr, float MuonRelIso) const;
  std::vector<reco::Muon> GetSelectedMuonsNonIso() const;
  std::vector<reco::Muon> GetSelectedMuonsNonIso(float PtThr, float EtaThr, float MuonRelIso) const;
  std::vector<reco::Muon> GetSelectedMuonsLoose() const;
  //MET
  // GetSelectedMETs included for completeness;  if you only want to know if it passed the cut use METpass instead
  std::vector<reco::CaloMET> GetSelectedMETs() const;
  bool METpass() const;
  
  ////////////////////////
  //Selection of objects and refill object memebers
  ////////////////////////
  
  //Jets
  void SelectJets() ;
  void SelectJets(float PtThr, float EtaThr, float EHThr = 0.1) ;
  
  //Jets
  void SelectJets(const edm::Event& iEvent, const edm::EventSetup& iSetup, const JetCorrector *acorrector) ;
  void SelectJets(float PtThr, float EtaThr, float EHThr , const edm::Event& iEvent, const edm::EventSetup& iSetup, const JetCorrector *acorrector) ;
  
  //NonIso means no isolation required
  
  //Electrons
  void SelectElectrons() ;
  void SelectElectrons(float PtThr, float EtaThr, float ElectronRelIso) ;
  void SelectElectronsDiLeptIso() ;
  void SelectElectronsDiLeptIso(float PtThr, float EtaThr, float ElectronRelIso) ;
  void SelectElectronsDiLeptNonIsoeID() ;
  void SelectElectronsDiLeptSimpleeID() ;
  void SelectElectronsDiLeptIsoeID() ;
  void SelectElectronsDiLeptIsoeID(float PtThr, float EtaThr, float ElectronRelIso) ;
  void SelectElectronsNonIso() ;
  void SelectElectronsNonIso(float PtThr, float EtaThr, float ElectronRelIso) ;
  
  void RemoveElecClose2Mu(float deltaREMCut);
  void RemoveJetClose2Muon(float deltaREMCut);
  void RemoveJetClose2Electron(float deltaREMCut);
  
  //Muons
  void SelectMuons() ;
  void SelectMuons(float PtThr, float EtaThr, float MuonRelIso) ;
  void SelectMuonsDiLeptIso() ;
  void SelectMuonsDiLeptSimpleSel() ;
  void SelectMuonsDiLeptIso(float PtThr, float EtaThr, float MuonRelIso) ;
  void SelectMuonsDiLeptNonIso() ;
  void SelectMuonsNonIso() ;
  void SelectMuonsNonIso(float PtThr, float EtaThr, float MuonRelIso) ;
  
  ////////////////////////
  //Get objects
  ////////////////////////
  
  
  //Jets
  std::vector<reco::CaloJet> GetJets() const {return jets;};
  //Electrons
  std::vector<reco::GsfElectron> GetElectrons() const {return electrons;};
  //Muons
  std::vector<reco::Muon> GetMuons() const {return muons;};
  
  std::vector<float > GeteID() const {return eID;};
  
  
  
  
  ////////////////////////
  //Selection of events
  ////////////////////////
  
  // at least Njets and at least Nleptons (Nmuons or Nelectrons). All is defined as inclusif here
  
  //selection semi-leptonic
  bool isSelected(unsigned int Njets = 4, std::string leptonType = std::string("muon"), unsigned int Nleptons = 1, bool Veto2ndLepton = false) const;
  bool isSelected (float PtThrJets, float EtaThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso, unsigned int Njets = 4, std::string leptonType = std::string("muon"), unsigned int Nleptons = 1, bool Veto2ndLepton = false) const;
  //selection di-leptonic 
  bool isSelected(unsigned int Njets = 4, unsigned int Nmuons = 1, unsigned int Nelectrons = 1) const;
  bool isSelected (float PtThrJets, float EtaThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso, unsigned int Njets = 4, unsigned int Nmuons = 1, unsigned int Nelectrons = 1) const; 
  
  bool isSelectedFromObjects(unsigned int Njets = 4, unsigned int Nmuons = 1, unsigned int Nelectrons = 1) const;
  bool isSelectedFromObjects (float PtThrJets, float EtaThrJets, float PtThrMuons, float EtaThrMuons, float MuonRelIso, float PtThrElectrons, float EtaThrElectrons, float ElectronRelIso, unsigned int Njets = 4, unsigned int Nmuons = 1, unsigned int Nelectrons = 1) const; 
  
  
  //selection other ?? :to be implemented if needed
  
  ////////////////////////
  //methods for lepton
  ////////////////////////
  
  bool VetoIsoDeposit(const reco::Muon& muon) const;
  float RelativeIso     (const reco::Muon& muon) const;
  float RelativeIso     (const reco::GsfElectron& electron) const;
  float RelativeIsoCalo (const reco::Muon& muon) const;
  float RelativeIsoCalo (const reco::GsfElectron& electron) const;
  float RelativeIsoTrk  (const reco::Muon& muon) const;
  float RelativeIsoTrk  (const reco::GsfElectron& electron) const;
  
  
  
  
  
 private:
  
  std::string leptonType_;
  reco::TrackBase::Point  beamSpot;
  std::vector<reco::CaloJet> jets;
  std::vector<reco::GsfElectron> electrons;
  std::vector<float > eID;
  std::vector<reco::Muon> muons;
  std::vector<reco::CaloMET> mets;
  ///////////
  //jets   //
  ///////////
  int Njets_;
  float JetPtThreshold_;
  float JetEtaThreshold_;
  float JetEHThreshold_;
  float JetDeltaRLeptonJetThreshold_;
  bool  applyLeptonJetDeltaRCut_;
  ///////////
  //muon   //
  ///////////
  float MuonPtThreshold_;
  float MuonEtaThreshold_;
  float MuonD0Cut_;
  int Chi2Cut_;
  int NofValidHits_;
  //iso
  float MuonRelIso_;
  float MuonRelIsoCalo_;
  float MuonRelIsoTrk_;
  float MuonVetoEM_;
  float MuonVetoHad_; 
  ///////////
  //electron/
  ///////////
  float ElectronPtThreshold_;
  float ElectronEtaThreshold_;
  float ElectronD0Cut_;
  bool vetoEBEETransitionRegion_;
  bool useElectronID_;
  //iso
  float ElectronRelIso_;
  float ElectronRelIsoCalo_;
  float ElectronRelIsoTrk_;
  ///////////
  //met    //
  ///////////
  float METThreshold_;
  /////////////////////////////////////////
  //Loose thresholds for 2nd lepton veto //
  /////////////////////////////////////////
  float PtThrMuonLoose_;           
  float EtaThrMuonLoose_;        
  float RelIsoThrMuonLoose_;     
  float PtThrElectronLoose_;     
  float EtaThrElectronLoose_;    
  float RelIsoThrElectronLoose_;   	      
};

#endif
