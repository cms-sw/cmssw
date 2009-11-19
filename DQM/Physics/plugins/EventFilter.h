// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
// DataFormat
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TMath.h"

//user
#include "../interface/Selection.h"

/**
   \class   EventFilter EventFilter.h "DQM/Physics/plugins/EventFilter.h"

   \brief   Add a one sentence description here...

  This module is an EDFilter
  It takes all the objects collection as input (edm::View):
   Electrons, Muons, CaloJets, CaloMETs, trigger
  Using the Selection class, it returns a boolean according to the configuration given
    by the configuration file
  true: event selected
  false: event not selected
  It's actually running for semi-leptonic or di-leptonic channel
*/

class EventFilter : public edm::EDFilter {
public:
  explicit EventFilter(const edm::ParameterSet&);
  ~EventFilter();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  
  edm::InputTag labelMuons_;
  edm::InputTag labelElectrons_;
  edm::InputTag labelJets_;
  edm::InputTag labelMETs_;
  edm::InputTag labelBeamSpot_;
  edm::InputTag labelTriggerResults_;
  
  bool verbose_;
  
  //Configuration
  //MET
  double METCut;//
  //Jets
  int NofJets;
  double PtThrJets;
  double EtaThrJets;
  double EHThrJets;
  //Muons
  int NofMuons;//
  double PtThrMuons;
  double EtaThrMuons;
  double MuonRelIso;
  double MuonVetoEM;
  double MuonVetoHad;
  double MuonD0Cut;
  int Chi2Cut;
  int NofValidHits;
  //Electrons
  int NofElectrons;//
  double PtThrElectrons;
  double EtaThrElectrons;
  double ElectronRelIso;
  double ElectronD0Cut;
  //
  bool Veto2ndLepton;
  //HLT
  std::string triggerPath;
};
