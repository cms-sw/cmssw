#ifndef MuonMiniAOD_H
#define MuonMiniAOD_H


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h" //
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/PatCandidates/interface/Flags.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


class MuonMiniAOD : public DQMEDAnalyzer {
 public:

  /// Constructor
  MuonMiniAOD(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~MuonMiniAOD();

  /// Inizialize parameters for histo binning
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
 

 private:
  // ----------member data ---------------------------
    
  edm::ParameterSet parameters;
  
  edm::EDGetTokenT< edm::View<pat::Muon> >   theMuonCollectionLabel_;
  // Switch for verbosity
  std::string metname;
    
  // Monitors:
  std::vector<MonitorElement*> workingPoints;
  /* MonitorElement* mediumMuons; */
  /* MonitorElement* looseMuons; */
  /* MonitorElement* softMuons; */
  /* MonitorElement* highPtMuons; */

  //Vertex requirements
  bool doPVCheck_;
  edm::EDGetTokenT<reco::VertexCollection> theVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot>         theBeamSpotLabel_;

  bool PassesCut_A(edm::View<pat::Muon>::const_iterator,reco::Vertex,TString);
  bool PassesCut_B(edm::View<pat::Muon>::const_iterator,reco::Vertex,TString);

};
#endif
