#ifndef MuonPFAnalyzer_H
#define MuonPFAnalyzer_H

/** \class MuRecoAnalyzer
 *
 *  DQM monitoring source for PF muons
 *
 *  $Date: 2013/06/14 10:28:23 $
 *  $Revision: 1.1 $
 *  \author C. Battilana - CIEMAT
 */



//Base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <map>
#include <string>

class MuonPFAnalyzer : public edm::EDAnalyzer {

public:

  typedef std::pair<const reco::Muon*, const reco::GenParticle*> RecoGenPair;
  typedef std::vector<RecoGenPair> RecoGenCollection; 

  /// Constructor
  explicit MuonPFAnalyzer(const edm::ParameterSet&);

  /// Destructor
  ~MuonPFAnalyzer();

  /// Initialize an book plots
  virtual void beginRun(edm::Run const &, edm::EventSetup const &);

  /// Perform the PF - TUNEP muon analysis
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
private:

  // Book histos for a given group of plots (e.g. for Tight TUNEP muons)
  void bookHistos(const std::string & group);

  // Get a specific plot for a given group
  MonitorElement* getPlot(const std::string & group, const std::string & type);

  // Algorithm to identify muon pt track type
  int muonTrackType(const reco::Muon * muon, bool usePF);
  
  // Compute comb. rel. iso. (RECO based) for a given muon
  inline float combRelIso(const reco::Muon * muon);

  // Compute delta phi taking into account overflows
  inline float fDeltaPhi(float phi1, float phi2);

  // Set labels for code plots
  void setCodeLabels(MonitorElement *plot, int nAxis); 

  // Fill plot within its range limits
  void fillInRange(MonitorElement *plot, int nAxis, double x, double y = 0); 

  // Perform reco-gen geometrical matching on a best effort basis
  // (if runOnMC == false or no matched gen particles are available gen is set to 0 in theRecoGen)
  void recoToGenMatch( edm::Handle<reco::MuonCollection>        & reco, 
		       edm::Handle<reco::GenParticleCollection> & gen );

  const reco::Vertex getPrimaryVertex( edm::Handle<reco::VertexCollection> &vertex,
				       edm::Handle<reco::BeamSpot> &beamSpot );


  edm::InputTag theGenLabel;
  edm::InputTag theRecoLabel;
  edm::InputTag theVertexLabel;
  edm::InputTag theBeamSpotLabel;

  std::vector<std::string> theMuonKinds;

  DQMStore *theDbe;

  std::map<std::string,std::map<std::string,MonitorElement*> > thePlots;
  RecoGenCollection theRecoGen;

  double theHighPtTh;
  double theRecoGenR;
  double theIsoCut;

  bool theRunOnMC;

  std::string theFolder;

};
#endif  


