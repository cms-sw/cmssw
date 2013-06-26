//#define DEBUG
// System include files
// --------------------
#include <memory>
#include <vector>
#include <string>

// User include files
// ------------------
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


// Collaborating Class Header
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "MuScleFitUtils.h"

#include <CLHEP/Vector/LorentzVector.h>

// Class declaration
// -----------------

class MuScleFitGenFilter : public edm::EDFilter {
 public:
  explicit MuScleFitGenFilter(const edm::ParameterSet&);
  ~MuScleFitGenFilter();

 private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() {};

  std::string genParticlesName_;
  unsigned int totalEvents_;
  unsigned int eventsPassingTheFilter_;
};

// Constructor
// -----------
MuScleFitGenFilter::MuScleFitGenFilter(const edm::ParameterSet& iConfig) :
  genParticlesName_( iConfig.getUntrackedParameter<std::string>("GenParticlesName", "genParticles") ),
  totalEvents_(0),
  eventsPassingTheFilter_(0)
{
  MuScleFitUtils::resfind = iConfig.getParameter<std::vector<int> >("resfind");
}

// Destructor
// ----------
MuScleFitGenFilter::~MuScleFitGenFilter()
{
  std::cout << "Total number of events = " << totalEvents_ << std::endl;
  std::cout << "Events passing the filter = " << eventsPassingTheFilter_ << std::endl;
}

// Method called for each event 
// ----------------------------
bool MuScleFitGenFilter::filter(edm::Event& event, const edm::EventSetup& iSetup)
{
  ++totalEvents_;

  edm::Handle<edm::HepMCProduct> evtMC;

  std::pair<lorentzVector,lorentzVector> genPair;

  event.getByLabel( genParticlesName_, evtMC );
  if( evtMC.isValid() ) {

    genPair = MuScleFitUtils::findGenMuFromRes(evtMC.product());
  }
  else {
    edm::Handle<reco::GenParticleCollection> genParticles;
    event.getByLabel( genParticlesName_, genParticles );
    if( genParticles.isValid() ) {
      
      genPair = MuScleFitUtils::findGenMuFromRes(genParticles.product());
    }
    else {
      std::cout << "ERROR: no generator info found" << std::endl;
      return false;
    }
  }
  lorentzVector emptyVec(0.,0.,0.,0.);
  if( (genPair.first == emptyVec) || (genPair.second == emptyVec) ) {
    return false;
  }

  ++eventsPassingTheFilter_;

  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuScleFitGenFilter);
