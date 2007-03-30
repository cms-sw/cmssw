#include "RecoMuon/MuonIsolation/interface/MuIsoByTrackPt.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"


using namespace std;

class IsolationExample : public edm::EDAnalyzer {
public:
  IsolationExample(const edm::ParameterSet& conf);
  ~IsolationExample();
  virtual void beginJob(const edm::EventSetup& iSetup);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() { }
private:
  string theMuonLabel;
  edm::ParameterSet theAlgorithmConfig;
  MuIsoByTrackPt * theIsolation;
  unsigned long theEventCount;
};

IsolationExample::IsolationExample(const edm::ParameterSet& conf)
  : theMuonLabel(conf.getUntrackedParameter<std::string>("MuonCollectionLabel")), 
    theAlgorithmConfig(conf.getParameter<edm::ParameterSet>("AlgorithmPSet")),
    theIsolation(0),
    theEventCount(0)
{
  LogDebug("IsolationExample") <<" CTOR"<<endl;
}

IsolationExample::~IsolationExample()
{
  delete theIsolation;
}

void IsolationExample::beginJob(const edm::EventSetup& iSetup)
{
  theIsolation = new MuIsoByTrackPt(theAlgorithmConfig);
}

void IsolationExample:: analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug("IsolationExample::analyze")<<" ============== analysis of event: "<< ++theEventCount;
  edm::Handle<reco::TrackCollection> trackCollection;
  ev.getByLabel(theMuonLabel, trackCollection);

  const reco::TrackCollection  muons = *(trackCollection.product());
  typedef reco::TrackCollection::const_iterator IT;
  for (IT it=muons.begin(), itEnd = muons.end();  it < itEnd; ++it) {
    LogTrace("") <<"muon pt="<< (*it).pt();
    for( int i=0; i<5; ++i) {
      float coneSize = 0.2*i;
      theIsolation->setConeSize(coneSize);
      float variable= theIsolation->isolation(ev,es,*it);
      LogTrace("") <<" dR cone: "<<coneSize<<" isolationvariable: " << variable;
    }
  }
}

DEFINE_FWK_MODULE(IsolationExample);
