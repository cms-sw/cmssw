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

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"


using namespace std;

class MuIsoDepositAnalyzer : public edm::EDAnalyzer {
public:
  MuIsoDepositAnalyzer(const edm::ParameterSet& conf);
  ~MuIsoDepositAnalyzer();
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() { }
private:
  edm::InputTag theMuonLabel;
  std::vector<edm::InputTag> theIsoDepositInputTags;
  unsigned long theEventCount;
};

MuIsoDepositAnalyzer::MuIsoDepositAnalyzer(const edm::ParameterSet& conf)
  : theMuonLabel(conf.getUntrackedParameter<edm::InputTag>("MuonCollectionLabel")), 
    theIsoDepositInputTags(conf.getUntrackedParameter<std::vector<edm::InputTag> >("DepositInputTags")),
    theEventCount(0)
{
  LogDebug("MuIsoDepositAnalyzer") <<" CTOR"<<endl;
}

MuIsoDepositAnalyzer::~MuIsoDepositAnalyzer()
{
}

void MuIsoDepositAnalyzer::beginJob()
{
}

void MuIsoDepositAnalyzer:: analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  LogDebug("MuIsoDepositAnalyzer::analyze")<<" ============== analysis of event: "<< ++theEventCount;
  edm::Handle<reco::TrackCollection> trackCollection;
  ev.getByLabel(theMuonLabel, trackCollection);

  unsigned int nDeps = theIsoDepositInputTags.size();
  std::vector<edm::Handle<reco::IsoDepositMap> > isoDeposits(nDeps);
  for (unsigned int iDep = 0; iDep < nDeps; ++iDep){
    ev.getByLabel(theIsoDepositInputTags[iDep], isoDeposits[iDep]);
  }

  const reco::TrackCollection&  muons = *(trackCollection.product());
  typedef reco::TrackCollection::const_iterator IT;
  unsigned int iMu = 0;
  for (IT it=muons.begin(), itEnd = muons.end();  it < itEnd; ++it) {
    LogTrace("") <<"muon pt="<< (*it).pt();
    reco::TrackRef muRef(trackCollection, iMu); ++iMu;
    for (unsigned int iDep=0; iDep < isoDeposits.size();++iDep){
      const reco::IsoDeposit& dep = (*isoDeposits[iDep])[muRef];
      LogTrace("") <<theIsoDepositInputTags[iDep]
		   <<dep.print();
    }
    for( int i=0; i<10; ++i) {
      float coneSize = 0.1*i;
      LogTrace("") <<" dR cone: "<<coneSize<<" isolationvariables: ";
      for (unsigned int iDep=0; iDep < isoDeposits.size();++iDep){
	const reco::IsoDeposit& dep = (*isoDeposits[iDep])[muRef];
	      LogTrace("") <<theIsoDepositInputTags[iDep]
			   <<" (eta, phi)= ("<<dep.eta()<<", "<<dep.phi()
			   <<") V(eta, phi)= ("<<dep.veto().vetoDir.eta()<<", "<<dep.veto().vetoDir.phi()
			   <<") muE= "<<dep.candEnergy()
			   <<" "<<dep.depositWithin(coneSize);
      }
    }
  }
}

DEFINE_FWK_MODULE(MuIsoDepositAnalyzer);
