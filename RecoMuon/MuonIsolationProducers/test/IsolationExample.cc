#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

using namespace std;

class IsolationExample : public edm::one::EDAnalyzer<> {
public:
  IsolationExample(const edm::ParameterSet& conf);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::InputTag theMuonTag;

  edm::InputTag theTkDepMapTag;
  edm::InputTag theEcalDepMapTag;
  edm::InputTag theHcalDepMapTag;

  unsigned long theEventCount;
};

IsolationExample::IsolationExample(const edm::ParameterSet& conf)
    : theMuonTag(conf.getUntrackedParameter<edm::InputTag>("MuonCollection", edm::InputTag("muons"))),
      theTkDepMapTag(conf.getUntrackedParameter<edm::InputTag>("TkMapCollection", edm::InputTag("muIsoDepositTk"))),
      theEcalDepMapTag(conf.getUntrackedParameter<edm::InputTag>(
          "EcalMapCollection", edm::InputTag("muIsoDepositCalByAssociatorTowers:ecal"))),
      theHcalDepMapTag(conf.getUntrackedParameter<edm::InputTag>(
          "HcalMapCollection", edm::InputTag("muIsoDepositCalByAssociatorTowers:hcal"))),
      theEventCount(0) {
  LogDebug("IsolationExample") << " CTOR" << endl;
}

void IsolationExample::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  static const std::string metname = "IsolationExample";
  LogDebug(metname) << " ============== analysis of event: " << ++theEventCount;
  edm::Handle<edm::View<reco::Muon> > trackCollection;
  ev.getByLabel(theMuonTag, trackCollection);
  const edm::View<reco::Muon>& muons = *trackCollection;

  //! take iso deposits for tracks (contains (eta,phi, pt) of tracks in R<X (1.0) around each muon)
  edm::Handle<reco::IsoDepositMap> tkMapH;
  ev.getByLabel(theTkDepMapTag, tkMapH);

  //! take iso deposits for ecal (contains (eta,phi, pt) of ecal in R<X (1.0) around each muon)
  edm::Handle<reco::IsoDepositMap> ecalMapH;
  ev.getByLabel(theEcalDepMapTag, ecalMapH);

  //! take iso deposits for hcal (contains (eta,phi, pt) of hcal in R<X (1.0) around each muon)
  edm::Handle<reco::IsoDepositMap> hcalMapH;
  ev.getByLabel(theHcalDepMapTag, hcalMapH);
  //! make a dummy veto list (used later)
  reco::IsoDeposit::Vetos dVetos;

  unsigned int nMuons = muons.size();
  for (unsigned int iMu = 0; iMu < nMuons; ++iMu) {
    LogTrace(metname) << "muon pt=" << muons[iMu].pt();

    //! let's look at sumPt in 5 different cones
    //! pick a deposit first (change to ..sit& when it works)
    const reco::IsoDeposit tkDep((*tkMapH)[muons.refAt(iMu)]);
    for (int i = 1; i < 6; ++i) {
      float coneSize = 0.1 * i;
      LogTrace(metname) << " iso sumPt in cone " << coneSize << " is " << tkDep.depositWithin(coneSize);
    }
    //! can count tracks too
    LogTrace(metname) << " N tracks in cone 0.5  is " << tkDep.depositAndCountWithin(0.5).second;

    //! now the same with pt>1.5 for each track
    LogTrace(metname) << " N tracks in cone 0.5  is " << tkDep.depositAndCountWithin(0.5, dVetos, 1.5).first;

    //! now the closest track
    LogTrace(metname) << " The closest track in dR is at " << tkDep.begin().dR() << " with pt "
                      << tkDep.begin().value();
  }
}

DEFINE_FWK_MODULE(IsolationExample);
