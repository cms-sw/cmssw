#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <iostream>

#include "TH1F.h"
#include "TFile.h"
#include "TObjArray.h"

using namespace std;

class PixelTrackTest : public edm::one::EDAnalyzer<> {
public:
  explicit PixelTrackTest(const edm::ParameterSet& conf);
  ~PixelTrackTest();
  virtual void beginJob() {}
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void endJob() {}

private:
  string collectionLabel;
  void myprint(const reco::Track& track) const;
  TObjArray hList;
};

PixelTrackTest::PixelTrackTest(const edm::ParameterSet& conf) {
  hList.Add(new TH1F("h_Pt", "h_Pt", 100, -1.2, 1.2));
  hList.Add(new TH1F("h_tip", "h_tip", 100, -0.03, 0.03));
  hList.SetOwner();
  collectionLabel = conf.getParameter<std::string>("TrackCollection");
  edm::LogInfo("PixelTrackTest") << " CTOR";
}

PixelTrackTest::~PixelTrackTest() {
  std::string fileName = collectionLabel + ".root";
  TFile f(fileName.c_str(), "RECREATE");
  hList.Write();
  f.Close();

  edm::LogInfo("PixelTrackTest") << " DTOR";
}

void PixelTrackTest::analyze(const edm::Event& ev, const edm::EventSetup& es) {
  using namespace edm;
  using namespace std;
  using namespace reco;

  edm::LogPrint("PixelTrackTest") << "*** PixelTrackTest, analyze event: " << ev.id();
  Handle<SimTrackContainer> simTrks;
  ev.getByLabel("g4SimHits", simTrks);
  //  std::edm::LogPrint("PixelTrackTest") << "simtrks " << simTrks->size() << std::endl;

  float pt_gen = 0.0;
  typedef SimTrackContainer::const_iterator IP;
  for (IP p = simTrks->begin(); p != simTrks->end(); p++) {
    if ((*p).noVertex())
      continue;
    if ((*p).type() == -99)
      continue;
    if ((*p).vertIndex() != 0)
      continue;
    if ((*p).momentum().Pt() > pt_gen)
      pt_gen = (*p).momentum().Pt();
  }
  //  edm::LogPrint("PixelTrackTest") << "pt_gen: " << pt_gen ;
  if (pt_gen < 0.9)
    return;

  typedef reco::TrackCollection::const_iterator IT;

  edm::Handle<reco::TrackCollection> trackCollection;
  ev.getByLabel(collectionLabel, trackCollection);
  const reco::TrackCollection tracks = *(trackCollection.product());
  edm::LogPrint("PixelTrackTest") << "Number of tracks: " << tracks.size() << " tracks" << std::endl;
  for (IT it = tracks.begin(); it != tracks.end(); it++) {
    //math::XYZVector mom_rec = (*it).momentum();
    float pt_rec = (*it).pt();
    myprint(*it);
    static_cast<TH1*>(hList.FindObject("h_Pt"))->Fill((pt_gen - pt_rec) / pt_gen);
    static_cast<TH1*>(hList.FindObject("h_tip"))->Fill((*it).d0());
  }

  edm::LogPrint("PixelTrackTest") << "------------------------------------------------";
}

void PixelTrackTest::myprint(const reco::Track& track) const {
  edm::LogPrint("PixelTrackTest") << "--- RECONSTRUCTED TRACK: ";
  edm::LogPrint("PixelTrackTest") << "\tmomentum: " << track.momentum() << "\tPT: " << track.pt();
  edm::LogPrint("PixelTrackTest") << "\tvertex: " << track.vertex() << "\t zip: " << track.dz() << "+/-"
                                  << track.dzError() << "\t tip: " << track.d0() << "+/-" << track.d0Error();
  edm::LogPrint("PixelTrackTest") << "\t chi2: " << track.chi2();
  edm::LogPrint("PixelTrackTest") << "\tcharge: " << track.charge();
}

DEFINE_FWK_MODULE(PixelTrackTest);
