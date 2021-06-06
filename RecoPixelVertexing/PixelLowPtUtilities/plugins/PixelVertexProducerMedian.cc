#include "PixelVertexProducerMedian.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>

#include "TROOT.h"
#include "TH1F.h"
#include "TF1.h"

/*****************************************************************************/
struct ComparePairs {
  bool operator()(const reco::Track* t1, const reco::Track* t2) { return (t1->vz() < t2->vz()); };
};

/*****************************************************************************/
PixelVertexProducerMedian::PixelVertexProducerMedian(const edm::ParameterSet& ps) : theConfig(ps) {
  thePtMin = theConfig.getParameter<double>("PtMin");
  produces<reco::VertexCollection>();
}

/*****************************************************************************/
PixelVertexProducerMedian::~PixelVertexProducerMedian() {}

/*****************************************************************************/
void PixelVertexProducerMedian::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& es) const {
  // Get pixel tracks
  edm::Handle<reco::TrackCollection> trackCollection;
  std::string trackCollectionName = theConfig.getParameter<std::string>("TrackCollection");
  ev.getByLabel(trackCollectionName, trackCollection);
  const reco::TrackCollection tracks_ = *(trackCollection.product());

  // Select tracks
  std::vector<const reco::Track*> tracks;
  for (unsigned int i = 0; i < tracks_.size(); i++) {
    if (tracks_[i].pt() > thePtMin) {
      reco::TrackRef recTrack(trackCollection, i);
      tracks.push_back(&(*recTrack));
    }
  }

  LogTrace("MinBiasTracking") << " [VertexProducer] selected tracks: " << tracks.size() << " (out of " << tracks_.size()
                              << ")";

  auto vertices = std::make_unique<reco::VertexCollection>();

  if (!tracks.empty()) {
    // Sort along vertex z position
    std::sort(tracks.begin(), tracks.end(), ComparePairs());

    // Median
    float med;
    if (tracks.size() % 2 == 0)
      med = (tracks[tracks.size() / 2 - 1]->vz() + tracks[tracks.size() / 2]->vz()) / 2;
    else
      med = tracks[tracks.size() / 2]->vz();

    LogTrace("MinBiasTracking") << "  [vertex position] median    = " << med << " cm";

    if (tracks.size() > 10) {
      // Binning around med, halfWidth
      int nBin = 100;
      float halfWidth = 0.1;  // cm

      // Most probable
      TH1F histo("histo", "histo", nBin, -halfWidth, halfWidth);

      for (std::vector<const reco::Track*>::const_iterator track = tracks.begin(); track != tracks.end(); track++)
        if (fabs((*track)->vz() - med) < halfWidth)
          histo.Fill((*track)->vz() - med);

      LogTrace("MinBiasTracking") << "  [vertex position] most prob = "
                                  << med + histo.GetBinCenter(histo.GetMaximumBin()) << " cm";

      // Fit above max/2
      histo.Sumw2();

      TF1 f1("f1", "[0]*exp(-0.5 * ((x-[1])/[2])^2) + [3]");
      f1.SetParameters(10., 0., 0.01, 1.);

      histo.Fit(&f1, "QN");

      LogTrace("MinBiasTracking") << "  [vertex position] fitted    = " << med + f1.GetParameter(1) << " +- "
                                  << f1.GetParError(1) << " cm";

      // Store
      reco::Vertex::Error err;
      err(2, 2) = f1.GetParError(1) * f1.GetParError(1);
      reco::Vertex ver(reco::Vertex::Point(0, 0, med + f1.GetParameter(1)), err, 0, 1, 1);
      vertices->push_back(ver);
    } else {
      // Store
      reco::Vertex::Error err;
      err(2, 2) = 0.1 * 0.1;
      reco::Vertex ver(reco::Vertex::Point(0, 0, med), err, 0, 1, 1);
      vertices->push_back(ver);
    }
  }
  ev.put(std::move(vertices));
}
