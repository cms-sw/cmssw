#ifndef HiEgammaAlgos_TrackIsoCalculator_h
#define HiEgammaAlgos_TrackIsoCalculator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/PatCandidates/interface/Photon.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class TrackIsoCalculator
{
public:

  TrackIsoCalculator(const edm::Event &iEvent, const edm::EventSetup &iSetup, const edm::Handle<reco::TrackCollection> trackLabel, const std::string trackQuality_) ;

  /// Return the tracker energy in a cone around the photon
  double getTrackIso(const reco::Photon clus, const double i, const double threshold, const double innerDR=0);
  /// Return the background-subtracted tracker energy in a cone around the photon
  double getBkgSubTrackIso(const reco::Photon clus, const double i, const double threshold, const double innerDR=0);

private:

  edm::Handle<reco::TrackCollection>  recCollection;
  std::string trackQuality_;
};

#endif
