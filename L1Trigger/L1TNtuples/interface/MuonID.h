#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include <DataFormats/PatCandidates/interface/Muon.h>

//vertex
// EDM formats
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
//local  data formats
#include "L1Trigger/L1TNtuples/interface/L1AnalysisRecoVertexDataFormat.h"

inline bool isLooseMuonCustom(const reco::Muon& recoMu) {
  bool flag = false;
  if (recoMu.isPFMuon() && (recoMu.isGlobalMuon() || recoMu.isTrackerMuon()))
    flag = true;

  return flag;
}

inline bool isMediumMuonCustom(const reco::Muon& recoMu) {
  bool goodGlob = recoMu.isGlobalMuon() && recoMu.globalTrack()->normalizedChi2() < 3 &&
                  recoMu.combinedQuality().chi2LocalPosition < 12 && recoMu.combinedQuality().trkKink < 20;
  bool isMedium = isLooseMuonCustom(recoMu) && recoMu.innerTrack()->validFraction() > 0.49 &&
                  muon::segmentCompatibility(recoMu) > (goodGlob ? 0.303 : 0.451);

  return isMedium;
}

inline bool isTightMuonCustom(const reco::Muon& recoMu, const reco::Vertex recoVtx) {
  //bp
  bool isTight = recoMu.isGlobalMuon() && recoMu.isPFMuon() && recoMu.globalTrack()->normalizedChi2() < 10. &&
                 recoMu.globalTrack()->hitPattern().numberOfValidMuonHits() > 0 &&
                 recoMu.numberOfMatchedStations() > 1 && fabs(recoMu.muonBestTrack()->dxy(recoVtx.position())) < 0.2 &&
                 fabs(recoMu.muonBestTrack()->dz(recoVtx.position())) < 0.5 &&
                 recoMu.innerTrack()->hitPattern().numberOfValidPixelHits() > 0 &&
                 recoMu.innerTrack()->hitPattern().trackerLayersWithMeasurement() > 5 &&
                 recoMu.globalTrack()->normalizedChi2() < 1;

  return isTight;
}
