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

using namespace muon;
using namespace std;

bool isLooseMuonCustom(const reco::Muon & recoMu)
{
  bool flag = false ;
  if(recoMu.isPFMuon() && (recoMu.isGlobalMuon() || recoMu.isTrackerMuon())) flag = true;

  return flag;
}

bool isMediumMuonCustom(const reco::Muon & recoMu) 
   {
      bool goodGlob = recoMu.isGlobalMuon() && 
                      recoMu.globalTrack()->normalizedChi2() < 3 && 
                      recoMu.combinedQuality().chi2LocalPosition < 12 && 
                      recoMu.combinedQuality().trkKink < 20; 
      bool isMedium = isLooseMuonCustom(recoMu) && 
                      recoMu.innerTrack()->validFraction() > 0.8 && 
                      segmentCompatibility(recoMu) > (goodGlob ? 0.303 : 0.451); 
      return isMedium; 
   }
