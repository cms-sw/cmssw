#include "KDTreeLinkerTrackHGC.h"

typedef KDTreeLinkerTrackHGC<reco::PFTrajectoryPoint::HGC_ECALEntrance> KDTreeTrackAndHGCEELinker;
typedef KDTreeLinkerTrackHGC<reco::PFTrajectoryPoint::HGC_HCALFEntrance> KDTreeTrackAndHGCHEFLinker;
typedef KDTreeLinkerTrackHGC<reco::PFTrajectoryPoint::HGC_HCALBEntrance> KDTreeTrackAndHGCHEBLinker;

DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, 
		  KDTreeTrackAndHGCEELinker, 
		  "KDTreeTrackAndHGCEELinker"); 

DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, 
		  KDTreeTrackAndHGCHEFLinker, 
		  "KDTreeTrackAndHGCHEFLinker"); 

DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, 
		  KDTreeTrackAndHGCHEBLinker, 
		  "KDTreeTrackAndHGCHEBLinker"); 
