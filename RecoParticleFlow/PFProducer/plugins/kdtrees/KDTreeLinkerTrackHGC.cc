#include "KDTreeLinkerTrackHGC.h"

typedef KDTreeLinkerTrackHGC<reco::PFTrajectoryPoint::HGC_ECALEntrance,2> KDTreeTrackAndHGCEELinker;
typedef KDTreeLinkerTrackHGC<reco::PFTrajectoryPoint::HGC_HCALFEntrance,2> KDTreeTrackAndHGCHEFLinker;
typedef KDTreeLinkerTrackHGC<reco::PFTrajectoryPoint::HGC_HCALBEntrance,2> KDTreeTrackAndHGCHEBLinker;

DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, 
		  KDTreeTrackAndHGCEELinker, 
		  "KDTreeTrackAndHGCEELinker"); 

DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, 
		  KDTreeTrackAndHGCHEFLinker, 
		  "KDTreeTrackAndHGCHEFLinker"); 

DEFINE_EDM_PLUGIN(KDTreeLinkerFactory, 
		  KDTreeTrackAndHGCHEBLinker, 
		  "KDTreeTrackAndHGCHEBLinker"); 
