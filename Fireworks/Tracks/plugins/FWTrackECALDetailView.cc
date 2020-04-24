#include "Fireworks/Calo/interface/FWECALDetailViewBase.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

class  FWTrackECALDetailView: public FWECALDetailViewBase<reco::TrackBase>
{
public:
   FWTrackECALDetailView() {}
   ~FWTrackECALDetailView() override {}

private:
   FWTrackECALDetailView(const FWTrackECALDetailView&) = delete; // stop default
   const FWTrackECALDetailView& operator=(const FWTrackECALDetailView&) = delete; // stop default


};

REGISTER_FWDETAILVIEW(FWTrackECALDetailView, ECAL );
/*
REGISTER_FWDETAILVIEW(FWTrackECALDetailView, ECAL,ecalRecHit );
REGISTER_FWDETAILVIEW(FWTrackECALDetailView, ECAL,reducedEcalRecHitsEB);
REGISTER_FWDETAILVIEW(FWPhotonDetailView,Photon,reducedEGamma);
*/
