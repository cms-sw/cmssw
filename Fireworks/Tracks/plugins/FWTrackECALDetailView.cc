#include "Fireworks/Calo/interface/FWECALDetailViewBase.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"

class  FWTrackECALDetailView: public FWECALDetailViewBase<reco::TrackBase>
{
public:
   FWTrackECALDetailView() {}
   virtual ~FWTrackECALDetailView() {}

private:
   FWTrackECALDetailView(const FWTrackECALDetailView&); // stop default
   const FWTrackECALDetailView& operator=(const FWTrackECALDetailView&); // stop default


};

REGISTER_FWDETAILVIEW(FWTrackECALDetailView, ECAL );
/*
REGISTER_FWDETAILVIEW(FWTrackECALDetailView, ECAL,ecalRecHit );
REGISTER_FWDETAILVIEW(FWTrackECALDetailView, ECAL,reducedEcalRecHitsEB);
REGISTER_FWDETAILVIEW(FWPhotonDetailView,Photon,reducedEGamma);
*/
