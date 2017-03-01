#include "Fireworks/Calo/interface/FWECALDetailViewBase.h"
#include "DataFormats/Candidate/interface/Candidate.h"

class  FWCandidateECALDetailView: public FWECALDetailViewBase<reco::Candidate>
{
public:
   FWCandidateECALDetailView() {}
   virtual ~FWCandidateECALDetailView() {}

private:
   FWCandidateECALDetailView(const FWCandidateECALDetailView&); // stop default
   const FWCandidateECALDetailView& operator=(const FWCandidateECALDetailView&); // stop default


};

REGISTER_FWDETAILVIEW(FWCandidateECALDetailView, ECAL);

/*
// reco
REGISTER_FWDETAILVIEW(FWCandidateECALDetailView, ECAL,ecalRecHit );
// aod
REGISTER_FWDETAILVIEW(FWCandidateECALDetailView, ECAL,reducedEcalRecHitsEB);
// miniaod
REGISTER_FWDETAILVIEW(FWCandidateECALDetailView, ECAL,reducedEgamma);
*/

