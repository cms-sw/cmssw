#include "PhysicsTools/PatAlgos/plugins/ValueMapSkimmer.h"

//#include "DataFormats/BTauReco/interface/JetTagFwd.h"
//#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"

using namespace pat::helper;

typedef ValueMapSkimmer<double> CandValueMapSkimmerDouble;
typedef ValueMapSkimmer<float>  CandValueMapSkimmerFloat;
typedef ValueMapSkimmer<int>    CandValueMapSkimmerInt;
typedef ValueMapSkimmer<float, edm::ValueMap<double> >  CandValueMapSkimmerDouble2Float;
typedef ValueMapSkimmer<float, edm::ValueMap<int>    >  CandValueMapSkimmerInt2Float;
typedef ValueMapSkimmer<reco::CandidateBaseRef>         CandRefValueMapSkimmer;

//typedef ValueMapSkimmer<reco::JetTagRef>                JetTagRefValueMapSkimmer;
typedef ValueMapSkimmer<pat::JetCorrFactors>            JetCorrFactorsValueMapSkimmer;

typedef ManyValueMapsSkimmer<float>  CandManyValueMapsSkimmerFloat;
typedef ManyValueMapsSkimmer<reco::IsoDeposit>  CandManyValueMapsSkimmerIsoDeposits;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandValueMapSkimmerDouble);
DEFINE_FWK_MODULE(CandValueMapSkimmerFloat);
DEFINE_FWK_MODULE(CandManyValueMapsSkimmerFloat);
DEFINE_FWK_MODULE(CandManyValueMapsSkimmerIsoDeposits);
//DEFINE_FWK_MODULE(CandValueMapSkimmerInt);
//DEFINE_FWK_MODULE(CandValueMapSkimmerDouble2Float);
//DEFINE_FWK_MODULE(CandValueMapSkimmerInt2Float);
//DEFINE_FWK_MODULE(CandRefValueMapSkimmer);

//DEFINE_FWK_MODULE(JetTagRefValueMapSkimmer);
DEFINE_FWK_MODULE(JetCorrFactorsValueMapSkimmer);
