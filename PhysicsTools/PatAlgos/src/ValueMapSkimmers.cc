#include "PhysicsTools/PatAlgos/interface/ValueMapSkimmer.h"

using namespace pat::helper;

typedef ValueMapSkimmer<double> CandValueMapSkimmerDouble;
typedef ValueMapSkimmer<float>  CandValueMapSkimmerFloat;
typedef ValueMapSkimmer<int>    CandValueMapSkimmerInt;
typedef ValueMapSkimmer<float, edm::ValueMap<double> >  CandValueMapSkimmerDouble2Float;
typedef ValueMapSkimmer<float, edm::ValueMap<int>    >  CandValueMapSkimmerInt2Float;
typedef ValueMapSkimmer<reco::CandidateBaseRef>         CandRefValueMapSkimmer;


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CandValueMapSkimmerDouble);
DEFINE_FWK_MODULE(CandValueMapSkimmerFloat);
//DEFINE_FWK_MODULE(CandValueMapSkimmerInt);
//DEFINE_FWK_MODULE(CandValueMapSkimmerDouble2Float);
//DEFINE_FWK_MODULE(CandValueMapSkimmerInt2Float);
//DEFINE_FWK_MODULE(CandRefValueMapSkimmer);
