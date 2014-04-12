#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithmBase.h"

PFRecoTauAlgorithmBase::PFRecoTauAlgorithmBase():
  TransientTrackBuilder_(0)
{
}


PFRecoTauAlgorithmBase::PFRecoTauAlgorithmBase(const edm::ParameterSet&):
  TransientTrackBuilder_(0)

{
}


PFRecoTauAlgorithmBase::~PFRecoTauAlgorithmBase()
{

}

void 
PFRecoTauAlgorithmBase::setTransientTrackBuilder(const TransientTrackBuilder* builder) 
{
    TransientTrackBuilder_ = builder;
}
