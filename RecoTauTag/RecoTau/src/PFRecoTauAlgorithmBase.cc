#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithmBase.h"

PFRecoTauAlgorithmBase::PFRecoTauAlgorithmBase():
  TransientTrackBuilder_(nullptr)
{
}


PFRecoTauAlgorithmBase::PFRecoTauAlgorithmBase(const edm::ParameterSet&):
  TransientTrackBuilder_(nullptr)

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
