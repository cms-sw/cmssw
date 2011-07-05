#ifndef RecoJets_FFTJetAlgorithms_classes_h
#define RecoJets_FFTJetAlgorithms_classes_h

#include <utility>

#include "DataFormats/Common/interface/Wrapper.h"
#include "RecoJets/FFTJetAlgorithms/interface/DiscretizedEnergyFlow.h"

namespace {
  struct dictionary {
    fftjetcms::DiscretizedEnergyFlow dflow;
    edm::Wrapper<fftjetcms::DiscretizedEnergyFlow> wr_dflow;
    std::pair<double,double> ddpair;
    edm::Wrapper<std::pair<double,double> > wr_ddpair;
  };
}

#endif
