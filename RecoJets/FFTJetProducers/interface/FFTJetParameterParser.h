#ifndef RecoJets_FFTJetProducers_FFTJetParameterParser_h
#define RecoJets_FFTJetProducers_FFTJetParameterParser_h

#include <memory>
#include <vector>

#include "fftjet/RecombinedJet.hh"
#include "fftjet/PeakFinder.hh"
#include "fftjet/Grid2d.hh"
#include "fftjet/AbsPeakSelector.hh"
#include "fftjet/ScaleSpaceKernel.hh"
#include "fftjet/ClusteringTreeSparsifier.hh"
#include "fftjet/AbsDistanceCalculator.hh"
#include "fftjet/LinearInterpolator1d.hh"
#include "fftjet/SimpleFunctors.hh"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "RecoJets/FFTJetAlgorithms/interface/fftjetTypedefs.h"

namespace fftjetcms {
    // Pseudo-constructors for various FFTJet classes using ParameterSet
    // objects as arguments
    std::auto_ptr<fftjet::Grid2d<Real> >
    fftjet_Grid2d_parser(const edm::ParameterSet& ps);

    std::auto_ptr<fftjet::Functor1<bool,fftjet::Peak> >
    fftjet_PeakSelector_parser(const edm::ParameterSet& ps);

    std::auto_ptr<fftjet::ScaleSpaceKernel>
    fftjet_MembershipFunction_parser(const edm::ParameterSet& ps);

    std::auto_ptr<AbsBgFunctor>
    fftjet_BgFunctor_parser(const edm::ParameterSet& ps);

    std::auto_ptr<std::vector<double> >
    fftjet_ScaleSet_parser(const edm::ParameterSet& ps);

    std::auto_ptr<fftjet::ClusteringTreeSparsifier<fftjet::Peak,long> >
    fftjet_ClusteringTreeSparsifier_parser(const edm::ParameterSet& ps);

    std::auto_ptr<fftjet::AbsDistanceCalculator<fftjet::Peak> >
    fftjet_DistanceCalculator_parser(const edm::ParameterSet& ps);

    std::auto_ptr<fftjet::LinearInterpolator1d>
    fftjet_LinearInterpolator1d_parser(const edm::ParameterSet& ps);

    std::auto_ptr<fftjet::Functor1<double,fftjet::Peak> >
    fftjet_PeakFunctor_parser(const edm::ParameterSet& ps);

    std::auto_ptr<fftjet::Functor1<double,fftjet::RecombinedJet<VectorLike> > >
    fftjet_JetFunctor_parser(const edm::ParameterSet& ps);

    std::auto_ptr<fftjet::Functor2<double,
                                   fftjet::RecombinedJet<VectorLike>,
                                   fftjet::RecombinedJet<VectorLike> > >
    fftjet_JetDistance_parser(const edm::ParameterSet& ps);

    std::auto_ptr<fftjet::Functor1<double,double> >
    fftjet_Function_parser(const edm::ParameterSet& ps);
}

#endif // RecoJets_FFTJetProducers_FFTJetParameterParser_h
