//=========================================================================
// PileupGrid2d.h
//
// Simple table in the eta-phi space which multiplies the pile-up rho
//
// I. Volobouev
// June 2011
//=========================================================================

#ifndef RecoJets_FFTJetAlgorithms_PileupGrid2d_h
#define RecoJets_FFTJetAlgorithms_PileupGrid2d_h

#include "fftjet/Grid2d.hh"

#include "RecoJets/FFTJetAlgorithms/interface/fftjetTypedefs.h"
#include "RecoJets/FFTJetAlgorithms/interface/AbsPileupCalculator.h"

namespace fftjetcms {
    class PileupGrid2d : public AbsPileupCalculator
    {
    public:
        inline explicit PileupGrid2d(const fftjet::Grid2d<Real>& g,
                                     const double rhoFactor)
	    : grid_(g), rhoFactor_(rhoFactor) {}

        inline virtual ~PileupGrid2d() {}

        inline virtual double operator()(
            const double eta, const double phi,
            const reco::FFTJetPileupSummary& summary) const
	{
	    return rhoFactor_ * summary.pileupRho() *
                   grid_.coordValue(eta, phi);
	}

        inline virtual bool isPhiDependent() const {return true;}

    private:
	fftjet::Grid2d<Real> grid_;
        double rhoFactor_;
    };
}

#endif // RecoJets_FFTJetAlgorithms_PileupGrid2d_h
