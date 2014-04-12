//=========================================================================
// gridConverters.h
//
// Utility functions for combining Grid2d objects templated on different
// types
//
// I. Volobouev
// June 2011
//=========================================================================

#ifndef RecoJets_FFTJetAlgorithms_gridConverters_h
#define RecoJets_FFTJetAlgorithms_gridConverters_h

#include <cassert>
#include "fftjet/Grid2d.hh"

namespace fftjetcms {
    template<typename Numeric>
    fftjet::Grid2d<float>* convert_Grid2d_to_float(
        const fftjet::Grid2d<Numeric>& grid);

    template<typename Numeric>
    fftjet::Grid2d<double>* convert_Grid2d_to_double(
        const fftjet::Grid2d<Numeric>& grid);

    template<typename F1, typename F2>
    void copy_Grid2d_data(
        fftjet::Grid2d<F2>* to, const fftjet::Grid2d<F1>& from);

    template<typename F1, typename F2>
    void add_Grid2d_data(
        fftjet::Grid2d<F2>* to, const fftjet::Grid2d<F1>& from);
}

////////////////////////////////////////////////////////////////////////
//
//  Implementation follows
//
////////////////////////////////////////////////////////////////////////

namespace fftjetcms {
    template<typename F1, typename F2>
    void copy_Grid2d_data(
        fftjet::Grid2d<F2>* to, const fftjet::Grid2d<F1>& from)
    {
        assert(to);
        assert(from.nEta() == to->nEta());
        assert(from.nPhi() == to->nPhi());
        const unsigned len = from.nEta()*from.nPhi();
        const F1* fromData = from.data();
        F2* toData = const_cast<F2*>(to->data());
        for (unsigned i=0; i<len; ++i)
            toData[i] = fromData[i];
    }

    template<typename F1, typename F2>
    void add_Grid2d_data(
        fftjet::Grid2d<F2>* to, const fftjet::Grid2d<F1>& from)
    {
        assert(to);
        assert(from.nEta() == to->nEta());
        assert(from.nPhi() == to->nPhi());
        const unsigned len = from.nEta()*from.nPhi();
        const F1* fromData = from.data();
        F2* toData = const_cast<F2*>(to->data());
        for (unsigned i=0; i<len; ++i)
            toData[i] += fromData[i];
    }

    template<typename Numeric>
    fftjet::Grid2d<float>* convert_Grid2d_to_float(
        const fftjet::Grid2d<Numeric>& grid)
    {
        fftjet::Grid2d<float>* to = new fftjet::Grid2d<float>(
            grid.nEta(), grid.etaMin(), grid.etaMax(),
            grid.nPhi(), grid.phiBin0Edge(), grid.title());
        copy_Grid2d_data(to, grid);
        return to;
    }

    template<typename Numeric>
    fftjet::Grid2d<double>* convert_Grid2d_to_double(
        const fftjet::Grid2d<Numeric>& grid)
    {
        fftjet::Grid2d<double>* to = new fftjet::Grid2d<double>(
            grid.nEta(), grid.etaMin(), grid.etaMax(),
            grid.nPhi(), grid.phiBin0Edge(), grid.title());
        copy_Grid2d_data(to, grid);
        return to;
    }
}

#endif // RecoJets_FFTJetAlgorithms_gridConverters_h
