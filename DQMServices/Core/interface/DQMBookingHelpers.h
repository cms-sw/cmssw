#ifndef DQMServices_Core_DQMBookingHelpers_h
#define DQMServices_Core_DQMBookingHelpers_h

// Package:    DQMServices/Core
//
/**\file DQMBookingHelpers.h DQMServices/Core/interface/DQMBookingHelpers.h

 Description: Utility functions for booking DQM histograms with non-standard
              axis configurations, in particular logarithmic axis binning.

              These supplement the standard DQMStore::IBooker interface, which
              deliberately provides only plain ROOT-argument booking. Callers
              that need log-scale axes can use these helpers to construct and
              configure the underlying ROOT object before handing it to IBooker.

 Motivation:  The rebinXToLog / make1DIfLogX pattern was previously duplicated
              across multiple validation packages (Validation/RecoTrack,
              Validation/MuonIdentification, and others). Centralising it here
              removes the duplication, ensures a single correct implementation
              of the threading-safe booking pattern (see note below), and makes
              log-scale booking discoverable for new DQM module authors.

 Threading note:
              rebinXToLog must be called before the histogram is registered with
              IBooker, i.e. before book1D / book2D is called. Calling it after
              booking (e.g. via getTH1()) is not thread-safe in stream-based
              DQMEDAnalyzers because multiple streams may be booking
              concurrently. The helpers below enforce the correct order by
              constructing and configuring the ROOT object first, then passing
              ownership to IBooker. See also:
              https://github.com/cms-sw/cmssw/pull/29224

 Usage:
              #include "DQMServices/Core/interface/DQMBookingHelpers.h"
              using namespace dqm::booking;

              // 1D histogram with log X axis
              auto h = book1DLogX(ibook, "name", "title;x;y", 50, 0.1, 1000.);

              // 2D histogram, log X axis, called conditionally
              auto h2 = book2DIfLogX(ibook, useLog, "name", "title", ...);
*/

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <memory>
#include <string>
#include <vector>

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TMath.h"

namespace dqm {
  namespace booking {

    // =========================================================================
    // Internal helpers — not part of the public interface
    // =========================================================================

    namespace detail {

      /// Rebin the X axis of h to have logarithmically-spaced bin edges.
      /// The axis min/max are interpreted as log10 values, consistent with
      /// how TAxis stores them after a SetRangeUser call.
      /// Must be called on the ROOT object BEFORE handing it to IBooker.
      void rebinXToLog(TH1* h);

      /// Rebin the Y axis of h to have logarithmically-spaced bin edges.
      /// Must be called on the ROOT object BEFORE handing it to IBooker.
      void rebinYToLog(TH1* h);

    }  // namespace detail

    // =========================================================================
    // Public booking helpers
    // =========================================================================

    using IBooker = dqm::reco::DQMStore::IBooker;
    using ME = dqm::reco::MonitorElement;

    // -------------------------------------------------------------------------
    // 1D histograms
    // -------------------------------------------------------------------------

    /// Book a 1D histogram, applying log binning on the X axis if logx=true.
    template <typename... Args>
    inline ME* book1DIfLogX(IBooker& ibook, bool logx, Args&&... args) {
      auto h = std::make_unique<TH1F>(std::forward<Args>(args)...);
      if (logx)
        detail::rebinXToLog(h.get());
      const std::string name = h->GetName();
      return ibook.book1D(name, h.release());
    }

    /// Book a 1D histogram with a logarithmic X axis.
    template <typename... Args>
    inline ME* book1DLogX(IBooker& ibook, Args&&... args) {
      return book1DIfLogX(ibook, true, std::forward<Args>(args)...);
    }

    // -------------------------------------------------------------------------
    // 2D histograms
    // -------------------------------------------------------------------------

    /// Book a 2D histogram, applying log binning on the X and Y axis if logx=true and logy=true.
    template <typename... Args>
    inline ME* book2DIfLogXIfLogY(IBooker& ibook, bool logx, bool logy, Args&&... args) {
      auto h = std::make_unique<TH2F>(std::forward<Args>(args)...);
      if (logx)
        detail::rebinXToLog(h.get());
      if (logy)
        detail::rebinYToLog(h.get());
      const std::string name = h->GetName();
      return ibook.book2D(name, h.release());
    }

    /// Book a 2D histogram with a logarithmic X and Y axis.
    template <typename... Args>
    inline ME* book2DLogXLogY(IBooker& ibook, Args&&... args) {
      return book2DIfLogXIfLogY(ibook, true, true, std::forward<Args>(args)...);
    }

    /// Book a 2D histogram, applying log binning on the X axis if logx=true.
    template <typename... Args>
    inline ME* book2DIfLogX(IBooker& ibook, bool logx, Args&&... args) {
      return book2DIfLogXIfLogY(ibook, logx, false, std::forward<Args>(args)...);
    }

    /// Book a 2D histogram with a logarithmic X axis.
    template <typename... Args>
    inline ME* book2DLogX(IBooker& ibook, Args&&... args) {
      return book2DIfLogX(ibook, true, std::forward<Args>(args)...);
    }

    /// Book a 2D histogram, applying log binning on the Y axis if logy=true.
    template <typename... Args>
    inline ME* book2DIfLogY(IBooker& ibook, bool logy, Args&&... args) {
      return book2DIfLogXIfLogY(ibook, false, logy, std::forward<Args>(args)...);
    }

    /// Book a 2D histogram with a logarithmic Y axis.
    template <typename... Args>
    inline ME* book2DLogY(IBooker& ibook, Args&&... args) {
      return book2DIfLogY(ibook, true, std::forward<Args>(args)...);
    }

    // -------------------------------------------------------------------------
    // TProfile histograms
    // -------------------------------------------------------------------------

    /// Book a TProfile, applying log binning on the X axis if logx=true.
    template <typename... Args>
    inline ME* bookProfileIfLogX(IBooker& ibook, bool logx, Args&&... args) {
      auto h = std::make_unique<TProfile>(std::forward<Args>(args)...);
      if (logx)
        detail::rebinXToLog(h.get());
      const std::string name = h->GetName();
      return ibook.bookProfile(name, h.release());
    }

    /// Book a TProfile with a logarithmic X axis.
    template <typename... Args>
    inline ME* bookProfileLogX(IBooker& ibook, Args&&... args) {
      return bookProfileIfLogX(ibook, true, std::forward<Args>(args)...);
    }

  }  // namespace booking
}  // namespace dqm

#endif  // DQMServices_Core_DQMBookingHelpers_h
