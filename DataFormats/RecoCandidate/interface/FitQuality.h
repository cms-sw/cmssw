#ifndef RecoCandidate_FitQuality_h
#define RecoCandidate_FitQuality_h
/** \class reco::FitQuality
 *
 * structure containg fit quality
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: FitQuality.h,v 1.2 2007/11/16 13:49:57 llista Exp $
 *
 */

#include "Rtypes.h"

namespace reco {

  class FitQuality {
  public:
    /// default constructor
    FitQuality() : chi2_(0), ndof_(0) { }
    /// constructor form values
    FitQuality( double chi2, double ndof ) :
      chi2_( chi2 ), ndof_( ndof ) { }
    /// chi-squared
    double chi2() const { return chi2_; }
    /// number of degrees of freedom
    double ndof() const { return ndof_; }
    /// normalized chi-squared
    double normalizedChi2() const { return chi2_ / ndof_; }

  private:
    Double32_t chi2_;
    Double32_t ndof_;
  };

}

#endif
