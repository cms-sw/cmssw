/**
 * Handles the cross sections for MuScleFit. <br>
 * What counts in the fit is the ratio of the cross sections. However it depends on which resonances are used in the fit.
 * If we are fitting only the Upsilon(1S), for example, we do not need to consider the cross section ratio, because the probability
 * of the other resonances will be 0. This is useful when running on MC to test the algorithm. <br>
 *
 * The constructor receives the array of cross sections and the array of resfind that tells which of the resonances are considered in the fit. <br>
 * It builds the relative cross section parameters for each of the resonance and it has a method that unlocks the parameter accordingly. <br>
 * If for example only the Upsilon(1S) is fitted, the relative cross section will be 1 and it will remain fixed. <br>
 * The relative cross sections are fitted only when a background fit is done. <br>
 *
 * Note that this handles only the initialization of the cross sections, so that it is consistent with the fitted resonances, and
 * the fix/release of the cross section parameters. <br>
 *
 * This assumes that resfind is the same during all the processing (it is saved internally when received in the constructor).
 */

#include <vector>
#include <numeric>

#include "TString.h"
#include "TMinuit.h"

class CrossSectionHandler
{
public:
  CrossSectionHandler(const std::vector<double> & crossSection, const std::vector<int> & resfind) :
    parNum_(0),
    numberOfResonances_(resfind.size())
  {
    // The number of parameters is the number of fitted resonances minus 1
    std::vector<int>::const_iterator it = resfind.begin();
    for( ; it != resfind.end(); ++it ) {
      if( *it != 0 ) ++parNum_;
    }
    if( parNum_ > 0 ) parNum_ = parNum_ - 1;

    vars_.resize(parNum_);

    computeRelativeCrossSections(crossSection, resfind);
    imposeConstraint();
  }

  /// Inputs the vars in a vector
  void addParameters(std::vector<double> & initpar)
  {
    std::vector<double>::const_iterator it = vars_.begin();
    for( ; it != vars_.end(); ++it ) {
      initpar.push_back(*it);
    }
  }

  /// Initializes the arrays needed by Minuit
  void setParameters( double* Start, double* Step, double* Mini, double* Maxi, int* ind, TString* parname,
                      const std::vector<double> & parCrossSection, const std::vector<int> & parCrossSectionOrder,
                      const std::vector<int> & resfind )
  {
    computeRelativeCrossSections(parCrossSection, resfind);
    imposeConstraint();

    double thisStep[] = {0.001, 0.001, 0.001, 0.001, 0.001};
    TString thisParName[] = {"cross section var 1",
                             "cross section var 2",
                             "cross section var 3",
                             "cross section var 4",
                             "cross section var 5"};
    double thisMini[] = {0., 0., 0., 0., 0.};
    double thisMaxi[] = {1000., 1000., 1000., 1000., 1000.};

    // This is used to unlock the parameters in a given order. It is not
    // a TMinuit parameter, but a MuScleFit parameter.
    for( unsigned int iPar=0; iPar<numberOfResonances_; ++iPar ) {
      ind[iPar] = parCrossSectionOrder[iPar];
    }

    if( parNum_ > 0 ) {
      for( unsigned int iPar=0; iPar<parNum_; ++iPar ) {
        Start[iPar] = vars_[iPar];
        Step[iPar] = thisStep[iPar];
        Mini[iPar] = thisMini[iPar];
        Maxi[iPar] = thisMaxi[iPar];
        parname[iPar] = thisParName[iPar];
      }
    }
  }

  /// Use the information in resfind, parorder and parfix to release the N-1 variables
  bool releaseParameters( TMinuit & rmin, const std::vector<int> & resfind, const std::vector<int> & parfix,
                          const int * ind, const int iorder, const unsigned int shift )
  {
    // Find the number of free cross section parameters in this iteration
    unsigned int freeParNum = 0;
    for( unsigned int ipar=0; ipar<numberOfResonances_; ++ipar ) {
      if( (parfix[shift+ipar]==0) && (ind[shift+ipar]<=iorder) && (resfind[ipar] == 1) ) {
        ++freeParNum;
      }
    }
    if( freeParNum > 0 ) {
      freeParNum = freeParNum - 1;
      // Free only the first (freeParNum-1) of the N-1 variables
      for( unsigned int i=0; i<freeParNum; ++i ) {
        rmin.Release( shift+i );
      }
      return true;
    }
    return false;
  }

  inline unsigned int parNum()
  {
    return parNum_;
  }

  /// Perform a variable transformation from N-1 to relative cross sections 
  std::vector<double> relativeCrossSections( const double * variables, const std::vector<int> & resfind )
  {
    // parNum_ is 0 in two cases:
    // 1) if only one resonance is being fitted, in which case the relative cross section is
    // fixed to one and there is no need to recompute it
    // 2) if no resonance is being fitted, in which case all the relative cross sections will
    // be set to 0.
    // In both cases there is no need to make the transformation of variables.
    if( parNum_ != 0 ) {
      double * partialProduct = new double[numberOfResonances_];
      double norm = 0.;
      // Loop on all relative cross sections (that are parNum_+1)
      for( unsigned int i=0; i<parNum_+1; ++i ) {
        partialProduct[i] = std::accumulate(variables, variables + i, 1., std::multiplies<double>());
        norm += partialProduct[i];
      }
      for( unsigned int i=0; i<parNum_+1; ++i ) {
        relativeCrossSectionVec_[i] = partialProduct[i]/norm;
      }
      delete[] partialProduct;
    }

    std::vector<double> allRelativeCrossSections;
    std::vector<int>::const_iterator it = resfind.begin();
    int smallerVectorIndex = 0;
    for( ; it != resfind.end(); ++it ) {
      if( *it == 0 ) {
        allRelativeCrossSections.push_back(0.);
      }
      else {
        allRelativeCrossSections.push_back(relativeCrossSectionVec_[smallerVectorIndex]);
        ++smallerVectorIndex;
      }
    }

    return allRelativeCrossSections;
  }

protected:
  /**
   * Initializes the relative cross sections for the range of resonances in [minRes, maxRes]. (note that both minRes and maxRes are included). <br>
   * Also sets the lock on resonances. If only one of the resonances in the range is fitted its relative cross section will be 1 and it will not
   * be fitted. If there are more than one only those that are fitted will have the relative cross section parameters unlocked during the fit.
   */
  // void computeRelativeCrossSections(const double crossSection[], const std::vector<int> resfind, const unsigned int minRes, const unsigned int maxRes)
  void computeRelativeCrossSections(const std::vector<double> & crossSection, const std::vector<int> & resfind)
  {
    relativeCrossSectionVec_.clear();
    double normalization = 0.;
    for( unsigned int ires = 0; ires < resfind.size(); ++ires ) {
      if( resfind[ires] ) {
        normalization += crossSection[ires];
      }
    }
    if( normalization != 0. ) {
      for( unsigned int ires = 0; ires < resfind.size(); ++ires ) {
        if( resfind[ires] ) {
          relativeCrossSectionVec_.push_back(crossSection[ires]/normalization);
        }
      }
    }
  }

  /// Change of variables so that we move from N to N-1 variables using the constraint that Sum(x_i) = 1.
  void imposeConstraint()
  {
    if( parNum_ > 0 ) {
      for( unsigned int iVar = 0; iVar < parNum_; ++iVar ) {
        vars_[iVar] = relativeCrossSectionVec_[iVar+1]/relativeCrossSectionVec_[iVar];
      }
    }
  }

  // Data members
  std::vector<double> relativeCrossSectionVec_;
  std::vector<double> vars_;
  unsigned int parNum_;
  unsigned int numberOfResonances_;
};
