// Author : Samvel Khalatian ( samvel at fnal dot gov )
// Created: 04/26/07

#include "DQMServices/Core/interface/MostProbableROOT.h"

#include <TF1.h>
#include <math.h>

/** 
* @brief 
*   Rational approximation for erfc( rdX) (Abramowitz & Stegun, Sec. 7.1.26)
*   Fifth order approximation. | error| <= 1.5e-7 for all rdX
* 
* @param rdX 
*   Significance value   
* 
* @return 
*   Probability
*/
double edm::qtests::fits::erfc( const double &rdX) {
  const double dP  = 0.3275911;
  const double dA1 = 0.254829592;
  const double dA2 = -0.284496736;
  const double dA3 = 1.421413741;
  const double dA4 = -1.453152027;
  const double dA5 = 1.061405429;

  const double dT  = 1.0 / ( 1.0 + dP * rdX);
  return ( dA1 + ( dA2 + ( dA3 + ( dA4 + dA5 * dT) * dT) * dT) * dT) * 
         exp( -rdX * rdX);
}

// --[ Fit: Base - MostProbableBaseROOT ]--------------------------------------
MostProbableBaseROOT::MostProbableBaseROOT():
  dMostProbable_( 0),
  dSigma_( 0),
  dXMin_( 0),
  dXMax_( 0) {}

/** 
* @brief 
*   Run QTest
* 
* @param poPLOT 
*   See Interface
* 
* @return 
*   See Interface
*/
float MostProbableBaseROOT::runTest( const TH1F *const poPLOT) {
  float dResult = -1;

  if( poPLOT && !isInvalid()) {
    // It is childrens responsibility to implement and actually fit Histogram.
    // Constness should be removed since TH1::Fit(...) method is declared as
    // non const.
    if( TH1F *const poPlot = const_cast<TH1F *const>( poPLOT)) {
      dResult = fit( poPlot);
    }
  }

  return dResult;
}

/** 
* @brief 
*   Check if given QTest is Invalid
* 
* @return 
*   See Interface
*/
bool MostProbableBaseROOT::isInvalid() {
  return !( dMostProbable_ > dXMin_ &&
            dMostProbable_ < dXMax_ &&
            dSigma_ > 0);
}

/** 
* @brief 
*   Function will compare two MostProbable values and return value representing
*   percent of match. For that two tests are used:
*     1. Distance between MostProbables should be less than 2 Sigmas of
*        estimated distribution, e.g.:
*
*          | Mn - Me |
*          ----------- < Alpha
*            Sigma_e
*
*        where:
*          Mn       Most Probable new value gotten from Fit or any other 
*                   algorithm
*          Me       Estimated value of Most Probable
*          Sigma_e  Sigma of Estimated distribution (Landau, Gauss, any other)
*          Alpha    Number of sigmas (at the moment value of 2 is used... but
*                   check the code below to make sure number is not outdated)

*     2. Significance that is calculated with formula:
*
*                | Mn - Me |
*          ------------------------- 
*          Sigma_e ^ 2 + Sigma_n ^ 2 
*         
*        where:
*          Mn       Most Probable new value gotten from Fit or any other 
*                   algorithm
*          Me       Estimated value of Most Probable
*          Sigma_e  Sigma of Estimated distribution (Landau, Gauss, any other)
*          Sigma_n  Sigma new value gotten from Fit or any other algorithm
*
*   Once both values are calculated Distance between MostProbables is checked
*   for Alpha. -1 is returned if given value is greater than Alpha, otherwise
*   Significance is transformed into probability [0,1] and the number is 
*   returned.
* 
* @param rdMP_FIT
* @param rdSIGMA_FIT
*   See Interface
* 
* @return 
*   See Interface
*/
double 
MostProbableBaseROOT::compareMostProbables( const double &rdMP_FIT,
                                            const double &rdSIGMA_FIT) const {

  double dDeltaMP = rdMP_FIT - dMostProbable_;
  if( dDeltaMP < 0) {
    dDeltaMP = -dDeltaMP;
  }

  return ( /* Check Deviation */ dDeltaMP / dSigma_ < 2 ? 
           edm::qtests::fits::erfc( ( 
             /* Calculate Significance */ 
             dDeltaMP / sqrt( rdSIGMA_FIT * rdSIGMA_FIT + dSigma_ * dSigma_) 
             /* Done with Significance */ )
             ) / sqrt( 2.0) :
           0);
}

// --[ Fit: Landau ]-----------------------------------------------------------
MostProbableLandauROOT::MostProbableLandauROOT(): dNormalization_( 0) {
  setMinimumEntries( 50);
}

/** 
* @brief 
*   Perform Landau Fit
* 
* @param poPlot
*   See Interface
* 
* @return 
*   See Interface
*/
float MostProbableLandauROOT::fit( TH1F *const poPlot) {
  double dResult = -1;

  // Create Fit Function
  TF1 *poFFit = new TF1( "Landau", "landau", getXMin(), getXMax());

  // Fit Parameters
  //   [0]  Normalisation coefficient.
  //   [1]  Most probable value.
  //   [2]  Lambda in most books ( the width of the distribution)
  poFFit->SetParameters( dNormalization_, getMostProbable(), getSigma());

  // Fit
  if( !poPlot->Fit( poFFit, "RB0")) {

    // Obtain Fit Parameters: We are interested in Most Probable and Sigma
    // so far.
    const double dMP_FIT    = poFFit->GetParameter( 1);
    const double dSIGMA_FIT = poFFit->GetParameter( 2);

    // Compare gotten values with expected ones.
    dResult = compareMostProbables( dMP_FIT, dSIGMA_FIT);
  }

  return dResult;
}
