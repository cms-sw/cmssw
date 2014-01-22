#ifndef DTSegment_DTLinearFit_h
#define DTSegment_DTLinearFit_h

/** \class DTLinearFit
 *
 * Description:
 *  
 * detailed description
 *
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */

/* C++ Headers */
#include <vector>
#include <iostream>

/* ====================================================================== */

/* Class DTLinearFit Interface */

class DTLinearFit{

  public:

/// Constructor
    DTLinearFit() ;

/// Destructor
    ~DTLinearFit() ;

/* Operations */ 
    void fit(const std::vector<float> & x,
             const std::vector<float> & y,
             int ndat, 
             const std::vector<float> & sigy,
             float& slope,
             float& intercept, 
             double& chi2,
             float& covss,
             float& covii,
             float& covsi) const;

    // General function for performing a 2, 3 or 4 parameter fit
    void fitNpar( const int npar,
                  const std::vector<float>& xfit,
                  const std::vector<float>& yfit,
                  const std::vector<int>& lfit,
                  const std::vector<double>& tfit,
                  const std::vector<float> & sigy, 
                  float& aminf,
                  float& bminf,
                  float& cminf,
                  float& vminf,
                  double& chi2fit,
                  const bool debug) const; 

    // wrapper for the 3 parameter fit
    void fit3par( const std::vector<float>& xfit,
                           const std::vector<float>& yfit,
                           const std::vector<int>& lfit,
                           const int nptfit,
                           const std::vector<float> & sigy, 
                           float& aminf,
                           float& bminf,
                           float& cminf,
                           double& chi2fit,
                           const bool debug) const; 


    void fit4Var( const std::vector<float>& xfit,
                  const std::vector<float>& yfit,
                  const std::vector<int>& lfit,
                  const std::vector<double>& tfit,
                  const int nptfit,
                  float& aminf,
                  float& bminf,
                  float& cminf,
                  float& vminf,
                  double& chi2fit,
                  const bool vdrift_4parfit,
                  const bool debug) const; 

  protected:

  private:

};
#endif // DTSegment_DTLinearFit_h
