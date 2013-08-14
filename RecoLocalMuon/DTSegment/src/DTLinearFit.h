#ifndef DTSegment_DTLinearFit_h
#define DTSegment_DTLinearFit_h

/** \class DTLinearFit
 *
 * Description:
 *  
 * detailed description
 *
 * $Date: 2008/12/03 12:52:22 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */

/* Collaborating Class Declarations */

/* C++ Headers */
#include <vector>

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
             float& covss,
             float& covii,
             float& covsi) const;

  protected:

  private:

};
#endif // DTSegment_DTLinearFit_h
