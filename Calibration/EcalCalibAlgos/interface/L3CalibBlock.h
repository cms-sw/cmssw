#ifndef __CINT__
#ifndef L3CalibBlock_H
#define L3CalibBlock_H

#include <map>
#include <string>
#include <vector>

#include "CLHEP/Matrix/GenMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Random/RandGaussQ.h"

#include "Calibration/Tools/interface/InvMatrixCommonDefs.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/Tools/interface/MinL3AlgoUniv.h"
#include "Calibration/EcalCalibAlgos/interface/VEcalCalibBlock.h"

/** \class L3CalibBlock
 
    \brief interface to the L3Univ class for testing  

*/
class L3CalibBlock : public VEcalCalibBlock
{
  public :
    //! ctor
    L3CalibBlock (const int numberOfElements, 
                  const int keventweight = 1) ;
    //! dtor
    ~L3CalibBlock () override ;
    
    //! insert an entry
    void Fill (std::map<int,double>::const_iterator,
               std::map<int,double>::const_iterator,
    	       double pTk,
               double pSubtract,
               double sigma = 1.) override ;

    //! reset the calib objects
    void reset () override ;
    //! To retrieve the coefficients
    double at ( int);
    //! get the coefficients
    int solve (int usingBlockSolver, double min, double max) override ;
  
  private :  

    //! the L3 algo
//    MinL3Algorithm * m_L3Algo ;
    //! the universal L3 algo
    MinL3AlgoUniv<unsigned int> * m_L3AlgoUniv ;

} ;


#endif
#endif

