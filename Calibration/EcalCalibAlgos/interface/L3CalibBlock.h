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

    $Date: 2008/01/23 11:04:54 $
    $Revision: 1.1.2.1 $
    $Id: L3CalibBlock.h,v 1.1.2.1 2008/01/23 11:04:54 govoni Exp $ 
    \author $Author: govoni $
*/
class L3CalibBlock : public VEcalCalibBlock
{
  public :
    //! ctor
    L3CalibBlock (const int numberOfElements, 
                  const int keventweight = 1) ;
    //! dtor
    ~L3CalibBlock () ;
    
    //! insert an entry
    void Fill (std::map<int,double>::const_iterator,
               std::map<int,double>::const_iterator,
    	       double pTk,
               double pSubtract,
               double sigma = 1.) ;

    //! reset the chi2 matrices
    void reset () ;
    //! To retrieve the coefficients
    double at ( int);
    //! solve the chi2 linear system
    void solve (int usingBlockSolver, double min, double max) ;
  
  private :  

    //! the L3 algo
//    MinL3Algorithm * m_L3Algo ;
    //! the universal L3 algo
    MinL3AlgoUniv<unsigned int> * m_L3AlgoUniv ;

} ;


#endif
#endif

/* TODOS and dubios
   ----------------

- fare l'inversione finale
- uso le CLHEP
- ritorno CLHEP o un vector<double>? forse la seconda e' meglio
- oppure una mappa con uindex, coeff? cosi' poi si possono incollare insieme
  tutte le mappe facilmente?
- CLHEP hanno qualche metodo utile per usare STL che non ho visto?
- rem se ho gia' invertito con una variabile: ha senso?

*/
