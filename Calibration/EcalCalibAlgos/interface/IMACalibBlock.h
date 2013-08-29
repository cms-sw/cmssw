#ifndef __CINT__
#ifndef EcalCalibBlock_H
#define EcalCalibBlock_H

#include <map>
#include <string>
#include <vector>

#include "CLHEP/Matrix/GenMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Random/RandGaussQ.h"

#include "Calibration/Tools/interface/InvMatrixCommonDefs.h"
#include "Calibration/EcalCalibAlgos/interface/VEcalCalibBlock.h"

/** \class EcalCalibBlock
 
    \brief element for the single ECAL block intercalibration  

*/
class IMACalibBlock : public VEcalCalibBlock
{
  public :
    //! ctor
    IMACalibBlock (const int) ;
    //! dtor
    ~IMACalibBlock () ;
    
    //! insert an entry
    void Fill (std::map<int,double>::const_iterator,
               std::map<int,double>::const_iterator,
               double pTk,
               double pSubtract,
               double sigma = 1.) ;

    //! reset the chi2 matrices
    void reset () ;
    //! solve the chi2 linear system
    int solve (int usingBlockSolver, double min, double max) ;
  private :
    
    //! give the size of a chi2 matrix
    int evalX2Size () ;
    //! complete the triangolar chi2 matrix to a sym one
    void complete () ;
    //! copy a vector into a CLHEP object
    void riempiMtr (const std::vector<double> & piena, CLHEP::HepMatrix & vuota) ;
    //! copy a vector into a CLHEP object
    void riempiVtr (const std::vector<double> & pieno, CLHEP::HepVector & vuoto) ;
    //! fill the coefficients map from the CLHEP vector solution
    void fillMap (const CLHEP::HepVector & result) ; 
  
  private :  

    //! vector for the chi2 inversion
    std::vector<double> m_kaliVector ; 
    //! matrix for the chi2 inversion
    std::vector<double> m_kaliMatrix ;
} ;


#endif
#endif

