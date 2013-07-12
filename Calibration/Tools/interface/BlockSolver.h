#ifndef __CINT__
#ifndef BlockSolver_H
#define BlockSolver_H

#include <map>
#include <string>
#include <vector>

#include "CLHEP/Matrix/GenMatrix.h"
#include "CLHEP/Matrix/Matrix.h"
#include "CLHEP/Matrix/Vector.h"
#include "CLHEP/Random/RandGaussQ.h"

#include "Calibration/Tools/interface/InvMatrixCommonDefs.h"

/** \class BlockSolver
 
    \brief solves at best the matrix invertion for calibration 

    $Date: 2008/02/25 17:50:16 $
    $Revision: 1.2 $
    $Id: BlockSolver.h,v 1.2 2008/02/25 17:50:16 malberti Exp $ 
    \author $Author: malberti $
*/
struct BlockSolver
{
  int operator () (const CLHEP::HepMatrix & matrix, 
                   const CLHEP::HepVector & vector,
                   CLHEP::HepVector & result) ;

  private :

   //! eliminate empty columns and rows                
   void shrink (const CLHEP::HepMatrix & matrix,
                CLHEP::HepMatrix & solution,
                const CLHEP::HepVector & result,
                CLHEP::HepVector & input,
                const std::vector<int> & where) ;
   //! pour results in bigger vector             
   void pour (CLHEP::HepVector & result,
              const CLHEP::HepVector & output,
              const std::vector<int> & where) ;
  
} ;


#endif
#endif

