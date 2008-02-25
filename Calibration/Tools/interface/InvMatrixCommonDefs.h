/** \class InvMatrixCommonDefs

    \brief common definitions that have to hold across different programs
    
    $Date: 2008/01/23 10:59:54 $
    $Revision: 1.1.2.1 $
    $Id: InvMatrixCommonDefs.h,v 1.1.2.1 2008/01/23 10:59:54 govoni Exp $ 
    \author $Author: govoni $
*/

#ifndef __CINT__
#ifndef  InvMatrixCommonDefs_h
#define  InvMatrixCommonDefs_h

#include "Calibration/Tools/interface/CalibCoeff.h"

#include <map>

#define SCMaxPhi 20
#define SCMaxEta 85
//const int  SCMaxPhi = 20 ;
//const int  SCMaxEta = 85 ;

namespace ecalIM 
{

  typedef std::map<int,CalibCoeff> coeffMap ;
  typedef std::map<int,CalibCoeff>::const_iterator coeffMapIt ;
  typedef std::pair<coeffMapIt,coeffMapIt> coeffBlock ;
  
  int uniqueIndex (int eta, int phi) ;

}
#endif
#endif

