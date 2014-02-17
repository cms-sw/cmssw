/**   
    $Date: 2008/02/25 17:42:31 $
    $Revision: 1.2 $
    $Id: InvMatrixCommonDefs.cc,v 1.2 2008/02/25 17:42:31 malberti Exp $ 
    \author $Author: malberti $
*/

#include "Calibration/Tools/interface/InvMatrixCommonDefs.h"

int
ecalIM::uniqueIndex (int eta, int phi)
{
  return eta * SCMaxPhi + phi ;
}
