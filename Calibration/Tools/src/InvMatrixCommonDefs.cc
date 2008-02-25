/**   
    $Date: 2008/01/23 10:59:54 $
    $Revision: 1.1.2.1 $
    $Id: InvMatrixCommonDefs.cc,v 1.1.2.1 2008/01/23 10:59:54 govoni Exp $ 
    \author $Author: govoni $
*/

#include "Calibration/Tools/interface/InvMatrixCommonDefs.h"

int
ecalIM::uniqueIndex (int eta, int phi)
{
  return eta * SCMaxPhi + phi ;
}
