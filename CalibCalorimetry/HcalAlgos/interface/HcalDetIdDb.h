#ifndef HcalDetIdDb_h
#define HcalDetIdDb_h

/** 
\author Fedor Ratnikov (UMd)
DB instance of HcalDetId
$Author: ratnikov
$Date: 2005/10/18 23:34:56 $
$Revision: 1.1 $
*/

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <vector>
#include <algorithm>


namespace HcalDetIdDb {
  unsigned long HcalDetIdDb (const HcalDetId& fId) {return fId.rawId ();}
  HcalDetId HcalDetId (unsigned long fId) {return ::HcalDetId (fId);}
}

#endif
