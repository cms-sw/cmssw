#ifndef HcalDetIdDb_h
#define HcalDetIdDb_h

/** 
\author Fedor Ratnikov (UMd)
DB instance of HcalDetId
$Author: ratnikov
$Date: 2005/10/28 01:29:29 $
$Revision: 1.1 $
*/

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include <vector>
#include <algorithm>


namespace HcalDetIdDb {
  inline unsigned long HcalDetIdDb (const HcalDetId& fId) {return fId.rawId ();}
  inline HcalDetId HcalDetId (unsigned long fId) {return ::HcalDetId (fId);}
}

#endif
