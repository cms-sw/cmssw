/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"

EcalXtalGroupId::EcalXtalGroupId() {
  id_ = 0;
}

EcalXtalGroupId::~EcalXtalGroupId() {

}

EcalXtalGroupId::EcalXtalGroupId(const unsigned int& id) {
  id_ = id;
}
