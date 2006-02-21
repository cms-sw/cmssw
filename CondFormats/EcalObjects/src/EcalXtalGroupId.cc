#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"

EcalXtalGroupId::EcalXtalGroupId() {
  id_ = 0;
}

EcalXtalGroupId::~EcalXtalGroupId() {

}

EcalXtalGroupId::EcalXtalGroupId(const unsigned int& id) {
  id_ = id;
}
