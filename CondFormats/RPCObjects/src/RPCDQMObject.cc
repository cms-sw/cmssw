#include "CondFormats/RPCObjects/interface/RPCDQMObject.h"
#include "FWCore/Utilities/interface/Exception.h"

RPCDQMObject* RPCDQMObject::Fake_RPCDQMObject() {
  RPCDQMObject* fakeObject = new RPCDQMObject();
  fakeObject->dqmv = -1;
  fakeObject->run = -1;
  return fakeObject;
}
