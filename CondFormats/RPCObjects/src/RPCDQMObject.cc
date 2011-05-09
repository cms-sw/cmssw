#include "CondFormats/RPCObjects/interface/RPCDQMObject.h"
#include "FWCore/Utilities/interface/Exception.h"

RPCDQMObject * RPCDQMObject::Fake_RPCDQMObject() {
	RPCDQMObject * fakeObject = new RPCDQMObject();
	return fakeObject;
}







