#include "CondFormats/RPCObjects/interface/RPCNoiseObject.h"
#include "FWCore/Utilities/interface/Exception.h"

RPCNoiseObject * RPCNoiseObject::Fake_RPCNoiseObject() {
	RPCNoiseObject * fakeObject = new RPCNoiseObject();
        fakeObject->version = -1; 
        fakeObject->run = -1;
	return fakeObject;
}







