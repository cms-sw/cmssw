#include "CondCore/DBTest/interface/Includes.h"
#include "CondCore/DBTest/interface/Primitives.h"
#include "CondCore/DBTest/interface/DataStructs.h"
#include "CondCore/DBTest/interface/Inheritances.h"

class TestPayloadClass
{	
private:
	Primitives primitives;
	DataStructs dataStructs;
	Inheritances inheritances;
public :
	TestPayloadClass(int payloadID);
	TestPayloadClass();
	bool DataToFile(std::string fname);
	bool operator ==(const TestPayloadClass& ref) const;
	bool operator !=(const TestPayloadClass& ref) const;
};
