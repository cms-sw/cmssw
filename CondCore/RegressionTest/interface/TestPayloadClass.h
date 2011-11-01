#include "CondCore/RegressionTest/interface/Includes.h"
#include "CondCore/RegressionTest/interface/Primitives.h"
#include "CondCore/RegressionTest/interface/DataStructs.h"
#include "CondCore/RegressionTest/interface/Inheritances.h"

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
