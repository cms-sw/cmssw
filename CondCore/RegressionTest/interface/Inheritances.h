#include "CondCore/RegressionTest/interface/Includes.h"


class TestData
{
public :
	int commonInt;
	std::vector<std::vector<int> > commonIntVector2d;
	TestData (int payloadID);
	TestData ();
	bool operator ==(const TestData& ref) const;
	bool operator !=(const TestData& ref) const;

};

class TestInheritance : public TestData
{
public :
	std::vector<std::string> dataStringVector;
	TestInheritance (int payloadID);
	TestInheritance ();
	bool operator ==(const TestInheritance& ref) const;
	bool operator !=(const TestInheritance& ref) const;
};

class Inheritances {
public:
	TestData testData; 
	TestInheritance testInheritance;
	Inheritances(int payloadID);
	Inheritances();
	bool operator ==(const Inheritances& ref) const;
	bool operator !=(const Inheritances& ref) const;
};
