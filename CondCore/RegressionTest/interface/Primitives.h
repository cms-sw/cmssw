#include "CondCore/RegressionTest/interface/Includes.h"
class Primitives {
public :
	int testInt;
	long int testLongInt;
	double testDouble;
	std::string testString;
	enum TestEnum { A =3, B, C= 101, D, E, F};
	TestEnum testEnum;
	typedef int TestTypedef;
	TestTypedef testTypedef;
	Primitives ();
	Primitives (int payloadID);
	bool operator ==(const Primitives& ref) const;
	bool operator !=(const Primitives& ref) const;
};
