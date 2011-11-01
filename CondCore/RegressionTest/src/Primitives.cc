#include "CondCore/RegressionTest/interface/Primitives.h"
Primitives::Primitives(int payloadID) :
	testInt(payloadID),
	testLongInt(payloadID),
	testDouble(payloadID),
	testString(),
	testEnum(),
	testTypedef(payloadID)
	
{
	std::stringstream st;
	st << payloadID;
	testString = st.str();
}
Primitives::Primitives() {}

bool Primitives::operator ==(const Primitives& ref) const
{
	if(testInt != ref.testInt)
	{
		return false;
	} 
	if(testLongInt != ref.testLongInt)
	{
		return false;
	}
	if(testDouble != ref.testDouble)
	{
		return false;
	}
	if(testTypedef != ref.testTypedef)
	{
		return false;
	}
	if(testString.compare(ref.testString) != 0)
	{
		return false;
	}
	return true;
}

bool Primitives::operator !=(const Primitives& ref) const
{
	return !operator==(ref);
}
