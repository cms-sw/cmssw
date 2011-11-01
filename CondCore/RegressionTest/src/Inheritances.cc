#include "CondCore/RegressionTest/interface/Inheritances.h"

TestData::TestData (int payloadID): commonInt(payloadID), commonIntVector2d() 
{
	for(int i =0; i<VSIZE; i++) 
	{
		commonIntVector2d.push_back ( std::vector<int>() );
		for(int j=0; j<VSIZE; j++) 
		{
			commonIntVector2d[i].push_back(payloadID); 
		}
	}
}
TestData::TestData () {}
bool TestData::operator ==(const TestData& ref) const
{
	if(commonInt != ref.commonInt)
	{
		std::cout<<commonInt<<std::endl;
		return false;
	} 
	for(int i=0; i<VSIZE; i++)
		for(int j=0; j<VSIZE; j++)
		{
			if(commonIntVector2d[i] != ref.commonIntVector2d[i])
			{	
				return false;
			}
		}
	return true;
}
bool TestData::operator !=(const TestData& ref) const
{
	return !operator==(ref);
}



TestInheritance::TestInheritance (int payloadID):
TestData(),
dataStringVector()
{
	commonInt = payloadID;
	for(int i =0; i<VSIZE; i++)
	{
		commonIntVector2d.push_back ( std::vector<int>() );
		for(int j=0; j<VSIZE; j++)
		{
			commonIntVector2d[i].push_back(payloadID); 
		}
	}
	for(int i=0; i<VSIZE; i++)
	{
		std::stringstream st;
		st << payloadID;
		dataStringVector.push_back(st.str());
	}
}
TestInheritance::TestInheritance () {}
bool TestInheritance::operator ==(const TestInheritance& ref) const
{
	if(commonInt != ref.commonInt)
	{
		return false;
	}
	for(int i=0; i<VSIZE; i++)
		for(int j=0; j<VSIZE; j++)
		{
			if(commonIntVector2d[i][j] != ref.commonIntVector2d[i][j])
			{
				return false;
			}
		}
	for(int i=0; i<VSIZE; i++)
	{
		if(dataStringVector[i].compare(ref.dataStringVector[i]) != 0)
		{
			return false;
		}
	}
	return true;
}
bool TestInheritance::operator !=(const TestInheritance& ref) const
{
	return !operator==(ref);
}

Inheritances::Inheritances(int payloadID) :
	testData(payloadID),
	testInheritance(payloadID)
{}
Inheritances::Inheritances() {}

bool Inheritances::operator ==(const Inheritances& ref) const
{
	if(testData != ref.testData)
	{
		return false;
	}
	if(testInheritance != ref.testInheritance)
	{
		return false;
	}
	return true;
}

bool Inheritances::operator !=(const Inheritances& ref) const
{
	return !operator==(ref);
}
