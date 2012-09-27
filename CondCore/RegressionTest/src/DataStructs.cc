#include "CondCore/RegressionTest/interface/DataStructs.h"
DataStructs::DataStructs(int payloadID) :
	//testCharArray(),
	//testIntArray(),
	testStruct(),
	testTripletMap(),
	//testTripletHashMap(),
	testStringList(),
	//testTestTypedefQueue(),
	//testCrope(RSIZE, 'a'),
	//testDeque(QSIZE, 'a'),
	testPair("payloadID", payloadID),
	testSet(),
	//testString(""),
	testStringVector(),
	testIntVector()
{
	std::stringstream st;
	st << payloadID;
	tmpColor.r = payloadID;
	tmpColor.g = payloadID;
	tmpColor.b = payloadID;
	testTripletMap["DATA"].push_back(tmpColor);
	//testTripletHashMap["YELLOW"].push_back(tmpColor);
	// testTripletHashMap["WHITE"].push_back(tmpColor);
	// testStringList.push_back("Last");
	// testStringList.push_front("First");
	// testStringList.insert(++testStringList.begin(), "Middle");
	
	// for(int i =0; i<QSIZE; i++)
	// {
		// testTestTypedefQueue.push(i);
	// }
	//testDeque.push_back("Last");
	//testDeque.push_front("First");
	testSet.insert((payloadID % 10) + 48);
	// for(int i=0; i<INTSIZE; i++)
	// {
		// testIntArray[i] = i;
	// }
		// for(int i=0; i<CHSIZE; i++)
	// {
		// testCharArray[i] = (payloadID % 10) + 48;
	// }
	testStringVector.reserve(VSIZE);
	for(int i=0; i<SVSIZE; i++)
	{
		testStringVector.push_back(st.str());
		testIntVector.push_back(i);
	}
	testStruct.testStructString = st.str();
	testStruct.testStructInt = payloadID;
}
DataStructs::DataStructs() {}

bool DataStructs::operator ==(const DataStructs& ref) const
{
	// for(int i =0; i<CHSIZE; i++)
	// {
		// if(testCharArray[i] != ref.testCharArray[i])
		// {
			// std::cout<<i<<"fail"<<std::endl;
			// std::cout<<testCharArray[i]<<" != "<<ref.testCharArray[i]<<std::endl;
			// return false;
		// }
	// }
	// for(int i =0; i<INTSIZE; i++)
	// {
		// if(testIntArray[i] != ref.testIntArray[i])
		// {
			// std::cout<<i<<"fail"<<std::endl;
			// std::cout<<testIntArray[i]<<" != "<<ref.testIntArray[i]<<std::endl;
			// return false;
		// }
	// }
	if(testTripletMap != ref.testTripletMap)
	{
		return false;
	}
	// if(testTripletHashMap != ref.testTripletHashMap)
	// {
		// return false;
	// }
	if(testStringList != ref.testStringList)
	{
		return false;
	}
	// if(testTestTypedefQueue != ref.testTestTypedefQueue)
	// {
		// return false;
	// }
	// if(testCrope != ref.testCrope)
	// {
		// return false;
	// }
	// if(testDeque != ref.testDeque)
	// {
		// return false;
	// }
	if(testPair != ref.testPair)
	{
		return false;
	}
	if(testSet != ref.testSet)
	{
		return false;
	}
	for(int i=0; i<VSIZE; i++)
	{
		if(testStringVector[i].compare(ref.testStringVector[i]) != 0)
		{
			return false;
		}
	}
	for(int i=0; i<SVSIZE; i++)
	{
		if(testIntVector != ref.testIntVector)
		{
			return false;
		}
	}
	if(testStruct.testStructString.compare(ref.testStruct.testStructString) != 0)
	{
		return false;
	}
	if(testStruct.testStructInt != ref.testStruct.testStructInt)
	{
		return false;
	}
	return true;
}

bool DataStructs::operator !=(const DataStructs& ref) const
{
	return !operator==(ref);
}
