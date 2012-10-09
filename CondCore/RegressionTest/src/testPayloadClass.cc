#include "CondCore/RegressionTest/interface/TestPayloadClass.h"

TestPayloadClass::TestPayloadClass(int payloadID) :
	primitives(payloadID),
	dataStructs(payloadID),
	inheritances(payloadID)
{}
TestPayloadClass::TestPayloadClass() {}

bool TestPayloadClass::DataToFile(std::string fname)
{
	std::ofstream myfile;
	myfile.open (fname.c_str());
	myfile << "TestPayloadClass : "<<std::endl;
	myfile << "primitives.testInt : "<<std::endl;
	myfile<<primitives.testInt<<std::endl;
	myfile << "primitives.testLongInt : "<<std::endl;
	myfile <<primitives.testLongInt<<std::endl;
	myfile << "primitives.testDouble : "<<std::endl;
	myfile <<primitives.testDouble<<std::endl;
	// myfile << "dataStructs.testCharArray : "<<std::endl;
	// myfile <<dataStructs.testCharArray<<std::endl;
	// myfile << "dataStructs.testIntArray : "<<std::endl;
	// for(int i=0; i<INTSIZE; i++)
	// {
		// myfile <<dataStructs.testIntArray[i]<<std::endl;
	// }
	myfile << "primitives.testTypedef : "<<std::endl;
	myfile <<primitives.testTypedef<<std::endl;
	myfile << "primitives.testString : "<<std::endl;
	myfile <<primitives.testString<<std::endl;
	myfile << "dataStructs.testStringVector : "<<std::endl;
	for(int i=0; i<VSIZE; i++)
	{
		myfile << "dataStructs.testStringVector["<<i<<"] :"<<std::endl;
		myfile <<dataStructs.testStringVector[i]<<std::endl;
	}
	myfile << "dataStructs.testIntVector : "<<std::endl;
	for(int i=0; i<SVSIZE; i++)
	{
		myfile << "dataStructs.testIntVector["<<i<<"] :"<<std::endl;
		myfile <<dataStructs.testIntVector[i]<<std::endl;
	}
	myfile << "inheritances.testData : "<<std::endl;
	myfile << "inheritances.testData.commonInt : "<<std::endl;
	myfile <<inheritances.testData.commonInt<<std::endl;
	myfile << "inheritances.testData.commonIntVector2d : "<<std::endl;
	for(int i=0; i<VSIZE; i++)
		for(int j =0; j<VSIZE; j++)
		{
			myfile << "inheritances.testdata.commonIntVector2d["<<i<<"]["<<j<<"] : "<<std::endl;
			myfile <<inheritances.testData.commonIntVector2d[i][j]<<std::endl;
		}
	myfile << "inheritances.testInheritance : "<<std::endl;
	myfile << "inheritances.testInheritance.commonInt : "<<std::endl;
	myfile <<inheritances.testInheritance.commonInt<<std::endl;
	myfile << "testInheritance.commonIntVector2d : "<<std::endl;
	for(int i=0; i<VSIZE; i++)
		for(int j =0; j<VSIZE; j++)
		{
			myfile << "inheritances.testInheritance.commonIntVector2d["<<i<<"]["<<j<<"] : "<<std::endl;
			myfile <<inheritances.testInheritance.commonIntVector2d[i][j]<<std::endl;
		}
	myfile << "inheritances.testInheritance.dataStringVector : "<<std::endl;
	for(int i=0; i<VSIZE; i++)
	{
		myfile << "inheritances.testInheritance.dataStringVector["<<i<<"] :"<<std::endl;
		myfile <<inheritances.testInheritance.dataStringVector[i]<<std::endl;
	}
	myfile << "dataStructs.testTripletMap : "<<std::endl;
	for (std::map<std::string, std::vector<DataStructs::Color> >::iterator it = dataStructs.testTripletMap.begin(); it != dataStructs.testTripletMap.end(); ++it)
		for(unsigned int i=0; i<(*it).second.size(); i++)
		{
			myfile << "dataStructs.testTripletMap["<<(*it).first<<"] :"<<std::endl;
			myfile <<(*it).second[i].r<<std::endl;
			myfile <<(*it).second[i].g<<std::endl;
			myfile <<(*it).second[i].b<<std::endl;
		}
		// myfile << "dataStructs.testTripletMap : "<<std::endl;
	// for (stdext::hash_map<std::string, std::vector<DataStructs::Color> >::iterator it = dataStructs.testTripletHashMap.begin(); it != dataStructs.testTripletHashMap.end(); ++it)
		// for(unsigned int i=0; i<(*it).second.size(); i++)
		// {
			// myfile << "dataStructs.testTripletHashMap["<<(*it).first<<"] :"<<std::endl;
			// myfile <<(*it).second[i].r<<std::endl;
			// myfile <<(*it).second[i].g<<std::endl;
			// myfile <<(*it).second[i].b<<std::endl;
		// }
	myfile << "dataStructs.testStringList : "<<std::endl;
	for (std::list<std::string>::iterator it = dataStructs.testStringList.begin(); it != dataStructs.testStringList.end(); ++it)
	{
		myfile <<*it<<std::endl;
	}
	// myfile<<"dataStructs.testTestTypedefQueue : "<<std::endl;
	// for(int i=0; i<QSIZE; i++)
	// {
	// myfile<<dataStructs.testTestTypedefQueue.front()<<std::endl;
	// dataStructs.testTestTypedefQueue.pop();
		
	// }
	// myfile<<"dataStructs.testCrope : "<<std::endl;
	// myfile<<dataStructs.testCrope<<std::endl;
	// myfile<<"dataStructs.testDeque : "<<std::endl;
	// for (int i=0; i<2; ++i)
	// {
		// myfile<<dataStructs.testDeque.at(i)<<std::endl;
	// }
	myfile<<"dataStructs.testPair :"<<std::endl;
	myfile<<dataStructs.testPair.first<<" "<<dataStructs.testPair.second<<std::endl;
	// myfile<<"dataStructs.testSet :"<<std::endl;
	// for ( std::set<char>::const_iterator it = dataStructs.testSet.begin(); it != dataStructs.testSet.end(); it++ )
	// {
		// myfile << *it <<std::endl;
	// }
	myfile.close();
	return true;
}
bool TestPayloadClass::operator ==(const TestPayloadClass& ref) const
{
	if(primitives != ref.primitives)
	{
		return false;
	}
	if(dataStructs != ref.dataStructs)
	{
		return false;
	}
	if(inheritances != ref.inheritances)
	{
		return false;
	}
	return true;
}

bool TestPayloadClass::operator !=(const TestPayloadClass& ref) const
{
	return !operator==(ref);
}
