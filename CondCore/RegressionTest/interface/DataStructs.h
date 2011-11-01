#include "CondCore/RegressionTest/interface/Includes.h"
// namespace __gnu_cxx
// {

	// #ifdef __GNUC__
	  // template <> struct hash<std::string> {
		// unsigned int operator()(const std::string& s) const { 
		  // return chash(s.c_str());
		// }
	  // private: 
		// hash<const char *> chash;
	  // };
	// #endif
// }


class  DataStructs {
public:
	struct TestStruct 
	{
		std::string testStructString;
		int testStructInt;
	};
	struct Color
	{
	public:
		int r;
		int g;
		int b;
		bool operator ==(const Color& ref) const
		{
			if(r != ref.r)
			{
				return false;
			} 
			if(g != ref.g)
			{
				return false;
			} 
			if(b != ref.b)
			{
				return false;
			} 
			return true;
		}
		bool operator !=(const Color& ref) const
		{
			return !operator==(ref);
		}
	};
	TestStruct testStruct;
	Color tmpColor;
	std::map<std::string, std::vector<Color> > testTripletMap;
	//__gnu_cxx::hash_map<std::string, std::vector<Color> > testTripletHashMap;
	std::list<std::string> testStringList;
	//std::queue<int> testTestTypedefQueue;
	 //__gnu_cxx::crope testCrope;
	//std::deque <std::string> testDeque;
	//std::deque <int> testDeque;
	//char testCharArray[CHSIZE];
	//int testIntArray[INTSIZE];
	std::pair<std::string, int> testPair; 
	std::set<char> testSet;
	std::vector<std::string> testStringVector;
	std::vector<int> testIntVector;
	DataStructs();
	DataStructs(int payloadID);
	bool operator ==(const DataStructs& ref) const;
	bool operator !=(const DataStructs& ref) const;

};
