
#include "CondCore/RegressionTest/interface/TestPayloadClass.h"
#include "CondCore/RegressionTest/interface/RegressionTestPayload.h"
namespace {
	struct dictionary {
	  std::vector<double>::iterator dummy1;
	  std::vector<int>::iterator dummy2;
	  std::vector<float>::iterator dummy5;
	  std::vector<std::string>::iterator dummy10;
	  std::vector<int>::iterator dummy11;
	  std::vector<std::vector<int> >::iterator dummy12;
	  std::vector<DataStructs::Color> dummy13;
	  std::pair<const std::string,std::vector<DataStructs::Color> > dummy14;
	  std::map<std::string, std::vector<DataStructs::Color> > dummy15;
	  __gnu_cxx::hash_map<int, int > dummy21;
	  std::list<std::string> dummy16;
	  //std::queue<int> dummy17;
	  //std::deque<std::string> dummy18;
	  
	  //__gnu_cxx::crope dummy19;
          std::pair<const int,Param> d00;

	};
}
