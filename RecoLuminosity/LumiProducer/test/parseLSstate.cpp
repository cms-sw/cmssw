#include <string>
#include <vector>
#include <map>
#include <iostream>
namespace lumitest{
  struct LSState{
    unsigned int lsnum;
    bool physicsbit;
    bool cmspause;
  };
  
  void stringSplit(const std::string& instr, char delim, std::vector<std::string>&results){
    size_t cutAt=0;
    std::string str=instr;
    while( (cutAt=str.find_first_of(delim))!=std::string::npos){
      if(cutAt>0){
	//std::cout<<"pushing back "<<str.substr(0,cutAt)<<std::endl;
	results.push_back(str.substr(0,cutAt));
	str=str.substr(cutAt+1);
      }
    }
    if(str.length()>0){
      //std::cout<<"pushing back "<<str<<std::endl;
      results.push_back(str);
    }
  }
  
  void fillMap(const std::vector<std::string>& inVector,
	       std::map<float,char>){
    std::vector<std::string>::const_iterator it;
    std::vector<std::string>::const_iterator itBeg=inVector.begin();
    std::vector<std::string>::const_iterator itEnd=inVector.end();
    for(it=itBeg;it!=itEnd;++it){
      std::cout<<*it<<std::endl;
      std::string rundelimeterStr(it->begin(),it->end()-1);
      std::string stateStr(it->end()-1,it->end());
      std::cout<<"rundelimeterStr "<<rundelimeterStr<<std::endl;
      std::cout<<"stateStr "<<stateStr<<std::endl;
    }
  }
}
int main(){
  //
  // float number as string for runsection delimiter,
  // "T" for true,"F" for false, "P" for pause
  //
  // the output of the parsing are 2 booleans per lumisection
  //
  // physicsbit
  // cmspause
  //
  // a lumisection can be considered for recorded luminosity only if
  // physicsbit && !cmspause
  // 
  // source of this decision is documented here:
  // https://savannah.cern.ch/support/?112921
  //
  std::string LSstateInputstr("1.0T,19.9P,21.6F,23.54P");
  std::vector<std::string> parseresult;
  unsigned int totalLS=40; //suppose I know there are 40 cms ls
  lumitest::stringSplit(LSstateInputstr,',',parseresult);
  std::map<float,char> delimiterMap;
  lumitest::fillMap(parseresult,delimiterMap);
}
