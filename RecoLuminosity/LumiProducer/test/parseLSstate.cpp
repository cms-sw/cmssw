#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "RecoLuminosity/LumiProducer/interface/Utils.h"
namespace lumitest{
  void stringSplit(const std::string& instr, char delim, std::vector<std::string>&results){
    size_t cutAt=0;
    std::string str=instr;
    while( (cutAt=str.find_first_of(delim))!=std::string::npos){
      if(cutAt>0){
	results.push_back(str.substr(0,cutAt));
	str=str.substr(cutAt+1);
      }
    }
    if(str.length()>0){
      results.push_back(str);
    }
  }
  
  void fillMap(const std::vector<std::string>& inVector,
	       std::map<unsigned int,std::string>& result){
    std::vector<std::string>::const_iterator it;
    std::vector<std::string>::const_iterator itBeg=inVector.begin();
    std::vector<std::string>::const_iterator itEnd=inVector.end();
    for(it=itBeg;it!=itEnd;++it){
      //std::cout<<*it<<std::endl;
      std::string rundelimeterStr(it->begin(),it->end()-1);
      std::string stateStr(it->end()-1,it->end());
      //std::cout<<"rundelimeterStr "<<rundelimeterStr<<std::endl;
      float rundelimeter=0.0;
      if(!lumi::from_string(rundelimeter,rundelimeterStr,std::dec)){
	std::cout<<"failed to convert string to float"<<std::endl;
      }
      //std::cout<<"stateStr "<<stateStr<<std::endl;
      //
      //logic of rounding:
      //for states physics_declared T,F, use ceil function to round up then convert to unsigned int, because T,F will be set at the next LS boundary
      //for state paused P, use floor function to round down then convert to unsigned int, because we count the LS as paused as long as it contains pause (this logic could be changed)
      //
      if(stateStr=="P"){
	result.insert(std::make_pair((unsigned int)std::floor(rundelimeter),stateStr));}else if(stateStr=="T"||stateStr=="F"){
	result.insert(std::make_pair((unsigned int)std::ceil(rundelimeter),stateStr));
      }else{
	throw std::runtime_error("unknown LS state");
      }
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
  // physicsDeclared
  // isPaused
  //
  // a lumisection can be considered for recorded luminosity only if
  // PhysicsDeclared && !isPaused
  // 
  // source of this decision is documented here:
  // https://savannah.cern.ch/support/?112921
  //
  std::string LSstateInputstr("1.0T,19.9P,21.6F,23.54P");//output of database query in one string format
  unsigned int totalLS=40; //suppose there are 40 cms ls, I want to assign T,F,P state to each of them
  std::vector<std::string> parseresult;
  lumitest::stringSplit(LSstateInputstr,',',parseresult);//split the input string into a vector of string pairs by ','
  std::map<unsigned int,std::string> delimiterMap;
  lumitest::fillMap(parseresult,delimiterMap);//parse the vector into state boundary LS(key) to LS state(value) map
  //keys [1,19,22,23]
  for(unsigned int ls=1;ls<=totalLS;++ls){ //loop over my LS comparing it to the state boundaries
    std::map<unsigned int,std::string>::const_iterator lsItUp;
    lsItUp=delimiterMap.upper_bound(ls);
    std::string r;
    if(lsItUp!=delimiterMap.end()){
      lsItUp=delimiterMap.upper_bound(ls);
      --lsItUp;
      r=(*lsItUp).second;
      //std::cout<<"LS "<<ls<<std::endl;
      //std::cout<<"boundary "<<(*lsItUp).first<<std::endl;
      //std::cout<<"state "<<r<<std::endl;
    }else{
      std::map<unsigned int,std::string>::reverse_iterator lsItLast=delimiterMap.rbegin();    
      r=(*lsItLast).second;
      //std::cout<<"LS "<<ls<<std::endl;
      //std::cout<<"boundary "<<(*lsItLast).first<<std::endl;
      //std::cout<<"state "<<r<<std::endl;
    }
    std::cout<<"LS : "<<ls<<" , state : "<<r<<std::endl; 
  }
}
