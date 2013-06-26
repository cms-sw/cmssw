#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
int main(){
  std::string input("TK_HV_ON&N/A&N/A%PIX_HV_ON&N/A&N/A%LHC_RAMPING&false&false%PHYSICS_DECLARED&false&false%");
  std::string nameinput("CMS.LVL0:RUNSECTION_DELIMITER_DCSLHCFLAGS_5");
  //const boost::regex e("TK_HV_ON&N/A&N/A%PIX_HV_ON&N/A&N/A%LHC_RAMPING&false&false%PHYSICS_DECLARED&(true|false|N/A)&(true|false|N/A)%$");
  const boost::regex e("%PHYSICS_DECLARED&(true|false|N/A)&(true|false|N/A)%");
  const boost::regex ename("^CMS.LVL0:RUNSECTION_DELIMITER_DCSLHCFLAGS_([0-9]+)");
  boost::match_results<std::string::const_iterator> what;
  boost::regex_search(input,what,e,boost::match_default);
  if(what[0].matched){
    std::cout<<"matched"<<std::endl;
    std::cout<<std::string(what[1].first,what[1].second)<<std::endl;
    std::cout<<std::string(what[2].first,what[2].second)<<std::endl;
  }
  boost::regex_match(nameinput,what,ename,boost::match_default);
  if(what[0].matched){ 
    std::cout<<"named matched"<<std::endl;
    std::cout<<std::string(what[1].first,what[1].second)<<std::endl;
  }
  std::string fname("fillsummary.dat");
  std::ifstream ins(fname.c_str());
  if (!ins.is_open()) {
    std::cout<<"cannot open file "<<fname<<std::endl;
    return 0;
  }
  typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
  std::vector< std::string > result;
  std::string line;
  while(std::getline(ins,line)){
    Tokenizer tok(line);
    result.assign(tok.begin(),tok.end());
    if (result.size()<3) continue;
    std::cout<<"fill num "<<result[0]<<" , fill scheme "<<result[1]<<", ncolliding bunches "<<result[2]<<std::endl;
    //std::copy(result.begin(),result.end(),std::ostream_iterator<std::string>(std::cout,","));
  }
}
