
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include <sstream>

LinkBoardSpec::LinkBoardSpec(bool m, int l, int n)
    : theMaster(m), theLinkBoardNumInLink(l), theCode(n) { }

void LinkBoardSpec::add(const FebConnectorSpec & feb)
{
  theFebs.push_back(feb);
}

const FebConnectorSpec * LinkBoardSpec::feb(int febInputNum) const
{
  //FIXME - temporary implementaion, to be replace by LUT (in preparation)
  typedef std::vector<FebConnectorSpec>::const_iterator IT;
  for (IT it=theFebs.begin(); it != theFebs.end(); it++) {
    if(febInputNum==it->linkBoardInputNum()) return &(*it);
  }
  return nullptr;
}

std::string LinkBoardSpec::linkBoardName() const
{
  std::ostringstream lbName;
  std::string char1Val[2]={"B","E"};                              // 1,2
  std::string char2Val[3]={"N","M","P"};                          // 0,1,2
  std::string char4Val[9]={"0","1","2","3","A","B","C","D","E"};  // 0,...,8
  int n3=theCode%10;
  int num3=(theCode%100)/10;
  int n2=(theCode%1000)/100;
  int n1=(theCode%10000)/1000;
  int wheel=(theCode%100000)/10000;
  if(n2==0)wheel=-wheel;
  int sector=theCode/100000;
  std::string sign="";
  if(wheel>0) sign="+";
  lbName <<"LB_R"<<char1Val[n1-1]<<sign<<wheel<<"_S"<<sector<<"_"<<char1Val[n1-1]<<char2Val[n2]<<num3<<char4Val[n3]<<"_CH"<<theLinkBoardNumInLink;
  return lbName.str();
}

std::string LinkBoardSpec::print(int depth ) const 
{
  std::ostringstream str;
  std::string type = (theMaster) ? "master" : "slave";
  str <<" LinkBoardSpec: " << std::endl
            <<" --->" <<type<<" linkBoardNumInLink: " << theLinkBoardNumInLink 
            << std::endl;
  depth--;
  if (depth >=0) {
    typedef std::vector<FebConnectorSpec>::const_iterator IT;
    for (IT it=theFebs.begin(); it != theFebs.end(); it++) str << (*it).print(depth);
  }
  return str.str();
}

