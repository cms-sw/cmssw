/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2002                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcParameters.h"
//#include "L1Trigger/RPCTrigger/src/L1RpcException.h"
#include <iostream>

//inline
int RPCParam::StringToInt(std::string str) {
  for(int i = 0; i < str.size(); i++)
    if(str[i] < '0' || str[i] > '9' )
      //throw L1RpcException("Error in StringToInt(): the string cannot be converted to a number");
      std::cout<< "Error in StringToInt(): the string cannot be converted to a number" <<std::endl;
  return atoi(str.c_str());
};

//inline
std::string RPCParam::IntToString(int number) {
  std::string str;
  /* Some problems. AK
  ostringstream ostr;
  ostr<<number;
  str = ostr.str();
  cout<<"std::string IntToString(int number)"<<endl;
  cout<<str<<endl;
  */
  char tmp[20];
  sprintf(tmp,"%d",number);
  str.append(tmp);
  return str;
};

bool RPCParam::L1RpcConeCrdnts::operator < (const L1RpcConeCrdnts& cone) const{
  if(Tower != cone.Tower)
    return (Tower < cone.Tower);
  if(LogSector != cone.LogSector)
    return (LogSector < cone.LogSector);
  if(LogSegment != cone.LogSegment)
    return (LogSegment < cone.LogSegment);

  return false;
}

bool RPCParam::L1RpcConeCrdnts::operator == (const L1RpcConeCrdnts& cone) const{
  if(Tower != cone.Tower)
    return false;
  if(LogSector != cone.LogSector)
    return false;
  if(LogSegment != cone.LogSegment)
    return false;

  return true;
}
