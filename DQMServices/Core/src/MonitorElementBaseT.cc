#include "DQMServices/Core/interface/MonitorElementBaseT.h"

using namespace std;

void MonitorElementFloat::Fill(float x, float y, float z, float w)
{
  Fill(x,w);
}
void MonitorElementFloat::Fill(float x, float y, float w)
{
  std::cerr << " *** Fill method for floats needs to be called"
	    << " with one argument! " << std::endl;
  
}
void MonitorElementFloat::Fill(float x)
{
  MonitorElementT<float>::operator*()=x;
}
void MonitorElementFloat::Reset()
{
  MonitorElementT<float>::operator*()=0;
}
string MonitorElementFloat::valueString() const
{
  ostringstream retval;
  retval << "f=" << MonitorElementT<float>::operator*();
  return retval.str();
}

void MonitorElementInt::Fill(float x, float y, float z, float w)
{
  Fill(x,w);
}
void MonitorElementInt::Fill(float x, float y, float w)
{
  std::cerr << " *** Fill method for integers needs to be called"
	    << " with one argument! " << std::endl;
}
void MonitorElementInt::Fill(float x)
{
  MonitorElementT<int>::operator*()=(int)x;
}
void MonitorElementInt::Reset()
{
  MonitorElementT<int>::operator*()=0;
}
string MonitorElementInt::valueString() const
{
  ostringstream retval;
  retval << "i="<<MonitorElementT<int>::operator*(); 
  return retval.str();
}

string MonitorElementString::valueString() const
{
  ostringstream retval;
  retval << "s="<<MonitorElementT<string>::operator*(); 
  return retval.str();
}
