#ifndef MonitorElementBaseT_h
#define MonitorElementBaseT_h


#include "DQMServices/Core/interface/MonitorElementT.h"

#include <sstream>
#include <string>

class MonitorElementFloat : public MonitorElementT<float>
{

 public:

  MonitorElementFloat(float *val, const std::string &name) : 
  MonitorElementT<float>(val,name){}
  virtual ~MonitorElementFloat(){}

  virtual void Fill(float x);
  void Reset();
  std::string valueString() const;

 private:
  void Fill(float x, float y, float w=1.);
  void Fill(float x, float y, float z, float w);

};

class MonitorElementInt : public MonitorElementT<int>
{

 public:

  MonitorElementInt(int *val, const std::string &name) : 
  MonitorElementT<int>(val,name){}
  virtual ~MonitorElementInt(){}
   
  virtual void Fill(float x);
  void Reset();
  std::string valueString() const;

 private:
  void Fill(float x, float y, float w=1.);
  void Fill(float x, float y, float z, float w);

};

class MonitorElementString : public MonitorElementT<std::string>
{

 public:

  MonitorElementString(std::string *val, const std::string &name) : 
  MonitorElementT<std::string>(val,name)
  // strings do not change! need to "update" in ctor!
  {update(); }
  // here just to satisfy interface!
  virtual ~MonitorElementString(){}
 private:
  void Fill(float x){}
  void Fill(float x, float y){}
  void Fill(float x, float y, float z, float w=1.){}
  void Reset() {}

 public:
  virtual std::string valueString() const;

};

#endif









