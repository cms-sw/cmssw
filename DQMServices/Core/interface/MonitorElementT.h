#ifndef MonitorElementT_h
#define MonitorElementT_h

#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>

template<class T>
class MonitorElementT : public MonitorElement
{
 public:
  MonitorElementT(T *val, const std::string name="") : 
  name_(name), val_(val) 
  {reference_ = 0;}
  virtual ~MonitorElementT() 
  {
    delete val_;
    if(reference_)deleteReference();
  } 

  void clear(){} 
  // pointer to val_
  T * operator->(){
    update();
    return val_;
  }
  // return *val_ by reference
  T & operator*()
    {
      update();
      return *val_;
    }
  // return *val_ by value
  T operator*() const {return *val_;}
  
  // noarg functor (do we need this?)
  T operator()(){return *val_;}

  // explicit cast overload
  operator T(){return *val_;}

  virtual std::string getName() const {return name_;}
  virtual T getValue() const {return *val_;}

  virtual void Reset()=0;

  virtual std::string valueString() const {return std::string();};
  
 private:
  
  std::string name_;
  T * val_;
 protected:

  T * reference_; // this is set to "val_" upon a "softReset"

  // delete reference_
  void deleteReference(void)
  {
    if(!reference_)
      {
	std::cerr << " *** Cannot delete null reference for " 
		  << getName() << std::endl;
	return;
      }
    delete reference_;
    reference_ = 0;
  }

  // for description: see DQMServices/Core/interface/MonitorElement.h
  void enableSoftReset(bool flag)
  {
    softReset_on = flag;

    std::cout << " \"soft-reset\" option has been";
    if(softReset_on)
      std::cout << " en";
    else
      std::cout << " dis";
    std::cout << "abled for " << getName() << std::endl;

    if(softReset_on)
	this->softReset();
    else
      if(reference_)deleteReference();
    
  }

  // this is really bad; unfortunately, gcc 3.2.3 won't let me define 
  // template classes, so I have to find a workaround for now
  // error: "...is not a template type" - christos May26, 2005
  friend class CollateMERootH1;
  friend class CollateMERootH2;
  friend class CollateMERootH3;
  friend class CollateMERootProf;

};
#endif









