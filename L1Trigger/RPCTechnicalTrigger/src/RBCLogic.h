#ifndef RBCEmulator_RBCLogic_h
#define RBCEmulator_RBCLogic_h
#include <iostream>
#include <string>
class RBCLogic{
 public:

  RBCLogic(){}


  virtual ~RBCLogic(){ std::cout<<"bye logic"<<std::endl;}

  
  virtual void action(){std::cout <<"Do still nothing"<<std::endl;}

};

#endif
