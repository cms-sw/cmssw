#include "TMVA/PyMethodBase.h"
#include <iostream>

void test()
{
  TMVA::PyMethodBase::PyInitialize();
  std::cout<<"TMVA::PyMethodBase::PyInitialize(): OK"<<std::endl;
  return;
}

int main( int argc, char** argv )
{
  test();
  return 0;
}
