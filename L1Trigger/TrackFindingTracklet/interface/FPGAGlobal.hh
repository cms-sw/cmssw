#ifndef FPGAGLOBAL_H
#define FPGAGLOBAL_H

#include "FPGAHistBase.hh"

using namespace std;

class FPGAGlobal{

public:

  static SLHCEvent*& event(){
    static SLHCEvent* theEvent=0;
    return theEvent;
  }


  static FPGAHistBase*& histograms(){
    static FPGAHistBase dummy;
    static FPGAHistBase* theHistBase=&dummy;
    return theHistBase;
  }

};



#endif



