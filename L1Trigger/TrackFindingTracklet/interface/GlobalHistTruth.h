#ifndef GLOBALHISTTRUTH_H
#define GLOBALHISTTRUTH_H

#include "HistBase.h"

using namespace std;

class GlobalHistTruth{

public:

  static SLHCEvent*& event(){
    static SLHCEvent* theEvent=0;
    return theEvent;
  }


  static HistBase*& histograms(){
    static HistBase dummy;
    static HistBase* theHistBase=&dummy;
    return theHistBase;
  }

};



#endif



