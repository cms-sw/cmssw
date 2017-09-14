#ifndef CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H
#define CONDCORE_ALIGNMENTPLUGINS_ALIGNMENTPAYLOADINSPECTORHELPER_H

#include <vector>
#include <numeric>
#include <string>
#include "TH1.h"
#include "TPaveText.h"

namespace AlignmentPI {

  enum index {
    XX=1,
    XY=2,
    XZ=3,
    YZ=4,
    YY=5,
    ZZ=6
  };

  /*--------------------------------------------------------------------*/
  std::string getStringFromIndex (AlignmentPI::index i)
  /*--------------------------------------------------------------------*/
  {
    switch(i){
    case XX : return "XX";
    case XY : return "XY";
    case XZ : return "XZ";
    case YZ : return "YX";
    case YY : return "YY";
    case ZZ : return "ZZ";
    default : return "should never be here!";
    }
  }

  std::pair<int,int> getIndices(AlignmentPI::index i){
    switch(i){
    case XX : return std::make_pair(0,0);
    case XY : return std::make_pair(0,1);
    case XZ : return std::make_pair(0,2);
    case YZ : return std::make_pair(1,0);
    case YY : return std::make_pair(1,1);
    case ZZ : return std::make_pair(2,2);
    default : return std::make_pair(-1,-1);
    }
  }
  
}

#endif
