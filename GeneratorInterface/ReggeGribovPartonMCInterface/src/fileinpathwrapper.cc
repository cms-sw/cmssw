#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <iostream>
extern "C"{
  const char* getfullpathfromfip_(char* fipname){
    //std::cout << "getfullpathfromfip_ " << fipname << std::endl;
    edm::FileInPath* fip = new edm::FileInPath(fipname); 
    return fip->fullPath().c_str(); 
  }
}  
