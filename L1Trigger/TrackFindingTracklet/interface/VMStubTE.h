#ifndef VMSTUBTE_H
#define VMSTUBTE_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>

#include "FPGAWord.h"
#include "Util.h"
#include "Constants.h"
#include "Stub.h"
#include "L1TStub.h"

using namespace std;

class VMStubTE{

public:

  VMStubTE() {}
  

  VMStubTE(std::pair<Stub*, L1TStub*> stub, FPGAWord finephi, FPGAWord finerz, FPGAWord bend, FPGAWord vmbits, FPGAWord allstubindex) {
    stub_=stub;
    finephi_=finephi;
    finerz_=finerz;
    bend_=bend;
    vmbits_=vmbits;
    allStubIndex_=allstubindex;
  }

 
  ~VMStubTE() {

  }

  FPGAWord finephi() const {
    return finephi_;
  }
  
  FPGAWord finerz() const {
    return finerz_;
  }

  FPGAWord bend() const {
    return bend_;
  }
  
  FPGAWord vmbits() const {
    return vmbits_;
  }

  std::pair<Stub*, L1TStub*> stub() const {
    return stub_;
  }

  bool isPSmodule() const {
    return stub_.first->isPSmodule();
  }

  FPGAWord stubindex() const {
    return allStubIndex_;
  }


  //return binary string for memory printout
  std::string str() const {
    
    string stub=allStubIndex_.str();
    stub+="|";
    stub+=bend_.str();
    stub+="|";
    stub+=finephi_.str();
    stub+="|";
    stub+=finerz_.str();
    stub+="|";
    stub+=vmbits_.str();

    return stub;
    
  }

  
private:

  FPGAWord finephi_;
  FPGAWord finerz_;
  FPGAWord bend_;
  FPGAWord vmbits_;
  FPGAWord allStubIndex_;
  std::pair<Stub*, L1TStub*> stub_;

};



#endif



