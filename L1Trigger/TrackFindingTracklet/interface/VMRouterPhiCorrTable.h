#ifndef VMROUTERPHICORRTABLE_H
#define VMROUTERPHICORRTABLE_H

#include "TETableBase.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class VMRouterPhiCorrTable: public TETableBase{

public:

  VMRouterPhiCorrTable() {
    nbits_ = 14; 
  }

  ~VMRouterPhiCorrTable() {

  }


  void init(int layer,
	    int bendbits,
	    int rbits
	    ) {

    assert(bendbits==3||bendbits==4); 
    
    layer_=layer;
    bendbits_=bendbits;
    rbits_=rbits;

    rbins_=(1<<rbits);
    rmin_=rmean[layer-1]-drmax;
    rmax_=rmean[layer-1]+drmax;
    dr_=2*drmax/rbins_;

    bendbins_=(1<<bendbits);
 
    rmean_=rmean[layer-1];

    for (int ibend=0;ibend<bendbins_;ibend++) {
      for (int irbin=0;irbin<rbins_;irbin++) {
	int value=getphiCorrValue(ibend,irbin);
	table_.push_back(value);
      }
    }

    if (writeVMTables) {
      writeVMTable("VMPhiCorrL"+std::to_string(layer_)+".txt", false);
    }

    
  }

  int getphiCorrValue(int ibend, int irbin){

    double bend=Stub::benddecode(ibend,layer_<=3);

    //for the rbin - calculate the distance to the nominal layer radius
    double Delta=(irbin+0.5)*dr_-drmax;

    //calculate the phi correction - this is a somewhat approximate formula
    double dphi=(Delta/0.18)*(bend*0.009)/rmean_;

    int idphi=0;
      
    if (layer_<=3) {
      idphi=dphi/kphi;
    } else {
      idphi=dphi/kphi1;
    }

    return idphi;
    
  }
  
  int lookupPhiCorr(int ibend, int rbin) {

    int index=ibend*rbins_+rbin;
    assert(index<(int)table_.size());
    return table_[index];
    
  }

  
private:


  double rmean_;

  double rmin_;
  double rmax_;
 
  double dr_;
  
  int bendbits_;
  int rbits_;

  int bendbins_;
  int rbins_;

  int layer_;
  
  
};



#endif



