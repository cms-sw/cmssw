#ifndef TETABLEOUTER_H
#define TETABLEOUTER_H

#include "TETableBase.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class TETableOuter: public TETableBase{

public:

  TETableOuter() {
    nbits_ = 6;
  }

  ~TETableOuter() {

  }


  void init(int layer,
	    int zbits,
	    int rbits
	    ) {

    layer_=layer;
    zbits_=zbits;
    rbits_=rbits;

    rbins_=(1<<rbits);
    rmin_=rmean[layer-1]-drmax;
    rmax_=rmean[layer-1]+drmax;
    dr_=2*drmax/rbins_;

    zbins_=(1<<zbits);
    zmin_=-zlength;
    zmax_=zlength;
    dz_=2*zlength/zbins_;

    rmean_=rmean[layer-1];

    for (int izbin=0;izbin<zbins_;izbin++) {
      for (int irbin=0;irbin<rbins_;irbin++) {
	//int ibin=irbin+izbin*rbins_;
	int value=getLookupValue(izbin,irbin);
	table_.push_back(value);
      }
    }


    if (writeVMTables) {
      writeVMTable("VMTableOuterL"+std::to_string(layer_)+".tab");
    }

    
  }

  // negative return means that seed can not be formed
  int getLookupValue(int izbin, int irbin){

    double z=zmin_+(izbin+0.5)*dz_;
    double r=rmin_+(irbin+0.5)*dr_;

    double zproj=z*rmean_/r;

    int NBINS=NLONGVMBINS*NLONGVMBINS;
    
    int zbin=NBINS*(zproj+zlength)/(2*zlength);

    if (zbin<0) zbin=0;
    if (zbin>=NBINS) zbin=NBINS-1;

    //cout << "izbin zbin z zproj "<<izbin<<" "<<zbin<<" "<<z<<" "<<zproj<<endl;
    
    return zbin;
    
  }

  int lookup(int zbin, int rbin) {

    int index=zbin*rbins_+rbin;
    return table_[index];
    
  }

  
private:


  double rmean_;

  double rmin_;
  double rmax_;

  double zmin_;
  double zmax_;

  double dr_;
  double dz_;
  
  int zbits_;
  int rbits_;

  int zbins_;
  int rbins_;

  int layer_;
  
  
};



#endif



