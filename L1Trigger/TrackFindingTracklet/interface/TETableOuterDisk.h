#ifndef TETABLEOUTERDISK_H
#define TETABLEOUTERDISK_H

#include "TETableBase.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class TETableOuterDisk:public TETableBase{

public:

  TETableOuterDisk() {
    nbits_ = 5;
  }

  ~TETableOuterDisk() {

  }


  void init(int disk,
	    int rbits,
	    int zbits
	    ) {

    disk_=disk;
    rbits_=rbits;
    zbits_=zbits;

    rbins_=(1<<rbits);
    rmin_=0;
    rmax_=rmaxdisk;
    dr_=rmaxdisk/rbins_;

    zbins_=(1<<zbits);
    zmin_=zmean[disk-1]-dzmax;
    zmax_=zmean[disk-1]+dzmax;
    dz_=2*dzmax/zbins_;

    zmean_=zmean[disk-1];

    for (int irbin=0;irbin<rbins_;irbin++) {
      for (int izbin=0;izbin<zbins_;izbin++) {
	int value=getLookupValue(irbin,izbin);
	table_.push_back(value);
      }
    }
    if (writeVMTables) {
      writeVMTable("VMTableOuterD"+std::to_string(disk_)+".txt");
    }
  }

  // negative return means that seed can not be formed
  int getLookupValue(int irbin, int izbin){

    double r=rmin_+(irbin+0.5)*dr_;
    double z=zmin_+(izbin+0.5)*dz_;

    double rproj=r*zmean_/z;

    int NBINS=NLONGVMBINS*NLONGVMBINS/2;  //divide by two for + vs - z disks
    
    int rbin=NBINS*(rproj-rmindiskvm)/(rmaxdiskvm-rmindiskvm);

    if (rbin<0) rbin=0;
    if (rbin>=NBINS) rbin=NBINS-1;

    return rbin;
    
  }


  int lookup(int rbin, int zbin) {

    int index=rbin*zbins_+zbin;
    return table_[index];
    
  }
    
private:

  double zmean_;

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

  int disk_;
  
  
};



#endif



