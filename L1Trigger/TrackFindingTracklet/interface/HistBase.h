#ifndef HISTBASE_H
#define HISTBASE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <assert.h>
#include <math.h>


using namespace std;

class HistBase{

public:

  HistBase() {}

  virtual ~HistBase() {}

  virtual void FillLayerResidual(int, int, double, double, 
				 double, double,bool){
    return; //default implementation does nothing
  }
  
  virtual void FillDiskResidual(int, int, double, double, 
				double, double, bool){
    return; //default implementation does nothing
  }

  //arguments are
  // int seedIndex
  // int iSector
  // double irinv, rinv
  // double iphi0, phi0
  // double ieta, eta
  // double iz0, z0
  // int tp
  virtual void fillTrackletParams(int, int, double, double,
				  double, double,
				  double, double, 
				  double, double, 
				  int ) {
    return; //default implementation does nothing
  }


  //int seedIndex
  //double etaTP
  //bool eff
  virtual void fillSeedEff(int, double, bool) {
    return; //default implementation does nothing
  }

  
private:

};



#endif



