#ifndef TETABLEINNER_H
#define TETABLEINNER_H

#include "TETableBase.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class TETableInner:public TETableBase{

public:

  TETableInner() {
    nbits_ = 20;
  }

  ~TETableInner() {

  }

  void init(int layer1,
	    int layer2,
	    int zbits,
	    int rbits
	    ) {
    init(layer1, layer2, -1, zbits, rbits);
  }

  void init(int layer1,
	    int layer2,
	    int layer3,
	    int zbits,
	    int rbits,
            bool thirdLayerIsDisk = false
	    ) {

    thirdLayerIsDisk_ = thirdLayerIsDisk;

    layer1_=layer1;
    layer2_=layer2;
    layer3_=layer3;
    zbits_=zbits;
    rbits_=rbits;

    bool extended=layer1==2&&layer2==3&&layer3_==1&&thirdLayerIsDisk_;
    bool extra=layer1==2&&layer2==3&&!extended;
    
    rbins_=(1<<rbits);
    rminl1_=rmean[layer1-1]-drmax;
    rmaxl1_=rmean[layer1-1]+drmax;
    dr_=2*drmax/rbins_;

    zbins_=(1<<zbits);
    zminl1_=-zlength;
    zminl2_=zlength;
    dz_=2*zlength/zbins_;

    if (layer1==1){
      rmindisk_=rmindiskvm;
      rmaxdisk_=rmaxdiskl1overlapvm;
    }

    if (layer1==2){
      rmindisk_=rmindiskl2overlapvm;
      rmaxdisk_=(extended?rmaxdisk:rmaxdiskvm);
    }

    rmeanl2_=rmean[layer2-1];
    if (layer3_>0) {
      rmeanl3_=rmean[layer3-1];
      zmeand3_=zmean[layer3-1];
    }
    else {
      rmeanl3_=0.;
      zmeand3_=0.;
    }

    for (int izbin=0;izbin<zbins_;izbin++) {
      for (int irbin=0;irbin<rbins_;irbin++) {
	//int ibin=irbin+izbin*rbins_;
	int value=getLookupValue(izbin,irbin,extra);
	table_.push_back(value);
      }
    }

    if (writeVMTables) {
      writeVMTable("VMTableInnerL"+std::to_string(layer1_)+"L"+std::to_string(layer2_)+".tab");
    }
    
  }

  // negative return means that seed can not be formed
  int getLookupValue(int izbin, int irbin, bool extra){

    double z1=zminl1_+izbin*dz_;
    double z2=zminl1_+(izbin+1)*dz_;

    double r1=rminl1_+irbin*dr_;
    double r2=rminl1_+(irbin+1)*dr_;

    if (extra and fabs(0.5*(z1+z2))<52.0) {   //This seeding combinations should not be for central region of detector
      return -1; 
    }
    
    double zmaxl2=-2*zlength;
    double zminl2=2*zlength;

    findzL2(z1,r1,zminl2,zmaxl2);
    findzL2(z1,r2,zminl2,zmaxl2);
    findzL2(z2,r1,zminl2,zmaxl2);
    findzL2(z2,r2,zminl2,zmaxl2);

    assert(zminl2<zmaxl2);

    if (zminl2>zlength) return -1;
    if (zmaxl2<-zlength) return -1;

    double zmaxl3=-2*zlength;
    double zminl3=2*zlength;

    findzL3(z1,r1,zminl3,zmaxl3);
    findzL3(z1,r2,zminl3,zmaxl3);
    findzL3(z2,r1,zminl3,zmaxl3);
    findzL3(z2,r2,zminl3,zmaxl3);

    assert(zminl3<zmaxl3);

    if (zminl3>zlength && layer3_>0 && !thirdLayerIsDisk_) return -1;
    if (zmaxl3<-zlength && layer3_>0 && !thirdLayerIsDisk_) return -1;


    int NBINS=NLONGVMBINS*NLONGVMBINS;
    
    // first pack zbinmin and deltaz for second layer

    int zbinmin=NBINS*(zminl2+zlength)/(2*zlength);
    int zbinmax=NBINS*(zmaxl2+zlength)/(2*zlength);

    //cout << "zbinmin zminl2 "<<zbinmin<<" "<<zminl2<<endl;
    //cout << "zbinmax zmaxl2 "<<zbinmax<<" "<<zmaxl2<<endl;
    
    if (zbinmin<0) zbinmin=0;
    if (zbinmax>=NBINS) zbinmax=NBINS-1;

    assert(zbinmin<=zbinmax);
    assert(zbinmax-zbinmin<=(int)NLONGVMBINS);

    int valueL2=zbinmin/8;
    valueL2*=2;
    if (zbinmax/8-zbinmin/8>0) valueL2+=1;
    valueL2*=8;
    valueL2+=(zbinmin&7);
    //cout << "zbinmax/8 zbinmin/8 valueL2 "<<zbinmax/8<<" "<<zbinmin/8<<" "<<valueL2<<endl;
    assert(valueL2/8<15);
    int deltaz=zbinmax-zbinmin;
    if (deltaz>7) {
      //cout << "deltaz = "<<deltaz<<endl;
      deltaz=7;
    }
    assert(deltaz<8);
    valueL2+=(deltaz<<7);

    // then pack zbinmin and deltaz for third layer

    zbinmin=NBINS*(zminl3+zlength)/(2*zlength);
    zbinmax=NBINS*(zmaxl3+zlength)/(2*zlength);

    //cout << "zbinmin zminl2 "<<zbinmin<<" "<<zminl2<<endl;
    //cout << "zbinmax zmaxl2 "<<zbinmax<<" "<<zmaxl2<<endl;
    
    if (zbinmin<0) zbinmin=0;
    if (zbinmax>=NBINS) zbinmax=NBINS-1;

    assert(zbinmin<=zbinmax);
    assert(zbinmax-zbinmin<=(int)NLONGVMBINS);

    int valueL3=zbinmin/8;
    valueL3*=2;
    if (zbinmax/8-zbinmin/8>0) valueL3+=1;
    valueL3*=8;
    valueL3+=(zbinmin&7);
    //cout << "zbinmax/8 zbinmin/8 valueL3 "<<zbinmax/8<<" "<<zbinmin/8<<" "<<valueL3<<endl;
    assert(valueL3/8<15);
    deltaz=zbinmax-zbinmin;
    if (deltaz>7) {
      //cout << "deltaz = "<<deltaz<<endl;
      deltaz=7;
    }
    assert(deltaz<8);
    valueL3+=(deltaz<<7);


    int valueD3 = 0;
    if (layer3_>0 && thirdLayerIsDisk_) {

      if (fabs(z1)<=z0cut) return -1;
      if (fabs(z2)<=z0cut) return -1;

      double rmaxd3=-2*rmaxdisk;
      double rmind3=2*rmaxdisk;

      findr(r1,z1,rmind3,rmaxd3);
      findr(r1,z2,rmind3,rmaxd3);
      findr(r2,z1,rmind3,rmaxd3);
      findr(r2,z2,rmind3,rmaxd3);

      assert(rmind3<rmaxd3);

      if (rmind3>rmaxdisk_) return -1;
      if (rmind3<rmindisk_) rmind3=rmindisk_;
      if (rmaxd3>rmaxdisk_) rmaxd3=rmaxdisk_;
      if (rmaxd3<rmindisk_) return -1;

      int NBINS=NLONGVMBINS*NLONGVMBINS/2; //divide by two for + and - z
      
      int rbinmin=NBINS*(rmind3-rmindiskvm)/(rmaxdisk-rmindiskvm);
      int rbinmax=NBINS*(rmaxd3-rmindiskvm)/(rmaxdisk-rmindiskvm);

      if (rmind3 < rmaxdiskvm)
        rbinmin = 0;
      if (rmaxd3 < rmaxdiskvm)
        rbinmax = 0;

      //cout << "zbinmin zminl2 "<<zbinmin<<" "<<zminl2<<endl;
      //cout << "zbinmax zmaxl2 "<<zbinmax<<" "<<zmaxl2<<endl;
      
      if (rbinmin<0) rbinmin=0;
      if (rbinmax>=NBINS) rbinmax=NBINS-1;

      assert(rbinmin<=rbinmax);
      //assert(rbinmax-rbinmin<=(int)NLONGVMBINS);

      valueD3=rbinmin/8;
      if (z1<0) valueD3+=4;
      valueD3*=2;
      if (rbinmax/8-rbinmin/8>0) valueD3+=1;
      valueD3*=8;
      valueD3+=(rbinmin&7);
      //cout << "zbinmax/8 zbinmin/8 valueD3 "<<zbinmax/8<<" "<<zbinmin/8<<" "<<valueD3<<endl;
      assert(valueD3/8<15);
      int deltar=rbinmax-rbinmin;
      if (deltar>7) {
        //cout << "deltar = "<<deltar<<endl;
        deltar=7;
      }
      assert(deltar<8);
      valueD3+=(deltar<<7);
      assert(valueD3<(1<<10));
    }

    // mask out the values for third layer if this is not a table for a triplet seed
    if (layer3_<=0) {
      valueL3=0;
      valueD3=0;
    }

    // finally pack values for second and third layer together

    int value = (valueL3<<10) + valueL2;
    if (thirdLayerIsDisk_)
      value = (valueD3<<10) + valueL2;

    return value;
    
  }


  void findzL2(double z, double r, double& zminl2, double& zmaxl2){

    double zl2=zinterceptL2(z0cut,z,r);

    if (zl2<zminl2) zminl2=zl2;
    if (zl2>zmaxl2) zmaxl2=zl2;
    
    zl2=zinterceptL2(-z0cut,z,r);

    if (zl2<zminl2) zminl2=zl2;
    if (zl2>zmaxl2) zmaxl2=zl2;

  }

  double zinterceptL2(double zcut, double z, double r) {

    return zcut+(z-zcut)*rmeanl2_/r;

  }

  void findzL3(double z, double r, double& zminl3, double& zmaxl3){

    double zl3=zinterceptL3(z0cut,z,r);

    if (zl3<zminl3) zminl3=zl3;
    if (zl3>zmaxl3) zmaxl3=zl3;
    
    zl3=zinterceptL3(-z0cut,z,r);

    if (zl3<zminl3) zminl3=zl3;
    if (zl3>zmaxl3) zmaxl3=zl3;

  }

  double zinterceptL3(double zcut, double z, double r) {

    return zcut+(z-zcut)*rmeanl3_/r;

  }

  void findr(double r, double z, double& rmind2, double& rmaxd2){

    double rd2=rintercept(z0cut,r,z);

    //cout << "rd2 : "<<r<<" "<<z<<" "<<rd2<<endl;
    
    if (rd2<rmind2) rmind2=rd2;
    if (rd2>rmaxd2) rmaxd2=rd2;
    
    rd2=rintercept(-z0cut,r,z);

    //cout << "rd2 : "<<rd2<<endl;

    if (rd2<rmind2) rmind2=rd2;
    if (rd2>rmaxd2) rmaxd2=rd2;

  }

  double rintercept(double zcut, double r, double z) {

    //cout << "zcut z "<<zcut<<" "<<z<<endl;

    double zmean=(z>0.0)?zmeand3_:-zmeand3_;
    
    return (zmean-zcut)*r/(z-zcut);
    
  }

  int lookup(int zbin, int rbin) {

    int index=zbin*rbins_+rbin;

    return table_[index];
    
  }


private:

  bool thirdLayerIsDisk_;

  int layer1_;
  int layer2_;
  int layer3_;
  int zbits_;
  int rbits_;
  
  int rbins_;
  double rminl1_;
  double rmaxl1_;
  double dr_;

  int zbins_;
  double zminl1_;
  double zminl2_;
  double dz_;
  
  double rmeanl2_;
  double rmeanl3_;
  double zmeand3_;
  
  double rmaxdisk_;
  double rmindisk_;
  
};



#endif



