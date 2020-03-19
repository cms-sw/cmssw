#ifndef TETABLEINNERDISK_H
#define TETABLEINNERDISK_H

#include "TETableBase.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class TETableInnerDisk:public TETableBase{

public:

  TETableInnerDisk() {
    nbits_ = 19;
  }

  ~TETableInnerDisk() {

  }

  void init(int disk1,
	    int disk2,
	    int rbits,
	    int zbits
	    ) {
    init(disk1, disk2, -1, rbits, zbits);
  }

  void init(int disk1,
	    int disk2,
	    int layer3,
	    int rbits,
	    int zbits
	    ) {

    disk1_=disk1;
    disk2_=disk2;
    layer3_=layer3;
    rbits_=rbits;
    zbits_=zbits;

    rbins_=(1<<rbits);
    rmind1_=0.0;
    rmaxd1_=rmaxdisk;
    dr_=rmaxdisk/rbins_;

    zbins_=(1<<zbits);
    zmind1_=zmean[disk1-1]-dzmax;
    zmaxd1_=zmean[disk1-1]+dzmax;
    dz_=2*dzmax/zbins_;

    zmeand2_=zmean[disk2-1];
    rmeanl3_=0.;
    if (layer3 > 0)
      rmeanl3_=rmean[layer3-1];

    for (int irbin=0;irbin<rbins_;irbin++) {
      for (int izbin=0;izbin<zbins_;izbin++) {
	//int ibin=irbin+izbin*rbins_;
	int value=getLookupValue(irbin,izbin);
	//cout << "table "<<table_.size()<<" "<<value<<" "<<rmeanl2_<<endl;
	table_.push_back(value);
      }
    }
    if (writeVMTables) {
      writeVMTable("VMTableInnerD"+std::to_string(disk1_)+"D"+std::to_string(disk2_)+".txt");
    }
  }

  // negative return means that seed can not be formed
  int getLookupValue(int irbin, int izbin){

    double r1=rmind1_+irbin*dr_;
    double r2=rmind1_+(irbin+1)*dr_;

    double z1=zmind1_+izbin*dz_;
    double z2=zmind1_+(izbin+1)*dz_;


    double rmaxd2=-2*rmaxdisk;
    double rmind2=2*rmaxdisk;

    findr(r1,z1,rmind2,rmaxd2);
    findr(r1,z2,rmind2,rmaxd2);
    findr(r2,z1,rmind2,rmaxd2);
    findr(r2,z2,rmind2,rmaxd2);

    assert(rmind2<rmaxd2);

    if (rmind2>rmaxdiskvm) return -1;
    if (rmaxd2<rmindiskvm) return -1;

    double zmaxl3=-2*zlength;
    double zminl3=2*zlength;

    findz(z1,r1,zminl3,zmaxl3);
    findz(z1,r2,zminl3,zmaxl3);
    findz(z2,r1,zminl3,zmaxl3);
    findz(z2,r2,zminl3,zmaxl3);

    assert(zminl3<zmaxl3);

    if (zminl3>zlength && layer3_>0) return -1;
    if (zmaxl3<-zlength && layer3_>0) return -1;

    int NBINS=NLONGVMBINS*NLONGVMBINS/2; //divide by two for + and - z
    
    // first pack rbinmin and deltar for second disk

    // This is a 9 bit word:
    // xxx|yy|z|rrr
    // xxx is the delta r window
    // yy is the r bin
    // z is flag to look in next bin
    // rrr fine r bin
    // NOTE : this encoding is not efficient z is one if xxx+rrr is greater than 8
    //        and xxx is only 1,2, or 3
    //        should also reject xxx=0 as this means projection is outside range
    
    int rbinmin=NBINS*(rmind2-rmindiskvm)/(rmaxdiskvm-rmindiskvm);
    int rbinmax=NBINS*(rmaxd2-rmindiskvm)/(rmaxdiskvm-rmindiskvm);

    //cout << "zbinmin zminl2 "<<zbinmin<<" "<<zminl2<<endl;
    //cout << "zbinmax zmaxl2 "<<zbinmax<<" "<<zmaxl2<<endl;
    
    if (rbinmin<0) rbinmin=0;
    if (rbinmax>=NBINS) rbinmax=NBINS-1;

    //cout <<"rbminmin rbinmax "<<rbinmin<<" "<<rbinmax<<endl;
    
    assert(rbinmin<=rbinmax);
    assert(rbinmax-rbinmin<=(int)NLONGVMBINS);

    int valueD2=rbinmin/8;
    valueD2*=2;
    if (rbinmax/8-rbinmin/8>0) valueD2+=1;
    valueD2*=8;
    valueD2+=(rbinmin&7);
    //cout << "zbinmax/8 zbinmin/8 valueD2 "<<zbinmax/8<<" "<<zbinmin/8<<" "<<valueD2<<endl;
    assert(valueD2/8<7);
    int deltar=rbinmax-rbinmin;
    assert(deltar<8);
    valueD2+=(deltar<<6);
    assert(valueD2<(1<<9));

    // then pack zbinmin and deltaz for third layer

    NBINS=NLONGVMBINS*NLONGVMBINS;

    int zbinmin=NBINS*(zminl3+zlength)/(2*zlength);
    int zbinmax=NBINS*(zmaxl3+zlength)/(2*zlength);

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
    int deltaz=zbinmax-zbinmin;
    if (deltaz>7) {
      //cout << "deltaz = "<<deltaz<<endl;
      deltaz=7;
    }
    assert(deltaz<8);
    valueL3+=(deltaz<<7);

    // mask out the values for third layer if this is not a table for a triplet seed
    if (layer3_<=0) valueL3=0;

    // finally pack values for second and third layer together

    int value = (valueL3<<9) + valueD2;

    return value;
    
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

    return (zmeand2_-zcut)*r/(z-zcut);
    
  }

  void findz(double z, double r, double& zminl3, double& zmaxl3){

    double zl3=zintercept(z0cut,z,r);

    if (zl3<zminl3) zminl3=zl3;
    if (zl3>zmaxl3) zmaxl3=zl3;
    
    zl3=zintercept(-z0cut,z,r);

    if (zl3<zminl3) zminl3=zl3;
    if (zl3>zmaxl3) zmaxl3=zl3;

  }

  double zintercept(double zcut, double z, double r) {

    return zcut+(z-zcut)*rmeanl3_/r;

  }

  int lookup(int rbin, int zbin) {

    int index=rbin*zbins_+zbin;
    assert(index<(int)table_.size());
    return table_[index];
    
  }
    
  /*
  
  void writephi(std::string fname) {

    ofstream out(fname.c_str());

    //cout << "writephi 2 phitableentries_ : "<<phitableentries_<<endl;

    for (int i=0;i<phitableentries_;i++){
      FPGAWord entry;
      //cout << "phitablebits_ : "<<phitablebits_<<endl;
      entry.set(i,phitablebits_);
      //out << entry.str()<<" "<<tablephi_[i]<<endl;
      out <<tablephi_[i]<<endl;
    }
    out.close();
  
  }

  */


private:

  int disk1_;
  int disk2_;
  int layer3_;
  int zbits_;
  int rbits_;
  
  int rbins_;
  double rmind1_;
  double rmaxd1_;
  double dr_;

  int zbins_;
  double zmind1_;
  double zmaxd1_;
  double dz_;
  
  double zmeand2_;
  double rmeanl3_;
  

  
};



#endif



