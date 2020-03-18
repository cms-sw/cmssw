#ifndef TETABLEINNEROVERLAP_H
#define TETABLEINNEROVERLAP_H

#include "TETableBase.h"
#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>


using namespace std;

class TETableInnerOverlap:public TETableBase{

public:

  TETableInnerOverlap() {
    nbits_ = 10;
  }

  ~TETableInnerOverlap() {

  }


  void init(int layer1,
	    int disk2,
	    int zbits,
	    int rbits
	    ) {

    layer1_=layer1;
    disk2_=disk2;
    zbits_=zbits;
    rbits_=rbits;

    rbins_=(1<<rbits);
    rminl1_=rmean[layer1-1]-drmax;
    rmaxl1_=rmean[layer1-1]-drmax;
    dr_=2*drmax/rbins_;

    zbins_=(1<<zbits);
    zminl1_=-zlength;
    zmaxl1_=zlength;
    dz_=2*zlength/zbins_;

    assert(layer1==1||layer1==2);
    
    if (layer1==1){
      rmindisk_=rmindiskvm;
      rmaxdisk_=rmaxdiskl1overlapvm;
    }

    if (layer1==2){
      rmindisk_=rmindiskl2overlapvm;
      rmaxdisk_=rmaxdiskvm;
    }

      
    zmeand2_=zmean[disk2-1];

    for (int izbin=0;izbin<zbins_;izbin++) {
      for (int irbin=0;irbin<rbins_;irbin++) {
	//int ibin=irbin+izbin*rbins_;
	int value=getLookupValue(izbin,irbin);
	//cout << "table "<<table_.size()<<" "<<value<<" "<<rmeanl2_<<endl;
	table_.push_back(value);
      }
    }

    if (writeVMTables) {
      writeVMTable("VMTableInnerL"+std::to_string(layer1_)+"D"+std::to_string(disk2_)+".txt");
    }
    
  }

  // negative return means that seed can not be formed
  int getLookupValue(int izbin, int irbin){

    bool print=false;
    //print=(izbin==127)&&(irbin==2);
    
    double r1=rminl1_+irbin*dr_;
    double r2=rminl1_+(irbin+1)*dr_;

    double z1=zminl1_+izbin*dz_;
    double z2=zminl1_+(izbin+1)*dz_;

    if (fabs(z1)<=z0cut) return -1;
    if (fabs(z2)<=z0cut) return -1;

    double rmaxd2=-2*rmaxdisk;
    double rmind2=2*rmaxdisk;

    findr(r1,z1,rmind2,rmaxd2);
    findr(r1,z2,rmind2,rmaxd2);
    findr(r2,z1,rmind2,rmaxd2);
    findr(r2,z2,rmind2,rmaxd2);

    if (print) cout << "PRINT layer1 rmind2 rmaxd2 z2 r1 "<<layer1_<<" "
		    <<rmind2<<" "<<rmaxd2<<" "<<z2<<" "<<r1<<endl;
    
    assert(rmind2<rmaxd2);

    if (rmind2>rmaxdisk_) return -1;
    if (rmind2<rmindisk_) rmind2=rmindisk_;
    if (rmaxd2>rmaxdisk_) rmaxd2=rmaxdisk_;
    if (rmaxd2<rmindisk_) return -1;

    int NBINS=NLONGVMBINS*NLONGVMBINS/2; //divide by two for + and - z
    
    int rbinmin=NBINS*(rmind2-rmindiskvm)/(rmaxdiskvm-rmindiskvm);
    int rbinmax=NBINS*(rmaxd2-rmindiskvm)/(rmaxdiskvm-rmindiskvm);

    //cout << "zbinmin zminl2 "<<zbinmin<<" "<<zminl2<<endl;
    //cout << "zbinmax zmaxl2 "<<zbinmax<<" "<<zmaxl2<<endl;
    
    if (rbinmin<0) rbinmin=0;
    if (rbinmax>=NBINS) rbinmax=NBINS-1;

    if (print) cout << "PRINT layer1 rmind2 rmaxd2 dr z2 r1 "<<layer1_<<" "
		    <<rmind2<<" "<<rmaxd2<<" "<<" "<<(rmaxdiskvm-rmindiskvm)/NBINS<<z2<<" "<<r1<<endl;

    if (print) cout <<"PRINT rbminmin rbinmax "<<rbinmin<<" "<<rbinmax<<endl;
    
    assert(rbinmin<=rbinmax);
    //assert(rbinmax-rbinmin<=(int)NLONGVMBINS);

    int value=rbinmin/8;
    if (z1<0) value+=4;
    value*=2;
    if (rbinmax/8-rbinmin/8>0) value+=1;
    value*=8;
    value+=(rbinmin&7);
    //cout << "zbinmax/8 zbinmin/8 value "<<zbinmax/8<<" "<<zbinmin/8<<" "<<value<<endl;
    assert(value/8<15);
    int deltar=rbinmax-rbinmin;
    if (deltar>7) {
      //cout << "deltar = "<<deltar<<endl;
      deltar=7;
    }
    assert(deltar<8);
    value+=(deltar<<7);
    assert(value<(1<<10));
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

    //cout << "zcut z "<<zcut<<" "<<z<<endl;

    double zmean=(z>0.0)?zmeand2_:-zmeand2_;
    
    return (zmean-zcut)*r/(z-zcut);
    
  }

  int lookup(int zbin, int rbin) {

    int index=zbin*rbins_+rbin;
    //cout << "index zbin rbin value "<<index<<" "<<zbin<<" "<<rbin<<" "<<table_[index]<<endl;
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

  int layer1_;
  int disk2_;
  int zbits_;
  int rbits_;
  
  int rbins_;
  double rminl1_;
  double rmaxl1_;
  double dr_;

  int zbins_;
  double zminl1_;
  double zmaxl1_;
  double dz_;
  
  double zmeand2_;

  double rmaxdisk_;
  double rmindisk_;

  
};



#endif



