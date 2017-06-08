#ifndef L1TSTUB_H
#define L1TSTUB_H

#include <iostream>
#include <assert.h>
using namespace std;

class L1TStub{

public:

  L1TStub() {
 
  }

  L1TStub(int simtrackid, int iphi, int iz, int layer, int ladder, int module, int strip,
	  double x, double y, double z, double sigmax, double sigmaz, double pt, double bend){
    simtrackid_=simtrackid;
    iphi_=iphi;
    iz_=iz;
    layer_=layer;
    ladder_=ladder;
    module_=module;
    strip_=strip;
    x_=x;
    y_=y;
    z_=z;
    sigmax_=sigmax;
    sigmaz_=sigmaz;
    pt_=pt;
    bend_ = bend;

    allstubindex_=999;

    if (layer_>999&&z_<0.0) {
      //cout <<"Flipping pt sign"<<endl;
      pt_=-pt_;
      bend_ = -bend_;
    }

  }

  void AddInnerDigi(int ladder, int module, int irphi,int iz){

    pair<int,int> tmplm(ladder,module);
    innerdigisladdermodule_.push_back(tmplm);

    pair<int,int> tmp(irphi,iz);
    innerdigis_.push_back(tmp);
  }

  void AddOuterDigi(int ladder, int module, int irphi,int iz){

    pair<int,int> tmplm(ladder,module);
    outerdigisladdermodule_.push_back(tmplm);

    pair<int,int> tmp(irphi,iz);
    outerdigis_.push_back(tmp);
  }

  void write(ofstream& out){
    
    out << "Stub: " 
	<< layer_ << "\t" 
	<< ladder_ << "\t" 
	<< module_ << "\t"
	<< strip_<< "\t"
	<< -1 << "\t"
	<< pt_ << "\t" 
	<< x_ << "\t" 
	<< y_ << "\t" 
	<< z_ << "\t" 
	<< bend_ << "\t" << endl; 

  }
  void write(ostream& out){
    
    out << "Stub: " 
	<< layer_ << "\t" 
	<< ladder_ << "\t" 
	<< module_ << "\t"
	<< strip_<< "\t"
	<< -1 << "\t"
	<< pt_ << "\t" 
	<< x_ << "\t" 
	<< y_ << "\t" 
	<< z_ << "\t" 
	<< bend_ << "\t" << endl; 
	
  }

  int ptsign() {
    int ptsgn=-1.0;
    if (diphi()<iphiouter()) ptsgn=-ptsgn;
    //if (layer_>999 && z_>0.0) ptsgn=-ptsgn; //sign fix for forward endcap
    return ptsgn;
  }

  double diphi() {
    if (innerdigis_.size()==0) {
      return 0.0;
    }
    double phi_tmp=0.0;
    for (unsigned int i=0;i<innerdigis_.size();i++){
      phi_tmp+=innerdigis_[i].first;
    }
    return phi_tmp/innerdigis_.size();
  }


  double iphiouter() {
    if (outerdigis_.size()==0) {
      return 0.0;
    }
    double phi_tmp=0.0;
    for (unsigned int i=0;i<outerdigis_.size();i++){
      phi_tmp+=outerdigis_[i].first;
    }
    return phi_tmp/outerdigis_.size();
  }

  double diz() {
    if (innerdigis_.size()==0) {
      return 0.0;
    }
    double z_tmp=0.0;
    for (unsigned int i=0;i<innerdigis_.size();i++){
      z_tmp+=innerdigis_[i].second;
    }
    return z_tmp/innerdigis_.size();
  }

  unsigned int layer() const { return layer_; }
  int disk() const {
    if (z_<0.0) {
      return -module_;
    }
    return module_; 
  }
  unsigned int ladder() const { return ladder_; }
  unsigned int module() const { return module_; }
  vector<pair<int,int> > innerdigis() const { return innerdigis_; }
  vector<pair<int,int> > outerdigis() const { return outerdigis_; }
  vector<pair<int,int> > innerdigisladdermodule() const { return innerdigisladdermodule_; }
  vector<pair<int,int> > outerdigisladdermodule() const { return outerdigisladdermodule_; }
  double x() const { return x_; }
  double y() const { return y_; }
  double z() const { return z_; }
  double r() const { return sqrt(x_*x_+y_*y_); }
  double pt() const { return pt_; }
  double r2() const { return x_*x_+y_*y_; }
  double bend() const { return bend_;}

  double phi() const { return atan2(y_,x_); }

  unsigned int iphi() const { return iphi_; }
  unsigned int iz() const { return iz_; }

  void setiphi(int iphi) {iphi_=iphi;}
  void setiz(int iz) {iz_=iz;}

  double sigmax() const {return sigmax_;}
  double sigmaz() const {return sigmaz_;}

  bool operator== (const L1TStub& other) const {
    if (other.iphi()==iphi_ &&
	other.iz()==iz_ &&
	other.layer()==layer_ &&
	other.ladder()==ladder_ &&
	other.module()==module_)
      return true;

    else
      return false;
  }

  void lorentzcor(double shift){
    double r=this->r();
    double phi=this->phi()-shift/r;
    this->x_=r*cos(phi);
    this->y_=r*sin(phi);
  }

  int simtrackid() const { return simtrackid_;}

  void setAllStubIndex(unsigned int index) { allstubindex_=index; }

  unsigned int allStubIndex() const { return allstubindex_; }

  unsigned int strip() const { return strip_; }

  double alpha() const {
    if (r()<57.0) return 0.0;
    if (z_>0.0) {
      return ((int)strip_-480.5)*0.009/r2();
    }
    return -((int)strip_-480.5)*0.009/r2();
  }

  double alphatruncated() const {
    if (r()<57.0) return 0.0;
    int striptruncated=strip_/1;
    striptruncated*=1;
    if (z_>0.0) {
      return (striptruncated-480.5)*0.009/r2();
    }
    return -(striptruncated-480.5)*0.009/r2();
  }

  void setXY(double x, double y){
    x_=x;
    y_=y;
  }

private:

  int simtrackid_;
  unsigned int iphi_;
  unsigned int iz_;
  unsigned int layer_;
  unsigned int ladder_;
  unsigned int module_;
  unsigned int strip_;
  double x_;
  double y_;
  double z_;
  double sigmax_;
  double sigmaz_;
  double pt_;
  double bend_;
  unsigned int allstubindex_;

  vector<pair<int,int> > innerdigis_;
  vector<pair<int,int> > innerdigisladdermodule_;
  vector<pair<int,int> > outerdigis_;
  vector<pair<int,int> > outerdigisladdermodule_;


};




#endif



