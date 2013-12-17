#ifndef SLHCEVENT_H
#define SLHCEVENT_H

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>
#include <map>
#include <ext/hash_set>
#include <math.h>
#include <assert.h>

#define NSECTORS 28

using namespace std;

//This is the number of strips in rphi and in z for a module.
//This should be in the header of the ASCII file, but for now
//just hardcoded here.

static int nrphi=1000;
static int nz=80;

static double two_pi=8*atan(1.0);

static double x_offset=0.199196*0.0;
static double y_offset=0.299922*0.0;



class ModuleGeometry {

public:

  ModuleGeometry() {

    x0_=0.0;
    y0_=0.0;
    z0_=0.0;

    x1_=0.0;
    y1_=0.0;
    z1_=0.0;

    x2_=0.0;
    y2_=0.0;
    z2_=0.0;

  }


  ModuleGeometry(double x0, double y0, double z0,
		 double x1, double y1, double z1,
		 double x2, double y2, double z2) {
    
    x0_=x0;
    y0_=y0;
    z0_=z0;

    x1_=x1;
    y1_=y1;
    z1_=z1;

    x2_=x2;
    y2_=y2;
    z2_=z2;

  }

  double phi() const {
    return atan2(y0_+y1_,x0_+x1_);
  }

  double phi1() const {
    return atan2(y0_,x0_);
  }

  double phi2() const {
    return atan2(y1_,x1_);
  }

  double r1() const {
    return sqrt(y0_*y0_+x0_*x0_);
  }

  double r2() const {
    return sqrt(y1_*y1_+x1_*x1_);
  }



private:

  double x0_;
  double y0_;
  double z0_;
  double x1_;
  double y1_;
  double z1_;
  double x2_;
  double y2_;
  double z2_;

};

class GeometryMap{

public:


  GeometryMap() {

    //empty contructor to be used with the filler function.

  }

  void addModule(int innerlayer, int innerladder, int innermodule,
		 int outerlayer, int outerladder, int,
		 double x0, double y0, double z0,
		 double x1, double y1, double z1,
		 double x2, double y2, double z2) {

    x0-=x_offset;
    y0-=y_offset;
    
    x1-=x_offset;
    y1-=y_offset;
    
    x2-=x_offset;
    y2-=y_offset;

    ModuleGeometry mg(x0,y0,z0,x1,y1,z1,x2,y2,z2);

    outerladder_[innerlayer][innerladder]=outerladder;
    innerladder_[outerlayer][outerladder]=innerladder;
    
    modulegeometry_[innerlayer][innerladder][innermodule]=mg;

  }


  GeometryMap(ifstream& in) {

    do {

      string tmp;

      in >> tmp;

      if (tmp=="EndMap") return;

      if (tmp!="Map:") {
	cout << "Expected to read 'Map:' or 'EndMap' but found:"<<tmp<<endl;
	abort();
      }

      int innerlayer, innerladder, innermodule;
      int outerlayer, outerladder, outermodule;

      double x0,y0,z0;
      double x1,y1,z1;
      double x2,y2,z2;

      in >> innerlayer >> innerladder >>  innermodule 
	 >> outerlayer >> outerladder >> outermodule
	 >> x0 >> y0 >> z0
	 >> x1 >> y1 >> z1
	 >> x2 >> y2 >> z2;

      innerlayer--;   //HACK 6_1
      outerlayer--;   //HACK 6_1
      //innerladder--;   //HACK 6_1
      //outerladder--;   //HACK 6_1


      //innermodule--; //Hack!!!
      //outermodule--; 

      x0-=x_offset;
      y0-=y_offset;

      x1-=x_offset;
      y1-=y_offset;

      x2-=x_offset;
      y2-=y_offset;


      ModuleGeometry mg(x0,y0,z0,x1,y1,z1,x2,y2,z2);


      //if (innermodule==1) {
      //	cout << "innerlayer="<<innerlayer<<" innerladder="<<innerladder<<endl;
      //}
      outerladder_[innerlayer][innerladder]=outerladder;
      innerladder_[outerlayer][outerladder]=innerladder;

      modulegeometry_[innerlayer][innerladder][innermodule]=mg;


    } while (true);

  }

  vector<int> ladders(int innerlayer){
    
    vector<int> tmp;

    map<int, map< int, int> >::const_iterator it1=outerladder_.find(innerlayer);
    assert(it1!=outerladder_.end());

    map< int, int>::const_iterator it2=it1->second.begin(); 

    while (it2!=it1->second.end()) {
      
      tmp.push_back(it2->first);

      ++it2;

    }

    return tmp;


  }

  int outerladder(int innerlayer, int innerladder) {
    map<int, map< int, int> >::const_iterator it1=outerladder_.find(innerlayer);
    assert(it1!=outerladder_.end());
    map< int, int>::const_iterator it2=it1->second.find(innerladder); 
    assert(it2!=it1->second.end());   
    return it2->second;
  }

  int innerladder(int outerlayer, int outerladder) {
    cout << "Called innerladder("<<outerlayer<<","<<outerladder<<")"<<endl;
    map<int, map< int, int> >::const_iterator it1=innerladder_.find(outerlayer);
    assert(it1!=innerladder_.end());
    map< int, int>::const_iterator it2=it1->second.find(outerladder); 
    assert(it2!=it1->second.end());   
    return it2->second;
  }

  bool inner(int layer, int ladder){

    map<int, map< int, int> >::const_iterator ita1=outerladder_.find(layer);
    assert(ita1!=outerladder_.end());
    map< int, int>::const_iterator ita2=ita1->second.find(ladder); 
    bool foundA=(ita2!=ita1->second.end());   

    

    map<int, map< int, int> >::const_iterator itb1=innerladder_.find(layer);
    assert(itb1!=innerladder_.end());
    map< int, int>::const_iterator itb2=itb1->second.find(ladder); 
    bool foundB=(itb2!=itb1->second.end());   
    
    assert(foundA||foundB);
    assert(!(foundA&&foundB));

    //if (foundA) cout << "layer="<<layer<<" ladder="<<ladder
    //		     <<" is inner module"<<endl;

    //if (foundB) cout << "layer="<<layer<<" ladder="<<ladder
    //		     <<" is outer module"<<endl;

    return foundA;

  }

  const ModuleGeometry& moduleGeometry(int layer,int ladder,int module) const {

    map<int, map< int, map<int, ModuleGeometry> > >::const_iterator ita1=modulegeometry_.find(layer);
    assert(ita1!=modulegeometry_.end());


    map< int, map<int, ModuleGeometry> >::const_iterator ita2=ita1->second.find(ladder);
    assert(ita2!=ita1->second.end());

    map<int, ModuleGeometry>::const_iterator ita3=ita2->second.find(module);
    assert(ita3!=ita2->second.end());
    
    return ita3->second;
    

  }

  bool moduleGeometryExists(int layer,int ladder,int module) const {

    map<int, map< int, map<int, ModuleGeometry> > >::const_iterator ita1=modulegeometry_.find(layer);
    assert(ita1!=modulegeometry_.end());


    map< int, map<int, ModuleGeometry> >::const_iterator ita2=ita1->second.find(ladder);
    assert(ita2!=ita1->second.end());

    map<int, ModuleGeometry>::const_iterator ita3=ita2->second.find(module);
    return ita3!=ita2->second.end();
    
   
  }

  bool print(int layer){

    map<int, map< int, map<int, ModuleGeometry> > >::const_iterator ita1=modulegeometry_.find(layer);
    assert(ita1!=modulegeometry_.end());
    
    map< int, map<int, ModuleGeometry> >::const_iterator ita2=ita1->second.begin(); 
    map< int, map<int, ModuleGeometry> >::const_iterator ita2end=ita1->second.end(); 
    while(ita2!=ita2end){
      cout << "ladder, phi:"<<ita2->first<<" "<<ita2->second.begin()->second.phi()<<endl;
      ++ita2;
    } 

  }


  //returns the outer ladder given inner ladder
  //  layer     ladder    
  map<int, map< int, int> > outerladder_;

  //returns the inner ladder given outer ladder
  //  layer     ladder    
  map<int, map< int, int> > innerladder_;

  //returns the geometry given inner ladder
  //  layer     ladder  module   
  map<int, map<int, map<int,ModuleGeometry> > > modulegeometry_;


};

class L1SimTrack{

public:

  L1SimTrack() {
   id_=-1; 
  }

  L1SimTrack(int id, int type, double pt, double eta, double phi, 
           double vx, double vy, double vz) {
    id_=id;
    type_=type;
    pt_=pt;
    eta_=eta;
    phi_=phi;
    vx_=vx;
    vy_=vy;
    vz_=vz;
  }

  void write(ofstream& out){
    
    out << "L1SimTrack: " 
	<< id_ << "\t" 
	<< type_ << "\t" 
	<< pt_ << "\t" 
	<< eta_ << "\t" 
	<< phi_ << "\t" 
	<< vx_ << "\t" 
	<< vy_ << "\t" 
	<< vz_ << "\t" << endl; 
	
  }
  
  int id() const { return id_; }
  int type() const { return type_; }
  double pt() { return pt_; }
  double eta() { return eta_; }
  double phi() { return phi_; }
  double vx() { return vx_; }
  double vy() { return vy_; }
  double vz() { return vz_; }

private:

  int id_;
  int type_;
  double pt_;
  double eta_;
  double phi_;
  double vx_;
  double vy_;
  double vz_;

};


class Digi{

public:


  Digi(int layer,int irphi, int iz, int sensorlayer,
       int ladder, int module, double x, double y, double z) {
    layer_=layer;
    irphi_=irphi;
    iz_=iz;
    sensorlayer_=sensorlayer;
    ladder_=ladder;
    module_=module;
    x_=x;
    y_=y;
    z_=z;
  }

  void AddSimtrack(int simtrackid){
    simtrackids_.push_back(simtrackid);
  }

  void write(ofstream& out){
    
    out << "Digi: " 
	<< layer_ << "\t" 
	<< irphi_ << "\t" 
	<< iz_ << "\t" 
	<< sensorlayer_ << "\t" 
	<< ladder_ << "\t" 
	<< module_ << "\t" 
	<< x_ << "\t" 
	<< y_ << "\t" 
	<< z_ << "\t" << endl; 

    for (unsigned int i=0;i<simtrackids_.size();i++){
      out << "SimTrackId: "<<simtrackids_[i]<<endl;
    }
	
  }

  int irphi() {return irphi_;}
  int iz() {return iz_;}
  int layer() {return layer_;}
  int sensorlayer() {return sensorlayer_;}
  int ladder() {return ladder_;}
  int module() {return module_;}
  double r() {return sqrt(x_*x_+y_*y_);}
  double z() {return z_;}
  double phi() {return atan2(y_,x_);}


  bool operator==(const Digi& anotherdigi) const {
    if (irphi_!=anotherdigi.irphi_) return false;
    if (iz_!=anotherdigi.iz_) return false;
    if (layer_!=anotherdigi.layer_) return false;
    if (ladder_!=anotherdigi.ladder_) return false;
    return module_==anotherdigi.module_;    
  }

  int hash() const {
    return irphi_+iz_*1009+layer_*10000003+ladder_*1000003+module_*10007;
  }

  int nsimtrack() {return simtrackids_.size();}
  int simtrackid(int isim) {return simtrackids_[isim];}
  bool matchsimtrackid(int simtrackid){
    for (unsigned int i=0;i<simtrackids_.size();i++){
      if (simtrackids_[i]==simtrackid) return true;
    }
    return false;
  }


private:

  int layer_;
  int irphi_;
  int iz_;
  int sensorlayer_;
  int ladder_;
  int module_;
  double x_;
  double y_;
  double z_;

  vector<int> simtrackids_;

};

struct HashOp {
  int operator()(const Digi &a) const {
    return a.hash();
  }
};
 
struct HashEqual {
  bool operator()(const Digi &a, const Digi &b) const {
    return a == b;
  }
};



class ChipDigi{

public:


  ChipDigi(int layer,int irphichip, int izchip,int irphi, int iz, 
	   int sensorlayer,
	   int ladder, int module,bool inner) {
    layer_=layer;
    irphi_=irphi;
    iz_=iz;
    irphichip_=irphichip;
    izchip_=izchip;
    sensorlayer_=sensorlayer;
    ladder_=ladder;
    module_=module;
    inner_=inner;
  }

  void AddSimtrack(int simtrackid){
    for(unsigned int i=0;i<simtrackids_.size();i++){
      if (simtrackids_[i]==simtrackid) return;
    }
    simtrackids_.push_back(simtrackid);
  }

  void write(ofstream& out){
    
    out << "ChipDigi: " 
	<< layer_ << "\t" 
	<< irphichip_ << "\t" 
	<< izchip_ << "\t" 
	<< irphi_ << "\t" 
	<< iz_ << "\t" 
	<< sensorlayer_ << "\t" 
	<< ladder_ << "\t" 
	<< module_ << "\t"
        << inner_ << "\t" 
	<< endl; 

    for (unsigned int i=0;i<simtrackids_.size();i++){
      out << "SimTrackId: "<<simtrackids_[i]<<endl;
    }
	
  }

  bool operator==(ChipDigi& aDigi) {
    return (layer_==aDigi.layer_)&&
      (irphichip_==aDigi.irphichip_)&&
      (izchip_==aDigi.izchip_)&&
      (sensorlayer_==aDigi.sensorlayer_)&&
      (irphi_==aDigi.irphi_)&&
      (iz_==aDigi.iz_)&&
      (ladder_==aDigi.ladder_)&&
      (module_==aDigi.module_);
  }


  int layer() {return layer_;}
  int module() {return module_;}
  int ladder() {return ladder_;}
  bool inner() {return inner_;}

  int irphichip() {return irphichip_;}
  int izchip() {return izchip_;}
  int irphi() {return irphi_;}
  int iz() {return iz_;}


private:

  int layer_;
  int irphichip_;
  int izchip_;
  int irphi_;
  int iz_;
  int sensorlayer_;
  int ladder_;
  int module_;
  bool inner_;

  vector<int> simtrackids_;

};


class Sector{

public:

  Sector() {
    sector_=-1;
    phi_=-999.999;
  }

  void setSector(int n){
    sector_=n;
    phi_=n*(two_pi/NSECTORS);    
  }

  //return true is the test phi is closer then ladder
  bool closer(const GeometryMap& map,int layer, int ladder,double phi){
    
    double phi_ladder=map.moduleGeometry(layer,ladder,1).phi();

    double delta_phi_ladder=phi_ladder-phi_;
    double delta_phi=phi-phi_;

    if (delta_phi_ladder>0.5*two_pi) delta_phi_ladder-=two_pi;
    if (delta_phi_ladder<-0.5*two_pi) delta_phi_ladder+=two_pi;

    if (delta_phi>0.5*two_pi) delta_phi-=two_pi;
    if (delta_phi<-0.5*two_pi) delta_phi+=two_pi;
    
    return (fabs(delta_phi)<fabs(delta_phi_ladder));
      

  }

  //Note maxLadder is actually one larger than the max!!!
  void addLadderInLayer(vector<int>& theLadder, unsigned int maxLadder, 
			const GeometryMap& map,
			int layer, int ladder){

    double phi=map.moduleGeometry(layer,ladder,1).phi();
    
    if (theLadder.size()<maxLadder) {
      theLadder.push_back(ladder);
      return;
    }
    
    bool found=false;
    do {
      found=false;
      for (unsigned int i=0;i<maxLadder;i++){
	if (closer(map,layer,theLadder[i],phi)) {
	  int ladder_tmp=theLadder[i];
	  theLadder[i]=ladder;
	  phi=map.moduleGeometry(layer,ladder_tmp,1).phi();
	  ladder=ladder_tmp;
	  found=true;
	}
      }
    } while (found);
	
    //sort
    do {
      found=false;
      for (unsigned int i=0;i<maxLadder-1;i++){
	double phi=map.moduleGeometry(layer,theLadder[i+1],1).phi();
	if (closer(map,layer,theLadder[i],phi)) {
	  int ladder_tmp=theLadder[i];
	    theLadder[i]=theLadder[i+1];
	    theLadder[i+1]=ladder_tmp;
	    found=true;
	}
      }
    } while (found);
       
  }

  void addLadder(const GeometryMap& map,int layer, int ladder){

    //double phi=map.moduleGeometry(layer,ladder,1).phi();
      
    if (layer==1){
      addLadderInLayer(ladder_1,3,map,layer,ladder);
      return;
    }


    if (layer==2){
      addLadderInLayer(ladder_2,3,map,layer,ladder);
      return;
    }

    if (layer==3){
      addLadderInLayer(ladder_3,4,map,layer,ladder);
      return;
    }

    if (layer==4){
      addLadderInLayer(ladder_4,4,map,layer,ladder);
      return;
    }

    if (layer==5){
      addLadderInLayer(ladder_5,4,map,layer,ladder);
      return;
    }

    if (layer==6){
      addLadderInLayer(ladder_6,4,map,layer,ladder);
      return;
    }

    if (layer==7){
      addLadderInLayer(ladder_7,4,map,layer,ladder);
      return;
    }

    if (layer==8){
      addLadderInLayer(ladder_8,4,map,layer,ladder);
      return;
    }

    if (layer==9){
      addLadderInLayer(ladder_9,7,map,layer,ladder);
      return;
    }

    if (layer==10){
      addLadderInLayer(ladder_10,7,map,layer,ladder);
      return;
    }

  }

  int inLadder(const vector<int>& theLadder,int ladder){
    for (unsigned int i=0;i<theLadder.size();i++){
      if (theLadder[i]==ladder) return i+1;
    }
    return 0;
  }
  

  int contain(int layer, int ladder){

    assert(layer>=1&&layer<=10);

    if (layer==1) {
      return inLadder(ladder_1,ladder);
      return 0;
    }

    if (layer==2) {
      return inLadder(ladder_2,ladder);
      return 0;
    }

    if (layer==3) {
      return inLadder(ladder_3,ladder);
    }

    if (layer==4) {
      return inLadder(ladder_4,ladder);
    }

    if (layer==5) {
      return inLadder(ladder_5,ladder);
    }

    if (layer==6) {
      return inLadder(ladder_6,ladder);
    }

    if (layer==7) {
      return inLadder(ladder_7,ladder);
    }

    if (layer==8) {
      return inLadder(ladder_8,ladder);
    }

    if (layer==9) {
      //cout << "ladder_9.size()="<<ladder_9.size()<<" : ";
      //for (unsigned int l=0;l<ladder_9.size();l++){
      //cout <<ladder_9[l]<<" ";
      //}
      //cout << endl;
      return inLadder(ladder_9,ladder);
    }

    if (layer==10) {
      return inLadder(ladder_10,ladder);
    }

    return 0;

  }


  void print(){

    cout << "Sector "<<sector_<<" "<<phi_<<endl;
    assert(ladder_1.size()==3);
    assert(ladder_2.size()==3);    
    cout << "Layer 1: "<<ladder_1[0]<<" "<<ladder_1[1]<<" "<<ladder_1[2]<<endl;
    cout << "Layer 2: "<<ladder_2[0]<<" "<<ladder_2[1]<<" "<<ladder_2[2]<<endl;
    assert(ladder_3.size()==4);
    assert(ladder_4.size()==4);
    cout << "Layer 3: "<<ladder_3[0]<<" "<<ladder_3[1]
	 << " "<<ladder_3[2]<<" "<<ladder_3[3]<<endl;
    cout << "Layer 4: "<<ladder_4[0]<<" "<<ladder_4[1]
	 << " "<<ladder_4[2]<<" "<<ladder_4[3]<<endl;
    //assert(ladder_9.size()==7);  HACK 6_1
    //assert(ladder_10.size()==7);
    //cout << "Layer 9: "<<ladder_9[0]<<" "
    //	 << ladder_9[1]<<" "<<ladder_9[2]<<" "<<ladder_9[3]<<" "
    //	 << ladder_9[4]<<" "<<ladder_9[5]<<" "<<ladder_9[6]<<endl;
    //cout << "Layer 10: "<<ladder_10[0]<<" "
    //	 << ladder_10[1]<<" "<<ladder_10[2]<<" "<<ladder_10[3]<<" "
    //	 << ladder_10[4]<<" "<<ladder_10[5]<<" "<<ladder_10[6]<<endl;
    
  }

  double sectorCenter() {
    return phi_;
  }


private:

  int sector_; // 0 to 23
  double phi_;

  vector<int> ladder_1;
  vector<int> ladder_2;
  vector<int> ladder_3;
  vector<int> ladder_4;
  vector<int> ladder_5;
  vector<int> ladder_6;
  vector<int> ladder_7;
  vector<int> ladder_8;
  vector<int> ladder_9;
  vector<int> ladder_10;

};


class Stub{

public:

  Stub() {
 
  }

  Stub(int layer,int ladder, int module, double pt,
       double x, double y, double z) {
    layer_=layer;
    ladder_=ladder;
    module_=module;
    pt_=pt;
    x_=x;
    y_=y;
    z_=z;

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
	<< pt_ << "\t" 
	<< x_ << "\t" 
	<< y_ << "\t" 
	<< z_ << "\t" << endl; 

    for (unsigned int i=0;i<outerdigis_.size();i++){
      out << "OuterStackDigi: "<<outerdigis_[i].first<<"\t"
	  << outerdigis_[i].second<<"\t"
	  << outerdigisladdermodule_[i].first<<"\t"
	  << outerdigisladdermodule_[i].second<<"\t"
	  <<endl;
    }

    for (unsigned int i=0;i<innerdigis_.size();i++){
      out << "InnerStackDigi: "<<innerdigis_[i].first<<"\t"
	  << innerdigis_[i].second<<"\t"
	  << innerdigisladdermodule_[i].first<<"\t"
	  << innerdigisladdermodule_[i].second
	  <<endl;
    }
	
  }

  int ptsign() {
    int ptsgn=-1.0;
    if (iphi()<iphiouter()) ptsgn=-ptsgn;
    //if (z_<0.0) ptsgn=-ptsgn;
    return ptsgn;
  }

  double iphi() {
    if (!innerdigis_.size()>0) {
      cout << "innerdigis_.size()="<<innerdigis_.size()<<endl;
      return 0.0;
    }
    double phi_tmp=0.0;
    for (unsigned int i=0;i<innerdigis_.size();i++){
      phi_tmp+=innerdigis_[i].first;
    }
    return phi_tmp/innerdigis_.size();
  }

  double iphiouter() {
    if (!outerdigis_.size()>0) {
      cout << "outerdigis_.size()="<<outerdigis_.size()<<endl;
      return 0.0;
    }
    double phi_tmp=0.0;
    for (unsigned int i=0;i<outerdigis_.size();i++){
      phi_tmp+=outerdigis_[i].first;
    }
    return phi_tmp/outerdigis_.size();
  }

  double iz() {
    if (!innerdigis_.size()>0) {
      cout << "innerdigis_.size()="<<innerdigis_.size()<<endl;
      return 0.0;
    }
    double z_tmp=0.0;
    for (unsigned int i=0;i<innerdigis_.size();i++){
      z_tmp+=innerdigis_[i].second;
    }
    return z_tmp/innerdigis_.size();
  }

  int layer() const { return layer_; }
  int ladder() const { return ladder_; }
  int module() const { return module_; }
  vector<pair<int,int> > innerdigis() const { return innerdigis_; }
  vector<pair<int,int> > outerdigis() const { return outerdigis_; }
  vector<pair<int,int> > innerdigisladdermodule() const { return innerdigisladdermodule_; }
  vector<pair<int,int> > outerdigisladdermodule() const { return outerdigisladdermodule_; }
  double x() const { return x_; }
  double y() const { return y_; }
  double z() const { return z_; }
  double r() const { return sqrt(x_*x_+y_*y_); }
  double pt() const { return pt_; }

private:

  int layer_;
  int ladder_;
  int module_;
  double pt_;
  double x_;
  double y_;
  double z_;

  vector<pair<int,int> > innerdigis_;
  vector<pair<int,int> > innerdigisladdermodule_;
  vector<pair<int,int> > outerdigis_;
  vector<pair<int,int> > outerdigisladdermodule_;


};



class SLHCEvent{

public:


  SLHCEvent() {

    //empty constructor to be used with 'filler' functions

    nchiprphi_=-1;
    nchipz_=-1;
    nstripz_=-1;
    Ntrack_=0;

  }

  void setIPx(double x) { x_offset=x;}
  void setIPy(double y) { y_offset=y;}

  void addL1SimTrack(int id,int type,double pt,double eta,double phi,
	      double vx,double vy,double vz){

    vx-=x_offset;
    vy-=y_offset;
    L1SimTrack simtrack(id,type,pt,eta,phi,vx,vy,vz);
    simtracks_.push_back(simtrack);

  }


  void addDigi(int layer,int irphi,int iz,int sensorlayer,int ladder,int module,
	  double x,double y,double z,vector<int> simtrackids){

    x-=x_offset;
    y-=y_offset;

    Digi digi(layer,irphi,iz,sensorlayer,ladder,
	      module,x,y,z);

    for (unsigned int i=0;i<simtrackids.size();i++){
      digi.AddSimtrack(simtrackids[i]);
    }    
  
    digis_.push_back(digi);
    digihash_.insert(digi);

  }


  bool addStub(int layer,int ladder,int module,double pt,
	   double x,double y,double z,
	   vector<bool> innerStack,
	   vector<int> irphi,
	   vector<int> iz,
	   vector<int> iladder,
	   vector<int> imodule){

    x-=x_offset;
    y-=y_offset;
  
    Stub stub(layer, ladder, module, pt, x, y, z);

    for(unsigned int i=0;i<innerStack.size();i++){
      if (innerStack[i]) {
	stub.AddInnerDigi(iladder[i],imodule[i],irphi[i],iz[i]);
      }
      else {
	stub.AddOuterDigi(iladder[i],imodule[i],irphi[i],iz[i]);
      }
    }   

    bool foundclose=false;

    for (unsigned int i=0;i<stubs_.size();i++) {
      if (fabs(stubs_[i].x()-stub.x())<0.2&&
	  fabs(stubs_[i].y()-stub.y())<0.2&&
	  fabs(stubs_[i].z()-stub.z())<2.0) {
	//foundclose=true;
      }
    }

    if (!foundclose) {
      stubs_.push_back(stub);
      return true;
    }

    return false;
    
  }


  SLHCEvent(istream& in) {

    nchiprphi_=-1;
    nchipz_=-1;
    nstripz_=-1;
    Ntrack_=0;
    string tmp;
    in >> tmp;
    if (tmp!="Event:") {
      cout << "Expected to read 'Event:' but found:"<<tmp<<endl;
      abort();
    }
    in >> eventnum_;

    cout << "Started to read event="<<eventnum_<<endl;

    // read the SimTracks

    bool first=true;

    in >> tmp;
    while (tmp!="SimTrackEnd"){
      if (!(tmp=="SimTrack:"||tmp=="SimTrackEnd")) {
	cout << "Expected to read 'SimTrack:' or 'SimTrackEnd' but found:"
	     << tmp << endl;
	abort();
      }
      int id;
      int type;
      double pt;
      double eta;
      double phi;
      double vx;
      double vy;
      double vz;
      in >> id >> type >> pt >> eta >> phi >> vx >> vy >> vz;
      //in >> id >> pt >> eta >> vx >> vy >> vz;
      if (first) {
	mc_rinv=0.00299792*3.8/pt;
	mc_phi0=phi;
	mc_z0=vz;
	mc_t=tan(0.25*two_pi-2.0*atan(exp(-eta)));
	event=eventnum_;
	first=false;
      }
      vx-=x_offset;
      vy-=y_offset;
      L1SimTrack simtrack(id,type,pt,eta,phi,vx,vy,vz);
      simtracks_.push_back(simtrack);
      in >> tmp;
    }


   
    //read te Digis
    in >> tmp;
    while (tmp!="DigiEnd"){
      //cout << "Here001:"<<tmp<<endl;
      if (!(tmp=="Digi:"||tmp=="DigiEnd")) {
	cout << "Expected to read 'Digi:' or 'DigiEnd' but found:"
	     << tmp << endl;
        abort();
      }
      int layer;
      int irphi;
      int iz;
      int sensorlayer;
      int ladder;
      int module;
      double x;
      double y;
      double z;

      in >> layer
	 >> irphi
	 >> iz
	 >> sensorlayer
	 >> ladder
	 >> module
	 >> x
	 >> y
	 >> z;

      x-=x_offset;
      y-=y_offset;


      Digi digi(layer,irphi,iz,sensorlayer,ladder,
		module,x,y,z);
      in >> tmp;
      while (tmp=="SimTrackId:"){
	int simtrackid;
	in >> simtrackid;
	digi.AddSimtrack(simtrackid);
	in >> tmp;
      }      
      digis_.push_back(digi);
      digihash_.insert(digi);
    }

    cout << "Read "<<digis_.size()<<" digis"<<endl;

    int nlayer[11];
    for (int i=0;i<10;i++) {
      nlayer[i]=0;
    }
    

    //read stubs
    in >> tmp;
    while (tmp!="StubEnd"){
      //cout << "Here002:"<<tmp<<endl;
      if (!in.good()) {
	cout << "File not good"<<endl;
	abort();
      };
      if (!(tmp=="Stub:"||tmp=="StubEnd")) {
	cout << "Expected to read 'Stub:' or 'StubEnd' but found:"
	     << tmp << endl;
	abort();
      }
      int layer;
      int ladder;
      int module;
      double pt;
      double x;
      double y;
      double z;

      in >> layer >> ladder >> module >> pt >> x >> y >> z;

      layer--;   //HACK 6_1
      //ladder++;

      x-=x_offset;
      y-=y_offset;

      if (layer < 10) nlayer[layer]++;

      Stub stub(layer, ladder, module, pt, x, y, z);

      in >> tmp;
      //cout << "Here003:"<<tmp<<endl;
      while (tmp=="InnerStackDigi:"||tmp=="OuterStackDigi:"){
	int irphi;
	int iz;
        int iladder;
        int imodule;
	in >> irphi;
	//cout << "irphi="<<irphi<<endl;
	in >> iz;
	//cout << "iz="<<iz<<endl;
	in >> iladder; 
	//cout << "iladder="<<iladder<<endl;
        in >> imodule;
	//cout << "imodule="<<imodule<<endl;
	if (tmp=="InnerStackDigi:") stub.AddInnerDigi(iladder,imodule,irphi,iz);
	if (tmp=="OuterStackDigi:") stub.AddOuterDigi(iladder,imodule,irphi,iz);
	in >> tmp;
	//cout << "Here004:"<<tmp<<endl;
      }   
      //cout << "tmp="<<tmp<<endl;

      bool foundclose=false;

      for (unsigned int i=0;i<stubs_.size();i++) {
	if (fabs(stubs_[i].x()-stub.x())<0.2&&
	    fabs(stubs_[i].y()-stub.y())<0.2&&
	    fabs(stubs_[i].z()-stub.z())<2.0) {
	  foundclose=true;
	}
      }

      if (!foundclose) {
	stubs_.push_back(stub);
      }
    }
    cout << "Read "<<stubs_.size()<<" stubs"<<endl;
    //for (int i=0;i<10;i++) {
    //  cout << "In layer "<<i<<" read "<<nlayer[i]<<" stubs"<<endl;
    //}
  }

  void write(ofstream& out){
    
    out << "Event: "<<eventnum_ << endl;
      
    for (unsigned int i=0; i<simtracks_.size(); i++) {
      simtracks_[i].write(out);
    }
    out << "SimTrackEnd" << endl;
    
    for (unsigned int i=0; i<digis_.size(); i++) {
      digis_[i].write(out);
    }
    out << "DigiEnd" << endl;

    for (unsigned int i=0; i<chipdigis_.size(); i++) {
      chipdigis_[i].write(out);
    }
    out << "ChipDigiEnd" << endl;
    
    for (unsigned int i=0; i<stubs_.size(); i++) {
      stubs_[i].write(out);
    }
    out << "StubEnd" << endl;
    
  }


  void writeChipData(ofstream& outchip, GeometryMap& geom){

    outchip << "Event "<<eventnum_ << endl;

    //  layer   ladder    module --- indexed by inner ladder!!!!
    map<int, map<int, map<int, vector<int> > > >  chipdigimap;

    cout << "number of chip digis="<<chipdigis_.size()<<endl;

    for (unsigned int i=0; i<chipdigis_.size(); i++) {
      ChipDigi tmp=chipdigis_[i];
      if (tmp.inner()){
	//cout << "added inner"<<endl;
	chipdigimap[tmp.layer()][tmp.ladder()][tmp.module()].push_back(i);
      }
      else{
	//cout << "added outer"<<endl;
	//get the ladder# coresponding to the inner ladder in the stack
	int innerladder=geom.innerladder(tmp.layer(),tmp.ladder());
	chipdigimap[tmp.layer()][innerladder][tmp.module()].push_back(i);
      }
    }

    map<int, map<int, map<int, vector<int> > > >::const_iterator itlayer=chipdigimap.begin();
    map<int, map<int, map<int, vector<int> > > >::const_iterator itlayerend=chipdigimap.end();

    for(;itlayer!=itlayerend;++itlayer) {
      
      map<int, map<int, vector<int> > >::const_iterator itladder=itlayer->second.begin();
      map<int, map<int, vector<int> > >::const_iterator itladderend=itlayer->second.end();

      for(;itladder!=itladderend;++itladder) {

	map<int, vector<int> >::const_iterator itmodule=itladder->second.begin();
	map<int, vector<int> >::const_iterator itmoduleend=itladder->second.end();

	for(;itmodule!=itmoduleend;++itmodule) {

	  cout << "layer="<<itlayer->first
	       << " ladder="<<itladder->first
	       << " module="<<itmodule->first
	       << " number of digis="<<itmodule->second.size()<<endl;

	  vector<int> digis=itmodule->second;
	  
	  int innermap[1000][1000];
	  int outermap[1000][1000];

	  //outchip << "layer="<<itlayer->first
	  //  << " ladder="<<itladder->first
	  //  << " module="<<itmodule->first << endl;

	  int nStubs=0;

	  //first we have to count the stubs...
	  for (unsigned int i=0;i<stubs_.size();i++){

	    Stub aStub=stubs_[i];

	    int innerladder=aStub.ladder();

	    if (aStub.layer()+1==itlayer->first&&
		innerladder==itladder->first&&
		aStub.module()==itmodule->first) {
	      nStubs++;
	    }

	  }

	  bool firstStub=true;
	  
	  for (unsigned int i=0;i<stubs_.size();i++){

	    Stub aStub=stubs_[i];

	    //int innerladder=geom.innerladder(aStub.layer()+1,aStub.ladder());
	    int innerladder=aStub.ladder();

	    cout << "layer="<<aStub.layer()+1
		 << " ladder="<<innerladder
		 << " module="<<aStub.module()
		 << " aStub"<<endl;
	      

	    if (aStub.layer()+1==itlayer->first&&
		innerladder==itladder->first&&
		aStub.module()==itmodule->first) {


	      char buf[500];


	      if (firstStub) {

		sprintf(buf,"FE_%02d_%03d_%03d_NSTUB %03d",
			itlayer->first,
			itladder->first,
			itmodule->first,
			nStubs);
	      
		outchip << buf << endl;;
		
	      }

	      firstStub=false;


	      sprintf(buf,"FE_%02d_%03d_%03d",
		      itlayer->first,
		      itladder->first,
		      itmodule->first);
	      
	      outchip << buf << "_STUB ";
	      
	      vector<pair<int,int> > innerdigis=aStub.innerdigis();
	      for (unsigned int j=0;j<innerdigis.size();j++) {

		int irphichip,irphi_on_chip,izchip,iz_on_chip;
		getChipCoordinates(innerdigis[j].first,
				   innerdigis[j].second,true,
				   irphichip,irphi_on_chip,izchip,iz_on_chip);
		
		sprintf(buf,"%02d_%02d_%03d_%03d",
			irphichip,izchip,irphi_on_chip,iz_on_chip);

		outchip << "SS("<<buf<<") ";

	      }
	      
	      vector<pair<int,int> > outerdigistmp=aStub.outerdigis();
	      
	      vector<pair<int,int> > outerdigis;
	      for (unsigned int j=0;j<outerdigistmp.size();j++) {
		bool found=false;
		for (unsigned int k=0;k<outerdigis.size();k++) {
		  if (outerdigistmp[j].first==outerdigis[k].first&&
		      outerdigistmp[j].second/4==outerdigis[k].second){
		    found=true;
		  }
		}
		if (!found){
		  outerdigis.push_back(pair<int,int>(outerdigistmp[j].first,
						     outerdigistmp[j].second/4));
		}
	      }
		    
	      for (unsigned int j=0;j<outerdigis.size();j++) {

		int irphichip,irphi_on_chip,izchip,iz_on_chip;
		getChipCoordinates(outerdigis[j].first,
				   outerdigis[j].second*nstripz_,false,
				   irphichip,irphi_on_chip,izchip,iz_on_chip);

		sprintf(buf,"%02d_%02d_%03d_%03d",
			irphichip,izchip,irphi_on_chip,iz_on_chip);

		outchip << "LS("<<buf<<") ";

	      }
	      outchip << endl;
	    }
	    
	  }



	  for(int izchip=0;izchip<5;izchip++){
	    for(int irphichip=0;irphichip<5;irphichip++){
	      
	      for(int i=0;i<1000;i++){
		for(int j=0;j<1000;j++){
		  innermap[i][j]=0;
		  outermap[i][j]=0;
		}
	      }	      

	      for (unsigned int i=0;i<digis.size();i++){
		ChipDigi tmp=chipdigis_[digis[i]];
		if (tmp.irphichip()!=irphichip) continue;
		if (tmp.izchip()!=izchip) continue;
		
		if (tmp.inner()) {
		  innermap[tmp.irphi()][tmp.iz()]=1;
		}
		else {
		  outermap[tmp.irphi()][tmp.iz()]=1;
		}
	      }


	      for(int iz=0;iz<16;iz++){
		if (iz%4==0) {
		  char buf[500];
		  sprintf(buf,"FE_%02d_%03d_%03d_%01d_%01d_%02d",
			  itlayer->first,
			  itladder->first,
			  itmodule->first,
			  izchip,
			  irphichip,
			  iz/4);
			  outchip<<buf<<"_LS_HITS ";
		  for(int irphi=0;irphi<200;irphi++){
		    outchip<<outermap[irphi][iz/4];
		  }
		  outchip <<endl;
		}
		
		char buf[500];
		sprintf(buf,"FE_%02d_%03d_%03d_%01d_%01d_%02d",
			itlayer->first,
			itladder->first,
			itmodule->first,
			izchip,
			irphichip,
			iz);
		outchip<<buf<<"_SS_HITS ";
		for(int irphi=0;irphi<200;irphi++){
		  outchip<<innermap[irphi][iz];
		}
		outchip <<endl;
	      }

	    }

	  }

	}

      }

    }
       
  }

  void setChipSize(int nchiprphi,int nchipz,int nstripz){

    nchiprphi_=nchiprphi;
    nchipz_=nchipz;
    nstripz_=nstripz;

  }

  void getChipCoordinates(int irphi,int iz,bool inner,
			  int &irphichip,int &irphi_on_chip,
			  int &izchip, int &iz_on_chip){


    int nrphistripchip=nrphi/nchiprphi_;
    int nzstripchip=nz/nchipz_;


    irphichip=irphi/nrphistripchip;

    irphi_on_chip=irphi-irphichip*nrphistripchip;

    izchip=iz/nzstripchip;
    iz_on_chip=-1;

    if (inner) {
      iz_on_chip=iz-izchip*nzstripchip;
    }
    else{
      iz_on_chip=(iz-izchip*nzstripchip)/nstripz_;
    }

  }
  

  void makeChipDigi(GeometryMap& geom){

    assert(nchiprphi_!=-1);
    assert(nchipz_!=-1);
    assert(nstripz_!=-1);

    //first check consistency

    if (nrphi%nchiprphi_!=0) {
      cout << "In makeChipDigi: nrphi="<<nrphi<<" nchiprphi="<<nchiprphi_<<endl;
      abort();
    }

    if (nz%(nchipz_*nstripz_)!=0) {
      cout << "In makeChipDigi: nz="<<nz<<" nchipz="<<nchipz_
	   << " nstripz="<<nstripz_<<endl;
      abort();
    }

    for (unsigned int i=0;i<digis_.size();i++){
      cout << "digi:"<<i<<endl;
      int irphi=digis_[i].irphi();
      int iz=digis_[i].iz();

      bool inner=geom.inner(digis_[i].layer(),digis_[i].ladder());

      int irphichip,irphi_on_chip,izchip,iz_on_chip;

      getChipCoordinates(irphi,iz,inner,
			 irphichip,irphi_on_chip,izchip,iz_on_chip);


      //int nrphistripchip=nrphi/nchiprphi_;
      //int nzstripchip=nz/nchipz_;


      //int irphichip=irphi/nrphistripchip;

      //int irphi_on_chip=irphi-irphichip*nrphistripchip;

      //int izchip=iz/nzstripchip;
      //int iz_on_chip=-1;

      //if (inner) {
      //  iz_on_chip=iz-izchip*nzstripchip;
      //}
      //else{
      //	iz_on_chip=(iz-izchip*nzstripchip)/nstripz_;
      //}

      ChipDigi tmp(digis_[i].layer(),
		   irphichip, 
		   izchip,
		   irphi_on_chip, 
		   iz_on_chip, 
		   digis_[i].sensorlayer(),
		   digis_[i].ladder(), 
		   digis_[i].module(),
		   inner);

      bool founddigi=false;
      for (unsigned int j=0;j<chipdigis_.size();j++){
	if (tmp==chipdigis_[j]) {
	  founddigi=true;
	  for (int isim=0;isim<digis_[i].nsimtrack();isim++){
	    chipdigis_[j].AddSimtrack(digis_[i].simtrackid(isim));
	  }
	}
      }
      if (!founddigi) {
	for (int isim=0;isim<digis_[i].nsimtrack();isim++){
	  tmp.AddSimtrack(digis_[i].simtrackid(isim));
	}	
	chipdigis_.push_back(tmp);
      }
    }
  }


  int simtrackid(const Stub& stub){

    std::vector<int> simtrackids;

    simtrackids=this->simtrackids(stub);

    if (simtrackids.size()==0) {
      //cout << "Warning no simtrackids"<<endl;
      return -1;
    }

    for (unsigned int i=1;i<simtrackids.size();i++){
      if (simtrackids[i]!=simtrackids[0]) return -1;
    }

    return simtrackids[0];

  }

  std::vector<int> simtrackids(const Stub& stub){

    //cout << "Entering simtrackids"<<endl;

    std::vector<int> simtrackids;

    int layer=stub.layer()+1;


    vector<pair<int,int> > innerdigis=stub.innerdigis();
    vector<pair<int,int> > outerdigis=stub.outerdigis();
    vector<pair<int,int> > innerdigisladdermodule=stub.innerdigisladdermodule();
    vector<pair<int,int> > outerdigisladdermodule=stub.outerdigisladdermodule();

    vector<pair<int,int> > alldigis=stub.innerdigis();
    alldigis.insert(alldigis.end(),outerdigis.begin(),outerdigis.end());
    vector<pair<int,int> > alldigisladdermodule=stub.innerdigisladdermodule();
    alldigisladdermodule.insert(alldigisladdermodule.end(),
				outerdigisladdermodule.begin(),
				outerdigisladdermodule.end());



    if (layer<1000) {

      for (unsigned int k=0;k<alldigis.size();k++){
	int irphi=alldigis[k].first;
	int iz=alldigis[k].second;
	int ladder=alldigisladdermodule[k].first;
	int module=alldigisladdermodule[k].second;
	Digi tmp(layer,irphi,iz,-1,ladder,module,0.0,0.0,0.0);
	__gnu_cxx::hash_set<Digi,HashOp,HashEqual>::const_iterator it=digihash_.find(tmp);
	if(it==digihash_.end()){
	  static int count=0;
	  count++;
	  if (count<5) {
	    cout << "Warning did not find digi"<<endl;
	  } 
 	}
	else{
	  Digi adigi=*it;
	  for(int idigi=0;idigi<adigi.nsimtrack();idigi++){
	    simtrackids.push_back(adigi.simtrackid(idigi));
	  }
	}	
      }
    }

    else{

      for (unsigned int k=0;k<alldigis.size();k++){
	int irphi=alldigis[k].first;
	int iz=alldigis[k].second;
	//int ladder=alldigisladdermodule[k].first;
	int module=alldigisladdermodule[k].second;
	int offset=1000;
	if (stub.z()<0.0) offset=2000;
	//cout << "Looking for: "<<stub.module()+offset<<" "<<irphi<<" "
	//     <<iz<<" "<<1<<" "<<module<<endl;
	Digi tmp(stub.module()+offset,irphi,iz,-1,1,module,0.0,0.0,0.0);
	__gnu_cxx::hash_set<Digi,HashOp,HashEqual>::const_iterator it=digihash_.find(tmp);
	if(it==digihash_.end()){
	  static int count=0;
	  count++;
	  if (count < 5) {
	    cout << "Warning did not find digi in disks"<<endl;
	  }
	}
	else{
	  //cout << "Warning found digi in disks"<<endl;
	  Digi adigi=*it;
	  for(int idigi=0;idigi<adigi.nsimtrack();idigi++){
	    simtrackids.push_back(adigi.simtrackid(idigi));
	  }
	}	
      }
    }

    return simtrackids;

  }

  void storeTrack(int* phibins, int Nlay,int Nbins) {

    cout << "In store track:"<<this<<endl;

    Nlay_=Nlay;
    Nbins_=Nbins;
    for (int i=0;i<Nlay;i++){
      phimap[Ntrack_][i]=phibins[i];
    }
    Ntrack_++;

  }



  int ndigis() { return digis_.size(); }

  Digi digi(int i) { return digis_[i]; }

  int nstubs() { return stubs_.size(); }

  Stub stub(int i) { return stubs_[i]; }

  int nChipDigis() { return chipdigis_.size(); }

  ChipDigi chipDigi(int i) { return chipdigis_[i]; }

  int nsimtracks() { return simtracks_.size(); }

  L1SimTrack simtrack(int i) { return simtracks_[i]; }

  int eventnum() const { return eventnum_; }

  int getSimtrackFromSimtrackid(int simtrackid) const {
    for(unsigned int i=0;i<simtracks_.size();i++){
      if (simtracks_[i].id()==simtrackid) return i;
    }
    return -1;
  }


  //hack!!

  int Nlay_;
  int Nbins_;
  int Ntrack_;
  int phimap[1000][10];

  static double mc_rinv;
  static double mc_phi0;
  static double mc_z0;
  static double mc_t;
  static int event;

private:

  int eventnum_;
  vector<L1SimTrack> simtracks_;
  vector<Digi> digis_;
  __gnu_cxx::hash_set<Digi,HashOp,HashEqual> digihash_;
  vector<ChipDigi> chipdigis_;
  vector<Stub> stubs_;

  int nchiprphi_;
  int nchipz_;
  int nstripz_;

};

double SLHCEvent::mc_rinv=0.0;
double SLHCEvent::mc_phi0=0.0;
double SLHCEvent::mc_z0=0.0;
double SLHCEvent::mc_t=0.0;
int SLHCEvent::event=0;

#endif



