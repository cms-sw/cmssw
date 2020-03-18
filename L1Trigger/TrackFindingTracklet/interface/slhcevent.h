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
#include "L1TStub.h"
#include "Constants.h"

using namespace std;

// --- these are instead set in Constants.hh ---
//static double two_pi=8*atan(1.0);
//static double ptcut=2.0;
//static unsigned int NSector=24;

static double x_offset=0.199196*0.0;
static double y_offset=0.299922*0.0;




class L1SimTrack{

public:

  L1SimTrack() {
    eventid_=-1; 
    trackid_=-1;   
  }

  L1SimTrack(int eventid, int trackid, int type, double pt, double eta, double phi, 
	     double vx, double vy, double vz) {
    eventid_=eventid;
    trackid_=trackid;
    type_=type;
    pt_=pt;
    eta_=eta;
    phi_=phi;
    vx_=vx;
    vy_=vy;
    vz_=vz;
  }

  void write(ofstream& out){
    
    if (pt_ > -2.0) {
    out << "SimTrack: " 
	<< eventid_ << "\t" 
	<< trackid_ << "\t" 
	<< type_ << "\t" 
	<< pt_ << "\t" 
	<< eta_ << "\t" 
	<< phi_ << "\t" 
	<< vx_ << "\t" 
	<< vy_ << "\t" 
	<< vz_ << "\t" << endl; 
    }
	
  }
  void write(ostream& out){
    
    if (pt_ > -2) {
    out << "SimTrack: " 
	<< eventid_ << "\t" 
	<< trackid_ << "\t" 
	<< type_ << "\t" 
	<< pt_ << "\t" 
	<< eta_ << "\t" 
	<< phi_ << "\t" 
	<< vx_ << "\t" 
	<< vy_ << "\t" 
	<< vz_ << "\t" << endl; 
    }

  }
  
  int eventid() const { return eventid_; }
  int trackid() const { return trackid_; }
  int type() const { return type_; }
  double pt() const { return pt_; }
  double rinv() const { return charge()*0.01*0.3*3.8/pt_; }
  double eta() const { return eta_; }
  double phi() const { return phi_; }
  double vx() const { return vx_; }
  double vy() const { return vy_; }
  double vz() const { return vz_; }
  double dxy() const { return -vx() * sin(phi()) + vy() * cos(phi()); }
  double d0() const { return -dxy(); }
  int charge() const {
     if (type_==11) return -1;
     if (type_==13) return -1;
     if (type_==-211) return -1;
     if (type_==-321) return -1;
     if (type_==-2212) return -1;
     return 1;
  }
  
private:

  int eventid_;
  int trackid_;
  int type_;
  double pt_;
  double eta_;
  double phi_;
  double vx_;
  double vy_;
  double vz_;

};

/*
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
  void write(ostream& out){
    
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

  unsigned int layer_;
  unsigned int ladder_;
  unsigned int module_;
  int irphi_;
  int iz_;
  int sensorlayer_;
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
*/



class SLHCEvent{

public:


  SLHCEvent() {
    //empty constructor to be used with 'filler' functions
    eventnum_=0;
  }

  void setIPx(double x) { x_offset=x;}
  void setIPy(double y) { y_offset=y;}

  void setEventNum(int eventnum) { eventnum_=eventnum; }

  void addL1SimTrack(int eventid,int trackid,int type,double pt,double eta,double phi,
		     double vx,double vy,double vz){

    vx-=x_offset;
    vy-=y_offset;
    L1SimTrack simtrack(eventid,trackid,type,pt,eta,phi,vx,vy,vz);
    simtracks_.push_back(simtrack);

  }

  /*
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
  */

  bool addStub(int layer,int ladder,int module, int strip, int eventid, vector<int> tps, 
              double pt,double bend,
              double x,double y,double z,
              vector<bool> innerStack,
              vector<int> irphi,
              vector<int> iz,
              vector<int> iladder,
              vector<int> imodule,
              int isPSmodule,
              int isFlipped){

    
    if (layer>999&&layer<1999&& z<0.0) {
      //cout << "Will change layer by addding 1000, before layer = " << layer <<endl;
      layer+=1000;
    }
    
    layer--;   
    x-=x_offset;
    y-=y_offset;

    
    L1TStub stub(eventid,tps,-1,-1,layer, ladder, module, strip, 
		 x, y, z, -1.0, -1.0, pt, bend, isPSmodule, isFlipped);

    for(unsigned int i=0;i<innerStack.size();i++){
      if (innerStack[i]) {
	stub.AddInnerDigi(iladder[i],imodule[i],irphi[i],iz[i]);
      }
      else {
	stub.AddOuterDigi(iladder[i],imodule[i],irphi[i],iz[i]);
      }
    }   

    stub.setiphi(stub.diphi());
    stub.setiz(stub.diz());


    double t=fabs(stub.z())/stub.r();
    double eta=asinh(t);
    
    double fact=1.0;
    if (ptcut>=2.7) fact=-1.2;
    fact=0.0;
    
    if (((fabs(stub.pt())>1.8*fact)&&(fabs(eta)<2.0))||
	((fabs(stub.pt())>1.4*fact)&&(fabs(eta)>2.0))||
	((fabs(stub.pt())>1.0*fact)&&(fabs(eta)>2.3))) {
      if (fabs(eta)<2.6) {
	stubs_.push_back(stub);
	return true;
      }
    }

    return false;
    
  }

  L1TStub lastStub(){
    return stubs_.back();
  }

  SLHCEvent(istream& in) {

    string tmp;
    in >> tmp;
    while (tmp=="Map:") {
      in>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp;
      in>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp>>tmp;
    }
    if (tmp=="EndMap") {
      in>>tmp;
    }
    if (tmp!="Event:") {
      cout << "Expected to read 'Event:' but found:"<<tmp<<endl;
      if (tmp=="") {
	cout << "WARNING: fewer events to process than specified!" << endl;
	return;
      }
      else {
	cout << "ERROR, aborting reading file" << endl;
	abort();
      }
    }
    in >> eventnum_;


    // read the SimTracks
    in >> tmp;
    while (tmp!="SimTrackEnd"){
      if (!(tmp=="SimTrack:"||tmp=="SimTrackEnd")) {
	cout << "Expected to read 'SimTrack:' or 'SimTrackEnd' but found:"
	     << tmp << endl;
	abort();
      }
      int eventid;
      int trackid;
      int type;
      string pt_str;
      string eta_str;
      string phi_str;
      string vx_str;
      string vy_str;
      string vz_str;
      double pt;
      double eta;
      double phi;
      double vx;
      double vy;
      double vz;
      in >> eventid >> trackid >> type >> pt_str >> eta_str >> phi_str >> vx_str >> vy_str >> vz_str;
      pt = strtod(pt_str.c_str(), NULL);
      eta = strtod(eta_str.c_str(), NULL);
      phi = strtod(phi_str.c_str(), NULL);
      vx = strtod(vx_str.c_str(), NULL);
      vy = strtod(vy_str.c_str(), NULL);
      vz = strtod(vz_str.c_str(), NULL);
      vx-=x_offset;
      vy-=y_offset;
      L1SimTrack simtrack(eventid,trackid,type,pt,eta,phi,vx,vy,vz);
      simtracks_.push_back(simtrack);
      in >> tmp;
    }


    //read te Digis
    /*
    in >> tmp;
    while (tmp!="DigiEnd"){
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
    */

    int nlayer[11];
    for (int i=0;i<10;i++) {
      nlayer[i]=0;
    }

    int oldlayer=0;
    int oldladder=0;
    int oldmodule=0;
    int oldcbc=-1;
    int count=1;
    double oldz=-1000.0;
    
    //read stubs
    in >> tmp;
    while (tmp!="StubEnd"){

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
      int eventid;
      vector<int> tps;
      int strip;
      double pt;
      double x;
      double y;
      double z;
      double bend;
      int isPSmodule;
      int isFlipped;

      unsigned int ntps;

      in >> layer >> ladder >> module >> strip >> eventid >> pt >> x >> y >> z >> bend >> isPSmodule >> isFlipped >> ntps;

      for(unsigned int itps=0;itps<ntps;itps++){
	int tp;
	in >> tp;
	tps.push_back(tp);
      }

      if (layer>999&&layer<1999&& z<0.0) {
	//cout << "Will change layer by addding 1000, before layer = " << layer <<endl;
	layer+=1000;
      }

      int cbc=strip/126;
      if (layer>3&&layer==oldlayer&&ladder==oldladder&&module==oldmodule&&cbc==oldcbc&&fabs(oldz-z)<1.0){
	count++;
      } else {
	oldlayer=layer;
	oldladder=ladder;
	oldmodule=module;
	oldcbc=cbc;
	oldz=z;
	count=1;
      }

      if (count>3) {
	//cout << "skipping count = "<<count<<" : "<<layer<<" "<<ladder<<" "<<module<<endl;
      }
      

      layer--;   
      x-=x_offset;
      y-=y_offset;

      if (layer < 10) nlayer[layer]++;

      /*
      if (layer>999&&z<0.0) {
	bend=-bend;
	pt=-pt;
      }
      */
      
      L1TStub stub(eventid,tps,-1,-1,layer, ladder, module, strip, x, y, z, -1.0, -1.0, pt, bend, isPSmodule, isFlipped);

      in >> tmp;

      while (tmp=="InnerStackDigi:"||tmp=="OuterStackDigi:"){
	int irphi;
	int iz;
        int iladder;
        int imodule;
	in >> irphi;
	in >> iz;
	in >> iladder; 
        in >> imodule;
	if (tmp=="InnerStackDigi:") stub.AddInnerDigi(iladder,imodule,irphi,iz);
	if (tmp=="OuterStackDigi:") stub.AddOuterDigi(iladder,imodule,irphi,iz);
	in >> tmp;
      }   

      double t=fabs(stub.z())/stub.r();
      double eta=asinh(t);

      double fact=1.0;
      if (ptcut>=2.7) fact=-1.2;
      fact=0.0;
      
      if (((fabs(stub.pt())>1.8*fact)&&(fabs(eta)<2.0))||
	  ((fabs(stub.pt())>1.4*fact)&&(fabs(eta)>2.0))||
	  ((fabs(stub.pt())>1.0*fact)&&(fabs(eta)>2.3))) {
	if (fabs(eta)<2.6&&count<=100) {
	  stubs_.push_back(stub);
	}
      }

    }
  }

  void allSector() {
    static double two_pi=8*atan(1.0);
    unsigned int nstub=stubs_.size();
    for (unsigned int i=0;i<nstub;i++) {
      for (unsigned int j=1;j<NSector;j++) {
	L1TStub tmp=stubs_[i];
	double phi=tmp.phi();
	double r=tmp.r();
	phi+=j*two_pi/NSector;
	double x=r*cos(phi);
	double y=r*sin(phi);
	tmp.setXY(x,y);
	stubs_.push_back(tmp);
      }
    }
  }

  void write(ofstream& out){
    
    out << "Event: "<<eventnum_ << endl;
      
    for (unsigned int i=0; i<simtracks_.size(); i++) {
      simtracks_[i].write(out);
    }
    out << "SimTrackEnd" << endl;

    /*
    for (unsigned int i=0; i<digis_.size(); i++) {
      digis_[i].write(out);
    }
    out << "DigiEnd" << endl;
    */

    for (unsigned int i=0; i<stubs_.size(); i++) {
      stubs_[i].write(out);
    }
    out << "StubEnd" << endl;
    
  }

  void write(ostream& out){
    
    out << "Event: "<<eventnum_ << endl;
      
    for (unsigned int i=0; i<simtracks_.size(); i++) {
      simtracks_[i].write(out);
    }
    out << "SimTrackEnd" << endl;
    
    /*
    for (unsigned int i=0; i<digis_.size(); i++) {
      digis_[i].write(out);
    }
    out << "DigiEnd" << endl;
    */

    for (unsigned int i=0; i<stubs_.size(); i++) {
      stubs_[i].write(out);
    }
    out << "StubEnd" << endl;
    
  }

  /*
  int simtrackid(const L1TStub& stub){

    std::vector<int> simtrackids;

    simtrackids=this->simtrackids(stub);

    if (simtrackids.size()==0) {
      return -1;
    }


    std::sort(simtrackids.begin(),simtrackids.end());

    int n_max = 0;
    int value_max = 0;
    int n_tmp = 1;
    int value_tmp = simtrackids[0];
    for (unsigned int i=1; i<simtrackids.size();i++) {
      if (simtrackids[i] == value_tmp) n_tmp++;
      else {
	if (n_tmp > n_max) {
	  n_max = n_tmp;
	  value_max = value_tmp;
	}
	n_tmp = 1;
	value_tmp = simtrackids[i];
      }
    }
    
    if (n_tmp > n_max) value_max = value_tmp;

    return value_max;

  }

  std::vector<int> simtrackids(const L1TStub& stub){

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
	  if (count<0) {
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
	int module=alldigisladdermodule[k].second;
	int offset=1000;
	if (stub.z()<0.0) offset=2000;
	Digi tmp(stub.module()+offset,irphi,iz,-1,1,module,0.0,0.0,0.0);
	__gnu_cxx::hash_set<Digi,HashOp,HashEqual>::const_iterator it=digihash_.find(tmp);
	if(it==digihash_.end()){
	  static int count=0;
	  count++;
	  if (count < 0) {
	    cout << "Warning did not find digi in disks"<<endl;
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

    return simtrackids;

  }

  int ndigis() { return digis_.size(); }

  Digi digi(int i) { return digis_[i]; }
  */

  void layersHit(int tpid, int &nlayers, int &ndisks){

    int l1=0;
    int l2=0;
    int l3=0;
    int l4=0;
    int l5=0;
    int l6=0;

    int d1=0;
    int d2=0;
    int d3=0;
    int d4=0;
    int d5=0;

    for (unsigned int istub=0; istub<stubs_.size(); istub++){
      if (stubs_[istub].tpmatch(tpid)){
	if (stubs_[istub].layer()==0) l1=1;
        if (stubs_[istub].layer()==1) l2=1;
        if (stubs_[istub].layer()==2) l3=1;
        if (stubs_[istub].layer()==3) l4=1;
	if (stubs_[istub].layer()==4) l5=1;
        if (stubs_[istub].layer()==5) l6=1;

        if (abs(stubs_[istub].disk())==1) d1=1;
        if (abs(stubs_[istub].disk())==2) d2=1;
        if (abs(stubs_[istub].disk())==3) d3=1;
        if (abs(stubs_[istub].disk())==4) d4=1;
        if (abs(stubs_[istub].disk())==5) d5=1;
      }

    }

    nlayers=l1+l2+l3+l4+l5+l6;
    ndisks=d1+d2+d3+d4+d5;


  }


  
  int nstubs() { return stubs_.size(); }

  L1TStub stub(int i) { return stubs_[i]; }

  unsigned int nsimtracks() { return simtracks_.size(); }

  L1SimTrack simtrack(int i) { return simtracks_[i]; }

  int eventnum() const { return eventnum_; }

  int getSimtrackFromSimtrackid(int simtrackid, int eventid=0) const {
    for(unsigned int i=0;i<simtracks_.size();i++){
      if (simtracks_[i].trackid()==simtrackid && simtracks_[i].eventid()==eventid) return i;
    }
    return -1;
  }


private:

  int eventnum_;
  vector<L1SimTrack> simtracks_;
  //vector<Digi> digis_;
  //__gnu_cxx::hash_set<Digi,HashOp,HashEqual> digihash_;
  vector<L1TStub> stubs_;


};

#endif



