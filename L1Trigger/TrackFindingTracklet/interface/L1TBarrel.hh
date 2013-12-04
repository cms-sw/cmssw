#ifndef L1TBARREL_H
#define L1TBARREL_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
using namespace std;

#include "L1TStub.hh"
#include "L1TTracklet.hh"
#include "L1TTrack.hh"
#include "L1TTracks.hh"
#include "L1TGeomBase.hh"
#include "L1TDisk.hh"

class L1TBarrel:public L1TGeomBase {

private:
  L1TBarrel(){
  }


public:

  L1TBarrel(double rmin,double rmax,double zmax, int NSector){
    rmin_=rmin;
    rmax_=rmax;
    zmax_=zmax;
    NSector_=NSector;
    stubs_.resize(NSector);
    tracklets_.resize(NSector);
    tracks_.resize(NSector);
  }

  bool addStub(const L1TStub& aStub){
    if (aStub.r()<rmin_||aStub.r()>rmax_||fabs(aStub.z())>zmax_) return false;
    double phi=aStub.phi();
    if (phi<0) phi+=two_pi;
    int nSector=NSector_*phi/two_pi;
    assert(nSector>=0);
    assert(nSector<NSector_);
    
    L1TStub tmp=aStub;
    tmp.lorentzcor(-40.0/10000.0);

    stubs_[nSector].push_back(tmp);
    return true;
  }

  void findTracklets(L1TBarrel* L, bool hermetic=false){

    //return;

    for(int iSector=0;iSector<NSector_;iSector++){
      for (int offset=-1;offset<2;offset++) {
	int jSector=iSector+offset;
	if (jSector<0) jSector+=NSector_;
	if (jSector>=NSector_) jSector-=NSector_;
	for (unsigned int i=0;i<stubs_[iSector].size();i++) {
	  for (unsigned int j=0;j<L->stubs_[jSector].size();j++) {
	    if (hermetic&&(stubs_[iSector][i].ladder()!=L->stubs_[jSector][j].ladder())) continue;
	    //cout << "ladders"<< stubs_[iSector][i].ladder()<<" "
	    //	 << L->stubs_[jSector][j].ladder()<<endl;
	    //cout << "r1 phi1 r2 phi2:"
	    //  <<stubs_[iSector][i].r()<<" "
	    //  <<stubs_[iSector][i].phi()<<" "
	    //  <<L->stubs_[jSector][j].r()<<" "
	    //  <<L->stubs_[jSector][j].phi()<<endl;
	    double r1=stubs_[iSector][i].r();
	    double z1=stubs_[iSector][i].z();
	    double phi1=stubs_[iSector][i].phi();

	    double r2=L->stubs_[jSector][j].r();
	    double z2=L->stubs_[jSector][j].z();
	    double phi2=L->stubs_[jSector][j].phi();
	    
	    double deltaphi=phi1-phi2;

	    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	    assert(fabs(deltaphi)<0.5*two_pi);


	    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
        
	    double rinv=2*sin(deltaphi)/dist;

	    double phi0=phi1+asin(0.5*r1*rinv);

	    if (phi0>0.5*two_pi) phi0-=two_pi;
	    if (phi0<-0.5*two_pi) phi0+=two_pi;
	    assert(fabs(phi0)<0.5*two_pi);

	    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
	    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
	    double t=(z1-z2)/(rhopsi1-rhopsi2);

	    double z0=z1-t*rhopsi1;

	    if (stubs_[iSector][i].sigmaz()>1.0) {
	      if (fabs(z1-z2)<10.0){
		z0=0.0;
		t=z1/rhopsi1;
	      }
	    }

	    if (fabs(z0)>30.0) continue;
	    if (fabs(rinv)>0.0057) continue;

	    double pt1=stubs_[iSector][i].pt();
	    double pt2=L->stubs_[jSector][j].pt();
	    double pttracklet=0.3*3.8/(rinv*100);
	    bool pass1=fabs(1.0/pt1-1.0/pttracklet)<0.25;
	    bool pass2=fabs(1.0/pt2-1.0/pttracklet)<0.25;
	    bool pass=pass1&&pass2;

	    if (0) {
	      static ofstream out("ptmatch.txt");
	      out << pt1<<" "
	          << pt2<<" "
	          << pttracklet<<" "
	          << pass
		  << endl;
	    }

	    if (!pass) continue;

	    L1TTracklet tracklet(rinv,phi0,t,z0);
	    tracklet.addStub(stubs_[iSector][i]);
	    tracklet.addStub(L->stubs_[jSector][j]);

	    tracklets_[iSector].push_back(tracklet);

	    if (0) {
	      static ofstream out("barreltracklets.txt");
	      out << iSector<<" "<<stubs_[iSector].size()<<" "
		  << jSector<<" "<<stubs_[jSector].size()<<" "
		  << i<<" "<<j<<" "<<z0<<" "<<rinv
		  << endl;
	    }

	    //cout << "rinv phi0 t z0:"<<
	    //  rinv<<" "<<phi0<<" "<<t<<" "<<z0<<endl;

	  }
	}
      }
    }
  }

  void findTracklets(L1TDisk* D){

    //return;

    for(int iSector=0;iSector<NSector_;iSector++){
      for (int offset=-1;offset<2;offset++) {
	int jSector=iSector+offset;
	if (jSector<0) jSector+=NSector_;
	if (jSector>=NSector_) jSector-=NSector_;
	for (unsigned int i=0;i<stubs_[iSector].size();i++) {
	  for (unsigned int j=0;j<D->stubs_[jSector].size();j++) {

	    double r1=stubs_[iSector][i].r();
	    double z1=stubs_[iSector][i].z();
	    double phi1=stubs_[iSector][i].phi();

	    double r2=D->stubs_[jSector][j].r();
	    double z2=D->stubs_[jSector][j].z();
	    double phi2=D->stubs_[jSector][j].phi();

	    if (r2>80.0) continue;

	    double deltaphi=phi1-phi2;

	    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	    assert(fabs(deltaphi)<0.5*two_pi);


	    double dist=sqrt(r2*r2+r1*r1-2*r1*r2*cos(deltaphi));
        
	    double rinv=2*sin(deltaphi)/dist;
	    
	    double phi0=phi1+asin(0.5*r1*rinv);

	    if (phi0>0.5*two_pi) phi0-=two_pi;
	    if (phi0<-0.5*two_pi) phi0+=two_pi;
	    if (!(fabs(phi0)<0.5*two_pi)) {
	      cout << "phi0 phi1 r1 rinv : "<<phi0<<" "<<phi1<<" "<<r1<<" "<<rinv<<endl;
	      continue;
	    }

	    double rhopsi1=2*asin(0.5*r1*rinv)/rinv;
	    
	    double rhopsi2=2*asin(0.5*r2*rinv)/rinv;
    
	    double t=(z1-z2)/(rhopsi1-rhopsi2);

	    double z0=z1-t*rhopsi1;

	    if (stubs_[iSector][i].sigmaz()>1.0) {
	      if (fabs(z1-z2)<10.0){
		z0=0.0;
		t=z1/rhopsi1;
	      }
	    }

	    if (fabs(z0)>30.0) continue;
	    if (fabs(rinv)>0.0057) continue;

	    L1TTracklet tracklet(rinv,phi0,t,z0);
	    tracklet.addStub(stubs_[iSector][i]);
	    tracklet.addStub(D->stubs_[jSector][j]);

	    tracklets_[iSector].push_back(tracklet);

	    //cout << "rinv phi0 t z0:"<<
	    //  rinv<<" "<<phi0<<" "<<t<<" "<<z0<<endl;

	  }
	}
      }
    }
  }

  void findMatches(L1TBarrel* L,double cutrphi, double cutrz){

    double scale=1.0;

    cutrphi*=scale;
    cutrz*=scale;
    

    for(int iSector=0;iSector<NSector_;iSector++){
      for (unsigned int i=0;i<tracklets_[iSector].size();i++) {
	L1TTracklet& aTracklet=tracklets_[iSector][i];
	double rinv=aTracklet.rinv();
	double phi0=aTracklet.phi0();
	double z0=aTracklet.z0();
	double t=aTracklet.t();

	L1TStub tmp;
	double distbest=2e30;

	for (int offset=-1;offset<2;offset++) {
	  int jSector=iSector+offset;
	  if (jSector<0) jSector+=NSector_;
	  if (jSector>=NSector_) jSector-=NSector_;
	  if (L->stubs_[jSector].size()==0) continue;
	  

	  double rapprox=L->stubs_[jSector][0].r();

	  double phiprojapprox=phi0-asin(0.5*rapprox*rinv);
	  double zprojapprox=z0+2*t*asin(0.5*rapprox*rinv)/rinv;
	  if (phiprojapprox-L->stubs_[jSector][0].phi()<-0.5*two_pi) phiprojapprox+=two_pi;  
	  if (phiprojapprox-L->stubs_[jSector][0].phi()>0.5*two_pi) phiprojapprox-=two_pi;  

	  for (unsigned int j=0;j<L->stubs_[jSector].size();j++) {
	    double z=L->stubs_[jSector][j].z();
	    if (fabs(z-zprojapprox)>10.0) continue;
	    double phi=L->stubs_[jSector][j].phi();
	    double deltaphiapprox=fabs(phi-phiprojapprox);
	    assert(deltaphiapprox<12.0);
	    if (deltaphiapprox*rapprox>5.0) continue;
	    double r=L->stubs_[jSector][j].r();
	    //cout << "r1 phi1 r2 phi2:"
	    //  <<stubs_[iSector][i].r()<<" "
	    //  <<stubs_[iSector][i].phi()<<" "
	    //  <<L->stubs_[jSector][j].r()<<" "
	    //  <<L->stubs_[jSector][j].phi()<<endl;

	    
	    double phiproj=phi0-asin(0.5*r*rinv);
	    double zproj=z0+2*t*asin(0.5*r*rinv)/rinv;

	    double deltaphi=phi-phiproj;
	    //cout << "deltaphi phi phiproj:"<<deltaphi<<" "<<phi<<" "<<phiproj<<" "<<phi0<<" "<<asin(0.5*r*rinv)<<endl;

	    if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	    if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	    assert(fabs(deltaphi)<0.5*two_pi);

	    double rdeltaphi=r*deltaphi;
            double deltaz=z-zproj;

	    if (0) {
	      static ofstream out("barrelmatch.txt");
	      out << aTracklet.r()<<" "<<r<<" "<<rdeltaphi<<" "<<deltaz
		       <<endl;
	    }
	    if (fabs(rdeltaphi)>cutrphi) continue;
	    if (fabs(deltaz)>cutrz) continue;

	    double dist=hypot(rdeltaphi/cutrphi,deltaz/cutrz);

	    if (dist<distbest){
	      tmp=L->stubs_[jSector][j];
	      distbest=dist;
	    }

	    //cout << "rdeltaphi deltaz:"<<rdeltaphi<<" "<<deltaz<<endl;
	    
	  }
	}
	if (distbest<1e30) {
	  tracklets_[iSector][i].addStub(tmp);
	}
      }
    }
  }


  void findMatches(L1TDisk* D){

    for(int iSector=0;iSector<NSector_;iSector++){
      for (unsigned int i=0;i<tracklets_[iSector].size();i++) {
	L1TTracklet& aTracklet=tracklets_[iSector][i];
	double rinv=aTracklet.rinv();
	double phi0=aTracklet.phi0();
	double z0=aTracklet.z0();
	double t=aTracklet.t();
	
	L1TStub tmp;
	double distbest=2e30;

	for (int offset=-1;offset<2;offset++) {
	  int jSector=iSector+offset;
	  if (jSector<0) jSector+=NSector_;
	  if (jSector>=NSector_) jSector-=NSector_;
	  for (unsigned int j=0;j<D->stubs_[jSector].size();j++) {
	    double r=D->stubs_[jSector][j].r();
	    double z=D->stubs_[jSector][j].z();
	    //skip stub if on disk
	    if (fabs(z-aTracklet.getStubs()[1].z())<5.0) continue;
	    double phi=D->stubs_[jSector][j].phi();
	    //cout << "r1 phi1 r2 phi2:"
	    //  <<stubs_[iSector][i].r()<<" "
	    //  <<stubs_[iSector][i].phi()<<" "
	    //  <<L->stubs_[jSector][j].r()<<" "
	    //  <<L->stubs_[jSector][j].phi()<<endl;

	    double r_track=2.0*sin(0.5*rinv*(z-z0)/t)/rinv;
	    double phi_track=phi0-0.5*rinv*(z-z0)/t;

	    int iphi=D->stubs_[jSector][j].iphi();
	    double width=4.608;
	    double nstrip=508.0;
	    if (r<60.0) {
	      width=4.8;
	      nstrip=480;
	    }
	    double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...
	    if (z>0.0) Deltai=-Deltai;
	    

	    double theta0=asin(Deltai/r);

	    double Delta=Deltai-r_track*sin(theta0-(phi_track-phi));



	    //double phiturn=0.5*(z-z0)*rinv/t;
	    //if (fabs(phiturn)>0.25*two_pi) continue;

	    //double phiproj=phi0-phiturn;
	    double rproj=2.0*sin(0.5*(z-z0)*rinv/t)/rinv;

	    //double deltaphi=phi-phiproj;
	    //cout << "deltaphi phi phiproj:"<<deltaphi<<" "<<phi<<" "<<phiproj<<" "<<phi0<<" "<<asin(0.5*r*rinv)<<endl;

	    //if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	    //if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	    //if (!(fabs(deltaphi)<0.5*two_pi)) {
	    //  cout << "deltaphi: "<<deltaphi<<" "<<phi<<" "<<phiproj<<endl;
	    //  cout << "z z0: "<<z<<" "<<z0<<endl;
	    //  cout << "phi0 rinv t: "<<phi0<<" "<<rinv<<" "<<t<<endl;
	    //  continue;
	    //}
	    //assert(fabs(deltaphi)<0.5*two_pi);

	    double rdeltaphi=Delta;
            double deltar=r-rproj;


	    if (fabs(rdeltaphi)>0.2) continue;
	    if (fabs(deltar)>3.0) continue;
	    
	    double dist=hypot(rdeltaphi/0.2,deltar/3.0);
	    
	    if (dist<distbest){
	      tmp=D->stubs_[jSector][j];
	      distbest=dist;
	    }
	    
	  }
	}
	if (distbest<1e30) {
	  tracklets_[iSector][i].addStub(tmp);
	}
      }
    }
  }



private:

  double rmin_;
  double rmax_;
  double zmax_;

};



#endif



