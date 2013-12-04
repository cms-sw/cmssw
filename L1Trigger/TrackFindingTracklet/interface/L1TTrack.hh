#ifndef L1TTRACK_H
#define L1TTRACK_H

#include <iostream>
#include <assert.h>

#include "L1TConstants.hh"
#include "L1TWord.hh"

using namespace std;

class L1TTrack{

public:

  L1TTrack() { };

  L1TTrack(const L1TTracklet& seed) {

    seed_=seed;
    rinv_=seed.rinv();
    phi0_=seed.phi0();
    z0_=seed.z0();
    t_=seed.t();
    stubs_=seed.getStubs();

    double largestresid;
    int ilargestresid;

    for (int i=0;i<1;i++){

      if (i>0) {
	rinv_=rinvfit_;
	phi0_=phi0fit_;
	z0_=z0fit_;
	t_=tfit_;
      }

      calculateDerivatives();
      
      linearTrackFit();
     

    }

    largestresid=-1.0;
    ilargestresid=-1;

    residuals(largestresid,ilargestresid);

    //cout << "Chisq largestresid: "<<chisq()<<" "<<largestresid<<endl;

    if (stubs_.size()>3&&chisq()>100.0&&largestresid>5.0) {
      //cout << "Refitting track"<<endl;
      stubs_.erase(stubs_.begin()+ilargestresid);
      rinv_=rinvfit_;
      phi0_=phi0fit_;
      z0_=z0fit_;
      t_=tfit_;
      calculateDerivatives();
      linearTrackFit();
      residuals(largestresid,ilargestresid);
    }


  }


  void invert(double M[4][8],unsigned int n){

    assert(n<=4);

    unsigned int i,j,k;
    double ratio,a;

    for(i = 0; i < n; i++){
      for(j = n; j < 2*n; j++){
	if(i==(j-n))
	  M[i][j] = 1.0;
	else
	  M[i][j] = 0.0;
      }
    }

    for(i = 0; i < n; i++){
      for(j = 0; j < n; j++){
	if(i!=j){
	  ratio = M[j][i]/M[i][i];
	  for(k = 0; k < 2*n; k++){
	    M[j][k] -= ratio * M[i][k];
	  }
	}
      }
    }

    for(i = 0; i < n; i++){
      a = M[i][i];
      for(j = 0; j < 2*n; j++){
	M[i][j] /= a;
      }
    }
  }



  void calculateDerivatives(){

    unsigned int n=stubs_.size();

    assert(n<=20);


    int j=0;

    for(unsigned int i=0;i<n;i++) {

      double ri=stubs_[i].r();
      double zi=stubs_[i].z();

      double sigmax=stubs_[i].sigmax();
      double sigmaz=stubs_[i].sigmaz();


      //cout << "i layer "<<i<<" "<<stubs_[i].layer()<<endl;

      if (stubs_[i].layer()<1000){
	//here we handle a barrel hit

	//first we have the phi position
	D_[0][j]=-0.5*ri*ri/sqrt(1-0.25*ri*ri*rinv_*rinv_)/sigmax;
	D_[1][j]=ri/sigmax;
	D_[2][j]=0.0;
	D_[3][j]=0.0;
	j++;
	//second the z position
	D_[0][j]=0.0;
	D_[1][j]=0.0;
	D_[2][j]=(2/rinv_)*asin(0.5*ri*rinv_)/sigmaz;
	D_[3][j]=1.0/sigmaz;
	j++;
      }
      else {
	//here we handle a disk hit
	//first we have the r position

	//int iphi=stubs_[i].iphi();
	//double phistub=(5.0/ri)*(iphi-508)/508.0;  //A bit of a hack...

	//cout << "iphi phistub rphistub: "<<100*stubs_[i].layer()+stubs_[i].module()<<" "<<iphi<<" "<<phistub<<" "
	//     << ri*phistub<<" "<<stubs_[i].r()<<" "<<stubs_[i].phi()<<endl;

	double r_track=2.0*sin(0.5*rinv_*(zi-z0_)/t_)/rinv_;
	double phi_track=phi0_-0.5*rinv_*(zi-z0_)/t_;

	int iphi=stubs_[i].iphi();
	double phii=stubs_[i].phi();

	double width=4.608;
	double nstrip=508.0;
	if (ri<60.0) {
	  width=4.8;
	  nstrip=480;
	}
	double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...

	if (stubs_[i].z()>0.0) Deltai=-Deltai;
	double theta0=asin(Deltai/ri);

	double rmultiplier=-sin(theta0-(phi_track-phii));
	double phimultiplier=r_track*cos(theta0-(phi_track-phii));


	double drdrinv=-2.0*sin(0.5*rinv_*(zi-z0_)/t_)/(rinv_*rinv_)
			+(zi-z0_)*cos(0.5*rinv_*(zi-z0_)/t_)/(rinv_*t_);
	double drdphi0=0;
	double drdt=-(zi-z0_)*cos(0.5*rinv_*(zi-z0_)/t_)/(t_*t_);
	double drdz0=-cos(0.5*rinv_*(zi-z0_)/t_)/t_;

	double dphidrinv=-0.5*(zi-z0_)/t_;
	double dphidphi0=1.0;
	double dphidt=0.5*rinv_*(zi-z0_)/(t_*t_);
	double dphidz0=0.5*rinv_/t_;
	
	D_[0][j]=drdrinv/sigmaz;
	D_[1][j]=drdphi0/sigmaz;
	D_[2][j]=drdt/sigmaz;
	D_[3][j]=drdz0/sigmaz;
	j++;
	//second the rphi position
	D_[0][j]=(phimultiplier*dphidrinv+rmultiplier*drdrinv)/sigmax;
	D_[1][j]=(phimultiplier*dphidphi0+rmultiplier*drdphi0)/sigmax;
	D_[2][j]=(phimultiplier*dphidt+rmultiplier*drdt)/sigmax;
	D_[3][j]=(phimultiplier*dphidz0+rmultiplier*drdz0)/sigmax;
        //old calculation
	//D_[0][j]=-0.5*(zi-z0_)/(t_*(sigmax/ri));
	//D_[1][j]=1.0/(sigmax/ri);
	//D_[2][j]=-0.5*rinv_*(zi-z0_)/(t_*t_*(sigmax/ri));
	//D_[3][j]=0.5*rinv_/((sigmax/ri)*t_);
	j++;
      }

      //cout << "Exact rinv derivative: "<<i<<" "<<D_[0][j-2]<<" "<<D_[0][j-1]<<endl;
      //cout << "Exact phi0 derivative: "<<i<<" "<<D_[1][j-2]<<" "<<D_[1][j-1]<<endl;
      //cout << "Exact t derivative   : "<<i<<" "<<D_[2][j-2]<<" "<<D_[2][j-1]<<endl;
      //cout << "Exact z0 derivative  : "<<i<<" "<<D_[3][j-2]<<" "<<D_[3][j-1]<<endl;
	
	
    }
    
    //cout << "D:"<<endl;
    //for(unsigned int j=0;j<2*n;j++){
    //  cout <<D_[0][j]<<" "<<D_[1][j]<<" "<<D_[2][j]<<" "<<D_[3][j]<<endl;
    //}

     



    for(unsigned int i1=0;i1<4;i1++){
      for(unsigned int i2=0;i2<4;i2++){
	M_[i1][i2]=0.0;
	for(unsigned int j=0;j<2*n;j++){
	  M_[i1][i2]+=D_[i1][j]*D_[i2][j];	  
	}
      }
    }

    invert(M_,4);

    for(unsigned int j=0;j<2*n;j++) {
      for(unsigned int i1=0;i1<4;i1++) {
	MinvDt_[i1][j]=0.0;
	for(unsigned int i2=0;i2<4;i2++) {
	  MinvDt_[i1][j]+=M_[i1][i2+4]*D_[i2][j];
	}
      }
    }

  }

  void residuals(double& largestresid,int& ilargestresid) {

    unsigned int n=stubs_.size();

    //Next calculate the residuals

    double delta[40];

    double chisq=0.0;

    unsigned int j=0;

    bool print=false;

    if (print) cout << "Residuals ("<<chisq1_<<") ["<<0.003*3.8/rinvfit_<<"]: ";

    largestresid=-1.0;
    ilargestresid=-1;

    for(unsigned int i=0;i<n;i++) {
      double ri=stubs_[i].r();
      double zi=stubs_[i].z();
      double phii=stubs_[i].phi();
      double sigmax=stubs_[i].sigmax();
      double sigmaz=stubs_[i].sigmaz();

      int layer=stubs_[i].layer();

      if (layer<1000) {
        //we are dealing with a barrel stub

	double deltaphi=phi0fit_-asin(0.5*ri*rinvfit_)-phii;
	if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	assert(fabs(deltaphi)<0.1*two_pi);

	delta[j++]=ri*deltaphi/sigmax;
	delta[j++]=(z0fit_+(2.0/rinvfit_)*tfit_*asin(0.5*ri*rinvfit_)-zi)/sigmaz;
	
      }
      else {
	//we are dealing with a disk hit

	double r_track=2.0*sin(0.5*rinvfit_*(zi-z0fit_)/tfit_)/rinvfit_;
	double phi_track=phi0fit_-0.5*rinvfit_*(zi-z0fit_)/tfit_;

	int iphi=stubs_[i].iphi();

	double width=4.608;
	double nstrip=508.0;
	if (ri<60.0) {
	  width=4.8;
	  nstrip=480;
	}
	double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...

	if (stubs_[i].z()>0.0) Deltai=-Deltai;

	double theta0=asin(Deltai/ri);

	double Delta=Deltai-r_track*sin(theta0-(phi_track-phii));

	delta[j++]=(r_track-ri)/sigmaz;
	delta[j++]=Delta/sigmax;
      }

      if (fabs(delta[j-2])>largestresid) {
	largestresid=fabs(delta[j-2]);
	ilargestresid=i;
      }

      if (fabs(delta[j-1])>largestresid) {
	largestresid=fabs(delta[j-1]);
	ilargestresid=i;
      }
      
      if (print) cout << delta[j-2]<<" "<<delta[j-1]<<" ";

      chisq+=delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1];

    }

    if (print) cout <<" ("<<chisq<<")"<<endl;

  }
  

  void linearTrackFit() {

    unsigned int n=stubs_.size();

    //Next calculate the residuals

    double delta[40];

    double chisq=0;

    unsigned int j=0;

    for(unsigned int i=0;i<n;i++) {
      double ri=stubs_[i].r();
      double zi=stubs_[i].z();
      double phii=stubs_[i].phi();
      double sigmax=stubs_[i].sigmax();
      double sigmaz=stubs_[i].sigmaz();

      int layer=stubs_[i].layer();

      if (layer<1000) {
        //we are dealing with a barrel stub

	double deltaphi=phi0_-asin(0.5*ri*rinv_)-phii;
	if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	assert(fabs(deltaphi)<0.1*two_pi);

	delta[j++]=ri*deltaphi/sigmax;
	delta[j++]=(z0_+(2.0/rinv_)*t_*asin(0.5*ri*rinv_)-zi)/sigmaz;


	//numerical derivative check

	for (int iii=0;iii<0;iii++){

	  double drinv=0.0;
	  double dphi0=0.0;
	  double dt=0.0;
	  double dz0=0.0;

	  if (iii==0) drinv=0.001*fabs(rinv_);
	  if (iii==1) dphi0=0.001;
	  if (iii==2) dt=0.001;
	  if (iii==3) dz0=0.01;

	  double deltaphi=phi0_+dphi0-asin(0.5*ri*(rinv_+drinv))-phii;
	  if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	  if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	  assert(fabs(deltaphi)<0.1*two_pi);

	  double delphi=ri*deltaphi/sigmax;
	  double deltaz=(z0_+dz0+(2.0/(rinv_+drinv))*(t_+dt)*asin(0.5*ri*(rinv_+drinv))-zi)/sigmaz;


	  if (iii==0) cout << "Numerical rinv derivative: "<<i<<" "
			   <<(delphi-delta[j-2])/drinv<<" "
			   <<(deltaz-delta[j-1])/drinv<<endl;

	  if (iii==1) cout << "Numerical phi0 derivative: "<<i<<" "
			   <<(delphi-delta[j-2])/dphi0<<" "
			   <<(deltaz-delta[j-1])/dphi0<<endl;

	  if (iii==2) cout << "Numerical t derivative: "<<i<<" "
			   <<(delphi-delta[j-2])/dt<<" "
			   <<(deltaz-delta[j-1])/dt<<endl;

	  if (iii==3) cout << "Numerical z0 derivative: "<<i<<" "
			   <<(delphi-delta[j-2])/dz0<<" "
			   <<(deltaz-delta[j-1])/dz0<<endl;

	}



      }
      else {
	//we are dealing with a disk hit

	double r_track=2.0*sin(0.5*rinv_*(zi-z0_)/t_)/rinv_;
	//cout <<"t_track 1: "<<r_track<<endl;
	double phi_track=phi0_-0.5*rinv_*(zi-z0_)/t_;

	int iphi=stubs_[i].iphi();

	double width=4.608;
	double nstrip=508.0;
	if (ri<60.0) {
	  width=4.8;
	  nstrip=480;
	}
	double Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...

	if (stubs_[i].z()>0.0) Deltai=-Deltai;

	double theta0=asin(Deltai/ri);

	double Delta=Deltai-r_track*sin(theta0-(phi_track-phii));

	delta[j++]=(r_track-ri)/sigmaz;
	//double deltaphi=phi_track-phii;
	//if (deltaphi>0.5*two_pi) deltaphi-=two_pi;
	//if (deltaphi<-0.5*two_pi) deltaphi+=two_pi;
	//assert(fabs(deltaphi)<0.1*two_pi);
	//delta[j++]=deltaphi/(sigmax/ri);
	delta[j++]=Delta/sigmax;

	//numerical derivative check

	for (int iii=0;iii<0;iii++){

	  double drinv=0.0;
	  double dphi0=0.0;
	  double dt=0.0;
	  double dz0=0.0;

	  if (iii==0) drinv=0.001*fabs(rinv_);
	  if (iii==1) dphi0=0.001;
	  if (iii==2) dt=0.001;
	  if (iii==3) dz0=0.01;

	  r_track=2.0*sin(0.5*(rinv_+drinv)*(zi-(z0_+dz0))/(t_+dt))/(rinv_+drinv);
	  //cout <<"t_track 2: "<<r_track<<endl;
	  phi_track=phi0_+dphi0-0.5*(rinv_+drinv)*(zi-(z0_+dz0))/(t_+dt);
	  
	  iphi=stubs_[i].iphi();

	  double width=4.608;
	  double nstrip=508.0;
	  if (ri<60.0) {
	    width=4.8;
	    nstrip=480;
	  }
	  Deltai=width*(iphi-nstrip)/nstrip;  //A bit of a hack...

	  if (stubs_[i].z()>0.0) Deltai=-Deltai;
	  theta0=asin(Deltai/ri);
	  
	  Delta=Deltai-r_track*sin(theta0-(phi_track-phii));

	  if (iii==0) cout << "Numerical rinv derivative: "<<i<<" "
			   <<((r_track-ri)/sigmaz-delta[j-2])/drinv<<" "
			   <<(Delta/sigmax-delta[j-1])/drinv<<endl;

	  if (iii==1) cout << "Numerical phi0 derivative: "<<i<<" "
			   <<((r_track-ri)/sigmaz-delta[j-2])/dphi0<<" "
			   <<(Delta/sigmax-delta[j-1])/dphi0<<endl;

	  if (iii==2) cout << "Numerical t derivative: "<<i<<" "
			   <<((r_track-ri)/sigmaz-delta[j-2])/dt<<" "
			   <<(Delta/sigmax-delta[j-1])/dt<<endl;

	  if (iii==3) cout << "Numerical z0 derivative: "<<i<<" "
			   <<((r_track-ri)/sigmaz-delta[j-2])/dz0<<" "
			   <<(Delta/sigmax-delta[j-1])/dz0<<endl;

	}

      }

      chisq+=(delta[j-2]*delta[j-2]+delta[j-1]*delta[j-1]);

    }

    double drinv=0.0;
    double dphi0=0.0;
    double dt=0.0;
    double dz0=0.0;

    double drinv_cov=0.0;
    double dphi0_cov=0.0;
    double dt_cov=0.0;
    double dz0_cov=0.0;



    for(unsigned int j=0;j<2*n;j++) {
      drinv-=MinvDt_[0][j]*delta[j];
      //cout << "MinvDt_[0][j] delta[j]:"<<MinvDt_[0][j]<<" "<<delta[j]<<endl;
      dphi0-=MinvDt_[1][j]*delta[j];
      dt-=MinvDt_[2][j]*delta[j];
      dz0-=MinvDt_[3][j]*delta[j];

      drinv_cov+=D_[0][j]*delta[j];
      dphi0_cov+=D_[1][j]*delta[j];
      dt_cov+=D_[2][j]*delta[j];
      dz0_cov+=D_[3][j]*delta[j];
    }
    

    double deltaChisq=drinv*drinv_cov+dphi0*dphi0_cov+dt*dt_cov+dz0*dz0_cov;

    //drinv=0.0; dphi0=0.0; dt=0.0; dz0=0.0;

    rinvfit_=rinv_+drinv;
    phi0fit_=phi0_+dphi0;

    tfit_=t_+dt;
    z0fit_=z0_+dz0;

    chisq1_=(chisq+deltaChisq);
    chisq2_=0.0;

    //cout << "Trackfit:"<<endl;
    //cout << "rinv_ drinv: "<<rinv_<<" "<<drinv<<endl;
    //cout << "phi0_ dphi0: "<<phi0_<<" "<<dphi0<<endl;
    //cout << "t_ dt      : "<<t_<<" "<<dt<<endl;
    //cout << "z0_ dz0    : "<<z0_<<" "<<dz0<<endl;

  }



  bool overlap(const L1TTrack& aTrack) const {
    
    int nSame=0;
    for(unsigned int i=0;i<stubs_.size();i++) {
      for(unsigned int j=0;j<aTrack.stubs_.size();j++) {
	if (stubs_[i]==aTrack.stubs_[j]) nSame++;
      }
    }

    return (nSame>=2);

  }


  int simtrackid(double& fraction) const {

    //cout << "In L1TTrack::simtrackid"<<endl;

    map<int, int> simtrackids;

    for(unsigned int i=0;i<stubs_.size();i++){
      //cout << "Stub simtrackid="<<stubs_[i].simtrackid()<<endl;
      simtrackids[stubs_[i].simtrackid()]++;
    }

    int simtrackid=0;
    int nsimtrack=0;

    map<int, int>::const_iterator it=simtrackids.begin();

    while(it!=simtrackids.end()) {
      //cout << it->first<<" "<<it->second<<endl;
      if (it->second>nsimtrack) {
	nsimtrack=it->second;
	simtrackid=it->first;
      }
      it++;
    }

    //cout << "L1TTrack::simtrackid done"<<endl;

    fraction=(1.0*nsimtrack)/stubs_.size();

    return simtrackid;

  }

  int npixelstrip() const {
    int count=0;
    for (unsigned int i=0;i<stubs_.size();i++){
      if (stubs_[i].sigmaz()<0.5) count++; 
    }
    return count;
  }

  L1TTracklet getSeed() const { return seed_; }
  vector<L1TStub> getStubs() const { return stubs_; }
  unsigned int nstub() const { return stubs_.size(); }
  double rinv() const { return rinv_; }
  double getPhi0() const { return phi0_; }
  double getZ0() const { return z0_; }
  double getT() const { return t_; }
  bool isCombinatorics() const { return isCombinatorics_; }
  double getSimTrackID() const { return SimTrackID_; }

  double pt(double bfield) const { return 0.00299792*bfield/rinvfit_; }
  //double ipt(double bfield) const { return 0.00299792*bfield/irinvfit(); }
  double ptseed(double bfield) const { return 0.00299792*bfield/rinv_; }

  double phi0() const { return phi0fit_;}
  //double iphi0() const { return iphi0fit();}
  double phi0seed() const { return phi0_;}

  double eta() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(tfit_)))); }
  //double ieta() const { static double two_pi=8*atan(1.0);
  //  return -log(tan(0.5*(0.25*two_pi-atan(itfit())))); }
  double etaseed() const { static double two_pi=8*atan(1.0);
    return -log(tan(0.5*(0.25*two_pi-atan(t_)))); }

  double z0() const { return z0fit_; }
  //double iz0() const { return iz0fit(); }
  double z0seed() const { return z0_; }

  double chisq1() const {return chisq1_;}
  double chisq2() const {return chisq2_;}

  double chisq1dof() const {return chisq1_/(stubs_.size()-2);}
  double chisq2dof() const {return chisq2_/(stubs_.size()-2);}
  
  double chisq() const {return chisq1_+chisq2_; }
  double chisqdof() const {return (chisq1_+chisq2_)/(2*stubs_.size()-4); }


private:

  L1TTracklet seed_;
  vector<L1TStub> stubs_;
  double rinv_;
  double phi0_;
  double z0_;
  double t_;
  bool isCombinatorics_;
  int SimTrackID_;
  double rinvfit_;
  double phi0fit_;
  double z0fit_;
  double tfit_;

  int irinvfit_;
  int iphi0fit_;
  int iz0fit_;
  int itfit_;

  double chisq1_;
  double chisq2_;

  int ichisq1_;
  int ichisq2_;

  double D_[4][40];
  
  double M_[4][8];
  
  double MinvDt_[4][40];


};


#endif
