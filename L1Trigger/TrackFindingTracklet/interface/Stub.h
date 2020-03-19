#ifndef STUB_H
#define STUB_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include "L1TStub.h"

#include "FPGAWord.h"
#include "Util.h"
#include "Constants.h"

using namespace std;

class Stub{

public:

  Stub() {
  
  }
  

  Stub(const L1TStub& stub,double phiminsec, double phimaxsec) {

    double r=stub.r();
    double z=stub.z();
    double sbend = stub.bend();

    isPSmodule_ = false;
    if (stub.isPSmodule()) isPSmodule_=true;

    
    int ibend=bendencode(sbend,isPSmodule_);

    int bendbits=3;
    if (!isPSmodule_) bendbits=4;
    
    bend_.set(ibend,bendbits,true,__LINE__,__FILE__);
    
    int layer = stub.layer()+1; 


    // hold the real values from L1Stub	
    double stubphi=stub.phi();

    if (layer<999) {

      disk_.set(0,4,false,__LINE__,__FILE__);

      assert(layer>=1&&layer<=6);
      double rmin=rmean[layer-1]-drmax;
      double rmax=rmean[layer-1]+drmax;

      if (r<rmin||r>rmax) cout << "Error r, rmin, rmeas,  rmax :"<<r
			       <<" "<<rmin<<" "<<0.5*(rmin+rmax)<<" "<<rmax<<endl;

      int irbits=nbitsrL123;
      if (layer>=4) irbits=nbitsrL456;
      
      int ir=round_int((1<<irbits)*((r-rmean[layer-1])/(rmax-rmin)));


      double zmin=-zlength;
      double zmax=zlength;
    
      if (z<zmin||z>zmax) cout << "Error z, zmin, zmax :"<<z
			     <<" "<<zmin<<" "<<zmax<<endl;
    
      int izbits=nbitszL123;
      if (layer>=4) izbits=nbitszL456;
      
      int iz=round_int((1<<izbits)*z/(zmax-zmin));

      if (z<zmin||z>zmax) cout << "Error z, zmin, zmax :"<<z
			       <<" "<<zmin<<" "<<zmax<<endl;
      
      assert(phimaxsec-phiminsec>0.0);

      if (stubphi<phiminsec-(phimaxsec-phiminsec)/6.0) {
	stubphi+=2*M_PI;
      }
      assert((phimaxsec-phiminsec)>0.0);

      int iphibits=nbitsphistubL123;
      if (layer>=4) iphibits=nbitsphistubL456;

      double deltaphi=Util::phiRange(stubphi-phiminsec);
      
      int iphi=(1<<iphibits)*deltaphi/(phimaxsec-phiminsec);

      layer_.set(layer-1,3,true,__LINE__,__FILE__);
      r_.set(ir,irbits,false,__LINE__,__FILE__);
      z_.set(iz,izbits,false,__LINE__,__FILE__);
      phi_.set(iphi,iphibits,true,__LINE__,__FILE__);


      phicorr_.set(iphi,iphibits,true,__LINE__,__FILE__);

    } else {
      
      // Here we handle the hits on disks.

      int disk=stub.module();
      assert(disk>=1&&disk<=5);
      int sign=1;
      if (z<0.0) sign=-1;

      double zmin=sign*(zmean[disk-1]-sign*dzmax);
      double zmax=sign*(zmean[disk-1]+sign*dzmax);

      if ((z>zmax)||(z<zmin)) {
	cout << "Error disk z, zmax, zmin: "<<z<<" "<<zmax<<" "<<zmin<<endl;
      }

      int iz=(1<<nzbitsdisk)*((z-sign*zmean[disk-1])/fabs(zmax-zmin));

      assert(phimaxsec-phiminsec>0.0);
      if (stubphi<phiminsec-(phimaxsec-phiminsec)/6.0) {
	stubphi+=2*M_PI;
      }

      assert(phimaxsec-phiminsec>0.0);
      if (stubphi<phiminsec-(phimaxsec-phiminsec)/6.0) {
	stubphi+=2*M_PI;
      }

      int iphibits=nbitsphistubL123;

      double deltaphi=Util::phiRange(stubphi-phiminsec);
      
      int iphi=(1<<iphibits)*deltaphi/(phimaxsec-phiminsec);
      
      double rmin=0;
      double rmax=rmaxdisk;
    
      if (r<rmin||r>rmax) cout << "Error disk r, rmin, rmax :"<<r
			     <<" "<<rmin<<" "<<rmax<<endl;
    
      int ir=(1<<nrbitsdisk)*(r-rmin)/(rmax-rmin);

      int irSS = -1;
      if (!isPSmodule_) {
	for (int i=0; i<10; ++i){
	  if (disk<=2) {
	    if (fabs(r-rDSSinner[i])<.2){
	      irSS = i;
	      break;
	    }
	  } else {
	    if (fabs(r-rDSSouter[i])<.2){
	      irSS = i;
	      break;
	    }
	  }
	}
	if (irSS<0) {
	  cout << "WARNING! didn't find rDSS value! r = " << r << endl;
	  assert(0);
	}
      }
      if(irSS < 0){
	//PS modules
	r_.set(ir,nrbitsdisk,true,__LINE__,__FILE__);
      }
      else {
	//SS modules
	r_.set(irSS,4,true,__LINE__,__FILE__);  // in case of SS modules, store index, not r itself
      }

      z_.set(iz,nzbitsdisk,false,__LINE__,__FILE__);
      phi_.set(iphi,iphibits,true,__LINE__,__FILE__);
      phicorr_.set(iphi,iphibits,true,__LINE__,__FILE__);
      
      disk_.set(sign*disk,4,false,__LINE__,__FILE__);    


      double alphanew=stub.alphanew();
      assert(fabs(alphanew)<1.0);
      int ialphanew=alphanew*(1<<(nbitsalpha-1));
      assert(ialphanew<(1<<(nbitsalpha-1)));
      assert(ialphanew>=-(1<<(nbitsalpha-1)));
      alphanew_.set(ialphanew,nbitsalpha,false,__LINE__,__FILE__);

      
    }

  }

 
  ~Stub() {

  }


  //Returns a number from 0 to 31
  unsigned int iphivmRaw() const {
    unsigned int iphivm=(phicorr_.value()>>(phicorr_.nbits()-5));
    assert(iphivm<32);
    return iphivm;
    
  }

  FPGAWord iphivmFineBins(int VMbits, int finebits) const {

    unsigned int finephi=(phicorr_.value()>>(phicorr_.nbits()-VMbits-finebits))&((1<<finebits)-1);

    return FPGAWord(finephi,finebits,true,__LINE__,__FILE__);

  }


  //Returns a number from 0 to 31
  int iphivmRawPlus() const {

    int iphivm=((phi_.value()+(1<<(phi_.nbits()-7)))>>(phi_.nbits()-5));
    if (iphivm<0) iphivm=0;
    if (iphivm>31) iphivm=0;
    return iphivm;
    
  }

  int iphivmRawMinus() const {

    int iphivm=((phi_.value()-(1<<(phi_.nbits()-7)))>>(phi_.nbits()-5));
    if (iphivm<0) iphivm=0;
    if (iphivm>31) iphivm=0;
    return iphivm;
    
  }
  
  std::string str() const {
    
    std::ostringstream oss;
    if (layer_.value()!=-1) {
      oss << r_.str()<<"|"
	  << z_.str()<<"|"<< phi_.str()<<"|"<<bend_.str();
    } else {
      if (isPSmodule())
	oss <<r_.str()<<"|"
	    << z_.str()<<"|"<< phi_.str()<<"|"<<bend_.str();
      else
	oss << "000"<<r_.str()<<"|"
	    << z_.str()<<"|"<< phi_.str()<<"|"<<alphanew_.str()<<"|"<<bend_.str();
    }
      
    return oss.str();

  }
  
  std::string strbare() const {
    
    std::ostringstream oss;
    oss << bend_.str()<<r_.str()
	<< z_.str()<< phi_.str();

    return oss.str();

  }


  std::string phiregionaddressstr() {
    assert(phiregion().value()>-1);
    return phiregion().str()+stubindex_.str();	
  }

	
  FPGAWord phiregion() const {
    // 3 bits
    if (layer_.value()>=0) {
      unsigned int nallstubs=nallstubslayers[layer_.value()];
      int iphiregion=iphivmRaw()/(32/nallstubs);
      FPGAWord phi;
      phi.set(iphiregion,3);
      return phi;
    }
    if (abs(disk_.value())>=1) {
      unsigned int nallstubs=nallstubsdisks[abs(disk_.value())-1];
      int iphiregion=iphivmRaw()/(32/nallstubs);
      FPGAWord phi;
      phi.set(iphiregion,3);
      return phi;
    }
    assert(0);     
  }

 
  void setAllStubIndex(int nstub){
    if (nstub>=(1<<7)){
      if (debug1) cout << "Warning too large stubindex!"<<endl;
      nstub=(1<<7)-1;
    }

    stubindex_.set(nstub,7);
  }

  void setPhiCorr(int phiCorr){

    int iphicorr=phi_.value()-phiCorr;
    
    if (iphicorr<0) iphicorr=0;
    if (iphicorr>=(1<<phi_.nbits())) iphicorr=(1<<phi_.nbits())-1;

    phicorr_.set(iphicorr,phi_.nbits(),true,__LINE__,__FILE__);

  }

 

  FPGAWord bend() const {return bend_; }

  FPGAWord r() const { return r_; }
  FPGAWord z() const { return z_; }
  FPGAWord phi() const { return phi_; }
  FPGAWord phicorr() const { return phicorr_; }
  FPGAWord alphanew() const { return alphanew_; }


  int ir() const { return r_.value(); }
  int iz() const { return z_.value(); }
  int iphi() const { return phi_.value(); }

  FPGAWord stubindex() const {return stubindex_;}

  FPGAWord layer() const {return layer_;}

  FPGAWord disk() const {return disk_;}

  bool isBarrel() const {return layer_.value()!=-1;}
  bool isDisk() const {return disk_.value()!=0;}

  bool isPSmodule() const {return isPSmodule_;}

  int round_int( double r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5); 
  }

  double rapprox() const {
    if (disk_.value()==0){
      int lr=1<<(8-nbitsrL123);
      if (layer_.value()>=3) {
	lr=1<<(8-nbitsrL456);
      }
      return r_.value()*kr*lr+rmean[layer_.value()];
    }
    return r_.value()*kr;
  }

  double zapprox() const {
    if (disk_.value()==0){
      int lz=1;
      if (layer_.value()>=3) {
	lz=16;
      }
      return z_.value()*kz*lz;
    }
    int sign=1;
    if (disk_.value()<0) sign=-1;
    if (sign<0) {
      return (z_.value()+1)*kz+sign*zmean[abs(disk_.value())-1];  //Not sure why this is needed to get agreement with integer calculations
    } else {
      return z_.value()*kz+sign*zmean[abs(disk_.value())-1];
    }
    //return z_.value()*kz+sign*zmean[abs(disk_.value())-1];
  }

  double phiapprox(double phimin, double) const {
    int lphi=1;
    if (layer_.value()>=3) {
      lphi=8;
    }
    return Util::phiRange(phimin+phi_.value()*kphi/lphi);
  }

  void setfiner(int finer) {
   finer_.set(finer,4,true,__LINE__,__FILE__);
  }

  FPGAWord finer() const {
    return finer_;
  }

  void setfinez(int finez) {
   finez_.set(finez,4,true,__LINE__,__FILE__);
  }

  FPGAWord finez() const {
    return finez_;
  }


  //Should be optimized by layer - now first implementation to make sure
  //it works OK
  static int bendencode(double bend, bool isPS) {
    
    int ibend=2.0*bend;

    assert(fabs(ibend-2.0*bend)<0.1);
    
    if (isPS) {

      if (ibend==0||ibend==1) return 0;
      if (ibend==2||ibend==3) return 1;
      if (ibend==4||ibend==5) return 2;
      if (ibend>=6) return 3;
      if (ibend==-1||ibend==-2) return 4;
      if (ibend==-3||ibend==-4) return 5;
      if (ibend==-5||ibend==-6) return 6;
      if (ibend<=-7) return 7;

      assert(0);

    }


    if (ibend==0||ibend==1) return 0;
    if (ibend==2||ibend==3) return 1;
    if (ibend==4||ibend==5) return 2;
    if (ibend==6||ibend==7) return 3;
    if (ibend==8||ibend==9) return 4;
    if (ibend==10||ibend==11) return 5;
    if (ibend==12||ibend==13) return 6;
    if (ibend>=14) return 7;
    if (ibend==-1||ibend==-2) return 8;
    if (ibend==-3||ibend==-4) return 9;
    if (ibend==-5||ibend==-6) return 10;
    if (ibend==-7||ibend==-8) return 11;
    if (ibend==-9||ibend==-10) return 12;
    if (ibend==-11||ibend==-12) return 13;
    if (ibend==-13||ibend==-14) return 14;
    if (ibend<=-15) return 15;

    cout << "bend ibend : "<<bend<<" "<<ibend<<endl;
    
    assert(0);
    
    
  }
  
  //Should be optimized by layer - now first implementation to make sure
  //it works OK
  static double benddecode(int ibend, bool isPS) {

    if (isPS) {

      if (ibend==0) return 0.25;
      if (ibend==1) return 1.25;
      if (ibend==2) return 2.25;
      if (ibend==3) return 3.25;
      if (ibend==4) return -0.75;
      if (ibend==5) return -1.75;
      if (ibend==6) return -2.75;
      if (ibend==7) return -3.75;

      assert(0);

    }

    if (ibend==0) return 0.25;
    if (ibend==1) return 1.25;
    if (ibend==2) return 2.25;
    if (ibend==3) return 3.25;
    if (ibend==4) return 4.25;
    if (ibend==5) return 5.25;
    if (ibend==6) return 6.25;
    if (ibend==7) return 7.25;
    if (ibend==8) return -0.75;
    if (ibend==9) return -1.75;
    if (ibend==10) return -2.75;
    if (ibend==11) return -3.75;
    if (ibend==12) return -4.75;
    if (ibend==13) return -5.75;
    if (ibend==14) return -6.75;
    if (ibend==15) return -7.75;

    assert(0);
    
    
  }
  
private:

  bool isPSmodule_;  //FIXME can be removed
  FPGAWord layer_;  
  FPGAWord disk_;  
  FPGAWord r_;
  FPGAWord z_;
  FPGAWord phi_;
  FPGAWord alphanew_;

  FPGAWord bend_;
  
  FPGAWord phicorr_;  //Corrected for bend to nominal radius
  
  FPGAWord stubindex_;

  FPGAWord finer_;   //FIXME should not be member data
  FPGAWord finez_;   //FIXME should not be member data
  


};



#endif



