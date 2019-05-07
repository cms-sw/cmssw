#ifndef FPGASTUB_H
#define FPGASTUB_H

#include <iostream>
#include <fstream>
#include <assert.h>
#include <math.h>
#include <vector>
#include "L1TStub.hh"

#include "FPGAWord.hh"
#include "FPGAUtil.hh"
#include "FPGAConstants.hh"

using namespace std;

class FPGAStub{

public:

  FPGAStub() {
  
  }
  

  FPGAStub(const L1TStub& stub,double phiminsec, double phimaxsec) {

    //cout << "FPGASTub : making stub"<<endl;
    
    double r=stub.r();
    double z=stub.z();
    double ptinv=1.0/stub.pt();
    double sbend = stub.bend();

    isPSmodule_ = false;
    if (stub.isPSmodule()) isPSmodule_=true;

    
    int ibend=bendencode(sbend,isPSmodule_);

    int bendbits=3;
    if (!isPSmodule_) bendbits=4;
    
    bend_.set(ibend,bendbits,true,__LINE__,__FILE__);
    
    
    //HACK!!! seems like stubs in negative disk has wrong sign!
    if (z<-120.0) ptinv=-ptinv;
    //cout << "z stub.pt() : "<<z<<" "<<stub.pt()<<endl;

    int layer = stub.layer()+1; 


    // hold the real values from L1Stub	
    stubphi_=stub.phi();
    stubr_  =stub.r();
    stubz_  =stub.z();
    stubrpt_=stub.pt();

    stubphimaxsec_ = phimaxsec;
    stubphiminsec_ = phiminsec;
   

    isbarrel_=false;

    if (layer<999) {

      isbarrel_=true;

      disk_.set(0,4,false,__LINE__,__FILE__);
	  
      double rmin=-1.0;
      double rmax=-1.0;

      if (layer==1) {rmin=rminL1; rmax=rmaxL1;}
      if (layer==2) {rmin=rminL2; rmax=rmaxL2;}
      if (layer==3) {rmin=rminL3; rmax=rmaxL3;}
      if (layer==4) {rmin=rminL4; rmax=rmaxL4;}
      if (layer==5) {rmin=rminL5; rmax=rmaxL5;}
      if (layer==6) {rmin=rminL6; rmax=rmaxL6;}


      assert(rmin>0.0);
      assert(rmax>0.0);
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

      if (z<zmin||z>zmax) cout << "Error phi, phimin, phimax :"<<stubphi_
			       <<" "<<phiminsec<<" "<<phimaxsec<<endl;
      
      assert(phimaxsec-phiminsec>0.0);

      if (stubphi_<phiminsec-(phimaxsec-phiminsec)/6.0) {
	stubphi_+=2*M_PI;
      }
      assert((phimaxsec-phiminsec)>0.0);

      int iphibits=nbitsphistubL123;
      if (layer>=4) iphibits=nbitsphistubL456;

      double deltaphi=FPGAUtil::phiRange(stubphi_-phiminsec);
      
      int iphi=(1<<iphibits)*deltaphi/(phimaxsec-phiminsec);

      phitmp_=stubphi_-phiminsec+(phimaxsec-phiminsec)/6.0;

      phimin_=phiminsec;

      layer_.set(layer-1,3,true,__LINE__,__FILE__);
      r_.set(ir,irbits,false,__LINE__,__FILE__);
      z_.set(iz,izbits,false,__LINE__,__FILE__);
      phi_.set(iphi,iphibits,true,__LINE__,__FILE__);


      phicorr_.set(iphi,iphibits,true,__LINE__,__FILE__);

      
      int iphivm=0;
      
      iphivm=(iphi>>(iphibits-(Nphibits+VMphibits)))&((1<<VMphibits)-1);
      
      if (layer==1||layer==3||layer==5) {
	iphivm^=(1<<(VMphibits-1));
      }

      phivm_.set(iphivm,VMphibits,true,__LINE__,__FILE__);

    } else {
      
      // Here we handle the hits on disks.

      int disk=stub.module();
      assert(disk>0);
      if (z<0.0) disk=-disk;
      int sign=1;
      if (disk<0) sign=-1;

      double zmin=0.0;
      double zmax=0.0;

      if (disk==1) {zmin=zminD1; zmax=zmaxD1;}
      if (disk==2) {zmin=zminD2; zmax=zmaxD2;}
      if (disk==3) {zmin=zminD3; zmax=zmaxD3;}
      if (disk==4) {zmin=zminD4; zmax=zmaxD4;}
      if (disk==5) {zmin=zminD5; zmax=zmaxD5;}

      if (disk==-1) {zmax=-zminD1; zmin=-zmaxD1;}
      if (disk==-2) {zmax=-zminD2; zmin=-zmaxD2;}
      if (disk==-3) {zmax=-zminD3; zmin=-zmaxD3;}
      if (disk==-4) {zmax=-zminD4; zmin=-zmaxD4;}
      if (disk==-5) {zmax=-zminD5; zmin=-zmaxD5;}

      if ((z>zmax)||(z<zmin)) {
	cout << "Error disk z, zmax, zmin: "<<z<<" "<<zmax<<" "<<zmin<<endl;
      }

      int iz=(1<<nzbitsdisk)*((z-sign*zmean[abs(disk)-1])/fabs(zmax-zmin));

      //if (disk<0) iz--;
      
      assert(phimaxsec-phiminsec>0.0);
      if (stubphi_<phiminsec-(phimaxsec-phiminsec)/6.0) {
	stubphi_+=2*M_PI;
      }

      assert(phimaxsec-phiminsec>0.0);
      if (stubphi_<phiminsec-(phimaxsec-phiminsec)/6.0) {
	stubphi_+=2*M_PI;
      }

      int iphibits=nbitsphistubL123;

      double deltaphi=FPGAUtil::phiRange(stubphi_-phiminsec);
      
      int iphi=(1<<iphibits)*deltaphi/(phimaxsec-phiminsec);
      
      double rmin=0;
      double rmax=rmaxdisk;
    
      if (r<rmin||r>rmax) cout << "Error disk r, rmin, rmax :"<<r
			     <<" "<<rmin<<" "<<rmax<<endl;
    
      int ir=(1<<nrbitsdisk)*(r-rmin)/(rmax-rmin);

      int irSS = -1;
      if (!isPSmodule_) {
	for (int i=0; i<10; ++i){
	  if (abs(disk)<=2) {
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
      	//cout << "ir irbits : "<<ir<<" "<<irbits<<endl;
	r_.set(ir,nrbitsdisk,true,__LINE__,__FILE__);
      }
      else {
	//SS modules
	r_.set(irSS,4,true,__LINE__,__FILE__);  // in case of SS modules, store index, not r itself
      }

      //cout << "iz izbits : "<<iz<<" "<<izbits<<" "<<disk<<endl;
      z_.set(iz,nzbitsdisk,false,__LINE__,__FILE__);
      phi_.set(iphi,iphibits,true,__LINE__,__FILE__);

      phicorr_.set(iphi,iphibits,true,__LINE__,__FILE__);    //Should disks also have phi correction? 
      
      int iphivm=0;

      iphivm=(iphi>>(iphibits-(Nphibits+VMphibits)))&((1<<VMphibits)-1);
      
      if ((abs(disk)%2)==0) {
        iphivm^=(1<<(VMphibits-1));
      }


      disk_.set(disk,4,false,__LINE__,__FILE__);    
      phivm_.set(iphivm,3,true,__LINE__,__FILE__);


      double alphanew=stub.alphanew();
      assert(fabs(alphanew)<1.0);
      int ialphanew=alphanew*(1<<(nbitsalpha-1));
      assert(ialphanew<(1<<(nbitsalpha-1)));
      assert(ialphanew>=-(1<<(nbitsalpha-1)));
      alphanew_.set(ialphanew,nbitsalpha,false,__LINE__,__FILE__);

      
    }

  }

 
  ~FPGAStub() {

  }


  //Returns a number from 0 to 31
  int iphivmRaw() const {
    int iphivm=(phicorr_.value()>>(phicorr_.nbits()-5));
    assert(iphivm>=0);
    assert(iphivm<32);
    return iphivm;
    
  }

  //VMbits is the number of bits for the fine bins. E.g. 32 bins would use VMbits=5
  //finebits is the number of bits within the VM 
  
  int iphivmFineBins(int VMbits, int finebits) const {

    return (phicorr_.value()>>(phicorr_.nbits()-VMbits-finebits))&((1<<finebits)-1);

  }


    //Returns a number from 0 to 31
  int iphivmRawPlus() const {

    //cout << layer_.value()<<" "<<disk_.value()<<endl;

    //cout << "bits : "<<phi_.value()<<" "<<(1<<(phi_.nbits()-8))<<endl;
    
    int iphivm=((phi_.value()+(1<<(phi_.nbits()-7)))>>(phi_.nbits()-5));
    if (iphivm<0) iphivm=0;
    if (iphivm>31) iphivm=0;
    return iphivm;
    
  }

  int iphivmRawMinus() const {

    //cout << layer_.value()<<" "<<disk_.value()<<endl;
    
    int iphivm=((phi_.value()-(1<<(phi_.nbits()-7)))>>(phi_.nbits()-5));
    if (iphivm<0) iphivm=0;
    if (iphivm>31) iphivm=0;
    return iphivm;
    
  }
  
  std::string str() const {
    
    std::ostringstream oss;
    oss << r_.str()<<"|"
        << z_.str()<<"|"<< phi_.str()<<"|"<<bend_.str();

    return oss.str();

  }
  std::string strdisk() const {
   
    std::ostringstream oss;
    if (isPSmodule())
      oss <<r_.str()<<"|"
          << z_.str()<<"|"<< phi_.str()<<"|"<<bend_.str();
    else
      oss << "000"<<r_.str()<<"|"
          << z_.str()<<"|"<< phi_.str()<<"|"<<alphanew_.str()<<"|"<<bend_.str();

    return oss.str();

  }

  std::string str_phys() const {

    std::ostringstream oss;
   
    int ilz_   = 1;
    float nbsr = 128.0;
    float nphibs = 16384;
    float rmean_ = rmean[layer_.value() ];
    int Layer = layer_.value() + 1;

    double rmin=1.0;
    double rmax=10.0;

    if (Layer==1) {rmin=rminL1; rmax=rmaxL1;}
    if (Layer==2) {rmin=rminL2; rmax=rmaxL2;}
    if (Layer==3) {rmin=rminL3; rmax=rmaxL3;}
    if (Layer==4) {rmin=rminL4; rmax=rmaxL4;}
    if (Layer==5) {rmin=rminL5; rmax=rmaxL5;}
    if (Layer==6) {rmin=rminL6; rmax=rmaxL6;}
    if(Layer > 3){
      ilz_   = 16;
      nbsr   = 256.0;
      nphibs = 131072;
    }

    double rreal = r_.value()/nbsr*(rmax - rmin) + rmean_;
    double phireal = ((phi_.value()*1.0)/nphibs  - 0.125 )*(1.0/0.75)*(stubphimaxsec_ - stubphiminsec_) + stubphiminsec_;

    oss <<rreal<<" "
        << z_.value()*kz*ilz_<<" "
        <<phireal;



    return oss.str();

  }





  std::string strbare() const {
    
    std::ostringstream oss;
    oss << bend_.str()<<r_.str()
	<< z_.str()<< phi_.str();

    return oss.str();

  }

  std::string strbareUNFLIPPED() const {
    
    std::ostringstream oss;
    oss << r_.str()
	<< z_.str()<< phi_.str()<<bend_.str();

    return oss.str();

  }

  std::string inputstr() const {
    
    std::ostringstream oss;
    oss << r_.str()<< z_.str()
	<< phi_.str()<<bend_.str();

    return oss.str();

  }


  std::string rawstr() const {
    
    std::ostringstream oss;
    oss << layer_.str()<<"|"<<bend_.str()<<"|"<< r_.str()<<"|" 
  	<< z_.str() <<"|"<< phi_.str();
    
    return oss.str();
    
  }

  std::string vmstr() const {
    
    std::ostringstream oss;
    oss << bend_.str() <<"|"<<stubindex_.str()<<"|"<< phivm_.str();

    return oss.str();

  }


  std::string phiregionaddressstr() {

    std::ostringstream oss;
	assert(phiregion().value()>-1);
	oss << phiregion().str() << stubindex_.str();

	return oss.str();
	
  }

  int ilink() const {

    //changed pow(2,phi_.nbits()) to (1<<phi_.nbits()), etc
    if (phi_.value()<0.33*(1<<phi_.nbits()) ) return 1;
    if (phi_.value()<0.66*(1<<phi_.nbits()) ) return 2;
    return 3;

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

  void setAllStubAddressTE(int nstub){
    if (nstub>=(1<<7)){
      if (debug1) cout << "Warning too large stubindex!"<<endl;
      nstub=(1<<7)-1;
    }

    assert(stubaddressaste_.nbits()==-1);//Check that we are not overwriting
    stubaddressaste_.set(nstub,7);
  }


  void setPhiCorr(int phiCorr){

    /*    
    int layer=layer_.value()+1;
    
    double Delta=stubr_-rmean[layer-1];
    double dphi=Delta*0.5*(bend_.value()-15.0)*0.009/0.18/rmean[layer-1];

    int idphi=0;

    if (layer<=3) {
      idphi=dphi/kphi;
    } else {
        idphi=dphi/kphi1;
    }
    */
     
    //cout << "iphi idphi "<<phi_.value()<<" "<<idphi<<endl;
    
    //int iphicorr=phi_.value()+idphi;
    int iphicorr=phi_.value()-phiCorr;

    //cout << "phiCorr: layer bend old, new : "<<layer<<" "<<bend_.value()<<" "<<idphi<<" "<<phiCorr<<endl;
    
    if (iphicorr<0) iphicorr=0;
    if (iphicorr>=(1<<phi_.nbits())) iphicorr=(1<<phi_.nbits())-1;

    phicorr_.set(iphicorr,phi_.nbits(),true,__LINE__,__FILE__);

  }
  
  void setVMBits(int bits){
    int nbits=-1;
    if (vmbits_.value()!=-1) cout << "VMbits already set : "<<vmbits_.value()<<" "<<bits<<endl;
    assert(vmbits_.value()==-1); //Should never change the value; -1 means uninitialized
    if (layer_.value()==0 or layer_.value()==2 or layer_.value()==4) { // L1, L3, L5
      nbits=2*(2*NLONGVMBITS+1+3);
    }
    if (layer_.value()==1 or layer_.value()==3 or layer_.value()==5) { // L2, L4, L6
      nbits=2*NLONGVMBITS; //vmstub really only use half of these bits.
    }
    int disk=abs(disk_.value());
    if (disk==1 or disk==3) { // D1, D3
      nbits=(2*NLONGVMBITS+3) + (2*NLONGVMBITS+1+3);
    }
    if (disk==2 or disk==4) { // D2, D4
      nbits=2*NLONGVMBITS;
    }
    //cout << "layer, disk : "<<layer_.value()<<" "<<disk_.value()<<endl;
    assert(nbits>=0);
    vmbits_.set(bits,nbits,true,__LINE__,__FILE__);
  }

  FPGAWord getVMBits() const { return vmbits_; }

  void setVMBitsExtended(int bits){
    int nbits=-1;
    if (vmbitsextended_.value()!=-1) cout << "VMbits already set : "<<vmbits_.value()<<" "<<bits<<endl;
    assert(vmbitsextended_.value()==-1); //Should never change the value; -1 means uninitialized
    if (layer_.value()== 1) { // L2 (first in L2L3D1, same as inner)
      nbits=(2*NLONGVMBITS+1+3) + (2*NLONGVMBITS+1+3);
    }
    if (layer_.value()== 2) { // L3 (second in L2L3D1, same as outer)
      nbits=2*NLONGVMBITS; //vmstub really only use half of these bits.
    }
    //cout << "layer, disk : "<<layer_.value()<<" "<<disk_.value()<<endl;
    assert(nbits>=0);
    vmbitsextended_.set(bits,nbits,true,__LINE__,__FILE__);
  }

  FPGAWord getVMBitsExtended() const { return vmbitsextended_; }

  void setVMBitsOverlap(int bits){
    int nbits=-1;
    if (vmbitsoverlap_.value()!=-1) cout << "VMbits already set : "<<vmbits_.value()<<" "<<bits<<endl;
    assert(vmbitsoverlap_.value()==-1); //Should never change the value; -1 means uninitialized
    if (layer_.value()==0 or layer_.value()==1 or layer_.value()==4) { // L1, L2
      nbits=2*NLONGVMBITS+1+3;
    }
    int disk=abs(disk_.value());
    if (disk==1 ) { // D1
      nbits=2*NLONGVMBITS;
    }
    //cout << "layer, disk : "<<layer_.value()<<" "<<disk_.value()<<endl;
    assert(nbits>=0);
    vmbitsoverlap_.set(bits,nbits,true,__LINE__,__FILE__);
  }

  FPGAWord getVMBitsOverlap() const { return vmbitsoverlap_; }
  

  void setVMBitsExtra(int bits){
    int nbits=-1;
    if (vmbitsextra_.value()!=-1) cout << "Error VMbits extra already set : "<<vmbits_.value()<<" "<<bits<<endl;
    assert(vmbitsextra_.value()==-1); //Should never change the value; -1 means uninitialized
    if (layer_.value()==1) { //L2
      nbits=(2*NLONGVMBITS+1+3) + (2*NLONGVMBITS+1+3);
    }
    if (layer_.value()==2) { //L3
      nbits=2*NLONGVMBITS;
    }
    //cout << "layer, disk : "<<layer_.value()<<" "<<disk_.value()<<endl;
    assert(nbits>=0);
    vmbitsextra_.set(bits,nbits,true,__LINE__,__FILE__);
  }

  FPGAWord getVMBitsExtra() const { return vmbitsextra_; }
  


  void setVMBitsOverlapExtended(int bits){
    int nbits=-1;
    if (vmbitsoverlapextended_.value()!=-1) cout << "VMbits already set : "<<vmbits_.value()<<" "<<bits<<endl;
    assert(vmbitsoverlapextended_.value()==-1); //Should never change the value; -1 means uninitialized
    if (layer_.value()==1 ) { // L2 (third in D1D2L2, same as outer)
      nbits=2*NLONGVMBITS;
    }
    int disk=abs(disk_.value());
    if (disk==1 ) { // D1 (third in L2L3D1, same as outer)
      nbits=2*NLONGVMBITS;
    }
    //cout << "layer, disk, bits : "<<layer_.value()<<" "<<disk_.value()<<" : "<<bits <<"("<<nbits<<")"<<endl;
    assert(nbits>=0);
    vmbitsoverlapextended_.set(bits,nbits,true,__LINE__,__FILE__);
  }

  FPGAWord getVMBitsOverlapExtended() const { return vmbitsoverlapextended_; }
  
  FPGAWord phivm() const {return phivm_; }

  FPGAWord bend() const {return bend_; }

  FPGAWord r() const { return r_; }
  FPGAWord z() const { return z_; }
  FPGAWord phi() const { return phi_; }
  FPGAWord phicorr() const { return phicorr_; }
  FPGAWord alphanew() const { return alphanew_; }


  int ir() const { return r_.value(); }
  int iz() const { return z_.value(); }
  int iphi() const { return phi_.value(); }

  double phitmp() const {return phitmp_;}
  double phimin() const {return phimin_;}

  FPGAWord stubindex() const {return stubindex_;}
  FPGAWord stubaddressaste() const {return stubaddressaste_;} 

  FPGAWord layer() const {return layer_;}

  FPGAWord disk() const {return disk_;}

  double stubr() const { return stubr_;}
  double stubphi() const { return stubphi_;}
  double stubz() const { return stubz_;}
  double stubrpt() const { return stubrpt_;}

  bool isBarrel() const {return isbarrel_;}
  bool isDisk() const {return !isbarrel_;}

  bool isPSmodule() const {return isPSmodule_;}

  int round_int( double r ) {
    return (r > 0.0) ? (r + 0.5) : (r - 0.5); 
  }

  double rapprox(){
    if (disk_.value()==0){
      int lr=1<<(8-nbitsrL123);
      if (layer_.value()>=3) {
	lr=1<<(8-nbitsrL456);
      }
      return r_.value()*kr*lr+rmean[layer_.value()];
    }
    return r_.value()*kr;
  }

  double zapprox() {
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

  double phiapprox(double phimin, double){
    int lphi=1;
    if (layer_.value()>=3) {
      lphi=8;
    }
    return FPGAUtil::phiRange(phimin+phi_.value()*kphi/lphi);
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
    if (ibend==-9||ibend==-9) return 12;
    if (ibend==-11||ibend==-10) return 13;
    if (ibend==-13||ibend==-12) return 14;
    if (ibend<=-14) return 15;

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

  bool isbarrel_;
  bool isPSmodule_;
  FPGAWord layer_;  
  FPGAWord disk_;  
  FPGAWord r_;
  FPGAWord z_;
  FPGAWord phi_;
  FPGAWord alphanew_;

  FPGAWord bend_;
  
  FPGAWord phicorr_;  //Corrected for bend to nominal radius
  
  FPGAWord phivm_;
  FPGAWord stubindex_;
  FPGAWord stubaddressaste_;

  FPGAWord vmbits_;
  FPGAWord vmbitsoverlap_;
  FPGAWord vmbitsextra_;
  FPGAWord vmbitsextended_;
  FPGAWord vmbitsoverlapextended_;

  FPGAWord finer_;
  FPGAWord finez_;
  
  double stubphi_;
  double stubr_;
  double stubz_;
  double stubrpt_;

  double stubphiminsec_;
  double stubphimaxsec_;


  double phitmp_;
  double phimin_;

};



#endif



