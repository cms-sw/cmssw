//Base class for processing modules
#ifndef DISKPROJECTION_H
#define DISKPROJECTION_H

using namespace std;

class DiskProjection{

public:

  DiskProjection(){
    valid_=false;
  }

  void init(int projdisk,
	    double zproj,
	    int iphiproj,
	    int irproj,
	    int iphider,
	    int irder,
	    double phiproj,
	    double rproj,
	    double phiprojder,
	    double rprojder,
	    double phiprojapprox,
	    double rprojapprox,
	    double phiprojderapprox,
	    double rprojderapprox) {

    assert(abs(projdisk)>=1);
    assert(abs(projdisk)<=5);

    if(debug1) cout << "Initiating projection to diak = "<<projdisk<< " at z = "<<zproj<<endl;
    
    valid_=true;

    zproj_=zproj;
    
    projdisk_=projdisk;

    assert(iphiproj>=0);
    
    fpgaphiproj_.set(iphiproj,nbitsphiprojL123,true,__LINE__,__FILE__);
    int iphivm=(iphiproj>>(nbitsphiprojL123-5))&0x7;
    if ((abs(projdisk_)%2)==1) {
      iphivm^=4;
    }
    fpgaphiprojvm_.set(iphivm,3,true,__LINE__,__FILE__);
    fpgarproj_.set(irproj,nrbitsprojdisk,false,__LINE__,__FILE__);
    int irvm=irproj>>(13-7)&0xf;
    fpgarprojvm_.set(irvm,4,true,__LINE__,__FILE__);
    fpgaphiprojder_.set(iphider,nbitsphiprojderL123,false,__LINE__,__FILE__);
    fpgarprojder_.set(irder,nrbitsprojderdisk,false,__LINE__,__FILE__);

    //FIXME the -3 and +3 should be evaluated and efficiency for matching hits checked.
    int rbin1=8.0*(irproj*krprojshiftdisk-3-rmindiskvm)/(rmaxdisk-rmindiskvm);
    int rbin2=8.0*(irproj*krprojshiftdisk+3-rmindiskvm)/(rmaxdisk-rmindiskvm);

    if (irproj*krprojshiftdisk<20.0) {
      cout <<" WARNING : irproj = "<<irproj<<" "<<irproj*krprojshiftdisk<<" "<<projdisk_<<endl;
    }

    if (rbin1<0) rbin1=0; 
    if (rbin2<0) rbin2=0; 
    if (rbin2>7) rbin2=7;
    assert(rbin1<=rbin2);
    assert(rbin2-rbin1<=1);

    int finer=64*((irproj*krprojshiftdisk-rmindiskvm)-rbin1*(rmaxdisk-rmindiskvm)/8.0)/(rmaxdisk-rmindiskvm);

    if (finer<0) finer=0;
    if (finer>15) finer=15;
    
    int diff=rbin1!=rbin2;
    if (irder<0) rbin1+=8;
    
    fpgarbin1projvm_.set(rbin1,4,true,__LINE__,__FILE__); // first r bin
    fpgarbin2projvm_.set(diff,1,true,__LINE__,__FILE__); // need to check adjacent r bin
 
    fpgafinervm_.set(finer,4,true,__LINE__,__FILE__); // fine r postions starting at rbin1
    
    phiproj_=phiproj;
    rproj_=rproj;
    phiprojder_=phiprojder;
    rprojder_=rprojder;

    phiprojapprox_=phiprojapprox;
    rprojapprox_=rprojapprox;
    phiprojderapprox_=phiprojderapprox;
    rprojderapprox_=rprojderapprox;

  }
  
  bool valid() const {
    return valid_;
  }

  
  virtual ~DiskProjection(){}

  int projdisk() const {
    assert(valid_);
    return projdisk_;
  };

  double zproj() const {
    assert(valid_);
    return zproj_;
  };

  
  FPGAWord fpgaphiproj() const {
    assert(valid_);
    return fpgaphiproj_;
  };
  
  FPGAWord fpgarproj() const {
    assert(valid_);
    return fpgarproj_;
  };

  FPGAWord fpgaphiprojder() const {
    assert(valid_);
    return fpgaphiprojder_;
  };

  FPGAWord fpgarprojder() const {
    assert(valid_);
    return fpgarprojder_;
  };

  FPGAWord fpgaphiprojvm() const {
    assert(valid_);
    return fpgaphiprojvm_;
  };

  FPGAWord fpgarprojvm() const {
    assert(valid_);
    return fpgarprojvm_;
  };

  double phiproj() const {
    assert(valid_);
    return phiproj_;
  };

  FPGAWord fpgarbin1projvm() const {
    assert(valid_);
    return fpgarbin1projvm_;
  };
  
  FPGAWord fpgarbin2projvm() const {
    assert(valid_);
    return fpgarbin2projvm_;
  };

  FPGAWord fpgafinervm() const {
    assert(valid_);
    return fpgafinervm_;
  };



  double rproj() const {
    assert(valid_);
    return rproj_;
  };

  double phiprojder() const {
    assert(valid_);
    return phiprojder_;
  };

  double rprojder() const {
    assert(valid_);
    return rprojder_;
  };

  double phiprojapprox() const {
    assert(valid_);
    return phiprojapprox_;
  };

  double rprojapprox() const {
    assert(valid_);
    return rprojapprox_;
  };

  double phiprojderapprox() const {
    assert(valid_);
    return phiprojderapprox_;
  };

  double rprojderapprox() const {
    assert(valid_);
    return rprojderapprox_;
  };

  void setBendIndex(int bendindex) {
    fpgabendindex_.set(bendindex,5,true,__LINE__,__FILE__);
  }

  FPGAWord getBendIndex() const {
    return fpgabendindex_;
  }

protected:

  bool valid_;

  int projdisk_;

  double zproj_;
  
  FPGAWord fpgaphiproj_;
  FPGAWord fpgarproj_;
  FPGAWord fpgaphiprojder_;
  FPGAWord fpgarprojder_;

  FPGAWord fpgaphiprojvm_;
  FPGAWord fpgarprojvm_;

  FPGAWord fpgarbin1projvm_;
  FPGAWord fpgarbin2projvm_;
  FPGAWord fpgafinervm_;

  FPGAWord fpgabendindex_;

  double phiproj_;
  double rproj_;
  double phiprojder_;
  double rprojder_;

  double phiprojapprox_;
  double rprojapprox_;
  double phiprojderapprox_;
  double rprojderapprox_;
    
};

#endif
