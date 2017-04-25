//Base class for processing modules
#ifndef FPGADISKPROJECTION_H
#define FPGADISKPROJECTION_H

using namespace std;

class FPGADiskProjection{

public:

  FPGADiskProjection(){
    valid_=false;
  }

  void init(int projdisk,
	    double zproj,
	    int iphiproj,
	    int irproj,
	    int iphider,
	    int irder,
	    bool minusNeighbor,
	    bool plusNeighbor,
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

    //cout << "Initiating projection to layer = "<<projlayer<< " at radius = "<<rproj<<endl;
    
    valid_=true;

    zproj_=zproj;
    
    projdisk_=projdisk;

    if (iphiproj<0) iphiproj=0; //FIXME should be assert?
    
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

    
    minusNeighbor_=minusNeighbor;
    plusNeighbor_=plusNeighbor;

    phiproj_=phiproj;
    rproj_=rproj;
    phiprojder_=phiprojder;
    rprojder_=rprojder;

    phiprojapprox_=phiprojapprox;
    rprojapprox_=rprojapprox;
    phiprojderapprox_=phiprojderapprox;
    rprojderapprox_=rprojderapprox;

  }

  virtual ~FPGADiskProjection(){}

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

  bool minusNeighbor() const {
    assert(valid_);
    return minusNeighbor_;
  };
  
  bool plusNeighbor() const {
    assert(valid_);
    return plusNeighbor_;
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

  

protected:

  bool valid_;

  int projdisk_;

  double zproj_;
  
  FPGAWord fpgaphiproj_;
  FPGAWord fpgarproj_;
  FPGAWord fpgaphiprojder_;
  FPGAWord fpgarprojder_;

  bool minusNeighbor_;
  bool plusNeighbor_;


  FPGAWord fpgaphiprojvm_;
  FPGAWord fpgarprojvm_;


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
