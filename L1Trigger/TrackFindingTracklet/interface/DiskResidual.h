//Base class for processing modules
#ifndef DISKRESIDUAL_H
#define DISKRESIDUAL_H

using namespace std;

class DiskResidual{

public:

  DiskResidual(){
    valid_=false;
  }

  void init(int disk,
	    int iphiresid,
	    int irresid,
	    int istubid,
	    double phiresid,
	    double rresid,
	    double phiresidapprox,
	    double rresidapprox,
	    double zstub,
	    double alpha,
	    FPGAWord ialpha,
	    std::pair<Stub*,L1TStub*> stubptrs) {
    
    assert(abs(disk)>=1);
    assert(abs(disk)<=5);

    //cout << "Initiating projection to disk = "<<projdisk<< " at radius = "<<rproj<<endl;
    
    if (valid_&&(fabs(iphiresid)>fabs(fpgaphiresid_.value()))) return;
    
    valid_=true;

    disk_=disk;

    fpgaphiresid_.set(iphiresid,phiresidbits,false,__LINE__,__FILE__);
    fpgarresid_.set(irresid,rresidbits,false,__LINE__,__FILE__);
    assert(istubid>=0);
    unsigned int nbitsstubid=10;
    fpgastubid_.set(istubid,nbitsstubid,true,__LINE__,__FILE__);
    assert(!fpgaphiresid_.atExtreme());

    phiresid_=phiresid;
    rresid_=rresid;
  
    phiresidapprox_=phiresidapprox;
    rresidapprox_=rresidapprox;

    zstub_=zstub;
    alpha_=alpha;
    ialpha_=ialpha;
    stubptrs_=stubptrs;
    
  }

  virtual ~DiskResidual(){}

  bool valid() const {
    return valid_;
  }
  
  FPGAWord fpgaphiresid() const {
    assert(valid_);
    return fpgaphiresid_;
  };

  FPGAWord fpgarresid() const {
    assert(valid_);
    return fpgarresid_;
  };
  
  FPGAWord fpgastubid() const {
    assert(valid_);
    return fpgastubid_;
  };

  
  
  double phiresid() const {
    assert(valid_);
    return phiresid_;
  };

  double rresid() const {
    assert(valid_);
    return rresid_;
  };


  double phiresidapprox() const {
    assert(valid_);
    return phiresidapprox_;
  };

  double rresidapprox() const {
    assert(valid_);
    return rresidapprox_;
  };
  
  double zstub()const {
    assert(valid_);
    return zstub_;
  };

  double alpha() const {
    assert(valid_);
    return alpha_;
  };

  FPGAWord ialpha() const {
    assert(valid_);
    return ialpha_;
  };
  
  std::pair<Stub*,L1TStub*> stubptrs() const {
    assert(valid_);
    return stubptrs_;
  };
  

protected:

  bool valid_;

  int disk_;

  FPGAWord fpgaphiresid_;
  FPGAWord fpgarresid_;
  FPGAWord fpgastubid_;

  double phiresid_;
  double rresid_;
  
  double phiresidapprox_;
  double rresidapprox_;

  double zstub_;
  double alpha_;
  FPGAWord ialpha_;
  std::pair<Stub*,L1TStub*> stubptrs_;
  
};

#endif
