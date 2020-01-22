//Base class for processing modules
#ifndef LAYERRESIDUAL_H
#define LAYERRESIDUAL_H

using namespace std;

class LayerResidual{

public:

  LayerResidual(){
    valid_=false;
  }

  void init(int layer,
	    int iphiresid,
	    int izresid,
	    int istubid,
	    double phiresid,
	    double zresid,
	    double phiresidapprox,
	    double zresidapprox,
	    double rstub,
	    std::pair<Stub*,L1TStub*> stubptrs) {
    
    assert(layer>=1);
    assert(layer<=6);

    //cout << "Initiating projection to layer = "<<projlayer<< " at radius = "<<rproj<<endl;
    
    if (valid_&&(fabs(iphiresid)>fabs(fpgaphiresid_.value()))) return;
    
    valid_=true;

    layer_=layer;

    fpgaphiresid_.set(iphiresid,phiresidbits,false,__LINE__,__FILE__);
    fpgazresid_.set(izresid,zresidbits,false,__LINE__,__FILE__);
    int nbitsid=10;
    fpgastubid_.set(istubid,nbitsid,true,__LINE__,__FILE__);
    assert(!fpgaphiresid_.atExtreme());
    
    phiresid_=phiresid;
    zresid_=zresid;
  
    phiresidapprox_=phiresidapprox;
    zresidapprox_=zresidapprox;

    rstub_=rstub;
    stubptrs_=stubptrs;

  }

  virtual ~LayerResidual(){}

  bool valid() const {
    return valid_;
  }
  
  FPGAWord fpgaphiresid() const {
    assert(valid_);
    return fpgaphiresid_;
  };

  FPGAWord fpgazresid() const {
    assert(valid_);
    return fpgazresid_;
  };
  
  FPGAWord fpgastubid() const {
    assert(valid_);
    return fpgastubid_;
  };

  
  
  double phiresid() const {
    assert(valid_);
    return phiresid_;
  };

  double zresid() const {
    assert(valid_);
    return zresid_;
  };


  double phiresidapprox() const {
    assert(valid_);
    return phiresidapprox_;
  };

  double zresidapprox() const {
    assert(valid_);
    return zresidapprox_;
  };
  

  double rstub() const {
    assert(valid_);
    return rstub_;
  }
    
  std::pair<Stub*,L1TStub*> stubptrs() const {
    assert(valid_);
    return stubptrs_;

  }

  

protected:

  bool valid_;

  int layer_;

  FPGAWord fpgaphiresid_;
  FPGAWord fpgazresid_;
  FPGAWord fpgastubid_;

  double phiresid_;
  double zresid_;
  
  double phiresidapprox_;
  double zresidapprox_;

  double rstub_;
  std::pair<Stub*,L1TStub*> stubptrs_;

  
};

#endif
