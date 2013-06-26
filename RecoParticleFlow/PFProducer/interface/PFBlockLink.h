#ifndef RecoParticleFlow_PFAlgo_PFBlockLink_h
#define RecoParticleFlow_PFAlgo_PFBlockLink_h 

#include <vector>
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"

/// \brief link between 2 elements
///
/// \author Colin Bernet
/// \date March 2006
class PFBlockLink {

 public:
    
  /// possible types. WARNING: have a look at PFBlockElement
  enum Type {
    NONE=0,
    ECALandECAL=0x8,
    TRACKandECAL=0x9,
    TRACKandHCAL=0x11,
    ECALandHCAL=0x18,
    PS1andECAL=0xA,
    PS2andECAL = 0xC,
    TRACKandPS1 = 0x3,
    TRACKandPS2 = 0x5,
    PS1andPS2 = 0x6,
    TRACKandTRACK = 0x1,
    ECALandGSF = 0x28,
    HCALandGSF = 0x30,
    TRACKandGSF = 0x21,
    GSFandBREM =0x60,
    GSFandGSF = 0x20,
    ECALandBREM = 0x48,
    HCALandBREM = 0x50,
    PS1andGSF = 0x22,
    PS2andGSF = 0x24,
    PS1andBREM = 0x42,
    PS2andBREM = 0x44,
    HFEMandHFHAD = 0x180,
    SCandECAL = 0x208,
    TRACKandHO= 0x401,
    HCALandHO= 0x410
  };
  
  /// default constructor
  /// \todo not sure it's useful
  PFBlockLink() : 
    type_(NONE), 
    test_(reco::PFBlock::LINKTEST_RECHIT),
    dist_(0),
    element1_( 0 ), 
    element2_( 0 ) {}  
  
  /// standard constructor
  PFBlockLink(Type type, 
	      reco::PFBlock::LinkTest test,
	      double dist,
	      unsigned elem1, 
	      unsigned elem2) 
    :  
    type_(type), 
    test_(test), 
    dist_(dist),
    element1_(elem1), 
    element2_(elem2) {}
  
  
  /// \return index to neighbouring element
  unsigned neighbour(unsigned elem) const {
    if( elem == element1_ ) return element2_;
    else if(elem == element2_ ) return element1_;
    else return elem;
  }

  /// \return the type
  Type type() const {return type_;}  

  /// \return the test: test used to compute the 
  /// value of the distance
  reco::PFBlock::LinkTest test() const {return test_;}  
  
  /// \return the distance
  double dist() const {return dist_;}
  
  /// \return index to first element
  unsigned element1() const {return element1_;}
  
  /// \return index to second element
  unsigned element2() const {return element2_;}
  
  
  /// print the link
  friend std::ostream& operator<<(std::ostream& out, const PFBlockLink& l); 
  
 private:
  /// type of the link
  Type    type_;
  
  /// type of test
  reco::PFBlock::LinkTest test_;

  /// distance of the link
  double dist_;
  
  /// element 1
  unsigned  element1_;

  /// element 2
  unsigned  element2_;
  
};  

#endif
