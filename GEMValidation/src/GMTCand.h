#ifndef GEMValidation_GMTCand_h
#define GEMValidation_GMTCand_h

#include "GEMCode/GEMValidation/src/GMTRegCand.h"

class GMTCand //: public GMTRegCand
{
 public:
  /// constructor
  GMTCand();
  /// copy constructor
  GMTCand(const GMTCand&);
  /// destructor
  ~GMTCand();

  void init(const L1MuGMTExtendedCand *t,
	    edm::ESHandle< L1MuTriggerScales > &muScales,
	    edm::ESHandle< L1MuTriggerPtScale > &muPtScale);

/*   GMTRegCand* gmtRegCand() const {return regCand_;} */
/*   GMTRegCand* gmtRegCandRPC() const {return regCandRPC_;} */
  const L1MuGMTExtendedCand* l1gmt() const {return l1GMT_;}

  double pt() const {return pt_;}
  double eta() const {return eta_;}
  double phi() const {return phi_;}
  double dr() const {return dr_;}
  double q() const {return q_;}
  double rank() const {return rank_;}
  double isCSC() const {return isCSC_;}
  double isCSC2s() const {return isCSC2s_;}
  double isCSC3s() const {return isCSC3s_;}
  double isCSC2q() const {return isCSC2q_;}
  double isCSC3q() const {return isCSC3q_;}
  double isDT() const {return isDT_;}
  double isRPCf() const {return isRPCf_;}
  double isRPCb() const {return isRPCb_;}


 private:
  double pt_;
  double eta_;
  double phi_;
  double dr_;
  int q_;
  int rank_;
  bool isCSC_, isCSC2s_, isCSC3s_, isCSC2q_, isCSC3q_;
  bool isDT_;
  bool isRPCf_;
  bool isRPCb_;

  const L1MuGMTExtendedCand * l1GMT_;
  const GMTRegCand * gmtRegCand_;
  const GMTRegCand * gmtRegCandRPC_;
};

#endif
