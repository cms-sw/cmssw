///////////////////////////////////////////////////////////////////////////////
// File: HcalCellType.h
// Description: Hcal readout cell definition given by eta boundary and depth
///////////////////////////////////////////////////////////////////////////////
#ifndef HcalCellType_h
#define HcalCellType_h

#include <iostream>

class HcalCellType {

public:

  struct HcalCell {
    bool   ok;
    double eta, deta, phi, dphi, rz, drz;
    bool   flagrz;
    HcalCell(bool fl=false, double et=0, double det=0, double fi=0,
             double dfi=0, double rzv=0, double drzv=0, bool frz=true) :
      ok(fl), eta(et), deta(det), phi(fi), dphi(dfi), rz(rzv), drz(drzv),
         flagrz(frz) {}
  };

  HcalCellType(int detType, int etaBin, int phiBin, int depthSegment,
	       HcalCell cell, int readoutDirection, double samplingFactor);
  HcalCellType(const HcalCellType&);
  ~HcalCellType();

  /// 1=HB, 2=HE, 3=HO, 4=HF (sub detector type)
  /// as in DataFormats/HcalDetId/interface/HcalSubdetector.h
  int detType() const {return theDetType;}
                                                                               
  /// which eta ring it belongs to, starting from one
  int etaBin() const {return theEtaBin;}
                                                                               
  /// which depth segment it is, starting from 1
  /// absolute within the tower, so HE depth of the
  /// overlap doesn't start at 1.
  int depthSegment() const {return theDepthSegment;}

  /// the number of these cells in a ring
  int nPhiBins() const {return theNumberOfPhiBins;}
                                                                               
  /// phi bin width, in degrees
  double phiBinWidth() const {return 360./theNumberOfPhiBins;}

  /// phi offset in degrees
  double phiOffset() const {return thePhiOffset;}
                                                                               
  /// which cell will actually do the readout for this cell
  /// 1 means move hits in this cell up, and -1 means down
  /// 0 means do nothing
  int actualReadoutDirection() const {return theActualReadoutDirection;}
                                                                               
  /// lower cell edge.  Always positive
  double etaMin() const {return theEtaMin;}
                                                                               
  /// cell edge, always positive & greater than etaMin
  double etaMax() const {return theEtaMax;}

  /// z or r position, depending on whether it's barrel or endcap
  double depth() const {return (theDepthMin+theDepthMax)/2;}
  double depthMin() const {return theDepthMin;}
  double depthMax() const {return theDepthMax;}
  bool   depthType() const {return theRzFlag;}
                                                                               
  /// ratio of real particle energy to deposited energy in the SimHi
  double samplingFactor() const {return theSamplingFactor;}

protected:
 
  HcalCellType();
                                                                               
private:

  int    theDetType;
  int    theEtaBin;
  int    theDepthSegment;
  int    theNumberOfPhiBins;
  int    theActualReadoutDirection;

  bool   theRzFlag;
                                                                               
  double theEtaMin;
  double theEtaMax;
  double thePhiOffset;
  double theDepthMin;
  double theDepthMax;
  double theSamplingFactor;
};
                                                                               
std::ostream& operator<<(std::ostream&, const HcalCellType&);
#endif
