#ifndef RecoBTag_SoftLepton_LeptonTaggerByPt_h
#define RecoBTag_SoftLepton_LeptonTaggerByPt_h

#include "TVector3.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerBase.h"

/**  \class LeptonTaggerByPt
 *
 *   Implementation of lepton b-tagging by the lepton relative transverse inpact parameter
 *
 *   $Date: 2006/12/07 02:53:05 $
 *   $Revision: 1.1 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class LeptonTaggerByPt : public LeptonTaggerBase {
public:
  LeptonTaggerByPt (void) {}
  virtual ~LeptonTaggerByPt (void) {}

  /// b-tag a jet based on track-to-jet parameters:
  virtual double discriminant (
      const TVector3 & axis,
      const TVector3 & lepton,
      const reco::SoftLeptonProperties & properties
  ) const 
  {
    return properties.ptRel;
  }

};

#endif // RecoBTag_SoftLepton_LeptonTaggerByPt_h
