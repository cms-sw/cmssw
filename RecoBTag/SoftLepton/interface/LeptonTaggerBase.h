#ifndef RecoBTag_SoftLepton_LeptonTaggerBase_h
#define RecoBTag_SoftLepton_LeptonTaggerBase_h

/**  \class LeptonTaggerBase
 *
 *   Abstract class for lepton b-tagging a jet, based on track-to-jet parameters
 *
 *   $Date: 2006/10/31 02:53:09 $
 *   $Revision: 1.1 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

class TVector3;

class LeptonTaggerBase {
public:
  LeptonTaggerBase (void) {}
  virtual ~LeptonTaggerBase (void) {}
 
  /// calculate a jet b-tagging discriminant based on track-to-jet parameters:
  virtual double discriminant (
      const TVector3 & axis,
      const TVector3 & lepton,
      const reco::SoftLeptonProperties & properties
  ) const = 0;

};

#endif // RecoBTag_SoftLepton_LeptonTaggerBase_h
