#ifndef RecoBTag_SoftLepton_LeptonTaggerDistance_h
#define RecoBTag_SoftLepton_LeptonTaggerDistance_h

#include "TVector3.h"
#include "RecoBTag/SoftLepton/interface/LeptonTaggerBase.h"

/**  \class LeptonTaggerDistance
 *
 *   Implementation of muon b-tagging returning 1 if a lepton is present in the jet, 0 otherwise
 *
 *   $Date: 2006/12/07 02:53:05 $
 *   $Revision: 1.1 $
 *
 *   \author Andrea 'fwyzard' Bocci, Scuola Normale Superiore, Pisa
 */

class LeptonTaggerDistance : public LeptonTaggerBase {
public:
  LeptonTaggerDistance (void) {}
  virtual ~LeptonTaggerDistance (void) {}

  /// b-tag a jet based on track-to-jet parameters
  virtual double discriminant (
      const TVector3 & axis,
      const TVector3 & lepton,
      const reco::SoftLeptonProperties & properties
  ) const 
  {
    return 1.0;
  }

};

#endif // RecoBTag_SoftLepton_LeptonTaggerDistance_h
