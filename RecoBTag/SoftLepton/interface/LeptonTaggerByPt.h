#ifndef RecoBTag_SoftLepton_LeptonTaggerByPt_h
#define RecoBTag_SoftLepton_LeptonTaggerByPt_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

/**  \class LeptonTaggerByPt
 *
 *   Implementation of muon b-tagging cutting on the lepton's transverse momentum relative to the jet axis
 *
 *   $Date: 2006/12/07 02:53:05 $
 *   $Revision: 1.1 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class LeptonTaggerByPt : public JetTagComputer {
public:

  /// default ctor
  LeptonTaggerByPt(void) { }

  /// explicit ctor 
  explicit LeptonTaggerByPt( __attribute__((unused)) const edm::ParameterSet & configuration) { }
  
  /// dtor
  virtual ~LeptonTaggerByPt() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const reco::BaseTagInfo & tagInfo) const;

};

#endif // RecoBTag_SoftLepton_LeptonTaggerByPt_h
