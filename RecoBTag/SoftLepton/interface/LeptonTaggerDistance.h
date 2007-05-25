#ifndef RecoBTag_SoftLepton_LeptonTaggerDistance_h
#define RecoBTag_SoftLepton_LeptonTaggerDistance_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

/**  \class LeptonTaggerDistance
 *
 *   Implementation of muon b-tagging returning 1 if a lepton is present in the jet, 0 otherwise
 *
 *   $Date: 2007/01/09 01:17:22 $
 *   $Revision: 1.1 $
 *
 *   \author Andrea 'fwyzard' Bocci, Scuola Normale Superiore, Pisa
 */

class LeptonTaggerDistance : public JetTagComputer {
public:

  /// default ctor
  LeptonTaggerDistance(void) : m_maxDistance(0.5) { }

  /// explicit ctor
  explicit LeptonTaggerDistance(const edm::ParameterSet & configuration) {
    m_maxDistance = configuration.getParameter<double>("distance");
  }

  /// dtor
  virtual ~LeptonTaggerDistance() { }

  /// b-tag a jet based on track-to-jet pseudo-angular distance
  virtual float discriminator(const reco::BaseTagInfo & tagInfo) const;

private:
  
  float m_maxDistance;

};

#endif // RecoBTag_SoftLepton_LeptonTaggerDistance_h
