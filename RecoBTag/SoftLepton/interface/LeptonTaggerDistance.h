#ifndef RecoBTag_SoftLepton_LeptonTaggerDistance_h
#define RecoBTag_SoftLepton_LeptonTaggerDistance_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

/**  \class LeptonTaggerDistance
 *
 *   Implementation of muon b-tagging returning 1 if a lepton is present in the jet, 0 otherwise
 *
 *   $Date: 2013/05/20 22:59:11 $
 *   $Revision: 1.5 $
 *
 *   \author Andrea 'fwyzard' Bocci, Scuola Normale Superiore, Pisa
 */

class LeptonTaggerDistance : public JetTagComputer {
public:

  /// default ctor
  LeptonTaggerDistance(void) : m_maxDistance(0.5) { uses("slTagInfos"); }

  /// explicit ctor
  explicit LeptonTaggerDistance(const edm::ParameterSet & configuration) {
    m_maxDistance = configuration.getParameter<double>("distance");
    uses("slTagInfos");
  }

  /// dtor
  virtual ~LeptonTaggerDistance() { }

  /// b-tag a jet based on track-to-jet pseudo-angular distance
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:
  
  float m_maxDistance;

};

#endif // RecoBTag_SoftLepton_LeptonTaggerDistance_h
