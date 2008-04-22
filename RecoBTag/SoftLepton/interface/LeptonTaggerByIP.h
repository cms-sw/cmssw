#ifndef RecoBTag_SoftLepton_LeptonTaggerByIP_h
#define RecoBTag_SoftLepton_LeptonTaggerByIP_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**  \class LeptonTaggerByIP
 *
 *   Implementation of muon b-tagging cutting on the lepton's transverse momentum relative to the jet axis
 *
 *   $Date: 2007/12/12 14:03:25 $
 *   $Revision: 1.1 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class LeptonTaggerByIP : public JetTagComputer {
public:

  /// default ctor
  LeptonTaggerByIP(void) :
    m_use3d( true )
  { uses("slTagInfos"); }

  /// explicit ctor 
  explicit LeptonTaggerByIP( const edm::ParameterSet & configuration) :
    m_use3d( configuration.getParameter<bool>("use3d") ) 
  { uses("slTagInfos"); }
  
  /// dtor
  virtual ~LeptonTaggerByIP() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:
  bool m_use3d;
  
};

#endif // RecoBTag_SoftLepton_LeptonTaggerByIP_h
