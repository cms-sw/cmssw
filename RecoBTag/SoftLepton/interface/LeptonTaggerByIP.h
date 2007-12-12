#ifndef RecoBTag_SoftLepton_LeptonTaggerByIP_h
#define RecoBTag_SoftLepton_LeptonTaggerByIP_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**  \class LeptonTaggerByIP
 *
 *   Implementation of muon b-tagging cutting on the lepton's transverse momentum relative to the jet axis
 *
 *   $Date: 2007/05/25 17:21:29 $
 *   $Revision: 1.2 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class LeptonTaggerByIP : public JetTagComputer {
public:

  /// default ctor
  LeptonTaggerByIP(void) :
    m_use3d( true )
  { }

  /// explicit ctor 
  explicit LeptonTaggerByIP( const edm::ParameterSet & configuration) :
    m_use3d( configuration.getParameter<bool>("use3d") ) 
  { }
  
  /// dtor
  virtual ~LeptonTaggerByIP() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const reco::BaseTagInfo & tagInfo) const;

private:
  bool m_use3d;
  
};

#endif // RecoBTag_SoftLepton_LeptonTaggerByIP_h
