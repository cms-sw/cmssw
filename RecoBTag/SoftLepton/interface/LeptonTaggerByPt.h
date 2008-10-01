#ifndef RecoBTag_SoftLepton_LeptonTaggerByPt_h
#define RecoBTag_SoftLepton_LeptonTaggerByPt_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"

/**  \class LeptonTaggerByPt
 *
 *   Implementation of muon b-tagging cutting on the lepton's transverse momentum relative to the jet axis
 *
 *   $Date: 2008/04/22 12:55:51 $
 *   $Revision: 1.3 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class LeptonTaggerByPt : public JetTagComputer {
public:

  /// explicit ctor 
  explicit LeptonTaggerByPt(const edm::ParameterSet & configuration) :
    m_selection( btag::LeptonSelector::option( configuration.getParameter<std::string>("ipSign") ) )
  { 
    uses("slTagInfos"); 
  }
  
  /// dtor
  virtual ~LeptonTaggerByPt() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:

  btag::LeptonSelector::sign m_selection;
  
};

#endif // RecoBTag_SoftLepton_LeptonTaggerByPt_h
