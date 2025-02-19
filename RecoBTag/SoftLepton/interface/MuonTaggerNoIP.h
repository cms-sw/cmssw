#ifndef RecoBTag_SoftLepton_MuonTaggerNoIP_h
#define RecoBTag_SoftLepton_MuonTaggerNoIP_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/src/MuonTaggerNoIPMLP.h"

/**  \class MuonTagger
 *
 *   Implementation of muon b-tagging using a softmax multilayer perceptron neural network
 *
 *   $Date: 2010/02/26 18:16:18 $
 *   $Revision: 1.6 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class MuonTaggerNoIP : public JetTagComputer {
public:

  /// explicit ctor
  explicit MuonTaggerNoIP(const edm::ParameterSet & configuration) : 
    theNet(),
    m_selector(configuration)
  { 
    uses("slTagInfos"); 
  }

  /// dtor
  virtual ~MuonTaggerNoIP() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:

  mutable MuonTaggerNoIPMLP theNet;

  btag::LeptonSelector m_selector;

};

#endif // RecoBTag_SoftLepton_MuonTaggerNoIP_h
