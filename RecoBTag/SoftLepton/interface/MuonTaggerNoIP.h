#ifndef RecoBTag_SoftLepton_MuonTaggerNoIP_h
#define RecoBTag_SoftLepton_MuonTaggerNoIP_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/src/MuonTaggerNoIPMLP.h"

/**  \class MuonTagger
 *
 *   Implementation of muon b-tagging using a softmax multilayer perceptron neural network
 *
 *   $Date: 2007/05/25 17:21:29 $
 *   $Revision: 1.2 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class MuonTaggerNoIP : public JetTagComputer {
public:

  /// default ctor
  MuonTaggerNoIP(void) : theNet() { uses("slTagInfos"); }

  /// explicit ctor
  explicit MuonTaggerNoIP( __attribute__((unused)) const edm::ParameterSet & configuration) : theNet() { uses("slTagInfos"); }

  /// dtor
  virtual ~MuonTaggerNoIP() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:

  mutable MuonTaggerNoIPMLP theNet;

};

#endif // RecoBTag_SoftLepton_MuonTaggerNoIP_h
