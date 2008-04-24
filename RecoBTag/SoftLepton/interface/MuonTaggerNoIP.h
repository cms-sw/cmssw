#ifndef RecoBTag_SoftLepton_MuonTaggerNoIP_h
#define RecoBTag_SoftLepton_MuonTaggerNoIP_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/src/MuonTaggerNoIPMLP.h"

/**  \class MuonTagger
 *
 *   Implementation of muon b-tagging using a softmax multilayer perceptron neural network
 *
 *   $Date: 2008/04/22 12:55:51 $
 *   $Revision: 1.3 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class MuonTaggerNoIP : public JetTagComputer {
public:

  /// default ctor
  MuonTaggerNoIP(void) : theNet() { uses("smTagInfos"); }

  /// explicit ctor
  explicit MuonTaggerNoIP( __attribute__((unused)) const edm::ParameterSet & configuration) : theNet() { uses("smTagInfos"); }

  /// dtor
  virtual ~MuonTaggerNoIP() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:

  mutable MuonTaggerNoIPMLP theNet;

};

#endif // RecoBTag_SoftLepton_MuonTaggerNoIP_h
