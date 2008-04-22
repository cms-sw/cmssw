#ifndef RecoBTag_SoftLepton_MuonTagger_h
#define RecoBTag_SoftLepton_MuonTagger_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/src/MuonTaggerMLP.h"

/**  \class MuonTagger
 *
 *   Implementation of muon b-tagging using a softmax multilayer perceptron neural network
 *
 *   $Date: 2007/05/25 17:21:29 $
 *   $Revision: 1.2 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class MuonTagger : public JetTagComputer {
public:

  /// default ctor
  MuonTagger(void) : theNet() { uses("slTagInfos"); }

  /// explicit ctor 
  explicit MuonTagger( __attribute__((unused)) const edm::ParameterSet & configuration) : theNet() { uses("slTagInfos"); }
  
  /// dtor
  virtual ~MuonTagger() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:

  mutable MuonTaggerMLP theNet;

};

#endif // RecoBTag_SoftLepton_MuonTagger_h
