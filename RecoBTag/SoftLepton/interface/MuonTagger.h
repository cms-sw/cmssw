#ifndef RecoBTag_SoftLepton_MuonTagger_h
#define RecoBTag_SoftLepton_MuonTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/src/MuonTaggerMLP.h"
#include "TRandom3.h"

/**  \class MuonTagger
 *
 *   Implementation of muon b-tagging using a softmax multilayer perceptron neural network
 *
 *   $Date: 2013/04/14 02:24:58 $
 *   $Revision: 1.7 $
 *
 *   \author Andrea 'fwyzard' Bocci, Universita' di Firenze
 */

class MuonTagger : public JetTagComputer {
public:

  /// explicit ctor 
  explicit MuonTagger(const edm::ParameterSet & configuration) : 
    theNet(),
    m_selector(configuration),
    randomNumberGenerator_(0)
  { 
    uses("smTagInfos"); 
  }
  
  /// dtor
  virtual ~MuonTagger() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:

  mutable MuonTaggerMLP theNet;

  btag::LeptonSelector m_selector;
  mutable TRandom3 randomNumberGenerator_;
};

#endif // RecoBTag_SoftLepton_MuonTagger_h
