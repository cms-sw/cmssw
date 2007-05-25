#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/src/ElectronTaggerMLP.h"

/** \class ElectronTagger
 *
 *  $Id: ElectronTagger.h,v 1.1 2007/02/14 17:10:24 demine Exp $
 *  $Date: 2007/02/14 17:10:24 $
 *  $Revision: 1.1 $
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve - Belgium
 *
 */

class ElectronTagger : public JetTagComputer {
public:

  /// default ctor
  ElectronTagger(void) : theNet() { }

  /// explicit ctor 
  explicit ElectronTagger( __attribute__((unused)) const edm::ParameterSet & configuration) : theNet() { }
  
  /// dtor
  virtual ~ElectronTagger() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const reco::BaseTagInfo & tagInfo) const;

private:

  mutable ElectronTaggerMLP theNet;

};

#endif // RecoBTag_SoftLepton_ElectronTagger_h
