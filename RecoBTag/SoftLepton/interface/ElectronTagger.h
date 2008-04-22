#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/src/ElectronTaggerMLP.h"

/** \class ElectronTagger
 *
 *  $Id: ElectronTagger.h,v 1.2 2007/05/25 17:21:29 fwyzard Exp $
 *  $Date: 2007/05/25 17:21:29 $
 *  $Revision: 1.2 $
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve - Belgium
 *
 */

class ElectronTagger : public JetTagComputer {
public:

  /// default ctor
  ElectronTagger(void) : theNet() { uses("slTagInfos"); }

  /// explicit ctor 
  explicit ElectronTagger( __attribute__((unused)) const edm::ParameterSet & configuration) : theNet() { uses("slTagInfos"); }
  
  /// dtor
  virtual ~ElectronTagger() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:

  mutable ElectronTaggerMLP theNet;

};

#endif // RecoBTag_SoftLepton_ElectronTagger_h
