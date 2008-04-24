#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/src/ElectronTaggerMLP.h"

/** \class ElectronTagger
 *
 *  $Id: ElectronTagger.h,v 1.3 2008/04/22 12:55:51 saout Exp $
 *  $Date: 2008/04/22 12:55:51 $
 *  $Revision: 1.3 $
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve - Belgium
 *
 */

class ElectronTagger : public JetTagComputer {
public:

  /// default ctor
  ElectronTagger(void) : theNet() { uses("seTagInfos"); }

  /// explicit ctor 
  explicit ElectronTagger( __attribute__((unused)) const edm::ParameterSet & configuration) : theNet() { uses("seTagInfos"); }
  
  /// dtor
  virtual ~ElectronTagger() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:

  mutable ElectronTaggerMLP theNet;

};

#endif // RecoBTag_SoftLepton_ElectronTagger_h
