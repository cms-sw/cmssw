#ifndef RecoBTag_SoftLepton_ElectronTagger_h
#define RecoBTag_SoftLepton_ElectronTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/src/ElectronTaggerMLP.h"

/** \class ElectronTagger
 *
 *  $Id: ElectronTagger.h,v 1.5 2008/10/01 12:35:34 fwyzard Exp $
 *  $Date: 2008/10/01 12:35:34 $
 *  $Revision: 1.5 $
 *
 *  \author P. Demin - UCL, Louvain-la-Neuve - Belgium
 *
 */

class ElectronTagger : public JetTagComputer {
public:

  /// explicit ctor 
  explicit ElectronTagger(const edm::ParameterSet & configuration) : 
    theNet(),
    m_selector(configuration)
  { 
    uses("seTagInfos"); 
  }
  
  /// dtor
  virtual ~ElectronTagger() { }

  /// b-tag a jet based on track-to-jet parameters in the extened info collection
  virtual float discriminator(const TagInfoHelper & tagInfo) const;

private:

  mutable ElectronTaggerMLP theNet;

  btag::LeptonSelector m_selector;

};

#endif // RecoBTag_SoftLepton_ElectronTagger_h
