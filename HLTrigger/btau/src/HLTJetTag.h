#ifndef HLTrigger_btau_HLTJetTag_h
#define HLTrigger_btau_HLTJetTag_h

/** \class HLTJetTag
 *
 *  This class is an HLTFilter (a spcialized EDFilter) implementing 
 *  tagged multi-jet trigger for b and tau. 
 *  It should be run after the normal multi-jet trigger.
 *
 *  $Date: 2010/02/11 00:12:20 $
 *  $Revision: 1.3 $
 *
 *  \author Arnaud Gay, Ian Tomalin
 *  \maintainer Andrea Bocci
 *
 */

#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTJetTag : public HLTFilter
{
public:
  explicit HLTJetTag(const edm::ParameterSet & config);
  ~HLTJetTag();

  virtual bool filter(edm::Event & event, const edm::EventSetup & setup);

private:
  edm::InputTag m_jetTag;       // module label of input JetTagCollection
  double m_minTag, m_maxTag;    // tag descriminator cuts applied to each jet
  int    m_minJets;             // min. number of jets required to be tagged
  bool   m_saveTags;             // save the tagged jets in the TriggerEvent

  std::string m_label;          // Label of this filter in configuration file.
};

#endif // HLTrigger_btau_HLTJetTag_h
