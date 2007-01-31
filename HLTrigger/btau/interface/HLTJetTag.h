#ifndef HLTJetTag_h
#define HLTJetTag_h

/** \class HLTJetTag
 *
 *  
 *  This class is an EDFilter implementing tagged multijet trigger
 *  (e.g., b or tau). It should be run after the normal multijet
 *  trigger.
 *
 *  $Date: 2006/11/14 07:37:47 $
 *  $Revision: 1.5 $
 *
 *  \author Arnaud Gay, Ian Tomalin
 *
 */
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTJetTag : public HLTFilter {

public:
  explicit HLTJetTag(const edm::ParameterSet&);
  ~HLTJetTag();

  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:
  edm::InputTag jetTag_;        // Module label of input JetTagCollection
  double min_Tag_, max_Tag_;    // tag descriminator cuts applied to each jet
  int    min_N_;                // Min. number of jets required to be tagged

  std::string label_;           // Label of this filter in configuration file.
};

#endif //HLTJetTag_h
