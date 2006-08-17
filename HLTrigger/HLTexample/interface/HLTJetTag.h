#ifndef HLTJetTag_h
#define HLTJetTag_h

/** \class HLTJetTag
 *
 *  
 *  This class is an EDFilter implementing tagged multijet trigger
 *  (e.g., b or tau). It should be run after the normal multijet
 *  trigger.
 *
 *  $Date: 3rd Aug 2006 $
 *  $Revision: 0.0 $
 *
 *  \author Arnaud Gay, Ian Tomalin
 *
 */
#include "HLTrigger/HLTcore/interface/HLTFilter.h"


// class decleration
//

class HLTJetTag : public HLTFilter {

public:
  explicit HLTJetTag(const edm::ParameterSet&);
  ~HLTJetTag();

  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:
  edm::InputTag JetTag_;     // Module label of JetTagCollection
  double Min_Tag_, Max_Tag_; // tag descriminator cuts applied to each jet
  int    Min_N_;             // Min. number of jets required to be tagged

};

#endif //HLTJetTag_h
