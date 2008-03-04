#ifndef HLTTauL25DoubleFilter_h
#define HLTTauL25DoubleFilter_h

/** \class HLTTauL25DoubleFilter
 *
 *  
 * 
 *
 *  $Date: 2006/11/13 17:28:34 $
 *  $Revision: 1.1 $
 *
 *  \author S. Gennai
 *
 */
#include "HLTrigger/HLTcore/interface/HLTFilter.h"


// class decleration
//

class HLTTauL25DoubleFilter : public HLTFilter {

public:
  explicit HLTTauL25DoubleFilter(const edm::ParameterSet&);
  ~HLTTauL25DoubleFilter();

  virtual bool filter(edm::Event&, const edm::EventSetup&);

private:
  std::string HLTTauL25DoubleFilterLabel_;    // Label of this filter in configuration file.
  edm::InputTag JetTag_;          // Module label of JetTagCollection
  edm::InputTag L1Code_; //Module with L1Coding convention for Tau Trigger;


};

#endif //HLTTauL25DoubleFilter_h
