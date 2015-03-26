#ifndef HLTBool_h
#define HLTBool_h

/** \class HLTBool
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) returning always the same
 *  configurable Boolean value (good for tests)
 *
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTBool : public edm::global::EDFilter<> {
public:
  explicit HLTBool(const edm::ParameterSet&);
  ~HLTBool();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual bool filter(edm::StreamID, edm::Event &, edm::EventSetup const &) const override final;

private:

  /// boolean result
  bool result_;

};

#endif //HLTBool_h
