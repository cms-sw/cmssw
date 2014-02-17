#ifndef HLTBool_h
#define HLTBool_h

/** \class HLTBool
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) returning always the same
 *  configurable Boolean value (good for tests)
 *
 *  $Date: 2012/01/22 22:15:43 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTBool : public edm::EDFilter {

  public:

    explicit HLTBool(const edm::ParameterSet&);
    ~HLTBool();
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    virtual bool filter(edm::Event&, const edm::EventSetup&);

  private:

    /// Boolean result
    bool result_;

};

#endif //HLTBool_h
