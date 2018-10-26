#ifndef EventBXFilter_h
#define EventBXFilter_h

// C++ headers
#include <vector>
#include <string>

// CMSSW headers
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class declaration
//

class EventBXFilter : public edm::EDFilter {

  public:

    explicit EventBXFilter(const edm::ParameterSet&);
    ~EventBXFilter() override;
    static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

    bool filter(edm::Event&, const edm::EventSetup&) override;

    /// input list of BXs
    std::vector<unsigned int> allowedBXs_;
    std::vector<unsigned int> vetoBXs_;
    
};

#endif //EventBXFilter_h
