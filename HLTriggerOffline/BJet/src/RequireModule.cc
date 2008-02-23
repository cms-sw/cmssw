#include <vector>

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "HLTriggerOffline/BJet/interface/ArbitraryType.h"

class RequireModule : public edm::EDFilter {
public:
  explicit RequireModule(const edm::ParameterSet& config);
  virtual ~RequireModule();

  bool filter(edm::Event & event, const edm::EventSetup & setup);

private:
  // input collections
  edm::InputTag m_requirement;
};


RequireModule::RequireModule(const edm::ParameterSet & config) :
  m_requirement( config.getParameter<edm::InputTag>("requirement") )
{
}

RequireModule::~RequireModule() 
{
}

bool RequireModule::filter(edm::Event & event, const edm::EventSetup & setup) 
{
  bool found = false;
  
  std::vector<const edm::Provenance *> provenances;
  event.getAllProvenance(provenances);

  edm::ArbitraryHandle handle;
  for (unsigned int i = 0; i < provenances.size(); ++i) {
    if ((m_requirement.label()    == provenances[i]->moduleLabel()) and
        (m_requirement.instance() == provenances[i]->productInstanceName()) and 
        (m_requirement.process()  == provenances[i]->processName() or m_requirement.process()  == "") and
        event.get(provenances[i]->productID(), handle) and
        handle.isValid()
    ) {
      found = true;
      break;
    } 
  }

  return found;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(RequireModule);
