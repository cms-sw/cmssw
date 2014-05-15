#include "GeneratorInterface/GenFilters/interface/MCPdgIndexFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

MCPdgIndexFilter::MCPdgIndexFilter(const edm::ParameterSet& cfg) :
  label_(cfg.getUntrackedParameter("moduleLabel",std::string("generator"))),
  pdgID(cfg.getParameter<std::vector<int> >("PdgId")),
  index(cfg.getParameter<std::vector<unsigned> >("Index")),
  maxIndex(*std::max_element(index.begin(),index.end())),
  taggingMode(cfg.getUntrackedParameter<bool>("TagMode",false)),
  tag(cfg.getUntrackedParameter<std::string>("Tag",""))
{
  if (pdgID.size() != index.size())
    edm::LogWarning("MCPdgIndexFilter")
      << "Configuration Error :"
      << "Sizes of array parameters 'PdgId' and 'Index' differ.";

  if (taggingMode) {
    produces<bool>(tag);
    edm::LogInfo("TagMode") << "Filter result in '" << tag << "', filtering disabled.";
  }
}


bool MCPdgIndexFilter::filter(edm::Event& evt, const edm::EventSetup&) {
  bool result = pass(evt);
  LogDebug("FilterResult") << (result?"Pass":"Fail");
  if (!taggingMode) return result;
  evt.put( std::auto_ptr<bool>(new bool(result)), tag);
  return true;
}


bool MCPdgIndexFilter::pass(const edm::Event& evt) {
  edm::Handle<edm::HepMCProduct> hepmc;
  evt.getByLabel(label_, hepmc);

  const HepMC::GenEvent * genEvent = hepmc->GetEvent();

  HepMC::GenEvent::particle_const_iterator
    p(genEvent->particles_begin()),
    p_end(genEvent->particles_end());

  for ( unsigned i=0; p!=p_end && i<=maxIndex; ++p, i++ ) {
    LogDebug("Particle") << "index: " << i << "   pdgID: " << (*p)->pdg_id();
    for (unsigned j = 0; j < pdgID.size(); j++) {
      if (i==index[j] && pdgID[j] != (*p)->pdg_id())
	return false;
    }
  }
  return true;
}
