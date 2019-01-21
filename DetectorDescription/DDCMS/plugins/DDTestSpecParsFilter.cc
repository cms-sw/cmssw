#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"

#include <iostream>

using namespace std;
using namespace cms;
using namespace edm;

class DDTestSpecParsFilter : public one::EDAnalyzer<> {
public:
  explicit DDTestSpecParsFilter(const ParameterSet& iConfig)
    : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")),
      m_attribute(iConfig.getUntrackedParameter<string>("attribute", "")),
      m_value(iConfig.getUntrackedParameter<string>("value", ""))
  {}

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:  
  const ESInputTag m_tag;
  const string m_attribute;
  const string m_value;
};

void
DDTestSpecParsFilter::analyze(const Event&, const EventSetup& iEventSetup)
{
  LogVerbatim("Geometry") << "DDTestSpecParsFilter::analyze: " << m_tag;
  ESTransientHandle<DDSpecParRegistry> registry;
  iEventSetup.get<DDSpecParRegistryRcd>().get(m_tag.module(), registry);

  LogVerbatim("Geometry") << "DDTestSpecParsFilter::analyze: " << m_tag.module() << " for attribute " << m_attribute << " and value " << m_value;
  LogVerbatim("Geometry") << "DD SpecPar Registry size: " << registry->specpars.size();

  DDSpecParRegistry myReg;
  bool found(false);
  for(const auto& i: registry->specpars) {
    found = false;
    for(const auto& l : i.second.spars) {
      if(l.first == m_attribute) {
	for(const auto& il : l.second) {
	  if(il == m_value) {
	    found = true;
	  }
	}
      }
    }
    if(found) {
      myReg.specpars.emplace(i);
    }
  }
  LogVerbatim("Geometry").log([&myReg](auto& log) {
      log << "Filtered DD SpecPar Registry size: " << myReg.specpars.size();
      for(const auto& t: myReg.specpars) {
	log << " " << t.first << " => ";
	for(const auto& ki : t.second.paths)
	  log << ki << ", ";
	for(const auto& kl : t.second.spars) {
	  log << kl.first << " => ";
	  for(const auto& kil : kl.second) {
	    log << kil << ", ";
	  }
	}
      }
    });
}

DEFINE_FWK_MODULE(DDTestSpecParsFilter);
