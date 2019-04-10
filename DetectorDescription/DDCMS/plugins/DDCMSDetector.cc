#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DetectorDescriptionRcd.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistryRcd.h"
#include "DetectorDescription/DDCMS/interface/DDVectorRegistry.h"
#include "DD4hep/Detector.h"

#include <memory>
#include <string>

using namespace std;
using namespace cms;
using namespace edm;
using namespace dd4hep;

class DDCMSDetector : public one::EDAnalyzer<> {
public:
  explicit DDCMSDetector(const ParameterSet& p);

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override;

private:  
  const ESInputTag m_tag;
};

DDCMSDetector::DDCMSDetector(const ParameterSet& iConfig)
  : m_tag(iConfig.getParameter<ESInputTag>("DDDetector"))
{}

void
DDCMSDetector::analyze(const Event&, const EventSetup& iEventSetup)
{
  ESTransientHandle<DDDetector> det;
  iEventSetup.get<DetectorDescriptionRcd>().get(m_tag.module(), det);

  LogInfo("DDCMS") << "Iterate over the detectors:\n";
  for( auto const& it : det->description()->detectors()) {
    dd4hep::DetElement det(it.second);
    LogInfo("DDCMS") << it.first << ": " << det.path() << "\n";
  }
  LogInfo("DDCMS") << "..done!\n";
  
  ESTransientHandle<DDVectorRegistry> registry;
  iEventSetup.get<DDVectorRegistryRcd>().get(m_tag.module(), registry);

  LogInfo("DDCMS") << "DD Vector Registry size: " << registry->vectors.size() << "\n";
  for( const auto& p: registry->vectors ) {
    LogInfo("DDCMS") << " " << p.first << " => ";
    for( const auto& i : p.second )
      LogInfo("DDCMS") << i << ", ";
    LogInfo("DDCMS") << '\n';
  }
}

void
DDCMSDetector::endJob()
{
}

DEFINE_FWK_MODULE( DDCMSDetector );
