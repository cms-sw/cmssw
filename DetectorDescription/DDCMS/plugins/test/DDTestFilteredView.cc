#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"

#include <iostream>

using namespace std;
using namespace cms;
using namespace edm;

class DDTestFilteredView : public one::EDAnalyzer<> {
public:
  explicit DDTestFilteredView(const ParameterSet& iConfig) : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")) {}

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const ESInputTag m_tag;
};

void DDTestFilteredView::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "DDTestFilteredView::analyze: " << m_tag;
  ESTransientHandle<DDCompactView> cpv;
  iEventSetup.get<IdealGeometryRecord>().get(m_tag, cpv);

  DDFilteredView fv(cpv->detector(), cpv->detector()->worldVolume());

  std::string attribute{"CMSCutsRegion"};
  cms::DDSpecParRefs ref;
  const cms::DDSpecParRegistry& mypar = cpv->specpars();
  mypar.filter(ref, attribute, "MuonChamber");
  fv.mergedSpecifics(ref);

  fv.printFilter();

  bool doLoop = fv.firstChild();
  LogVerbatim("Geometry") << "DDTestFilteredView::analyze: first child " << fv.path();
  while (doLoop) {
    edm::LogVerbatim("Geometry") << "DDTestFilteredView::analyze: next node = " << fv.path();
    doLoop = fv.firstChild();
  }

  LogVerbatim("Geometry") << "DDTestFilteredView::analyze done.";
}

DEFINE_FWK_MODULE(DDTestFilteredView);
