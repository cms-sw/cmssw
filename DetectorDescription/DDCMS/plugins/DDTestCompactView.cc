#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include <iostream>

using namespace std;
using namespace cms;
using namespace edm;

class DDTestCompactView : public one::EDAnalyzer<> {
public:
  explicit DDTestCompactView(const ParameterSet& iConfig) : m_tag(iConfig.getParameter<ESInputTag>("DDDetector")) {}

  void beginJob() override {}
  void analyze(Event const& iEvent, EventSetup const&) override;
  void endJob() override {}

private:
  const ESInputTag m_tag;
};

void DDTestCompactView::analyze(const Event&, const EventSetup& iEventSetup) {
  LogVerbatim("Geometry") << "DDTestCompactView::analyze: " << m_tag;
  ESTransientHandle<DDCompactView> cpv;
  iEventSetup.get<IdealGeometryRecord>().get(m_tag, cpv);

  LogVerbatim("Geometry") << "DDTestCompactView::analyze done.";
}

DEFINE_FWK_MODULE(DDTestCompactView);
