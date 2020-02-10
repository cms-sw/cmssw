#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"

#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <string>

//----------------------------------------------------------------------------------------------------

class GEMDQMSourceDigi : public DQMEDAnalyzer {
public:
  GEMDQMSourceDigi(const edm::ParameterSet& cfg);
  ~GEMDQMSourceDigi() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override{};
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& e, edm::EventSetup const& eSetup) override;

private:
  edm::EDGetToken tagDigi_;

  const GEMGeometry* initGeometry(edm::EventSetup const& iSetup);
  int findVFAT(float min_, float max_, float x_, int roll_);

  const GEMGeometry* GEMGeometry_;

  std::vector<GEMChamber> gemChambers_;

  std::unordered_map<UInt_t, MonitorElement*> Digi_2D_;
  std::unordered_map<UInt_t, MonitorElement*> Digi_1D_;
  std::unordered_map<UInt_t, MonitorElement*> BxVsVFAT;
};

using namespace std;
using namespace edm;

int GEMDQMSourceDigi::findVFAT(float min_, float max_, float x_, int roll_) {
  float step = max_ / 3;
  if (x_ < (min_ + step)) {
    return 8 - roll_;
  } else if (x_ < (min_ + 2 * step)) {
    return 16 - roll_;
  } else {
    return 24 - roll_;
  }
}

const GEMGeometry* GEMDQMSourceDigi::initGeometry(edm::EventSetup const& iSetup) {
  const GEMGeometry* GEMGeometry_ = nullptr;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  } catch (edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("MuonGEMBaseValidation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return nullptr;
  }

  return GEMGeometry_;
}

GEMDQMSourceDigi::GEMDQMSourceDigi(const edm::ParameterSet& cfg) {
  tagDigi_ = consumes<GEMDigiCollection>(cfg.getParameter<edm::InputTag>("digisInputLabel"));
}

void GEMDQMSourceDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digisInputLabel", edm::InputTag("muonGEMDigis", ""));
  descriptions.add("GEMDQMSourceDigi", desc);
}

void GEMDQMSourceDigi::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const& iSetup) {
  GEMGeometry_ = initGeometry(iSetup);
  if (GEMGeometry_ == nullptr)
    return;

  const std::vector<const GEMSuperChamber*>& superChambers_ = GEMGeometry_->superChambers();
  for (auto sch : superChambers_) {
    int n_lay = sch->nChambers();
    for (int l = 0; l < n_lay; l++) {
      Bool_t bExist = false;
      for (auto ch : gemChambers_)
        if (ch.id() == sch->chamber(l + 1)->id())
          bExist = true;
      if (bExist)
        continue;

      gemChambers_.push_back(*sch->chamber(l + 1));
    }
  }
  ibooker.cd();
  ibooker.setCurrentFolder("GEM/digi");
  for (auto ch : gemChambers_) {
    GEMDetId gid = ch.id();

    std::string strIdxName = "Gemini_" + to_string(gid.chamber()) + "_GE" + (gid.region() > 0 ? "p" : "m") +
                             to_string(gid.station()) + "_" + to_string(gid.layer());
    std::string strIdxTitle = "GEMINIm" + to_string(gid.chamber()) + " in GE" + (gid.region() > 0 ? "+" : "-") +
                              to_string(gid.station()) + "/" + to_string(gid.layer());

    string hName_digi = "Digi_Strips_" + strIdxName;
    string hTitle_digi = "Digi Strip " + strIdxTitle;
    string hAxis_digi = ";Strip;iEta";
    Digi_2D_[ch.id()] = ibooker.book2D(hName_digi, hTitle_digi + hAxis_digi, 384, 1, 385, 8, 0.5, 8.5);
    Digi_1D_[ch.id()] = ibooker.book1D(hName_digi + "_VFAT", hTitle_digi + " VFAT" + hAxis_digi, 24, 0, 24);

    string hNameBx = "bx_vs_VFAT_" + strIdxName;
    string hTitleBx = "bx vs VFAT " + strIdxTitle;
    hTitleBx += ";Bunch crossing;VFAT";
    BxVsVFAT[ch.id()] = ibooker.book2D(hNameBx, hTitleBx, 10, -5, 5, 24, 0, 24);
  }
}

void GEMDQMSourceDigi::analyze(edm::Event const& event, edm::EventSetup const& eventSetup) {
  const GEMGeometry* GEMGeometry_ = initGeometry(eventSetup);
  if (GEMGeometry_ == nullptr)
    return;

  edm::Handle<GEMDigiCollection> gemDigis;
  event.getByToken(this->tagDigi_, gemDigis);
  for (auto ch : gemChambers_) {
    GEMDetId cId = ch.id();
    for (auto roll : ch.etaPartitions()) {
      GEMDetId rId = roll->id();
      const auto& digis_in_det = gemDigis->get(rId);
      for (auto d = digis_in_det.first; d != digis_in_det.second; ++d) {
        auto nVFAT = findVFAT(1, roll->nstrips(), d->strip(), rId.roll());
        Digi_2D_[cId]->Fill(d->strip(), rId.roll());
        Digi_1D_[cId]->Fill(nVFAT);
        BxVsVFAT[cId]->Fill(d->bx(), nVFAT);
      }
    }
  }
}

DEFINE_FWK_MODULE(GEMDQMSourceDigi);
