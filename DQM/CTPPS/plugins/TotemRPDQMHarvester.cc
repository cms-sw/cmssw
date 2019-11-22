/****************************************************************************
*
* This is a part of TotemDQM and TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/CTPPSDetId/interface/TotemRPDetId.h"

//----------------------------------------------------------------------------------------------------

class TotemRPDQMHarvester : public DQMEDHarvester {
public:
  TotemRPDQMHarvester(const edm::ParameterSet &ps);
  ~TotemRPDQMHarvester() override;

protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override {}

  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             const edm::LuminosityBlock &,
                             const edm::EventSetup &) override;

private:
  void MakeHitNumberRatios(unsigned int id, DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter);

  void MakePlaneEfficiencyHistograms(unsigned int id,
                                     DQMStore::IBooker &ibooker,
                                     DQMStore::IGetter &igetter,
                                     bool &rpPlotInitialized);
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

TotemRPDQMHarvester::TotemRPDQMHarvester(const edm::ParameterSet &ps) {}

//----------------------------------------------------------------------------------------------------

TotemRPDQMHarvester::~TotemRPDQMHarvester() {}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMHarvester::MakeHitNumberRatios(unsigned int id, DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter) {
  // get source histogram
  string path;
  TotemRPDetId(id).rpName(path, TotemRPDetId::nPath);

  MonitorElement *activity = igetter.get(path + "/activity in planes (2D)");

  if (!activity)
    return;

  // book new histogram, if not yet done
  const string hit_ratio_name = "hit ratio in hot spot";
  MonitorElement *hit_ratio = igetter.get(path + "/" + hit_ratio_name);

  if (hit_ratio == nullptr) {
    ibooker.setCurrentFolder(path);
    string title;
    TotemRPDetId(id).rpName(title, TotemRPDetId::nFull);
    hit_ratio = ibooker.book1D(hit_ratio_name, title + ";plane;N_hits(320<strip<440) / N_hits(all)", 10, -0.5, 9.5);
  } else {
    hit_ratio->getTH1F()->Reset();
  }

  // calculate ratios
  TAxis *y_axis = activity->getTH2F()->GetYaxis();
  for (int bix = 1; bix <= activity->getNbinsX(); ++bix) {
    double S_full = 0., S_sel = 0.;
    for (int biy = 1; biy <= activity->getNbinsY(); ++biy) {
      double c = activity->getBinContent(bix, biy);
      double s = y_axis->GetBinCenter(biy);

      S_full += c;

      if (s > 320. && s < 440.)
        S_sel += c;
    }

    double r = (S_full > 0.) ? S_sel / S_full : 0.;

    hit_ratio->setBinContent(bix, r);
  }
}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMHarvester::MakePlaneEfficiencyHistograms(unsigned int id,
                                                        DQMStore::IBooker &ibooker,
                                                        DQMStore::IGetter &igetter,
                                                        bool &rpPlotInitialized) {
  TotemRPDetId detId(id);

  // get source histograms
  string path;
  detId.planeName(path, TotemRPDetId::nPath);

  MonitorElement *efficiency_num = igetter.get(path + "/efficiency num");
  MonitorElement *efficiency_den = igetter.get(path + "/efficiency den");

  if (!efficiency_num || !efficiency_den)
    return;

  // book new plane histogram, if not yet done
  const string efficiency_name = "efficiency";
  MonitorElement *efficiency = igetter.get(path + "/" + efficiency_name);

  if (efficiency == nullptr) {
    string title;
    detId.planeName(title, TotemRPDetId::nFull);
    TAxis *axis = efficiency_den->getTH1()->GetXaxis();
    ibooker.setCurrentFolder(path);
    efficiency = ibooker.book1D(
        efficiency_name, title + ";track position   (mm)", axis->GetNbins(), axis->GetXmin(), axis->GetXmax());
  } else {
    efficiency->getTH1F()->Reset();
  }

  // book new RP histogram, if not yet done
  CTPPSDetId rpId = detId.rpId();
  rpId.rpName(path, TotemRPDetId::nPath);
  const string rp_efficiency_name = "plane efficiency";
  MonitorElement *rp_efficiency = igetter.get(path + "/" + rp_efficiency_name);

  if (rp_efficiency == nullptr) {
    string title;
    rpId.rpName(title, TotemRPDetId::nFull);
    TAxis *axis = efficiency_den->getTH1()->GetXaxis();
    ibooker.setCurrentFolder(path);
    rp_efficiency = ibooker.book2D(rp_efficiency_name,
                                   title + ";plane;track position   (mm)",
                                   10,
                                   -0.5,
                                   9.5,
                                   axis->GetNbins(),
                                   axis->GetXmin(),
                                   axis->GetXmax());
    rpPlotInitialized = true;
  } else {
    if (!rpPlotInitialized)
      rp_efficiency->getTH2F()->Reset();
    rpPlotInitialized = true;
  }

  // calculate and fill efficiencies
  for (signed int bi = 1; bi <= efficiency->getNbinsX(); bi++) {
    double num = efficiency_num->getBinContent(bi);
    double den = efficiency_den->getBinContent(bi);

    if (den > 0) {
      double p = num / den;
      double p_unc = sqrt(p * (1. - p) / den);
      efficiency->setBinContent(bi, p);
      efficiency->setBinError(bi, p_unc);

      int pl_bi = detId.plane() + 1;
      rp_efficiency->setBinContent(pl_bi, bi, p);
    } else {
      efficiency->setBinContent(bi, 0.);
    }
  }
}

//----------------------------------------------------------------------------------------------------

void TotemRPDQMHarvester::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker,
                                                DQMStore::IGetter &igetter,
                                                const edm::LuminosityBlock &,
                                                const edm::EventSetup &) {
  // loop over arms
  for (unsigned int arm = 0; arm < 2; arm++) {
    // loop over stations
    for (unsigned int st = 0; st < 3; st += 2) {
      // loop over RPs
      for (unsigned int rp = 0; rp < 6; ++rp) {
        if (st == 2) {
          // unit 220-nr is not equipped
          if (rp <= 2)
            continue;

          // RP 220-fr-hr contains pixels
          if (rp == 3)
            continue;
        }

        TotemRPDetId rpId(arm, st, rp);

        MakeHitNumberRatios(rpId, ibooker, igetter);

        bool rpPlotInitialized = false;

        // loop over planes
        for (unsigned int pl = 0; pl < 10; ++pl) {
          TotemRPDetId plId(arm, st, rp, pl);

          MakePlaneEfficiencyHistograms(plId, ibooker, igetter, rpPlotInitialized);
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemRPDQMHarvester);
