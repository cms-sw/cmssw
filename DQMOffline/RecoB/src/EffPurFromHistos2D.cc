#include "DQMOffline/RecoB/interface/EffPurFromHistos2D.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "TStyle.h"
#include "TCanvas.h"

#include <iostream>
#include <cmath>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace std;
using namespace RecoBTag;

EffPurFromHistos2D::EffPurFromHistos2D(const std::string& ext,
                                       TH2F* h_d,
                                       TH2F* h_u,
                                       TH2F* h_s,
                                       TH2F* h_c,
                                       TH2F* h_b,
                                       TH2F* h_g,
                                       TH2F* h_ni,
                                       TH2F* h_dus,
                                       TH2F* h_dusg,
                                       TH2F* h_pu,
                                       const std::string& label,
                                       unsigned int mc,
                                       int nBinX,
                                       double startOX,
                                       double endOX)
    : fromDiscriminatorDistr(false),
      mcPlots_(mc),
      doCTagPlots_(false),
      label_(label),
      histoExtension(ext),
      effVersusDiscr_d(h_d),
      effVersusDiscr_u(h_u),
      effVersusDiscr_s(h_s),
      effVersusDiscr_c(h_c),
      effVersusDiscr_b(h_b),
      effVersusDiscr_g(h_g),
      effVersusDiscr_ni(h_ni),
      effVersusDiscr_dus(h_dus),
      effVersusDiscr_dusg(h_dusg),
      effVersusDiscr_pu(h_pu),
      nBinOutputX(nBinX),
      startOutputX(startOX),
      endOutputX(endOX) {
  // consistency check
  check();
}

EffPurFromHistos2D::EffPurFromHistos2D(const FlavourHistograms2D<double, double>& dDiscriminatorFC,
                                       const std::string& label,
                                       unsigned int mc,
                                       DQMStore::IBooker& ibook,
                                       int nBinX,
                                       double startOX,
                                       double endOX)
    : fromDiscriminatorDistr(true),
      mcPlots_(mc),
      doCTagPlots_(false),
      label_(label),
      nBinOutputX(nBinX),
      startOutputX(startOX),
      endOutputX(endOX) {
  histoExtension = "_" + dDiscriminatorFC.baseNameTitle();

  discrNoCutEffic =
      std::make_unique<FlavourHistograms2D<double, double>>("totalEntries" + histoExtension,
                                                            "Total Entries: " + dDiscriminatorFC.baseNameDescription(),
                                                            dDiscriminatorFC.nBinsX(),
                                                            dDiscriminatorFC.lowerBoundX(),
                                                            dDiscriminatorFC.upperBoundX(),
                                                            dDiscriminatorFC.nBinsY(),
                                                            dDiscriminatorFC.lowerBoundY(),
                                                            dDiscriminatorFC.upperBoundY(),
                                                            false,
                                                            label,
                                                            mcPlots_,
                                                            false,
                                                            ibook);

  // conditional discriminator cut for efficiency histos
  discrCutEfficScan = std::make_unique<FlavourHistograms2D<double, double>>(
      "effVsDiscrCut" + histoExtension,
      "Eff. vs Disc. Cut: " + dDiscriminatorFC.baseNameDescription(),
      dDiscriminatorFC.nBinsX(),
      dDiscriminatorFC.lowerBoundX(),
      dDiscriminatorFC.upperBoundX(),
      dDiscriminatorFC.nBinsY(),
      dDiscriminatorFC.lowerBoundY(),
      dDiscriminatorFC.upperBoundY(),
      false,
      label,
      mcPlots_,
      false,
      ibook);
  discrCutEfficScan->SetMinimum(1E-4);

  if (mcPlots_) {
    if (mcPlots_ > 2) {
      effVersusDiscr_d = discrCutEfficScan->histo_d();
      effVersusDiscr_u = discrCutEfficScan->histo_u();
      effVersusDiscr_s = discrCutEfficScan->histo_s();
      effVersusDiscr_g = discrCutEfficScan->histo_g();
      effVersusDiscr_dus = discrCutEfficScan->histo_dus();
    } else {
      effVersusDiscr_d = nullptr;
      effVersusDiscr_u = nullptr;
      effVersusDiscr_s = nullptr;
      effVersusDiscr_g = nullptr;
      effVersusDiscr_dus = nullptr;
    }
    effVersusDiscr_c = discrCutEfficScan->histo_c();
    effVersusDiscr_b = discrCutEfficScan->histo_b();
    effVersusDiscr_ni = discrCutEfficScan->histo_ni();
    effVersusDiscr_dusg = discrCutEfficScan->histo_dusg();
    effVersusDiscr_pu = discrCutEfficScan->histo_pu();

    if (mcPlots_ > 2) {
      effVersusDiscr_d->SetXTitle("Discriminant");
      effVersusDiscr_d->GetXaxis()->SetTitleOffset(0.75);
      effVersusDiscr_u->SetXTitle("Discriminant");
      effVersusDiscr_u->GetXaxis()->SetTitleOffset(0.75);
      effVersusDiscr_s->SetXTitle("Discriminant");
      effVersusDiscr_s->GetXaxis()->SetTitleOffset(0.75);
      effVersusDiscr_g->SetXTitle("Discriminant");
      effVersusDiscr_g->GetXaxis()->SetTitleOffset(0.75);
      effVersusDiscr_dus->SetXTitle("Discriminant");
      effVersusDiscr_dus->GetXaxis()->SetTitleOffset(0.75);
    }
    effVersusDiscr_c->SetXTitle("Discriminant");
    effVersusDiscr_c->GetXaxis()->SetTitleOffset(0.75);
    effVersusDiscr_b->SetXTitle("Discriminant");
    effVersusDiscr_b->GetXaxis()->SetTitleOffset(0.75);
    effVersusDiscr_ni->SetXTitle("Discriminant");
    effVersusDiscr_ni->GetXaxis()->SetTitleOffset(0.75);
    effVersusDiscr_dusg->SetXTitle("Discriminant");
    effVersusDiscr_dusg->GetXaxis()->SetTitleOffset(0.75);
    effVersusDiscr_pu->SetXTitle("Discriminant");
    effVersusDiscr_pu->GetXaxis()->SetTitleOffset(0.75);
  } else {
    effVersusDiscr_d = nullptr;
    effVersusDiscr_u = nullptr;
    effVersusDiscr_s = nullptr;
    effVersusDiscr_c = nullptr;
    effVersusDiscr_b = nullptr;
    effVersusDiscr_g = nullptr;
    effVersusDiscr_ni = nullptr;
    effVersusDiscr_dus = nullptr;
    effVersusDiscr_dusg = nullptr;
    effVersusDiscr_pu = nullptr;
  }

  // discr. for computation
  vector<TH2F*> discrCfHistos = dDiscriminatorFC.getHistoVector();
  // discr no cut
  vector<TH2F*> discrNoCutHistos = discrNoCutEffic->getHistoVector();
  // discr no cut
  vector<TH2F*> discrCutHistos = discrCutEfficScan->getHistoVector();

  const int& dimHistos = discrCfHistos.size();  // they all have the same size

  // DISCR-CUT LOOP:
  // fill the histos for eff-pur computations by scanning the discriminatorFC histogram

  // better to loop over bins -> discrCut no longer needed
  const int& nBinsX = dDiscriminatorFC.nBinsX();
  const int& nBinsY = dDiscriminatorFC.nBinsY();

  // loop over flavours
  for (int iFlav = 0; iFlav < dimHistos; iFlav++) {
    if (discrCfHistos[iFlav] == nullptr)
      continue;
    discrNoCutHistos[iFlav]->SetXTitle("Discriminant A");
    discrNoCutHistos[iFlav]->GetXaxis()->SetTitleOffset(0.75);
    discrNoCutHistos[iFlav]->SetYTitle("Discriminant B");
    discrNoCutHistos[iFlav]->GetYaxis()->SetTitleOffset(0.75);

    // In Root histos, bin counting starts at 1 to nBins.
    // bin 0 is the underflow, and nBins+1 is the overflow.
    const double& nJetsFlav = discrCfHistos[iFlav]->GetEntries();

    for (int iDiscrX = nBinsX; iDiscrX > 0; --iDiscrX) {
      for (int iDiscrY = nBinsY; iDiscrY > 0; --iDiscrY) {
        // fill all jets into NoCut histo
        discrNoCutHistos[iFlav]->SetBinContent(iDiscrX, iDiscrY, nJetsFlav);
        discrNoCutHistos[iFlav]->SetBinError(iDiscrX, iDiscrY, sqrt(nJetsFlav));
        const double& sum = nJetsFlav - discrCfHistos[iFlav]->Integral(0, iDiscrX - 1, 0, iDiscrY - 1);
        discrCutHistos[iFlav]->SetBinContent(iDiscrX, iDiscrY, sum);
        discrCutHistos[iFlav]->SetBinError(iDiscrX, iDiscrY, sqrt(sum));
      }
    }
  }

  // divide to get efficiency vs. discriminator cut from absolute numbers
  discrCutEfficScan->divide(*discrNoCutEffic);  // does: histos including discriminator cut / flat histo
  discrCutEfficScan->setEfficiencyFlag();
}

EffPurFromHistos2D::~EffPurFromHistos2D() {}

void EffPurFromHistos2D::epsPlot(const std::string& name) {
  if (fromDiscriminatorDistr) {
    discrNoCutEffic->epsPlot(name);
    discrCutEfficScan->epsPlot(name);
  }
  plot(name, ".eps");
}

void EffPurFromHistos2D::psPlot(const std::string& name) { plot(name, ".ps"); }

void EffPurFromHistos2D::plot(const std::string& name, const std::string& ext) {
  std::string hX = "";
  std::string Title = "";
  if (!doCTagPlots_) {
    hX = "FlavEffVsBEff";
    Title = "b";
  } else {
    hX = "FlavEffVsCEff";
    Title = "c";
  }
  TCanvas tc((hX + histoExtension).c_str(),
             ("Flavour misidentification vs. " + Title + "-tagging efficiency " + histoExtension).c_str());
  plot(&tc);
  tc.Print((name + hX + histoExtension + ext).c_str());
}

void EffPurFromHistos2D::plot(TPad* plotCanvas /* = 0 */) {
  setTDRStyle()->cd();

  if (plotCanvas)
    plotCanvas->cd();

  gPad->UseCurrentStyle();
  gPad->SetFillColor(0);
  gPad->SetLogy(1);
  gPad->SetGridx(1);
  gPad->SetGridy(1);
}

void EffPurFromHistos2D::check() {
  // number of bins
  int nBinsX_d = 0;
  int nBinsX_u = 0;
  int nBinsX_s = 0;
  int nBinsX_g = 0;
  int nBinsX_dus = 0;
  int nBinsY_d = 0;
  int nBinsY_u = 0;
  int nBinsY_s = 0;
  int nBinsY_g = 0;
  int nBinsY_dus = 0;
  if (mcPlots_ > 2) {
    nBinsX_d = effVersusDiscr_d->GetNbinsX();
    nBinsX_u = effVersusDiscr_u->GetNbinsX();
    nBinsX_s = effVersusDiscr_s->GetNbinsX();
    nBinsX_g = effVersusDiscr_g->GetNbinsX();
    nBinsX_dus = effVersusDiscr_dus->GetNbinsX();
    nBinsY_d = effVersusDiscr_d->GetNbinsY();
    nBinsY_u = effVersusDiscr_u->GetNbinsY();
    nBinsY_s = effVersusDiscr_s->GetNbinsY();
    nBinsY_g = effVersusDiscr_g->GetNbinsY();
    nBinsY_dus = effVersusDiscr_dus->GetNbinsY();
  }
  const int& nBinsX_c = effVersusDiscr_c->GetNbinsX();
  const int& nBinsX_b = effVersusDiscr_b->GetNbinsX();
  const int& nBinsX_ni = effVersusDiscr_ni->GetNbinsX();
  const int& nBinsX_dusg = effVersusDiscr_dusg->GetNbinsX();
  const int& nBinsX_pu = effVersusDiscr_pu->GetNbinsX();
  const int& nBinsY_c = effVersusDiscr_c->GetNbinsY();
  const int& nBinsY_b = effVersusDiscr_b->GetNbinsY();
  const int& nBinsY_ni = effVersusDiscr_ni->GetNbinsY();
  const int& nBinsY_dusg = effVersusDiscr_dusg->GetNbinsY();
  const int& nBinsY_pu = effVersusDiscr_pu->GetNbinsY();

  const bool& lNBinsX =
      ((nBinsX_d == nBinsX_u && nBinsX_d == nBinsX_s && nBinsX_d == nBinsX_c && nBinsX_d == nBinsX_b &&
        nBinsX_d == nBinsX_g && nBinsX_d == nBinsX_ni && nBinsX_d == nBinsX_dus && nBinsX_d == nBinsX_dusg) ||
       (nBinsX_c == nBinsX_b && nBinsX_c == nBinsX_dusg && nBinsX_c == nBinsX_ni && nBinsX_c == nBinsX_pu));

  const bool& lNBinsY =
      ((nBinsY_d == nBinsY_u && nBinsY_d == nBinsY_s && nBinsY_d == nBinsY_c && nBinsY_d == nBinsY_b &&
        nBinsY_d == nBinsY_g && nBinsY_d == nBinsY_ni && nBinsY_d == nBinsY_dus && nBinsY_d == nBinsY_dusg) ||
       (nBinsY_c == nBinsY_b && nBinsY_c == nBinsY_dusg && nBinsY_c == nBinsY_ni && nBinsY_c == nBinsY_pu));

  if (!lNBinsX || !lNBinsY) {
    throw cms::Exception("Configuration") << "Input histograms do not all have the same number of bins!\n";
  }

  // start
  float sBin_d = 0;
  float sBin_u = 0;
  float sBin_s = 0;
  float sBin_g = 0;
  float sBin_dus = 0;
  if (mcPlots_ > 2) {
    sBin_d = effVersusDiscr_d->GetBinCenter(1);
    sBin_u = effVersusDiscr_u->GetBinCenter(1);
    sBin_s = effVersusDiscr_s->GetBinCenter(1);
    sBin_g = effVersusDiscr_g->GetBinCenter(1);
    sBin_dus = effVersusDiscr_dus->GetBinCenter(1);
  }
  const float& sBin_c = effVersusDiscr_c->GetBinCenter(1);
  const float& sBin_b = effVersusDiscr_b->GetBinCenter(1);
  const float& sBin_ni = effVersusDiscr_ni->GetBinCenter(1);
  const float& sBin_dusg = effVersusDiscr_dusg->GetBinCenter(1);
  const float& sBin_pu = effVersusDiscr_pu->GetBinCenter(1);

  const bool& lSBin = ((sBin_d == sBin_u && sBin_d == sBin_s && sBin_d == sBin_c && sBin_d == sBin_b &&
                        sBin_d == sBin_g && sBin_d == sBin_ni && sBin_d == sBin_dus && sBin_d == sBin_dusg) ||
                       (sBin_c == sBin_b && sBin_c == sBin_dusg && sBin_c == sBin_ni && sBin_c == sBin_pu));

  if (!lSBin) {
    throw cms::Exception("Configuration")
        << "EffPurFromHistos::check() : Input histograms do not all have the same start bin!\n";
  }

  // end
  float eBin_d = 0;
  float eBin_u = 0;
  float eBin_s = 0;
  float eBin_g = 0;
  float eBin_dus = 0;
  const int& binEnd = effVersusDiscr_b->GetBin(nBinsX_b - 1, nBinsY_b - 1);
  if (mcPlots_ > 2) {
    eBin_d = effVersusDiscr_d->GetBinCenter(binEnd);
    eBin_u = effVersusDiscr_u->GetBinCenter(binEnd);
    eBin_s = effVersusDiscr_s->GetBinCenter(binEnd);
    eBin_g = effVersusDiscr_g->GetBinCenter(binEnd);
    eBin_dus = effVersusDiscr_dus->GetBinCenter(binEnd);
  }
  const float& eBin_c = effVersusDiscr_c->GetBinCenter(binEnd);
  const float& eBin_b = effVersusDiscr_b->GetBinCenter(binEnd);
  const float& eBin_ni = effVersusDiscr_ni->GetBinCenter(binEnd);
  const float& eBin_dusg = effVersusDiscr_dusg->GetBinCenter(binEnd);
  const float& eBin_pu = effVersusDiscr_pu->GetBinCenter(binEnd);

  const bool& lEBin = ((eBin_d == eBin_u && eBin_d == eBin_s && eBin_d == eBin_c && eBin_d == eBin_b &&
                        eBin_d == eBin_g && eBin_d == eBin_ni && eBin_d == eBin_dus && eBin_d == eBin_dusg) ||
                       (eBin_c == eBin_b && eBin_c == eBin_dusg && eBin_c == eBin_ni && eBin_c == eBin_pu));

  if (!lEBin) {
    throw cms::Exception("Configuration")
        << "EffPurFromHistos::check() : Input histograms do not all have the same end bin!\n";
  }
}

void EffPurFromHistos2D::compute(DQMStore::IBooker& ibook, vector<double> fixedEff) {
  if (!mcPlots_ || fixedEff.empty()) {
    return;
  }

  // to have shorter names ......
  const std::string& hE = histoExtension;
  std::string hX = "DUSG_vs_B_eff_at_fixedCeff_";
  if (!doCTagPlots_)
    hX = "DUSG_vs_C_eff_at_fixedBeff_";

  // create histograms from base name and extension as given from user
  HistoProviderDQM prov("Btag", label_, ibook);

  for (unsigned int ieff = 0; ieff < fixedEff.size(); ieff++) {
    std::string fixedEfficiency = std::to_string(fixedEff[ieff]);
    fixedEfficiency.replace(1, 1, "_");
    X_vs_Y_eff_at_fixedZeff.push_back(
        (prov.book1D(hX + fixedEfficiency + hE, hX + fixedEfficiency + hE, nBinOutputX, startOutputX, endOutputX)));
    X_vs_Y_eff_at_fixedZeff[ieff]->setEfficiencyFlag();

    X_vs_Y_eff_at_fixedZeff[ieff]->setAxisTitle("Light mistag");
    X_vs_Y_eff_at_fixedZeff[ieff]->getTH1F()->SetYTitle("B mistag");
    if (!doCTagPlots_)
      X_vs_Y_eff_at_fixedZeff[ieff]->getTH1F()->SetYTitle("C mistag");
    X_vs_Y_eff_at_fixedZeff[ieff]->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
    X_vs_Y_eff_at_fixedZeff[ieff]->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
  }
  // loop over eff. vs. discriminator cut b-histo and look in which bin the closest entry is;
  // use fact that eff decreases monotonously

  // any of the histos to be created can be taken here:
  MonitorElement* EffFlavVsXEff = X_vs_Y_eff_at_fixedZeff[0];

  const int& nBinX = EffFlavVsXEff->getTH1F()->GetNbinsX();

  for (int iBinX = 1; iBinX <= nBinX; iBinX++) {  // loop over the bins on the x-axis of the histograms to be filled
    const float& effBinWidthX = EffFlavVsXEff->getTH1F()->GetBinWidth(iBinX);
    const float& effMidX = EffFlavVsXEff->getTH1F()->GetBinCenter(iBinX);  // middle of efficiency bin
    const float& effLeftX = effMidX - 0.5 * effBinWidthX;                  // left edge of bin
    const float& effRightX = effMidX + 0.5 * effBinWidthX;                 // right edge of bin

    vector<int> binClosest;
    if (doCTagPlots_)
      binClosest =
          findBinClosestYValueAtFixedZ(effVersusDiscr_dusg, effMidX, effLeftX, effRightX, effVersusDiscr_c, fixedEff);
    else
      binClosest =
          findBinClosestYValueAtFixedZ(effVersusDiscr_dusg, effMidX, effLeftX, effRightX, effVersusDiscr_b, fixedEff);

    for (unsigned int ieff = 0; ieff < binClosest.size(); ieff++) {
      const bool& binFound = (binClosest[ieff] > 0);
      //
      if (binFound) {
        // fill the histos
        if (doCTagPlots_) {
          X_vs_Y_eff_at_fixedZeff[ieff]->Fill(effMidX, effVersusDiscr_b->GetBinContent(binClosest[ieff]));
          X_vs_Y_eff_at_fixedZeff[ieff]->getTH1F()->SetBinError(iBinX, effVersusDiscr_b->GetBinError(binClosest[ieff]));
        } else {
          X_vs_Y_eff_at_fixedZeff[ieff]->Fill(effMidX, effVersusDiscr_c->GetBinContent(binClosest[ieff]));
          X_vs_Y_eff_at_fixedZeff[ieff]->getTH1F()->SetBinError(iBinX, effVersusDiscr_c->GetBinError(binClosest[ieff]));
        }
      }
    }
  }
}

#include <typeinfo>
