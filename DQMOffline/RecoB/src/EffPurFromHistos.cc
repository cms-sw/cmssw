#include "DQMOffline/RecoB/interface/EffPurFromHistos.h"
#include "DQMOffline/RecoB/interface/Tools.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TStyle.h"
#include "TCanvas.h"

#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

using namespace std;
using namespace RecoBTag;

EffPurFromHistos::EffPurFromHistos(const std::string& ext,
                                   TH1F* h_d,
                                   TH1F* h_u,
                                   TH1F* h_s,
                                   TH1F* h_c,
                                   TH1F* h_b,
                                   TH1F* h_g,
                                   TH1F* h_ni,
                                   TH1F* h_dus,
                                   TH1F* h_dusg,
                                   TH1F* h_pu,
                                   const std::string& label,
                                   unsigned int mc,
                                   int nBin,
                                   double startO,
                                   double endO)
    : fromDiscriminatorDistr(false),
      mcPlots_(mc),
      signalFlavour_("B"),
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
      nBinOutput(nBin),
      startOutput(startO),
      endOutput(endO) {
  // consistency check
  check();
}

EffPurFromHistos::EffPurFromHistos(const FlavourHistograms<double>& dDiscriminatorFC,
                                   const std::string& label,
                                   unsigned int mc,
                                   DQMStore::IBooker& ibook,
                                   int nBin,
                                   double startO,
                                   double endO)
    : fromDiscriminatorDistr(true),
      mcPlots_(mc),
      signalFlavour_("B"),
      label_(label),
      nBinOutput(nBin),
      startOutput(startO),
      endOutput(endO) {
  histoExtension = "_" + dDiscriminatorFC.baseNameTitle();

  discrNoCutEffic =
      std::make_unique<FlavourHistograms<double>>("totalEntries" + histoExtension,
                                                  "Total Entries: " + dDiscriminatorFC.baseNameDescription(),
                                                  dDiscriminatorFC.nBins(),
                                                  dDiscriminatorFC.lowerBound(),
                                                  dDiscriminatorFC.upperBound(),
                                                  false,
                                                  true,
                                                  false,
                                                  "b",
                                                  label,
                                                  mcPlots_,
                                                  ibook);

  // conditional discriminator cut for efficiency histos

  discrCutEfficScan =
      std::make_unique<FlavourHistograms<double>>("effVsDiscrCut" + histoExtension,
                                                  "Eff. vs Disc. Cut: " + dDiscriminatorFC.baseNameDescription(),
                                                  dDiscriminatorFC.nBins(),
                                                  dDiscriminatorFC.lowerBound(),
                                                  dDiscriminatorFC.upperBound(),
                                                  false,
                                                  true,
                                                  false,
                                                  "b",
                                                  label,
                                                  mcPlots_,
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
  vector<TH1F*> discrCfHistos = dDiscriminatorFC.getHistoVector();

  // discr no cut
  vector<TH1F*> discrNoCutHistos = discrNoCutEffic->getHistoVector();

  // discr no cut
  vector<TH1F*> discrCutHistos = discrCutEfficScan->getHistoVector();

  const int& dimHistos = discrCfHistos.size();  // they all have the same size

  // DISCR-CUT LOOP:
  // fill the histos for eff-pur computations by scanning the discriminatorFC histogram

  // better to loop over bins -> discrCut no longer needed
  const int& nBins = dDiscriminatorFC.nBins();

  // loop over flavours
  for (int iFlav = 0; iFlav < dimHistos; iFlav++) {
    if (discrCfHistos[iFlav] == nullptr)
      continue;
    discrNoCutHistos[iFlav]->SetXTitle("Discriminant");
    discrNoCutHistos[iFlav]->GetXaxis()->SetTitleOffset(0.75);

    // In Root histos, bin counting starts at 1 to nBins.
    // bin 0 is the underflow, and nBins+1 is the overflow.
    const double& nJetsFlav = discrCfHistos[iFlav]->GetEntries();
    double sum = discrCfHistos[iFlav]->GetBinContent(nBins + 1);  //+1 to get the overflow.

    for (int iDiscr = nBins; iDiscr > 0; --iDiscr) {
      // fill all jets into NoCut histo
      discrNoCutHistos[iFlav]->SetBinContent(iDiscr, nJetsFlav);
      discrNoCutHistos[iFlav]->SetBinError(iDiscr, sqrt(nJetsFlav));
      sum += discrCfHistos[iFlav]->GetBinContent(iDiscr);
      discrCutHistos[iFlav]->SetBinContent(iDiscr, sum);
      discrCutHistos[iFlav]->SetBinError(iDiscr, sqrt(sum));
    }
  }

  // divide to get efficiency vs. discriminator cut from absolute numbers
  discrCutEfficScan->divide(*discrNoCutEffic);  // does: histos including discriminator cut / flat histo
  discrCutEfficScan->setEfficiencyFlag();
}

EffPurFromHistos::~EffPurFromHistos() {}

void EffPurFromHistos::epsPlot(const std::string& name) {
  if (fromDiscriminatorDistr) {
    discrNoCutEffic->epsPlot(name);
    discrCutEfficScan->epsPlot(name);
  }
  plot(name, ".eps");
}

void EffPurFromHistos::psPlot(const std::string& name) { plot(name, ".ps"); }

void EffPurFromHistos::plot(const std::string& name, const std::string& ext) {
  const std::string hX = "FlavEffVs" + signalFlavour_ + "Eff";
  std::string Title = signalFlavour_;
  std::transform(Title.begin(), Title.end(), Title.begin(), ::tolower);
  TCanvas tc((hX + histoExtension).c_str(),
             ("Flavour misidentification vs. " + Title + "-tagging efficiency " + histoExtension).c_str());
  plot(&tc);
  tc.Print((name + hX + histoExtension + ext).c_str());
}

void EffPurFromHistos::plot(TPad* plotCanvas /* = 0 */) {
  //fixme:
  bool btppNI = false;
  bool btppColour = true;

  //   if ( !btppTitle ) gStyle->SetOptTitle( 0 );
  setTDRStyle()->cd();

  if (plotCanvas)
    plotCanvas->cd();

  gPad->UseCurrentStyle();
  gPad->SetFillColor(0);
  gPad->SetLogy(1);
  gPad->SetGridx(1);
  gPad->SetGridy(1);

  int col_c;
  int col_g;
  int col_dus;
  int col_ni;

  int mStyle_c;
  int mStyle_g;
  int mStyle_dus;
  int mStyle_ni;

  // marker size(same for all)
  float mSize = gPad->GetWh() * gPad->GetHNDC() / 500.;  //1.2;

  if (btppColour) {
    col_c = 6;
    col_g = 3;    // g in green
    col_dus = 4;  // uds in blue
    col_ni = 5;   // ni in ??
    mStyle_c = 20;
    mStyle_g = 20;
    mStyle_dus = 20;
    mStyle_ni = 20;
  } else {
    col_c = 1;
    col_g = 1;
    col_dus = 1;
    col_ni = 1;
    mStyle_c = 22;
    mStyle_g = 29;
    mStyle_dus = 20;
    mStyle_ni = 27;
  }

  TString Title = signalFlavour_;
  Title.ToLower();

  // for the moment: plot c,dus,g
  if (mcPlots_ > 2) {
    EffFlavVsXEff_dus->getTH1F()->GetXaxis()->SetTitle(Title + "-jet efficiency");
    EffFlavVsXEff_dus->getTH1F()->GetYaxis()->SetTitle("non " + Title + "-jet efficiency");
    EffFlavVsXEff_dus->getTH1F()->GetYaxis()->SetTitleOffset(0.25);
    EffFlavVsXEff_dus->getTH1F()->SetMaximum(1.1);
    EffFlavVsXEff_dus->getTH1F()->SetMinimum(1.e-5);
    EffFlavVsXEff_dus->getTH1F()->SetMarkerColor(col_dus);
    EffFlavVsXEff_dus->getTH1F()->SetLineColor(col_dus);
    EffFlavVsXEff_dus->getTH1F()->SetMarkerSize(mSize);
    EffFlavVsXEff_dus->getTH1F()->SetMarkerStyle(mStyle_dus);
    EffFlavVsXEff_dus->getTH1F()->SetStats(false);
    EffFlavVsXEff_dus->getTH1F()->Draw("pe");

    EffFlavVsXEff_g->getTH1F()->SetMarkerColor(col_g);
    EffFlavVsXEff_g->getTH1F()->SetLineColor(col_g);
    EffFlavVsXEff_g->getTH1F()->SetMarkerSize(mSize);
    EffFlavVsXEff_g->getTH1F()->SetMarkerStyle(mStyle_g);
    EffFlavVsXEff_g->getTH1F()->SetStats(false);
    EffFlavVsXEff_g->getTH1F()->Draw("peSame");
  }
  EffFlavVsXEff_c->getTH1F()->SetMarkerColor(col_c);
  EffFlavVsXEff_c->getTH1F()->SetLineColor(col_c);
  EffFlavVsXEff_c->getTH1F()->SetMarkerSize(mSize);
  EffFlavVsXEff_c->getTH1F()->SetMarkerStyle(mStyle_c);
  EffFlavVsXEff_c->getTH1F()->SetStats(false);
  EffFlavVsXEff_c->getTH1F()->Draw("peSame");

  if (mcPlots_ > 2) {
    EffFlavVsXEff_d->getTH1F()->SetMinimum(0.01);
    EffFlavVsXEff_u->getTH1F()->SetMinimum(0.01);
    EffFlavVsXEff_s->getTH1F()->SetMinimum(0.01);
    EffFlavVsXEff_g->getTH1F()->SetMinimum(0.01);
    EffFlavVsXEff_dus->getTH1F()->SetMinimum(0.01);
  }
  EffFlavVsXEff_c->getTH1F()->SetMinimum(0.01);
  EffFlavVsXEff_b->getTH1F()->SetMinimum(0.01);
  EffFlavVsXEff_ni->getTH1F()->SetMinimum(0.01);
  EffFlavVsXEff_dusg->getTH1F()->SetMinimum(0.01);
  EffFlavVsXEff_pu->getTH1F()->SetMinimum(0.01);

  // plot separately u,d and s
  //  EffFlavVsXEff_d ->GetXaxis()->SetTitle( Title + "-jet efficiency" );
  //  EffFlavVsXEff_d ->GetYaxis()->SetTitle( "non " + Title + "-jet efficiency" );
  //  EffFlavVsXEff_d ->GetYaxis()->SetTitleOffset( 1.25 );
  //  EffFlavVsXEff_d ->SetMaximum    ( 1.1 );
  //  EffFlavVsXEff_d ->SetMinimum    ( 1.e-5 );
  //  EffFlavVsXEff_d ->SetMarkerColor( col_dus );
  //  EffFlavVsXEff_d ->SetLineColor  ( col_dus );
  //  EffFlavVsXEff_d ->SetMarkerSize ( mSize );
  //  EffFlavVsXEff_d ->SetMarkerStyle( mStyle_dus );
  //  EffFlavVsXEff_d ->SetStats    ( false );
  //  EffFlavVsXEff_d ->Draw("pe");
  //
  //  EffFlavVsXEff_u   ->SetMarkerColor( col_g );
  //  EffFlavVsXEff_u   ->SetLineColor  ( col_g );
  //  EffFlavVsXEff_u   ->SetMarkerSize ( mSize );
  //  EffFlavVsXEff_u   ->SetMarkerStyle( mStyle_g );
  //  EffFlavVsXEff_u   ->SetStats    ( false );
  //  EffFlavVsXEff_u   ->Draw("peSame");
  //
  //  EffFlavVsXEff_s   ->SetMarkerColor( col_c );
  //  EffFlavVsXEff_s   ->SetLineColor  ( col_c );
  //  EffFlavVsXEff_s   ->SetMarkerSize ( mSize );
  //  EffFlavVsXEff_s   ->SetMarkerStyle( mStyle_c );
  //  EffFlavVsXEff_s   ->SetStats    ( false );
  //  EffFlavVsXEff_s   ->Draw("peSame");

  // only if asked: NI
  if (btppNI) {
    EffFlavVsXEff_ni->getTH1F()->SetMarkerColor(col_ni);
    EffFlavVsXEff_ni->getTH1F()->SetLineColor(col_ni);
    EffFlavVsXEff_ni->getTH1F()->SetMarkerSize(mSize);
    EffFlavVsXEff_ni->getTH1F()->SetMarkerStyle(mStyle_ni);
    EffFlavVsXEff_ni->getTH1F()->SetStats(false);
    EffFlavVsXEff_ni->getTH1F()->Draw("peSame");
  }
}

void EffPurFromHistos::check() {
  // number of bins

  int nBins_d = 0;
  int nBins_u = 0;
  int nBins_s = 0;
  int nBins_g = 0;
  int nBins_dus = 0;
  if (mcPlots_ > 2) {
    nBins_d = effVersusDiscr_d->GetNbinsX();
    nBins_u = effVersusDiscr_u->GetNbinsX();
    nBins_s = effVersusDiscr_s->GetNbinsX();
    nBins_g = effVersusDiscr_g->GetNbinsX();
    nBins_dus = effVersusDiscr_dus->GetNbinsX();
  }
  const int& nBins_c = effVersusDiscr_c->GetNbinsX();
  const int& nBins_b = effVersusDiscr_b->GetNbinsX();
  const int& nBins_ni = effVersusDiscr_ni->GetNbinsX();
  const int& nBins_dusg = effVersusDiscr_dusg->GetNbinsX();
  const int& nBins_pu = effVersusDiscr_pu->GetNbinsX();

  const bool& lNBins = ((nBins_d == nBins_u && nBins_d == nBins_s && nBins_d == nBins_c && nBins_d == nBins_b &&
                         nBins_d == nBins_g && nBins_d == nBins_ni && nBins_d == nBins_dus && nBins_d == nBins_dusg) ||
                        (nBins_c == nBins_b && nBins_c == nBins_dusg && nBins_c == nBins_ni && nBins_c == nBins_pu));

  if (!lNBins) {
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
  if (mcPlots_ > 2) {
    eBin_d = effVersusDiscr_d->GetBinCenter(nBins_d - 1);
    eBin_u = effVersusDiscr_u->GetBinCenter(nBins_d - 1);
    eBin_s = effVersusDiscr_s->GetBinCenter(nBins_d - 1);
    eBin_g = effVersusDiscr_g->GetBinCenter(nBins_d - 1);
    eBin_dus = effVersusDiscr_dus->GetBinCenter(nBins_d - 1);
  }
  const float& eBin_c = effVersusDiscr_c->GetBinCenter(nBins_d - 1);
  const float& eBin_b = effVersusDiscr_b->GetBinCenter(nBins_d - 1);
  const float& eBin_ni = effVersusDiscr_ni->GetBinCenter(nBins_d - 1);
  const float& eBin_dusg = effVersusDiscr_dusg->GetBinCenter(nBins_d - 1);
  const float& eBin_pu = effVersusDiscr_pu->GetBinCenter(nBins_d - 1);

  const bool& lEBin = ((eBin_d == eBin_u && eBin_d == eBin_s && eBin_d == eBin_c && eBin_d == eBin_b &&
                        eBin_d == eBin_g && eBin_d == eBin_ni && eBin_d == eBin_dus && eBin_d == eBin_dusg) ||
                       (eBin_c == eBin_b && eBin_c == eBin_dusg && eBin_c == eBin_ni && eBin_c == eBin_pu));

  if (!lEBin) {
    throw cms::Exception("Configuration")
        << "EffPurFromHistos::check() : Input histograms do not all have the same end bin!\n";
  }
}

TH1F* EffPurFromHistos::effVersusDiscr(const std::string& flavour) const {
  if (flavour == "B")
    return effVersusDiscr_b;
  if (flavour == "C")
    return effVersusDiscr_c;
  if (flavour == "DUSG")
    return effVersusDiscr_dusg;
  if (flavour == "NI")
    return effVersusDiscr_ni;
  if (flavour == "PU")
    return effVersusDiscr_pu;
  if (mcPlots_ > 2) {
    if (flavour == "D")
      return effVersusDiscr_d;
    if (flavour == "U")
      return effVersusDiscr_u;
    if (flavour == "S")
      return effVersusDiscr_s;
    if (flavour == "G")
      return effVersusDiscr_g;
    if (flavour == "DUS")
      return effVersusDiscr_dus;
  }
  return nullptr;
}

double EffPurFromHistos::nJets(const std::string& flavour) const {
  // only known when built from the discriminator distributions (totalEntries = number of jets in every bin);
  // the d/u/s/g/dus histograms exist only for mcPlots > 2, so they are not touched otherwise
  if (!fromDiscriminatorDistr || !discrNoCutEffic || effVersusDiscr(flavour) == nullptr)
    return -1.;
  const FlavourHistograms<double>& total = *discrNoCutEffic;
  TH1F* h = nullptr;
  if (flavour == "B")
    h = total.histo_b();
  else if (flavour == "C")
    h = total.histo_c();
  else if (flavour == "DUSG")
    h = total.histo_dusg();
  else if (flavour == "NI")
    h = total.histo_ni();
  else if (flavour == "PU")
    h = total.histo_pu();
  else if (flavour == "D")
    h = total.histo_d();
  else if (flavour == "U")
    h = total.histo_u();
  else if (flavour == "S")
    h = total.histo_s();
  else if (flavour == "G")
    h = total.histo_g();
  else if (flavour == "DUS")
    h = total.histo_dus();
  return h == nullptr ? -1. : h->GetBinContent(1);
}

void EffPurFromHistos::compute(DQMStore::IBooker& ibook) {
  if (!mcPlots_) {
    EffFlavVsXEff_d = nullptr;
    EffFlavVsXEff_u = nullptr;
    EffFlavVsXEff_s = nullptr;
    EffFlavVsXEff_c = nullptr;
    EffFlavVsXEff_b = nullptr;
    EffFlavVsXEff_g = nullptr;
    EffFlavVsXEff_ni = nullptr;
    EffFlavVsXEff_dus = nullptr;
    EffFlavVsXEff_dusg = nullptr;
    EffFlavVsXEff_pu = nullptr;
    return;
  }

  // to have shorter names ......
  const std::string& hE = histoExtension;
  // the signal flavour defines the x axis; fall back to b if it is not available at this MC level
  if (effVersusDiscr(signalFlavour_) == nullptr) {
    edm::LogWarning("EffPurFromHistos") << "Signal flavour '" << signalFlavour_ << "' not available for "
                                        << histoExtension << " (MC level " << mcPlots_ << "), using B instead";
    signalFlavour_ = "B";
  }
  const std::string hX = "FlavEffVs" + signalFlavour_ + "Eff_";
  TString Title = signalFlavour_;
  Title.ToLower();

  // create histograms from base name and extension as given from user
  // BINNING MUST BE IDENTICAL FOR ALL OF THEM!!
  HistoProviderDQM prov("Btag", label_, ibook);
  if (mcPlots_ > 2) {
    EffFlavVsXEff_d = (prov.book1D(hX + "D" + hE, hX + "D" + hE, nBinOutput, startOutput, endOutput));
    EffFlavVsXEff_d->setEfficiencyFlag();
    EffFlavVsXEff_u = (prov.book1D(hX + "U" + hE, hX + "U" + hE, nBinOutput, startOutput, endOutput));
    EffFlavVsXEff_u->setEfficiencyFlag();
    EffFlavVsXEff_s = (prov.book1D(hX + "S" + hE, hX + "S" + hE, nBinOutput, startOutput, endOutput));
    EffFlavVsXEff_s->setEfficiencyFlag();
    EffFlavVsXEff_g = (prov.book1D(hX + "G" + hE, hX + "G" + hE, nBinOutput, startOutput, endOutput));
    EffFlavVsXEff_g->setEfficiencyFlag();
    EffFlavVsXEff_dus = (prov.book1D(hX + "DUS" + hE, hX + "DUS" + hE, nBinOutput, startOutput, endOutput));
    EffFlavVsXEff_dus->setEfficiencyFlag();
  } else {
    EffFlavVsXEff_d = nullptr;
    EffFlavVsXEff_u = nullptr;
    EffFlavVsXEff_s = nullptr;
    EffFlavVsXEff_g = nullptr;
    EffFlavVsXEff_dus = nullptr;
  }
  EffFlavVsXEff_c = (prov.book1D(hX + "C" + hE, hX + "C" + hE, nBinOutput, startOutput, endOutput));
  EffFlavVsXEff_c->setEfficiencyFlag();
  EffFlavVsXEff_b = (prov.book1D(hX + "B" + hE, hX + "B" + hE, nBinOutput, startOutput, endOutput));
  EffFlavVsXEff_b->setEfficiencyFlag();
  EffFlavVsXEff_ni = (prov.book1D(hX + "NI" + hE, hX + "NI" + hE, nBinOutput, startOutput, endOutput));
  EffFlavVsXEff_ni->setEfficiencyFlag();
  EffFlavVsXEff_dusg = (prov.book1D(hX + "DUSG" + hE, hX + "DUSG" + hE, nBinOutput, startOutput, endOutput));
  EffFlavVsXEff_dusg->setEfficiencyFlag();
  EffFlavVsXEff_pu = (prov.book1D(hX + "PU" + hE, hX + "PU" + hE, nBinOutput, startOutput, endOutput));
  EffFlavVsXEff_pu->setEfficiencyFlag();

  if (mcPlots_ > 2) {
    EffFlavVsXEff_d->getTH1F()->SetXTitle(Title + "-jet efficiency");
    EffFlavVsXEff_d->getTH1F()->SetYTitle("non " + Title + "-jet efficiency");
    EffFlavVsXEff_d->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
    EffFlavVsXEff_d->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
    EffFlavVsXEff_u->getTH1F()->SetXTitle(Title + "-jet efficiency");
    EffFlavVsXEff_u->getTH1F()->SetYTitle("non " + Title + "-jet efficiency");
    EffFlavVsXEff_u->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
    EffFlavVsXEff_u->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
    EffFlavVsXEff_s->getTH1F()->SetXTitle(Title + "-jet efficiency");
    EffFlavVsXEff_s->getTH1F()->SetYTitle("non " + Title + "-jet efficiency");
    EffFlavVsXEff_s->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
    EffFlavVsXEff_s->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
    EffFlavVsXEff_g->getTH1F()->SetXTitle(Title + "-jet efficiency");
    EffFlavVsXEff_g->getTH1F()->SetYTitle("non " + Title + "-jet efficiency");
    EffFlavVsXEff_g->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
    EffFlavVsXEff_g->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
    EffFlavVsXEff_dus->getTH1F()->SetXTitle(Title + "-jet efficiency");
    EffFlavVsXEff_dus->getTH1F()->SetYTitle("non " + Title + "-jet efficiency");
    EffFlavVsXEff_dus->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
    EffFlavVsXEff_dus->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
  }
  EffFlavVsXEff_c->getTH1F()->SetXTitle(Title + "-jet efficiency");
  EffFlavVsXEff_c->getTH1F()->SetYTitle("c-jet efficiency");
  EffFlavVsXEff_c->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
  EffFlavVsXEff_c->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
  EffFlavVsXEff_b->getTH1F()->SetXTitle(Title + "-jet efficiency");
  EffFlavVsXEff_b->getTH1F()->SetYTitle("b-jet efficiency");
  EffFlavVsXEff_b->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
  EffFlavVsXEff_b->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
  EffFlavVsXEff_ni->getTH1F()->SetXTitle(Title + "-jet efficiency");
  EffFlavVsXEff_ni->getTH1F()->SetYTitle("non " + Title + "-jet efficiency");
  EffFlavVsXEff_ni->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
  EffFlavVsXEff_ni->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
  EffFlavVsXEff_dusg->getTH1F()->SetXTitle(Title + "-jet efficiency");
  EffFlavVsXEff_dusg->getTH1F()->SetYTitle("non " + Title + "-jet efficiency");
  EffFlavVsXEff_dusg->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
  EffFlavVsXEff_dusg->getTH1F()->GetYaxis()->SetTitleOffset(0.75);
  EffFlavVsXEff_pu->getTH1F()->SetXTitle(Title + "-jet efficiency");
  EffFlavVsXEff_pu->getTH1F()->SetYTitle("non " + Title + "-jet efficiency");
  EffFlavVsXEff_pu->getTH1F()->GetXaxis()->SetTitleOffset(0.75);
  EffFlavVsXEff_pu->getTH1F()->GetYaxis()->SetTitleOffset(0.75);

  // Loop over the x bins (signal efficiency) and take, for each, the discriminator cut whose signal efficiency is
  // closest to the bin centre within the bin; bins without such a cut stay empty (the efficiency decreases
  // monotonically with the cut). A flavour without jets gets no curve (its efficiency would be 0/0).
  const TH1F* effSignal = effVersusDiscr(signalFlavour_);
  if (nJets(signalFlavour_) == 0.)
    return;

  struct Curve {
    MonitorElement* out;
    const TH1F* eff;
  };
  std::vector<Curve> curves;
  auto addCurve = [&](MonitorElement* out, const std::string& flavour) {
    if (out != nullptr && nJets(flavour) != 0.)
      curves.push_back({out, effVersusDiscr(flavour)});
  };
  if (mcPlots_ > 2) {
    addCurve(EffFlavVsXEff_d, "D");
    addCurve(EffFlavVsXEff_u, "U");
    addCurve(EffFlavVsXEff_s, "S");
    addCurve(EffFlavVsXEff_g, "G");
    addCurve(EffFlavVsXEff_dus, "DUS");
  }
  addCurve(EffFlavVsXEff_c, "C");
  addCurve(EffFlavVsXEff_b, "B");
  addCurve(EffFlavVsXEff_ni, "NI");
  addCurve(EffFlavVsXEff_dusg, "DUSG");
  addCurve(EffFlavVsXEff_pu, "PU");

  const int nBinX = EffFlavVsXEff_b->getTH1F()->GetNbinsX();
  for (int iBinX = 1; iBinX <= nBinX; iBinX++) {
    const float effXBinWidth = EffFlavVsXEff_b->getTH1F()->GetBinWidth(iBinX);
    const float effXMid = EffFlavVsXEff_b->getTH1F()->GetBinCenter(iBinX);
    const float effXLeft = effXMid - 0.5 * effXBinWidth;
    const float effXRight = effXMid + 0.5 * effXBinWidth;
    const int binClosest = findBinClosestYValue(effSignal, effXMid, effXLeft, effXRight);
    if (binClosest <= 0)
      continue;
    for (const auto& curve : curves) {
      curve.out->Fill(effXMid, curve.eff->GetBinContent(binClosest));
      curve.out->getTH1F()->SetBinError(iBinX, curve.eff->GetBinError(binClosest));
    }
  }
}

#include <memory>

#include <typeinfo>
