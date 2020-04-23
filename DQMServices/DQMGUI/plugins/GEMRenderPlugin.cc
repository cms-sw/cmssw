#include "DQMServices/DQMGUI/interface/DQMRenderPlugin.h"

#include "GEMRenderPlugin_SummaryChamber.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"

#include <cassert>

//----------------------------------------------------------------------------------------------------

class GEMRenderPlugin : public DQMRenderPlugin {
private:
  SummaryChamber summaryCh;

  int drawTimeHisto(TH2F *h2Curr) {
    std::string strNewName = std::string(h2Curr->GetName()).substr(std::string("per_time_").size());
    std::string strTmpPrefix = "tmp_";

    Int_t nNbinsX = h2Curr->GetNbinsX();
    Int_t nNbinsY = h2Curr->GetNbinsY();
    Int_t nNbinsXActual = 0;

    for (nNbinsXActual = 0; nNbinsXActual < nNbinsX; nNbinsXActual++) {
      if (h2Curr->GetBinContent(nNbinsXActual + 1, 0) <= 0)
        break;
    }

    std::string strTitle = std::string(h2Curr->GetTitle());

    //std::string strAxisX = h2Curr->GetXaxis()->GetTitle();
    std::string strAxisX = "Bin per " + std::to_string((Int_t)h2Curr->GetBinContent(0, 0)) + " events";
    std::string strAxisY = h2Curr->GetYaxis()->GetTitle();
    //strTitle += ";" + strAxisX + ";" + strAxisY;

    strTitle = "";
    TH2F *h2New = new TH2F((strTmpPrefix + strNewName).c_str(),
                           strTitle.c_str(),
                           nNbinsXActual,
                           0.0,
                           (Double_t)nNbinsXActual,
                           nNbinsY,
                           0.0,
                           (Double_t)nNbinsY);

    h2New->GetXaxis()->SetTitle(strAxisX.c_str());
    h2New->GetYaxis()->SetTitle(strAxisY.c_str());

    for (Int_t i = 1; i <= nNbinsY; i++) {
      std::string strLabel = h2Curr->GetYaxis()->GetBinLabel(i);
      if (!strLabel.empty())
        h2New->GetYaxis()->SetBinLabel(i, strLabel.c_str());
    }

    for (Int_t j = 0; j <= nNbinsY; j++)
      for (Int_t i = 1; i <= nNbinsXActual; i++) {
        h2New->SetBinContent(i, j, h2Curr->GetBinContent(i, j));
      }

    h2New->Draw("colz");

    return 0;
  };

public:
  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &) override {
    if ((o.name.find("GEM/") != std::string::npos))
      return true;

    return false;
  }

  virtual void preDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &) override {
    c->cd();

    gStyle->SetOptStat(10);

    if (dynamic_cast<TH1F *>(o.object))
      preDrawTH1F(c, o);

    if (dynamic_cast<TH2F *>(o.object))
      preDrawTH2F(c, o);
  }

  virtual void postDraw(TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &) override {
    c->cd();

    gStyle->SetOptStat(10);

    if (o.name.rfind("summary_statusChamber") != std::string::npos ||
        o.name.rfind("reportSummaryMap") != std::string::npos) {
      TH2 *obj2 = dynamic_cast<TH2 *>(o.object);
      //assert(obj2);
      if (obj2 == NULL)
        return;

      summaryCh.drawStats(obj2);

      //obj2->SetStats(false);
      gStyle->SetOptStat("e");
      obj2->SetOption("");
      gPad->SetGridx();
      gPad->SetGridy();

      return;
    }

    if (dynamic_cast<TH1F *>(o.object))
      postDrawTH1F(c, o);

    if (dynamic_cast<TH2F *>(o.object))
      postDrawTH2F(c, o);
  }

private:
  void preDrawTH1F(TCanvas *, const VisDQMObject &o) {
    bool setColor = true;
    if (o.name.rfind(" U") == o.name.size() - 2)
      setColor = false;
    if (o.name.rfind(" V") == o.name.size() - 2)
      setColor = false;
    if (o.name.find("events per BX") != std::string::npos)
      setColor = false;

    TH1F *obj = dynamic_cast<TH1F *>(o.object);
    assert(obj);

    //obj->SetStats(false);

    if (setColor)
      obj->SetLineColor(2);

    obj->SetLineWidth(2);
  }

  void preDrawTH2F(TCanvas *, const VisDQMObject &o) {
    TH2F *obj = dynamic_cast<TH2F *>(o.object);
    assert(obj);

    obj->SetOption("colz");
    //obj->SetStats(false);

    gPad->SetGridx();
    gPad->SetGridy();
  }

  void postDrawTH1F(TCanvas *c, const VisDQMObject &o) {
    TH1F *obj = dynamic_cast<TH1F *>(o.object);
    assert(obj);

    //obj->SetStats(false);
    c->SetGridx();
    c->SetGridy();
  }

  void postDrawTH2F(TCanvas *c, const VisDQMObject &o) {
    TH2F *obj = dynamic_cast<TH2F *>(o.object);
    assert(obj);

    if (o.name.find("GEM/StatusDigi") != std::string::npos) {
      obj->SetStats(false);
    }

    if (o.name.find("GEM/StatusDigi/per_time_") != std::string::npos) {
      drawTimeHisto(dynamic_cast<TH2F *>(o.object));
    }

    //obj->SetStats(false);
    c->SetGridx();
    c->SetGridy();
  }
};

//----------------------------------------------------------------------------------------------------
static GEMRenderPlugin instance;
