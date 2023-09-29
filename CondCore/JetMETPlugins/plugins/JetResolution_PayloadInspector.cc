#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/JetMETObjects/interface/JetResolution.h"
#include "CondFormats/JetMETObjects/interface/JetResolutionObject.h"
#include "CondFormats/JetMETObjects/interface/Utilities.h"

#include <memory>
#include <sstream>
#include <fstream>
#include <iostream>

// include ROOT
#include "TH2F.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"

#define MIN_ETA -5.05
#define MAX_ETA 5.05
#define NBIN_ETA 51
#define MIN_PT -5.
#define MAX_PT 3005.
#define NBIN_PT 301

namespace JME {

  using namespace cond::payloadInspector;

  enum index { NORM = 0, DOWN = 1, UP = 2 };

  /*******************************************************
 *    
 *         1d histogram of JetResolution of 1 IOV 
 *
   *******************************************************/

  // inherit from one of the predefined plot class: Histogram1D

  class JetResolutionVsEta : public cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV> {
  public:
    JetResolutionVsEta()
        : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>(
              "Jet Resolution", "#eta", NBIN_ETA, MIN_ETA, MAX_ETA, "Resolution") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Rho");
    }

    bool fill() override {
      double par_Pt = 100.;
      double par_Eta = 1.;
      double par_Rho = 20.;

      // Default values will be used if no input parameters
      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_Pt = std::stod(ip->second);
        edm::LogPrint("JER_PI") << "Jet Pt: " << par_Pt;
      }
      ip = paramValues.find("Jet_Rho");
      if (ip != paramValues.end()) {
        par_Rho = std::stod(ip->second);
        edm::LogPrint("JER_PI") << "Jet Rho: " << par_Rho;
      }

      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if (!payload->getRecords().empty() &&  // No formula for SF
              payload->getDefinition().getFormulaString().compare("") == 0)
            return false;

          for (size_t idx = 0; idx <= NBIN_ETA; idx++) {
            par_Eta = (idx + 0.5) * (MAX_ETA - MIN_ETA) / NBIN_ETA + MIN_ETA;

            JetParameters j_param;
            j_param.setJetPt(par_Pt);
            j_param.setRho(par_Rho);
            j_param.setJetEta(par_Eta);

            if (payload->getRecord(j_param) == nullptr) {
              continue;
            }

            const JetResolutionObject::Record record = *(payload->getRecord(j_param));
            float res = payload->evaluateFormula(record, j_param);
            fillWithBinAndValue(idx, res);
          }  // x-axis
        } else
          return false;
      }
      return true;
    }
  };  // class

  class JetResolutionVsPt : public cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV> {
  public:
    JetResolutionVsPt()
        : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>(
              "Jet Energy Resolution", "p_T", NBIN_PT, MIN_PT, MAX_PT, "Resolution") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Eta");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Rho");
    }

    bool fill() override {
      double par_Pt = 100.;
      double par_Eta = 1.;
      double par_Rho = 20.;

      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Eta");
      if (ip != paramValues.end()) {
        par_Eta = std::stod(ip->second);
        edm::LogPrint("JER_PI") << "Jet Eta: " << par_Eta;
      }
      ip = paramValues.find("Jet_Rho");
      if (ip != paramValues.end()) {
        par_Rho = std::stod(ip->second);
        edm::LogPrint("JER_PI") << "Jet Rho: " << par_Rho;
      }

      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if (!payload->getRecords().empty() &&  // No formula for SF
              payload->getDefinition().getFormulaString().compare("") == 0)
            return false;

          for (size_t idx = 0; idx <= NBIN_PT; idx++) {
            par_Pt = (idx + 0.5) * (MAX_PT - MIN_PT) / NBIN_PT + MIN_PT;

            JetParameters j_param;
            j_param.setJetEta(par_Eta);
            j_param.setRho(par_Rho);
            j_param.setJetPt(par_Pt);

            if (payload->getRecord(j_param) == nullptr) {
              continue;
            }

            const JetResolutionObject::Record record = *(payload->getRecord(j_param));
            float res = payload->evaluateFormula(record, j_param);
            fillWithBinAndValue(idx + 1, res);
          }  // x-axis
        } else
          return false;
      }
      return true;
    }  // fill
  };

  class JetResolutionSummary : public cond::payloadInspector::PlotImage<JetResolutionObject, SINGLE_IOV> {
  public:
    JetResolutionSummary()
        : cond::payloadInspector::PlotImage<JetResolutionObject, SINGLE_IOV>("Jet Resolution Summary") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Eta");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Rho");
    }

    bool fill() override {
      double par_Pt = 100.;
      double par_Eta = 1.;
      double par_Rho = 20.;

      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_Pt = std::stod(ip->second);
      }
      ip = paramValues.find("Jet_Eta");
      if (ip != paramValues.end()) {
        par_Eta = std::stod(ip->second);
      }
      ip = paramValues.find("Jet_Rho");
      if (ip != paramValues.end()) {
        par_Rho = std::stod(ip->second);
      }

      TH1D* resol_eta = new TH1D("Jet Resolution vs #eta", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* resol_pt = new TH1D("Jet Resolution vs p_T", "", NBIN_PT, MIN_PT, MAX_PT);
      TLegend* leg_eta = new TLegend(0.26, 0.73, 0.935, 0.90);
      TLegend* leg_pt = new TLegend(0.26, 0.73, 0.935, 0.90);

      leg_eta->SetBorderSize(0);
      leg_eta->SetLineStyle(0);
      leg_eta->SetFillStyle(0);

      leg_eta->SetTextFont(42);
      leg_pt->SetBorderSize(0);
      leg_pt->SetLineStyle(0);
      leg_pt->SetFillStyle(0);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<JetResolutionObject> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      std::string tagname = tag.name;
      std::stringstream ss_tagname(tag.name);
      std::string stmp;

      std::string tag_ver;
      std::string tag_res;
      std::string tag_jet;

      getline(ss_tagname, stmp, '_');  // drop first
      getline(ss_tagname, stmp, '_');  // year
      tag_ver = stmp;
      getline(ss_tagname, stmp, '_');  // ver
      tag_ver += '_' + stmp;
      getline(ss_tagname, stmp, '_');  // cmssw
      tag_ver += '_' + stmp;
      getline(ss_tagname, stmp, '_');  // data/mc
      tag_ver += '_' + stmp;
      getline(ss_tagname, stmp, '_');  // bin
      tag_res = stmp;
      getline(ss_tagname, stmp, '_');  // jet algorithm
      tag_jet = stmp;

      if (payload.get()) {
        if (!payload->getRecords().empty() &&  // No formula for SF
            payload->getDefinition().getFormulaString().compare("") == 0)
          return false;

        for (size_t idx = 0; idx <= NBIN_ETA; idx++) {
          double x_axis = (idx + 0.5) * (MAX_ETA - MIN_ETA) / NBIN_ETA + MIN_ETA;

          JetParameters j_param;
          j_param.setJetPt(par_Pt);
          j_param.setRho(par_Rho);
          j_param.setJetEta(x_axis);

          if (payload->getRecord(j_param) == nullptr) {
            continue;
          }

          const JetResolutionObject::Record record = *(payload->getRecord(j_param));
          float res = payload->evaluateFormula(record, j_param);
          resol_eta->SetBinContent(idx + 1, res);
        }  // x-axis

        for (size_t idx = 0; idx <= NBIN_PT; idx++) {
          double x_axis = (idx + 0.5) * (MAX_PT - MIN_PT) / NBIN_PT + MIN_PT;

          JetParameters j_param;
          j_param.setJetEta(par_Eta);
          j_param.setRho(par_Rho);
          j_param.setJetPt(x_axis);

          if (payload->getRecord(j_param) == nullptr) {
            continue;
          }

          const JetResolutionObject::Record record = *(payload->getRecord(j_param));
          float res = payload->evaluateFormula(record, j_param);
          resol_pt->SetBinContent(idx + 1, res);
        }  // x-axis

        gStyle->SetOptStat(0);
        gStyle->SetLabelFont(42, "XYZ");
        gStyle->SetLabelSize(0.05, "XYZ");
        gStyle->SetFrameLineWidth(3);

        std::string title = Form("Summary Run %i", run);
        TCanvas canvas("Jet Resolution Summary", title.c_str(), 800, 1200);
        canvas.Divide(1, 2);

        canvas.cd(1);
        resol_eta->SetTitle("Jet Resolution vs. #eta");
        resol_eta->SetXTitle("#eta");
        resol_eta->SetYTitle("Resolution");
        resol_eta->SetLineWidth(3);
        resol_eta->SetMaximum(resol_eta->GetMaximum() * 1.25);
        resol_eta->Draw("");

        leg_eta->AddEntry(resol_eta, (tag_ver + '_' + tag_jet).c_str(), "l");
        leg_eta->AddEntry((TObject*)nullptr, Form("JetPt=%.2f; JetRho=%.2f", par_Pt, par_Rho), "");
        leg_eta->Draw();

        canvas.cd(2);
        resol_pt->SetTitle("Jet Resolution vs. p_{T}");
        resol_pt->SetXTitle("p_{T} [GeV]");
        resol_pt->SetYTitle("Resolution");
        resol_pt->SetLineWidth(3);
        resol_pt->Draw("][");

        leg_pt->AddEntry(resol_pt, (tag_ver + '_' + tag_jet).c_str(), "l");
        leg_pt->AddEntry((TObject*)nullptr, Form("JetEta=%.2f; JetRho=%.2f", par_Eta, par_Rho), "");
        leg_pt->Draw();

        canvas.SaveAs(m_imageFileName.c_str());

        return true;
      } else  // no payload.get()
        return false;
    }  // fill

  };  // class

  class JetResolutionCompare : public cond::payloadInspector::PlotImage<JetResolutionObject, SINGLE_IOV, 2> {
  public:
    JetResolutionCompare()
        : cond::payloadInspector::PlotImage<JetResolutionObject, SINGLE_IOV, 2>("Jet Resolution Compare") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Eta");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Rho");
    }

    bool fill() override {
      double par_Pt = 100.;
      double par_Eta = 1.;
      double par_Rho = 20.;

      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_Pt = std::stod(ip->second);
      }
      ip = paramValues.find("Jet_Eta");
      if (ip != paramValues.end()) {
        par_Eta = std::stod(ip->second);
      }
      ip = paramValues.find("Jet_Rho");
      if (ip != paramValues.end()) {
        par_Rho = std::stod(ip->second);
      }

      TH1D* resol_eta1 = new TH1D("Jet Resolution vs #eta one", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* resol_eta2 = new TH1D("Jet Resolution vs #eta two", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* resol_pt1 = new TH1D("Jet Resolution vs p_T one", "", NBIN_PT, MIN_PT, MAX_PT);
      TH1D* resol_pt2 = new TH1D("Jet Resolution vs p_T two", "", NBIN_PT, MIN_PT, MAX_PT);
      TLegend* leg_eta = new TLegend(0.26, 0.73, 0.935, 0.90);
      TLegend* leg_pt = new TLegend(0.26, 0.73, 0.935, 0.90);

      leg_eta->SetBorderSize(0);
      leg_eta->SetLineStyle(0);
      leg_eta->SetFillStyle(0);

      leg_eta->SetTextFont(42);
      leg_pt->SetBorderSize(0);
      leg_pt->SetLineStyle(0);
      leg_pt->SetFillStyle(0);

      auto tag1 = PlotBase::getTag<0>();
      auto iov1 = tag1.iovs.front();
      std::shared_ptr<JetResolutionObject> payload1 = fetchPayload(std::get<1>(iov1));
      auto tag2 = PlotBase::getTag<1>();
      auto iov2 = tag2.iovs.front();
      std::shared_ptr<JetResolutionObject> payload2 = fetchPayload(std::get<1>(iov2));

      std::stringstream ss_tagname1(tag1.name);
      std::stringstream ss_tagname2(tag2.name);
      std::string stmp;

      std::string tag_ver1;
      std::string tag_ver2;

      getline(ss_tagname1, stmp, '_');  // drop first
      getline(ss_tagname1, stmp);       // year
      tag_ver1 = stmp;

      getline(ss_tagname2, stmp, '_');  // drop first
      getline(ss_tagname2, stmp);       // year
      tag_ver2 = stmp;

      if (payload1.get() && payload2.get()) {
        if (!payload1->getRecords().empty() &&  // No formula for SF
            payload1->getDefinition().getFormulaString().compare("") == 0)
          return false;

        for (size_t idx = 0; idx <= NBIN_ETA; idx++) {
          double x_axis = (idx + 0.5) * (MAX_ETA - MIN_ETA) / NBIN_ETA + MIN_ETA;

          JetParameters j_param;
          j_param.setJetPt(par_Pt);
          j_param.setRho(par_Rho);
          j_param.setJetEta(x_axis);

          if (payload1->getRecord(j_param) == nullptr || payload2->getRecord(j_param) == nullptr) {
            continue;
          }

          const JetResolutionObject::Record record1 = *(payload1->getRecord(j_param));
          const JetResolutionObject::Record record2 = *(payload2->getRecord(j_param));
          float res1 = payload1->evaluateFormula(record1, j_param);
          resol_eta1->SetBinContent(idx + 1, res1);
          float res2 = payload2->evaluateFormula(record2, j_param);
          resol_eta2->SetBinContent(idx + 1, res2);
        }  // x-axis

        for (size_t idx = 0; idx <= NBIN_PT; idx++) {
          double x_axis = (idx + 0.5) * (MAX_PT - MIN_PT) / NBIN_PT + MIN_PT;

          JetParameters j_param;
          j_param.setJetEta(par_Eta);
          j_param.setRho(par_Rho);
          j_param.setJetPt(x_axis);

          if (payload1->getRecord(j_param) == nullptr || payload2->getRecord(j_param) == nullptr) {
            continue;
          }

          const JetResolutionObject::Record record1 = *(payload1->getRecord(j_param));
          const JetResolutionObject::Record record2 = *(payload2->getRecord(j_param));
          float res1 = payload1->evaluateFormula(record1, j_param);
          resol_pt1->SetBinContent(idx + 1, res1);
          float res2 = payload2->evaluateFormula(record2, j_param);
          resol_pt2->SetBinContent(idx + 1, res2);
        }  // x-axis

        gStyle->SetOptStat(0);
        gStyle->SetLabelFont(42, "XYZ");
        gStyle->SetLabelSize(0.05, "XYZ");
        gStyle->SetFrameLineWidth(3);

        std::string title = Form("Comparison between %s and %s", tag1.name.c_str(), tag2.name.c_str());
        TCanvas canvas("Jet Resolution Comparison", title.c_str(), 800, 1200);
        canvas.Divide(1, 2);

        canvas.cd(1);
        resol_eta1->SetTitle("Jet Resolution Comparison vs. #eta");
        resol_eta1->SetXTitle("#eta");
        resol_eta1->SetYTitle("Resolution");
        resol_eta1->SetLineWidth(3);
        resol_eta1->SetMaximum(resol_eta1->GetMaximum() * 1.25);
        resol_eta1->Draw("][");

        resol_eta2->SetLineColor(2);
        resol_eta2->SetLineWidth(3);
        resol_eta2->SetLineStyle(2);
        resol_eta2->Draw("][same");

        leg_eta->AddEntry(resol_eta1, tag_ver1.c_str(), "l");
        leg_eta->AddEntry(resol_eta2, tag_ver2.c_str(), "l");
        leg_eta->AddEntry((TObject*)nullptr, Form("JetPt=%.2f; JetRho=%.2f", par_Pt, par_Rho), "");
        leg_eta->Draw();

        canvas.cd(2);
        resol_pt1->SetTitle("Jet Resolution Comparison vs. p_{T}");
        resol_pt1->SetXTitle("p_{T} [GeV]");
        resol_pt1->SetYTitle("Resolution");
        resol_pt1->SetLineWidth(3);
        resol_pt1->Draw("][");

        resol_pt2->SetLineColor(2);
        resol_pt2->SetLineWidth(3);
        resol_pt2->SetLineStyle(2);
        resol_pt2->Draw("][same");

        leg_pt->AddEntry(resol_pt1, tag_ver1.c_str(), "l");
        leg_pt->AddEntry(resol_pt2, tag_ver2.c_str(), "l");
        leg_pt->AddEntry((TObject*)nullptr, Form("JetEta=%.2f; JetRho=%.2f", par_Eta, par_Rho), "");
        leg_pt->Draw();

        canvas.SaveAs(m_imageFileName.c_str());

        return true;
      } else  // no payload.get()
        return false;
    }  // fill

  };  // class

  template <index ii>
  class JetScaleFactorVsEta : public cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV> {
  public:
    JetScaleFactorVsEta()
        : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>(
              "Jet Energy Scale Factor", "#eta", NBIN_ETA, MIN_ETA, MAX_ETA, "Scale Factor") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Eta");
    }

    bool fill() override {
      double par_Pt = 100.;
      double par_Eta = 1.;

      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_Pt = std::stod(ip->second);
        edm::LogPrint("JER_PI") << "Jet Pt: " << par_Pt;
      }
      ip = paramValues.find("Jet_Eta");
      if (ip != paramValues.end()) {
        par_Eta = std::stod(ip->second);
        edm::LogPrint("JER_PI") << "Jet Eta: " << par_Eta;
      }

      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if (!payload->getRecords().empty() &&  // No formula for SF
              payload->getDefinition().getFormulaString().compare("") != 0)
            return false;

          for (size_t idx = 0; idx <= NBIN_ETA; idx++) {
            par_Eta = (idx + 0.5) * (MAX_ETA - MIN_ETA) / NBIN_ETA + MIN_ETA;

            JetParameters j_param;
            j_param.setJetPt(par_Pt);
            j_param.setJetEta(par_Eta);

            if (payload->getRecord(j_param) == nullptr) {
              continue;
            }

            const JetResolutionObject::Record record = *(payload->getRecord(j_param));
            if (record.getParametersValues().size() == 3) {  // norm, down, up
              double sf = record.getParametersValues()[ii];
              fillWithBinAndValue(idx + 1, sf);
            }
          }  // x-axis
        } else
          return false;
      }  // for
      return true;
    }  // fill
  };   // class

  typedef JetScaleFactorVsEta<NORM> JetScaleFactorVsEtaNORM;
  typedef JetScaleFactorVsEta<DOWN> JetScaleFactorVsEtaDOWN;
  typedef JetScaleFactorVsEta<UP> JetScaleFactorVsEtaUP;

  template <index ii>
  class JetScaleFactorVsPt : public cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV> {
  public:
    JetScaleFactorVsPt()
        : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>(
              "Jet Energy Scale Factor", "p_T", NBIN_PT, MIN_PT, MAX_PT, "Scale Factor") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Eta");
    }

    bool fill() override {
      double par_Pt = 100.;
      double par_Eta = 1.;

      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_Pt = std::stod(ip->second);
        edm::LogPrint("JER_PI") << "Jet Pt: " << par_Pt;
      }
      ip = paramValues.find("Jet_Eta");
      if (ip != paramValues.end()) {
        par_Eta = std::stod(ip->second);
        edm::LogPrint("JER_PI") << "Jet Eta: " << par_Eta;
      }

      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if (!payload->getRecords().empty() &&  // No formula for SF
              payload->getDefinition().getFormulaString().compare("") != 0)
            return false;

          for (size_t idx = 0; idx <= NBIN_PT; idx++) {
            par_Pt = (idx + 0.5) * (MAX_PT - MIN_PT) / NBIN_PT + MIN_PT;

            JetParameters j_param;
            j_param.setJetEta(par_Eta);
            j_param.setJetPt(par_Pt);

            if (payload->getRecord(j_param) == nullptr) {
              continue;
            }

            const JetResolutionObject::Record record = *(payload->getRecord(j_param));
            if (record.getParametersValues().size() == 3) {  // norm, down, up
              double sf = record.getParametersValues()[ii];
              fillWithBinAndValue(idx + 1, sf);
            }
          }  // x-axis
        } else
          return false;
      }  // for
      return true;
    }  // fill
  };   // class

  typedef JetScaleFactorVsPt<NORM> JetScaleFactorVsPtNORM;
  typedef JetScaleFactorVsPt<DOWN> JetScaleFactorVsPtDOWN;
  typedef JetScaleFactorVsPt<UP> JetScaleFactorVsPtUP;

  class JetScaleFactorSummary : public cond::payloadInspector::PlotImage<JetResolutionObject, SINGLE_IOV> {
  public:
    JetScaleFactorSummary()
        : cond::payloadInspector::PlotImage<JetResolutionObject, SINGLE_IOV>("Jet ScaleFactor Summary") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Eta");
    }

    bool fill() override {
      double par_Pt = 100.;
      double par_Eta = 1.;

      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_Pt = std::stod(ip->second);
      }
      ip = paramValues.find("Jet_Eta");
      if (ip != paramValues.end()) {
        par_Eta = std::stod(ip->second);
      }

      TH1D* sf_eta_norm = new TH1D("Jet SF vs #eta NORM", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* sf_eta_down = new TH1D("Jet SF vs #eta DOWN", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* sf_eta_up = new TH1D("Jet SF vs #eta UP", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* sf_pt_norm = new TH1D("Jet SF vs p_T NORM", "", NBIN_PT, MIN_PT, MAX_PT);
      TH1D* sf_pt_down = new TH1D("Jet SF vs p_T DOWN", "", NBIN_PT, MIN_PT, MAX_PT);
      TH1D* sf_pt_up = new TH1D("Jet SF vs p_T UP", "", NBIN_PT, MIN_PT, MAX_PT);

      TLegend* leg_eta = new TLegend(0.26, 0.73, 0.935, 0.90);
      TLegend* leg_pt = new TLegend(0.26, 0.73, 0.935, 0.90);

      leg_eta->SetBorderSize(0);
      leg_eta->SetLineStyle(0);
      leg_eta->SetFillStyle(0);

      leg_eta->SetTextFont(42);
      leg_pt->SetBorderSize(0);
      leg_pt->SetLineStyle(0);
      leg_pt->SetFillStyle(0);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();
      std::shared_ptr<JetResolutionObject> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      std::string tagname = tag.name;
      std::stringstream ss_tagname(tag.name);
      std::string stmp;

      std::string tag_ver;
      std::string tag_res;
      std::string tag_jet;

      getline(ss_tagname, stmp, '_');  // drop first
      getline(ss_tagname, stmp, '_');  // year
      tag_ver = stmp;
      getline(ss_tagname, stmp, '_');  // ver
      tag_ver += '_' + stmp;
      getline(ss_tagname, stmp, '_');  // cmssw
      tag_ver += '_' + stmp;
      getline(ss_tagname, stmp, '_');  // data/mc
      tag_ver += '_' + stmp;
      getline(ss_tagname, stmp, '_');  // bin
      tag_res = stmp;
      getline(ss_tagname, stmp, '_');  // jet algorithm
      tag_jet = stmp;

      bool is_2bin = false;

      if (payload.get()) {
        if (!payload->getRecords().empty() &&  // No formula for SF
            payload->getDefinition().getFormulaString().compare("") != 0)
          return false;

        is_2bin = false;
        for (size_t idx = 0; idx <= NBIN_ETA; idx++) {
          double x_axis = (idx + 0.5) * (MAX_ETA - MIN_ETA) / NBIN_ETA + MIN_ETA;

          JetParameters j_param;
          j_param.setJetPt(par_Pt);
          j_param.setJetEta(x_axis);

          if (payload->getRecord(j_param) == nullptr) {
            continue;
          }

          const JetResolutionObject::Record record = *(payload->getRecord(j_param));
          if (record.getBinsRange().size() > 1)
            is_2bin = true;

          if (record.getParametersValues().size() == 3) {  // norm, down, up
            sf_eta_norm->SetBinContent(idx + 1, record.getParametersValues()[0]);
            sf_eta_down->SetBinContent(idx + 1, record.getParametersValues()[1]);
            sf_eta_up->SetBinContent(idx + 1, record.getParametersValues()[2]);
          }
        }  // x-axis

        for (size_t idx = 0; idx <= NBIN_PT; idx++) {
          double x_axis = (idx + 0.5) * (MAX_PT - MIN_PT) / NBIN_PT + MIN_PT;

          JetParameters j_param;
          j_param.setJetEta(par_Eta);
          j_param.setJetPt(x_axis);

          if (payload->getRecord(j_param) == nullptr) {
            continue;
          }

          const JetResolutionObject::Record record = *(payload->getRecord(j_param));
          if (record.getParametersValues().size() == 3) {  // norm, down, up
            sf_pt_norm->SetBinContent(idx + 1, record.getParametersValues()[0]);
            sf_pt_down->SetBinContent(idx + 1, record.getParametersValues()[1]);
            sf_pt_up->SetBinContent(idx + 1, record.getParametersValues()[2]);
          }
        }  // x-axis

        gStyle->SetOptStat(0);
        gStyle->SetLabelFont(42, "XYZ");
        gStyle->SetLabelSize(0.05, "XYZ");
        gStyle->SetFrameLineWidth(3);

        std::string title = Form("Summary Run %i", run);
        TCanvas canvas("Jet ScaleFactor Summary", title.c_str(), 800, 1200);
        canvas.Divide(1, 2);

        canvas.cd(1);
        sf_eta_up->SetTitle("ScaleFactor vs. #eta");
        sf_eta_up->SetXTitle("#eta");
        sf_eta_up->SetYTitle("Scale Factor");
        sf_eta_up->SetLineStyle(7);
        sf_eta_up->SetLineWidth(3);
        sf_eta_up->SetFillColorAlpha(kGray, 0.5);
        sf_eta_up->SetMaximum(sf_eta_up->GetMaximum() * 1.25);
        sf_eta_up->SetMinimum(0.);
        sf_eta_up->Draw("][");

        sf_eta_down->SetLineStyle(7);
        sf_eta_down->SetLineWidth(3);
        sf_eta_down->SetFillColorAlpha(kWhite, 1);
        sf_eta_down->Draw("][ same");

        sf_eta_norm->SetLineStyle(1);
        sf_eta_norm->SetLineWidth(5);
        sf_eta_norm->SetFillColor(0);
        sf_eta_norm->Draw("][ same");
        sf_eta_norm->Draw("axis same");

        leg_eta->AddEntry(sf_eta_norm, (tag_ver + '_' + tag_jet).c_str(), "l");
        leg_eta->AddEntry((TObject*)nullptr, Form("JetPt=%.2f", par_Pt), "");
        leg_eta->Draw();

        if (is_2bin == true) {
          canvas.cd(2);
          sf_pt_up->SetTitle("ScaleFactor vs. p_{T}");
          sf_pt_up->SetXTitle("p_{T} [GeV]");
          sf_pt_up->SetYTitle("Scale Factor");
          sf_pt_up->SetLineStyle(7);
          sf_pt_up->SetLineWidth(3);
          sf_pt_up->SetFillColorAlpha(kGray, 0.5);
          sf_pt_up->SetMaximum(sf_pt_up->GetMaximum() * 1.25);
          sf_pt_up->SetMinimum(0.);
          sf_pt_up->Draw("][");

          sf_pt_down->SetLineStyle(7);
          sf_pt_down->SetLineWidth(3);
          sf_pt_down->SetFillColorAlpha(kWhite, 1);
          sf_pt_down->Draw("][ same");

          sf_pt_norm->SetLineStyle(1);
          sf_pt_norm->SetLineWidth(5);
          sf_pt_norm->SetFillColor(0);
          sf_pt_norm->Draw("][ same");
          sf_pt_norm->Draw("axis same");

          leg_pt->AddEntry(sf_pt_norm, (tag_ver + '_' + tag_jet).c_str(), "l");
          leg_pt->AddEntry((TObject*)nullptr, Form("JetEta=%.2f", par_Eta), "");
          leg_pt->Draw();
        }

        canvas.SaveAs(m_imageFileName.c_str());

        return true;
      } else  // no payload.get()
        return false;
    }  // fill

  };  // class

  class JetScaleFactorCompare : public cond::payloadInspector::PlotImage<JetResolutionObject, SINGLE_IOV, 2> {
  public:
    JetScaleFactorCompare()
        : cond::payloadInspector::PlotImage<JetResolutionObject, SINGLE_IOV, 2>("Jet ScaleFactor Summary") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Eta");
    }

    bool fill() override {
      double par_Pt = 100.;
      double par_Eta = 1.;

      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_Pt = std::stod(ip->second);
      }
      ip = paramValues.find("Jet_Eta");
      if (ip != paramValues.end()) {
        par_Eta = std::stod(ip->second);
      }

      TH1D* sf_eta_one = new TH1D("Jet SF vs #eta one", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* sf_eta_two = new TH1D("Jet SF vs #eta two", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* sf_pt_one = new TH1D("Jet SF vs p_T one", "", NBIN_PT, MIN_PT, MAX_PT);
      TH1D* sf_pt_two = new TH1D("Jet SF vs p_T two", "", NBIN_PT, MIN_PT, MAX_PT);

      TLegend* leg_eta = new TLegend(0.26, 0.73, 0.935, 0.90);
      TLegend* leg_pt = new TLegend(0.26, 0.73, 0.935, 0.90);

      leg_eta->SetBorderSize(0);
      leg_eta->SetLineStyle(0);
      leg_eta->SetFillStyle(0);

      leg_eta->SetTextFont(42);
      leg_pt->SetBorderSize(0);
      leg_pt->SetLineStyle(0);
      leg_pt->SetFillStyle(0);

      auto tag1 = PlotBase::getTag<0>();
      auto iov1 = tag1.iovs.front();
      std::shared_ptr<JetResolutionObject> payload1 = fetchPayload(std::get<1>(iov1));

      auto tag2 = PlotBase::getTag<1>();
      auto iov2 = tag2.iovs.front();
      std::shared_ptr<JetResolutionObject> payload2 = fetchPayload(std::get<1>(iov2));

      std::stringstream ss_tagname1(tag1.name);
      std::stringstream ss_tagname2(tag2.name);
      std::string stmp;

      std::string tag_ver1;
      std::string tag_ver2;

      getline(ss_tagname1, stmp, '_');  // drop first
      getline(ss_tagname1, stmp);       // year
      tag_ver1 = stmp;

      getline(ss_tagname2, stmp, '_');  // drop first
      getline(ss_tagname2, stmp);       // year
      tag_ver2 = stmp;

      bool is_2bin = false;

      if (payload1.get() && payload2.get()) {
        if (!payload1->getRecords().empty() &&  // No formula for SF
            payload1->getDefinition().getFormulaString().compare("") != 0)
          return false;

        is_2bin = false;
        for (size_t idx = 0; idx <= NBIN_ETA; idx++) {
          double x_axis = (idx + 0.5) * (MAX_ETA - MIN_ETA) / NBIN_ETA + MIN_ETA;

          JetParameters j_param;
          j_param.setJetPt(par_Pt);
          j_param.setJetEta(x_axis);

          const JetResolutionObject::Record* rcd_p1 = payload1->getRecord(j_param);
          const JetResolutionObject::Record* rcd_p2 = payload2->getRecord(j_param);
          if (rcd_p1 == nullptr || rcd_p2 == nullptr) {
            continue;
          }

          const JetResolutionObject::Record record1 = *(rcd_p1);
          const JetResolutionObject::Record record2 = *(rcd_p2);

          if (record1.getBinsRange().size() > 1 && record2.getBinsRange().size() > 1)
            is_2bin = true;

          if (record1.getParametersValues().size() == 3) {  // norm, down, up
            sf_eta_one->SetBinContent(idx + 1, record1.getParametersValues()[0]);
          }
          if (record2.getParametersValues().size() == 3) {  // norm, down, up
            sf_eta_two->SetBinContent(idx + 1, record2.getParametersValues()[0]);
          }
        }  // x-axis

        for (size_t idx = 0; idx <= NBIN_PT; idx++) {
          double x_axis = (idx + 0.5) * (MAX_PT - MIN_PT) / NBIN_PT + MIN_PT;

          JetParameters j_param;
          j_param.setJetEta(par_Eta);
          j_param.setJetPt(x_axis);

          const JetResolutionObject::Record* rcd_p1 = payload1->getRecord(j_param);
          const JetResolutionObject::Record* rcd_p2 = payload2->getRecord(j_param);
          if (rcd_p1 == nullptr || rcd_p2 == nullptr) {
            continue;
          }

          const JetResolutionObject::Record record1 = *(rcd_p1);
          const JetResolutionObject::Record record2 = *(rcd_p2);

          if (record1.getParametersValues().size() == 3) {  // norm, down, up
            sf_pt_one->SetBinContent(idx + 1, record1.getParametersValues()[0]);
          }
          if (record2.getParametersValues().size() == 3) {  // norm, down, up
            sf_pt_two->SetBinContent(idx + 1, record2.getParametersValues()[0]);
          }
        }  // x-axis

        gStyle->SetOptStat(0);
        gStyle->SetLabelFont(42, "XYZ");
        gStyle->SetLabelSize(0.05, "XYZ");
        gStyle->SetFrameLineWidth(3);

        std::string title = Form("Comparison");
        TCanvas canvas("Jet ScaleFactor", title.c_str(), 800, 1200);
        canvas.Divide(1, 2);

        canvas.cd(1);
        sf_eta_one->SetTitle("Jet ScaleFactor Comparison vs. #eta");
        sf_eta_one->SetXTitle("#eta");
        sf_eta_one->SetYTitle("Scale Factor");
        sf_eta_one->SetLineWidth(3);
        sf_eta_one->SetMaximum(sf_eta_one->GetMaximum() * 1.25);
        sf_eta_one->SetMinimum(0.);
        sf_eta_one->Draw("][");

        sf_eta_two->SetLineColor(2);
        sf_eta_two->SetLineWidth(3);
        sf_eta_two->SetLineStyle(2);
        sf_eta_two->Draw("][ same");

        leg_eta->AddEntry(sf_eta_one, tag_ver1.c_str(), "l");
        leg_eta->AddEntry(sf_eta_two, tag_ver2.c_str(), "l");
        leg_eta->AddEntry((TObject*)nullptr, Form("JetPt=%.2f", par_Pt), "");
        leg_eta->Draw();

        if (is_2bin == true) {
          canvas.cd(2);
          sf_pt_one->SetTitle("Jet ScaleFactor Comparison vs. p_{T}");
          sf_pt_one->SetXTitle("p_{T} [GeV]");
          sf_pt_one->SetYTitle("Scale Factor");
          sf_pt_one->SetLineWidth(3);
          sf_pt_one->SetMaximum(sf_pt_one->GetMaximum() * 1.25);
          sf_pt_one->SetMinimum(0.);
          sf_pt_one->Draw("][");

          sf_pt_two->SetLineColor(2);
          sf_pt_two->SetLineWidth(3);
          sf_pt_two->SetLineStyle(2);
          sf_pt_two->Draw("][ same");

          leg_pt->AddEntry(sf_pt_one, tag_ver1.c_str(), "l");
          leg_pt->AddEntry(sf_pt_two, tag_ver2.c_str(), "l");
          leg_pt->AddEntry((TObject*)nullptr, Form("JetEta=%.2f", par_Eta), "");
          leg_pt->Draw();
        }

        canvas.SaveAs(m_imageFileName.c_str());

        return true;
      } else  // no payload.get()
        return false;
    }  // fill

  };  // class

  // Register the classes as boost python plugin
  PAYLOAD_INSPECTOR_MODULE(JetResolutionObject) {
    PAYLOAD_INSPECTOR_CLASS(JetResolutionVsEta);
    PAYLOAD_INSPECTOR_CLASS(JetResolutionVsPt);
    PAYLOAD_INSPECTOR_CLASS(JetScaleFactorVsEtaNORM);
    PAYLOAD_INSPECTOR_CLASS(JetScaleFactorVsEtaDOWN);
    PAYLOAD_INSPECTOR_CLASS(JetScaleFactorVsEtaUP);
    PAYLOAD_INSPECTOR_CLASS(JetScaleFactorVsPtNORM);
    PAYLOAD_INSPECTOR_CLASS(JetScaleFactorVsPtDOWN);
    PAYLOAD_INSPECTOR_CLASS(JetScaleFactorVsPtUP);

    PAYLOAD_INSPECTOR_CLASS(JetResolutionSummary);
    PAYLOAD_INSPECTOR_CLASS(JetScaleFactorSummary);
    PAYLOAD_INSPECTOR_CLASS(JetResolutionCompare);
    PAYLOAD_INSPECTOR_CLASS(JetScaleFactorCompare);
  }

}  // namespace JME
