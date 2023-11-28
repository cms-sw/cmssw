#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParametersHelper.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/SimpleJetCorrectionUncertainty.h"
#include "CondFormats/JetMETObjects/interface/Utilities.h"

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

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

namespace {

  using namespace cond::payloadInspector;

  enum levelt {
    L1Offset = 0,
    L1JPTOffset = 7,
    L1FastJet = 10,
    L2Relative = 1,
    L3Absolute = 2,
    L2L3Residual = 8,
    L4EMF = 3,
    L5Flavor = 4,
    L6UE = 5,
    L7Parton = 6,
    Uncertainty = 9,
    UncertaintyAbsolute = 11,
    UncertaintyHighPtExtra = 12,
    UncertaintySinglePionECAL = 13,
    UncertaintySinglePionHCAL = 27,
    UncertaintyFlavor = 14,
    UncertaintyTime = 15,
    UncertaintyRelativeJEREC1 = 16,
    UncertaintyRelativeJEREC2 = 17,
    UncertaintyRelativeJERHF = 18,
    UncertaintyRelativePtEC1 = 28,
    UncertaintyRelativePtEC2 = 29,
    UncertaintyRelativePtHF = 30,
    UncertaintyRelativeStatEC2 = 19,
    UncertaintyRelativeStatHF = 20,
    UncertaintyRelativeFSR = 21,
    UncertaintyRelativeSample = 31,
    UncertaintyPileUpDataMC = 22,
    UncertaintyPileUpOOT = 23,
    UncertaintyPileUpPtBB = 24,
    UncertaintyPileUpPtEC = 32,
    UncertaintyPileUpPtHF = 33,
    UncertaintyPileUpBias = 25,
    UncertaintyPileUpJetRate = 26,
    L1RC = 34,
    L1Residual = 35,
    UncertaintyAux3 = 36,
    UncertaintyAux4 = 37,
    N_LEVELS = 38
  };

  const std::vector<std::string> labels_ = {
      "L1Offset",
      "L2Relative",
      "L3Absolute",
      "L4EMF",
      "L5Flavor",
      "L6UE",
      "L7Parton",
      "L1JPTOffset",
      "L2L3Residual",
      "Uncertainty",
      "L1FastJet",
      "UncertaintyAbsolute",
      "UncertaintyHighPtExtra",
      "UncertaintySinglePionECAL",
      "UncertaintyFlavor",
      "UncertaintyTime",
      "UncertaintyRelativeJEREC1",
      "UncertaintyRelativeJEREC2",
      "UncertaintyRelativeJERHF",
      "UncertaintyRelativeStatEC2",
      "UncertaintyRelativeStatHF",
      "UncertaintyRelativeFSR",
      "UncertaintyPileUpDataMC",
      "UncertaintyPileUpOOT",
      "UncertaintyPileUpPtBB",
      "UncertaintyPileUpBias",
      "UncertaintyPileUpJetRate",
      "UncertaintySinglePionHCAL",
      "UncertaintyRelativePtEC1",
      "UncertaintyRelativePtEC2",
      "UncertaintyRelativePtHF",
      "UncertaintyRelativeSample",
      "UncertaintyPileUpPtEC",
      "UncertaintyPileUpPtHF",
      "L1RC",
      "L1Residual",
      "UncertaintyAux3",
      "UncertaintyAux4",
  };

  bool fill_eta_hist(JetCorrectorParameters const& JCParam,
                     TH1D* hist,
                     const std::map<std::string, std::string>& paramValues) {
    if (!(JCParam.isValid())) {
      edm::LogWarning("JEC_PI") << "JetCorrectorParameter is not valid.";
      return false;
    }

    std::vector<double> vars;
    std::vector<float> bins;
    std::vector<double> params;

    double par_JetPt = 100.;
    double par_JetEta = 0.;
    double par_JetA = 0.5;
    double par_Rho = 20.;
    double par_JetPhi = 0.;
    double par_JetE = 150.;

    // Default values will be used if no input parameters
    auto ip = paramValues.find("Jet_Pt");
    if (ip != paramValues.end()) {
      par_JetPt = std::stod(ip->second);
    }
    ip = paramValues.find("Jet_Rho");
    if (ip != paramValues.end()) {
      par_Rho = std::stod(ip->second);
    }

    int ir = -1;

    for (size_t idx = 0; idx <= NBIN_ETA; idx++) {
      par_JetEta = (idx + 0.5) * (MAX_ETA - MIN_ETA) / NBIN_ETA + MIN_ETA;

      if (JCParam.definitions().formula().compare("1") == 0) {  // unity
        hist->SetBinContent(idx + 1, 1.);
        continue;
      } else if (JCParam.definitions().level().compare("Uncertainty") == 0 &&
                 JCParam.definitions().formula().compare("\"\"") == 0) {
        JetCorrectionUncertainty JCU(JCParam);
        JCU.setJetEta(par_JetEta);
        JCU.setJetPt(par_JetPt);
        JCU.setJetPhi(par_JetPhi);
        JCU.setJetE(par_JetE);

        float unc = JCU.getUncertainty(true);
        hist->SetBinContent(idx + 1, unc);

        continue;
      }

      reco::FormulaEvaluator formula(JCParam.definitions().formula());

      vars.clear();
      bins.clear();
      params.clear();

      for (size_t i = 0; i < JCParam.definitions().nBinVar(); i++) {
        // fill up the parameter variables
        if (JCParam.definitions().binVar(i).compare("JetPt") == 0)
          bins.push_back(par_JetPt);
        if (JCParam.definitions().binVar(i).compare("JetEta") == 0)
          bins.push_back(par_JetEta);
        if (JCParam.definitions().binVar(i).compare("JetA") == 0)
          bins.push_back(par_JetA);
        if (JCParam.definitions().binVar(i).compare("Rho") == 0)
          bins.push_back(par_Rho);
      }

      for (size_t i = 0; i < JCParam.definitions().nParVar(); i++) {
        // fill up the parameter variables
        if (JCParam.definitions().parVar(i).compare("JetPt") == 0)
          vars.push_back(par_JetPt);
        if (JCParam.definitions().parVar(i).compare("JetEta") == 0)
          vars.push_back(par_JetEta);
        if (JCParam.definitions().parVar(i).compare("JetA") == 0)
          vars.push_back(par_JetA);
        if (JCParam.definitions().parVar(i).compare("Rho") == 0)
          vars.push_back(par_Rho);
      }

      ir = JCParam.binIndex(bins);

      if (ir < 0 || ir > (int)JCParam.size())
        continue;

      // Extract JEC formula parameters from payload
      for (size_t i = 2 * vars.size(); i < JCParam.record(ir).nParameters(); i++) {
        double par = JCParam.record(ir).parameter(i);
        params.push_back(par);
      }

      double jec = formula.evaluate(vars, params);
      hist->SetBinContent(idx + 1, jec);

    }  // x_axis
    return true;

  }  // fill_eta_hist()

  /*******************************************************
 *    
 *         1d histogram of JetCorectorParameters of 1 IOV 
 *
   *******************************************************/

  // inherit from one of the predefined plot class: Histogram1D

  template <levelt ii>
  class JetCorrectorVsEta : public cond::payloadInspector::Histogram1D<JetCorrectorParametersCollection, SINGLE_IOV> {
  public:
    JetCorrectorVsEta()
        : cond::payloadInspector::Histogram1D<JetCorrectorParametersCollection, SINGLE_IOV>(
              "Jet Corrector", "#eta", NBIN_ETA, MIN_ETA, MAX_ETA, "Corrector") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Rho");
    }

    bool fill() override {
      double par_JetPt = 100.;
      double par_Rho = 20.;

      // Default values will be used if no input parameters
      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_JetPt = std::stod(ip->second);
        edm::LogPrint("JEC_PI") << "Jet Pt: " << par_JetPt;
      }
      ip = paramValues.find("Jet_Rho");
      if (ip != paramValues.end()) {
        par_Rho = std::stod(ip->second);
        edm::LogPrint("JEC_PI") << "Rho: " << par_Rho;
      }

      TH1D* jec_hist = new TH1D("JEC vs. #eta", "", NBIN_ETA, MIN_ETA, MAX_ETA);

      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<JetCorrectorParametersCollection> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          // JetCorrectorParametersCollection::key_type
          std::vector<key_t> keys;
          // Get valid keys in this payload
          (*payload).validKeys(keys);

          if (std::find(keys.begin(), keys.end(), ii) == keys.end()) {
            edm::LogWarning("JEC_PI") << "Jet corrector level " << (*payload).findLabel(ii) << " is not available.";
            return false;
          }
          auto JCParams = (*payload)[ii];

          edm::LogInfo("JEC_PI") << "Jet corrector level " << (*payload).findLabel(ii) << " as "
                                 << JCParams.definitions().level() << " has " << JCParams.definitions().nParVar()
                                 << " parameters with " << JCParams.definitions().nBinVar() << " bin(s).";
          edm::LogInfo("JEC_PI") << "JCParam size: " << JCParams.size();
          edm::LogInfo("JEC_PI") << "JCParam def: " << JCParams.definitions().formula();

          fill_eta_hist(JCParams, jec_hist, paramValues);

          for (size_t idx = 0; idx <= NBIN_ETA; idx++) {
            fillWithBinAndValue(idx + 1, jec_hist->GetBinContent(idx + 1));
          }  // for eta

          return true;
        }
        return false;
      }  // for iovs
      return false;
    }  // fill
  };   // class

  class JetCorrectorVsEtaSummary
      : public cond::payloadInspector::PlotImage<JetCorrectorParametersCollection, SINGLE_IOV> {
  public:
    JetCorrectorVsEtaSummary()
        : cond::payloadInspector::PlotImage<JetCorrectorParametersCollection, SINGLE_IOV>("Jet Correction Summary") {
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Rho");
    }

    bool fill() override {
      double par_JetPt = 100.;
      double par_JetA = 0.5;
      double par_Rho = 40.;

      // Default values will be used if no input parameters (legend)
      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_JetPt = std::stod(ip->second);
      }
      ip = paramValues.find("Jet_Rho");
      if (ip != paramValues.end()) {
        par_Rho = std::stod(ip->second);
      }

      TH1D* jec_l1fj = new TH1D("JEC L1FastJet vs. #eta", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* jec_l2rel = new TH1D("JEC L2Relative vs. #eta", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* jec_l2l3 = new TH1D("JEC L2L3Residual vs. #eta", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* jec_l1rc = new TH1D("JEC L1RC vs. #eta", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* jec_uncert = new TH1D("JEC Uncertainty vs. #eta", "", NBIN_ETA, MIN_ETA, MAX_ETA);

      TLegend* leg_eta = new TLegend(0.50, 0.73, 0.935, 0.90);
      TLegend* leg_eta2 = new TLegend(0.50, 0.83, 0.935, 0.90);

      leg_eta->SetBorderSize(0);
      leg_eta->SetLineStyle(0);
      leg_eta->SetFillStyle(0);
      leg_eta->SetTextFont(42);

      leg_eta2->SetBorderSize(0);
      leg_eta2->SetLineStyle(0);
      leg_eta2->SetFillStyle(0);
      leg_eta2->SetTextFont(42);

      auto tag = PlotBase::getTag<0>();
      auto iov = tag.iovs.front();

      std::shared_ptr<JetCorrectorParametersCollection> payload = fetchPayload(std::get<1>(iov));
      unsigned int run = std::get<0>(iov);
      std::string tagname = tag.name;
      std::stringstream ss_tagname(tag.name);
      std::string stmp;

      std::string tag_ver;

      getline(ss_tagname, stmp, '_');  // drop first
      getline(ss_tagname, stmp);       // get the rest
      tag_ver = stmp;

      if (payload.get()) {
        auto JCParam_L1FJ = (*payload)[L1FastJet];
        auto JCParam_L2 = (*payload)[L2Relative];
        auto JCParam_L2L3 = (*payload)[L2L3Residual];
        auto JCParam_L1RC = (*payload)[L1RC];
        auto JCParam_Unc = (*payload)[Uncertainty];

        auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();

        fill_eta_hist(JCParam_L1FJ, jec_l1fj, paramValues);
        fill_eta_hist(JCParam_L2, jec_l2rel, paramValues);
        fill_eta_hist(JCParam_L2L3, jec_l2l3, paramValues);
        fill_eta_hist(JCParam_L1RC, jec_l1rc, paramValues);
        fill_eta_hist(JCParam_Unc, jec_uncert, paramValues);

        gStyle->SetOptStat(0);
        gStyle->SetLabelFont(42, "XYZ");
        gStyle->SetLabelSize(0.05, "XYZ");
        gStyle->SetFrameLineWidth(3);

        std::string title = Form("Summary Run %i", run);
        TCanvas canvas("Jet Energy Correction", title.c_str(), 800, 1200);
        canvas.Divide(1, 2);

        canvas.cd(1);
        jec_l1fj->SetTitle(tag_ver.c_str());
        jec_l1fj->SetXTitle("#eta");
        jec_l1fj->SetMaximum(1.6);
        jec_l1fj->SetMinimum(0.0);
        jec_l1fj->SetLineWidth(3);
        jec_l1fj->Draw("][");

        jec_l2rel->SetLineColor(2);
        jec_l2rel->SetLineWidth(3);
        jec_l2rel->Draw("][same");

        jec_l2l3->SetLineColor(8);
        jec_l2l3->SetLineWidth(3);
        jec_l2l3->Draw("][same");

        jec_l1rc->SetLineColor(9);
        jec_l1rc->SetLineWidth(3);
        jec_l1rc->Draw("][same");

        leg_eta->AddEntry(jec_l1fj, (*payload).findLabel(L1FastJet).c_str(), "l");
        leg_eta->AddEntry(jec_l2rel, (*payload).findLabel(L2Relative).c_str(), "l");
        leg_eta->AddEntry(jec_l2l3, (*payload).findLabel(L2L3Residual).c_str(), "l");
        leg_eta->AddEntry(jec_l1rc, (*payload).findLabel(L1RC).c_str(), "l");
        leg_eta->AddEntry((TObject*)nullptr, Form("JetPt=%.2f; JetA=%.2f; Rho=%.2f", par_JetPt, par_JetA, par_Rho), "");
        leg_eta->Draw();

        canvas.cd(2);
        jec_uncert->SetTitle(tag_ver.c_str());
        jec_uncert->SetXTitle("#eta");
        jec_uncert->SetMaximum(0.1);
        jec_uncert->SetMinimum(0.0);
        jec_uncert->SetLineColor(6);
        jec_uncert->SetLineWidth(3);
        jec_uncert->Draw("][");

        leg_eta2->AddEntry(jec_uncert, (*payload).findLabel(Uncertainty).c_str(), "l");
        leg_eta2->AddEntry(
            (TObject*)nullptr, Form("JetPt=%.2f; JetA=%.2f; Rho=%.2f", par_JetPt, par_JetA, par_Rho), "");
        leg_eta2->Draw();

        canvas.SaveAs(m_imageFileName.c_str());

        return true;
      } else  // no payload.get()
        return false;
    }  // fill

  };  // class

  class JetCorrectorVsEtaCompare
      : public cond::payloadInspector::PlotImage<JetCorrectorParametersCollection, SINGLE_IOV, 2> {
  public:
    JetCorrectorVsEtaCompare()
        : cond::payloadInspector::PlotImage<JetCorrectorParametersCollection, SINGLE_IOV, 2>(
              "Jet Correction Compare Two Tags") {
      cond::payloadInspector::PlotBase::addInputParam("Corr_Level");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Pt");
      cond::payloadInspector::PlotBase::addInputParam("Jet_Rho");
    }

    bool fill() override {
      double par_JetPt = 100.;
      double par_JetA = 0.5;
      double par_Rho = 40.;
      key_t par_Level = L1FastJet;

      // Default values will be used if no input parameters (legend)
      auto paramValues = cond::payloadInspector::PlotBase::inputParamValues();
      auto ip = paramValues.find("Jet_Pt");
      if (ip != paramValues.end()) {
        par_JetPt = std::stod(ip->second);
      }
      ip = paramValues.find("Jet_Rho");
      if (ip != paramValues.end()) {
        par_Rho = std::stod(ip->second);
      }

      TH1D* jec_one = new TH1D("JEC vs. #eta one", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D* jec_two = new TH1D("JEC vs. #eta two", "", NBIN_ETA, MIN_ETA, MAX_ETA);

      TLegend* leg_eta = new TLegend(0.50, 0.73, 0.935, 0.90);

      leg_eta->SetBorderSize(0);
      leg_eta->SetLineStyle(0);
      leg_eta->SetFillStyle(0);
      leg_eta->SetTextFont(42);

      auto tag1 = PlotBase::getTag<0>();
      auto iov1 = tag1.iovs.front();

      auto tag2 = PlotBase::getTag<1>();
      auto iov2 = tag2.iovs.front();

      std::shared_ptr<JetCorrectorParametersCollection> payload1 = fetchPayload(std::get<1>(iov1));
      std::string tagname1 = tag1.name;

      std::shared_ptr<JetCorrectorParametersCollection> payload2 = fetchPayload(std::get<1>(iov2));
      std::string tagname2 = tag2.name;

      ip = paramValues.find("Corr_Level");
      if (ip != paramValues.end()) {
        // input level as index
        if (ip->second.length() < 3) {
          int ii = std::stoi(ip->second);
          par_Level = static_cast<key_t>(ii);
        } else {
          // input level as text
          std::vector<std::string>::const_iterator found3 = find(labels_.begin(), labels_.end(), ip->second);
          if (found3 != labels_.end()) {
            par_Level = static_cast<key_t>(found3 - labels_.begin());
          }
        }
      }

      std::string stmp;
      std::stringstream ss_tagname(tag1.name);

      std::string tag_ver1;
      std::string tag_ver2;

      getline(ss_tagname, stmp, '_');  // drop first
      getline(ss_tagname, stmp);       // get the rest of the string
      tag_ver1 = stmp;

      std::stringstream ss_tagname2(tag2.name);

      getline(ss_tagname2, stmp, '_');  // drop first
      getline(ss_tagname2, stmp);       // get the rest
      tag_ver2 = stmp;

      if (payload1.get() && payload2.get()) {
        // JetCorrectorParametersCollection::key_type
        std::vector<key_t> keys;
        // Get valid keys in this payload
        (*payload1).validKeys(keys);

        if (std::find(keys.begin(), keys.end(), par_Level) == keys.end()) {
          edm::LogWarning("JEC_PI") << "Jet corrector level " << (*payload1).findLabel(par_Level)
                                    << " is not available for tag one.";
          return false;
        }

        keys.clear();
        (*payload2).validKeys(keys);
        if (std::find(keys.begin(), keys.end(), par_Level) == keys.end()) {
          edm::LogWarning("JEC_PI") << "Jet corrector level " << (*payload2).findLabel(par_Level)
                                    << " is not available for twg two.";
          return false;
        }

        auto JCParam_one = (*payload1)[par_Level];
        auto JCParam_two = (*payload2)[par_Level];

        fill_eta_hist(JCParam_one, jec_one, paramValues);
        fill_eta_hist(JCParam_two, jec_two, paramValues);

        gStyle->SetOptStat(0);
        gStyle->SetLabelFont(42, "XYZ");
        gStyle->SetLabelSize(0.05, "XYZ");
        gStyle->SetFrameLineWidth(3);

        std::string title = Form("Comparison between %s and %s", tag1.name.c_str(), tag2.name.c_str());
        TCanvas canvas("Jet Energy Correction", title.c_str(), 800, 600);

        canvas.cd();
        jec_one->SetTitle(("JetCorrector comparison for " + (*payload1).findLabel(par_Level)).c_str());
        jec_one->SetXTitle("#eta");
        jec_one->SetMaximum(1.6);
        if (par_Level == Uncertainty) {
          jec_one->SetMaximum(0.1);
        }
        jec_one->SetMinimum(0.0);
        jec_one->SetLineWidth(3);
        jec_one->Draw("][");

        jec_two->SetLineColor(2);
        jec_two->SetLineWidth(3);
        jec_two->SetLineStyle(2);
        jec_two->Draw("][same");

        leg_eta->AddEntry(jec_one, tag_ver1.c_str(), "l");
        leg_eta->AddEntry(jec_two, tag_ver2.c_str(), "l");
        leg_eta->AddEntry((TObject*)nullptr, Form("JetPt=%.2f; JetA=%.2f; Rho=%.2f", par_JetPt, par_JetA, par_Rho), "");
        leg_eta->Draw();

        canvas.SaveAs(m_imageFileName.c_str());

        return true;
      } else  // no payload.get()
        return false;
    }  // fill

  };  // class

  typedef JetCorrectorVsEta<L1Offset> JetCorrectorVsEtaL1Offset;
  typedef JetCorrectorVsEta<L1FastJet> JetCorrectorVsEtaL1FastJet;
  typedef JetCorrectorVsEta<L2Relative> JetCorrectorVsEtaL2Relative;
  typedef JetCorrectorVsEta<L3Absolute> JetCorrectorVsEtaL3Absolute;
  typedef JetCorrectorVsEta<L2L3Residual> JetCorrectorVsEtaL2L3Residual;
  typedef JetCorrectorVsEta<Uncertainty> JetCorrectorVsEtaUncertainty;
  typedef JetCorrectorVsEta<L1RC> JetCorrectorVsEtaL1RC;

  // Register the classes as boost python plugin
  PAYLOAD_INSPECTOR_MODULE(JetCorrectorParametersCollection) {
    PAYLOAD_INSPECTOR_CLASS(JetCorrectorVsEtaL1Offset);
    PAYLOAD_INSPECTOR_CLASS(JetCorrectorVsEtaL1FastJet);
    PAYLOAD_INSPECTOR_CLASS(JetCorrectorVsEtaL2Relative);
    PAYLOAD_INSPECTOR_CLASS(JetCorrectorVsEtaL3Absolute);
    PAYLOAD_INSPECTOR_CLASS(JetCorrectorVsEtaL2L3Residual);
    PAYLOAD_INSPECTOR_CLASS(JetCorrectorVsEtaUncertainty);
    PAYLOAD_INSPECTOR_CLASS(JetCorrectorVsEtaL1RC);

    PAYLOAD_INSPECTOR_CLASS(JetCorrectorVsEtaSummary);
    PAYLOAD_INSPECTOR_CLASS(JetCorrectorVsEtaCompare);
  }

}  // namespace
