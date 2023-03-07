#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/JetMETObjects/interface/JetResolution.h"
#include "CondFormats/JetMETObjects/interface/JetResolutionObject.h"
#include <CondFormats/JetMETObjects/interface/Utilities.h>

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

namespace JME{

  using namespace cond::payloadInspector;

  enum index{NORM=0, DOWN=1, UP=2};

  /*******************************************************
 *    
 *         1d histogram of JetResolution of 1 IOV 
 *
   *******************************************************/

  // inherit from one of the predefined plot class: Histogram1D

  class JetResolutionVsEta : public cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV> {
    public:
      static const int MIN_ETA = -5.05;
      static const int MAX_ETA =  5.05;
      static const int NBIN = 51;

    JetResolutionVsEta() : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>( "Jet Resolution", "#eta", NBIN, MIN_ETA, MAX_ETA, "Resolution"){
      Base::setSingleIov( true );
    }

    bool fill()override{
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
          std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if(payload->getRecords().size() > 0 &&  // No formula for SF
             payload->getDefinition().getFormulaString().compare("")==0) return false; 

          for(const auto& record : payload->getRecords()){ 
            // Check Pt & Rho
            if(record.getVariablesRange().size()>0 && 
               payload->getDefinition().getVariableName(0) == "JetPt" &&
               record.getVariablesRange()[0].is_inside(100.)){
              if(record.getBinsRange().size()>1 &&
                 payload->getDefinition().getBinName(1) == "Rho" &&
                 record.getBinsRange()[1].is_inside(20.)){
                if(record.getBinsRange().size()>0 &&
                   payload->getDefinition().getBinName(0) == "JetEta"){
                  reco::FormulaEvaluator f(payload->getDefinition().getFormulaString());

                  for(size_t idx = 0; idx < NBIN; idx++){
                    double x_axis = (idx+0.5)*(MAX_ETA-MIN_ETA)/NBIN+MIN_ETA;
                    if(record.getBinsRange()[0].is_inside(x_axis)){
                      std::vector<double> var={100.};
                      std::vector<double> param;
                      for(size_t i = 0; i < record.getParametersValues().size(); i++){
                        double par = record.getParametersValues()[i];
                        param.push_back(par);
                      }
                      float res = f.evaluate(var, param);
                      fillWithBinAndValue(idx+1, res);
                    }
                  }
                }
              }
            }
          }  // records
            return true;    
        }
      }
      return false;
    }
  };  // class

  class JetResolutionVsPt : public cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV> {
    public:
      static const int MIN_PT = -0.0;
      static const int MAX_PT =  3000.0;
      static const int NBIN = 300;

    JetResolutionVsPt() : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>( "Jet Energy Resolution", "p_T", NBIN, MIN_PT, MAX_PT, "Resolution"){
      Base::setSingleIov( true );
    }

    bool fill()override{
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if(payload->getRecords().size() > 0 &&  // No formula for SF
             payload->getDefinition().getFormulaString().compare("")==0) return false; 

          for(const auto& record : payload->getRecords()){ 
            // Check Eta & Rho
            if(record.getBinsRange().size()>0 && 
               payload->getDefinition().getBinName(0) == "JetEta" &&
               record.getBinsRange()[0].is_inside(2.30)){
              if(record.getBinsRange().size()>1 &&
                 payload->getDefinition().getBinName(1) == "Rho" &&
                 record.getBinsRange()[1].is_inside(15.)){
                if(record.getVariablesRange().size()>0 && 
                   payload->getDefinition().getVariableName(0) == "JetPt"){
                   reco::FormulaEvaluator f(payload->getDefinition().getFormulaString());

                  for(size_t idx = 0; idx <= NBIN; idx++){
                      double x_axis = (idx+0.5)*(MAX_PT-MIN_PT)/NBIN+MIN_PT;
                    if(record.getVariablesRange()[0].is_inside(x_axis)){
                       std::vector<double> var={x_axis};
                       std::vector<double> param;
                      for(size_t i = 0; i < record.getParametersValues().size(); i++){
                        double par = record.getParametersValues()[i];
                        param.push_back(par);
                      }
                      float res = f.evaluate(var, param);
                      fillWithBinAndValue(idx+1, res);
                    }
                  }
                }
              }
            }
          }
          return true;    
        }
      }
      return false;
    }
  };

  class JetResolutionSummary : public cond::payloadInspector::PlotImage<JetResolutionObject> {
    public:
      static const int MIN_ETA = -5.05;
      static const int MAX_ETA =  5.05;
      static const int NBIN_ETA = 51;
      static const int MIN_PT = -5.0;
      static const int MAX_PT =  3005.0;
      static const int NBIN_PT = 302;

    JetResolutionSummary() : cond::payloadInspector::PlotImage<JetResolutionObject>( "Jet Resolution Summary"){
      Base::setSingleIov( true );
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs)override{
      TH1D *resol_eta = new TH1D("Jet Resolution vs #eta", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D *resol_pt = new TH1D("Jet Resolution vs p_T", "", NBIN_PT, MIN_PT, MAX_PT);
      unsigned int run = 0;
      std::string tagname;

      auto iov = iovs.front();
      std::shared_ptr<JetResolutionObject> payload = fetchPayload(std::get<1>(iov));
      run = std::get<0>(iov);
      tagname = cond::payloadInspector::PlotBase::getTag<0>().name;

      if (payload.get()) {
        if(payload->getRecords().size() > 0 &&  // No formula for SF
           payload->getDefinition().getFormulaString().compare("")==0) return false; 

          for(const auto& record : payload->getRecords()){ 
            // Check Pt & Rho
            if(record.getVariablesRange().size()>0 && 
               payload->getDefinition().getVariableName(0) == "JetPt" &&
               record.getVariablesRange()[0].is_inside(100.)){
              if(record.getBinsRange().size()>1 &&
                 payload->getDefinition().getBinName(1) == "Rho" &&
                 record.getBinsRange()[1].is_inside(20.)){
                if(record.getBinsRange().size()>0 &&
                   payload->getDefinition().getBinName(0) == "JetEta"){
                  reco::FormulaEvaluator f(payload->getDefinition().getFormulaString());

                  for(size_t idx = 0; idx < NBIN_ETA; idx++){
                    double x_axis = (idx+0.5)*(MAX_ETA-MIN_ETA)/NBIN_ETA+MIN_ETA;
                    if(record.getBinsRange()[0].is_inside(x_axis)){
                      std::vector<double> var={100.};
                      std::vector<double> param;
                      for(size_t i = 0; i < record.getParametersValues().size(); i++){
                        double par = record.getParametersValues()[i];
                        param.push_back(par);
                      }
                      float res = f.evaluate(var, param);
                      resol_eta->SetBinContent(idx+1, res);
                    }
                  }
                }
              }
            }

            if(record.getBinsRange().size()>0 && 
               payload->getDefinition().getBinName(0) == "JetEta" &&
               record.getBinsRange()[0].is_inside(2.30)){
              if(record.getBinsRange().size()>1 &&
                 payload->getDefinition().getBinName(1) == "Rho" &&
                 record.getBinsRange()[1].is_inside(15.)){
                if(record.getVariablesRange().size()>0 && 
                   payload->getDefinition().getVariableName(0) == "JetPt"){
                   reco::FormulaEvaluator f(payload->getDefinition().getFormulaString());

                  for(size_t idx = 0; idx <= NBIN_PT+2; idx++){
                      double x_axis = (idx+0.5)*(MAX_PT-MIN_PT)/NBIN_PT+MIN_PT;
                    if(record.getVariablesRange()[0].is_inside(x_axis)){
                       std::vector<double> var={x_axis};
                       std::vector<double> param;
                      for(size_t i = 0; i < record.getParametersValues().size(); i++){
                        double par = record.getParametersValues()[i];
                        param.push_back(par);
                      }
                      float res = f.evaluate(var, param);
                      resol_pt->SetBinContent(idx+1, res);
                    }
                  }
                }
              }
            }
          }  // records
          gStyle->SetOptStat(0);

          std::string title = Form("Summary Run %i", run);
          TCanvas canvas("Jet Resolution Summary", title.c_str(), 800, 1200);
          canvas.Divide(1, 2);

          canvas.cd(1);
          resol_eta->SetTitle(tagname.c_str());
          resol_eta->SetXTitle("#eta");
          resol_eta->SetYTitle("Resolution");
          resol_eta->Draw();

          canvas.cd(2);
          resol_pt->SetTitle(tagname.c_str());
          resol_pt->SetXTitle("p_{T} [GeV]");
          resol_pt->SetYTitle("Resolution");
          resol_pt->Draw();

          canvas.SaveAs(m_imageFileName.c_str());

          return true;    
      }//else // no payload.get()
        return false;
    }  // fill

  };  // class

  template <index ii> class JetScaleFactorVsEta : public cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV> {
    public:
      static const int MIN_ETA = -5.0;
      static const int MAX_ETA =  5.0;
      static const int NBIN = 50;

    JetScaleFactorVsEta() : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>( "Jet Energy Scale Factor", "#eta", NBIN, MIN_ETA, MAX_ETA, "Scale Factor"){
      Base::setSingleIov( true );
    }

    bool fill()override{
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
          std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if(payload->getRecords().size() > 0 &&  // No formula for SF
             payload->getDefinition().getFormulaString().compare("")!=0) return false; 

          for(const auto& record : payload->getRecords()){ 
            if(record.getBinsRange().size()>0 &&
              payload->getDefinition().getBinName(0) == "JetEta" &&
              record.getParametersValues().size() == 3){ // norm, down, up

              for(size_t it = 0; it < NBIN; it++){
                  double x_axis = (it+0.5)*(MAX_ETA-MIN_ETA)/NBIN+MIN_ETA;
                if(record.getBinsRange()[0].is_inside(x_axis)){
                  double sf = 0.;
                  sf = record.getParametersValues()[ii];
                  fillWithBinAndValue(it, sf);
                }
              }
            }
          }  // records
          return true;    
        }
      }
      return false;
    }
  };  // class

  typedef JetScaleFactorVsEta<NORM> JetScaleFactorVsEtaNORM;
  typedef JetScaleFactorVsEta<DOWN> JetScaleFactorVsEtaDOWN;
  typedef JetScaleFactorVsEta<UP> JetScaleFactorVsEtaUP;

  class JetScaleFactorSummary : public cond::payloadInspector::PlotImage<JetResolutionObject> {
    public:
      static const int MIN_ETA = -5.05;
      static const int MAX_ETA =  5.05;
      static const int NBIN_ETA = 51;

    JetScaleFactorSummary() : cond::payloadInspector::PlotImage<JetResolutionObject>( "Jet ScaleFactor Summary"){
      Base::setSingleIov( true );
    }

    bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> > &iovs)override{
      TH1D *sf_eta_norm = new TH1D("Jet SF vs #eta NORM", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D *sf_eta_down = new TH1D("Jet SF vs #eta DOWN", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      TH1D *sf_eta_up = new TH1D("Jet SF vs #eta UP", "", NBIN_ETA, MIN_ETA, MAX_ETA);
      unsigned int run = 0;
      std::string tagname;

      auto iov = iovs.front();
      std::shared_ptr<JetResolutionObject> payload = fetchPayload(std::get<1>(iov));
      run = std::get<0>(iov);
      tagname = cond::payloadInspector::PlotBase::getTag<0>().name;

      if (payload.get()) {
        if(payload->getRecords().size() > 0 &&  // No formula for SF
           payload->getDefinition().getFormulaString().compare("")!=0) return false; 

        for(const auto& record : payload->getRecords()){ 
          if(record.getBinsRange().size()>0 &&
            payload->getDefinition().getBinName(0) == "JetEta" &&
            record.getParametersValues().size() == 3){ // norm, down, up

            for(size_t it = 0; it <= NBIN_ETA; it++){
                double x_axis = (it+0.5)*(MAX_ETA-MIN_ETA)/NBIN_ETA+MIN_ETA;
              if(record.getBinsRange()[0].is_inside(x_axis)){
                sf_eta_norm->SetBinContent(it+1, record.getParametersValues()[0]);
                sf_eta_down->SetBinContent(it+1, record.getParametersValues()[1]);
                sf_eta_up->SetBinContent(it+1, record.getParametersValues()[2]);
              }
            }
          }
        }  // records
        gStyle->SetOptStat(0);
        //gStyle->SetPadLeftMargin(0.16);
        //gStyle->SetPadRightMargin(0.02);

        std::string title = Form("Summary Run %i", run);
        TCanvas canvas("Jet ScaleFactor Summary", title.c_str(), 800, 600);

        canvas.cd();
        sf_eta_up->SetTitle(tagname.c_str());
        sf_eta_up->SetXTitle("#eta");
        sf_eta_up->SetYTitle("Scale Factor");
        sf_eta_up->SetLineStyle(7);
        sf_eta_up->SetLineWidth(3);
        sf_eta_up->SetFillColorAlpha(kGray, 0.5);
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

        canvas.SaveAs(m_imageFileName.c_str());

        return true;
      }//else // no payload.get()
      return false;
    }  // fill

  };  // class
// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( JetResolutionObject ){
  PAYLOAD_INSPECTOR_CLASS( JetResolutionVsEta );
  PAYLOAD_INSPECTOR_CLASS( JetResolutionVsPt );
  PAYLOAD_INSPECTOR_CLASS( JetScaleFactorVsEtaNORM );
  PAYLOAD_INSPECTOR_CLASS( JetScaleFactorVsEtaDOWN );
  PAYLOAD_INSPECTOR_CLASS( JetScaleFactorVsEtaUP );
  PAYLOAD_INSPECTOR_CLASS( JetResolutionSummary );
  PAYLOAD_INSPECTOR_CLASS( JetScaleFactorSummary );
}

}  // namespace

