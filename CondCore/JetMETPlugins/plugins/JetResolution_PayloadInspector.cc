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

  /*******************************************************
 *    
 *         1d histogram of JetResolution of 1 IOV 
 *
   *******************************************************/

  // inherit from one of the predefined plot class: Histogram1D

  class JetResolutionVsEta : public cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV> {
    public:
      static const int MIN_ETA = -5.0;
      static const int MAX_ETA =  5.0;
      static const int NBIN = 50;

    JetResolutionVsEta() : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>( "Jet Energy Resolution", "#eta", NBIN, MIN_ETA, MAX_ETA, "Resolution"){
      Base::setSingleIov( true );
    }

    bool fill()override{
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
          std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if(payload->getRecords().size() > 0 &&  // No formula for SF
             payload->getDefinition().getFormulaString() == "") return false; 

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
                      fillWithBinAndValue(idx, res);
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

    JetResolutionVsPt() : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>( "Jet Energy Resolution", "PT", NBIN, MIN_PT, MAX_PT, "Resolution"){
      Base::setSingleIov( true );
    }

    bool fill()override{
      auto tag = PlotBase::getTag<0>();
      for (auto const& iov : tag.iovs) {
        std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
        if (payload.get()) {
          if(payload->getRecords().size() > 0 &&  // No formula for SF
             payload->getDefinition().getFormulaString() == "") return false; 

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

                  for(size_t idx = 0; idx < NBIN; idx++){
                      double x_axis = (idx+0.5)*(MAX_PT-MIN_PT)/NBIN+MIN_PT;
                    if(record.getVariablesRange()[0].is_inside(x_axis)){
                       std::vector<double> var={x_axis};
                       std::vector<double> param;
                      for(size_t i = 0; i < record.getParametersValues().size(); i++){
                        double par = record.getParametersValues()[i];
                        param.push_back(par);
                      }
                      float res = f.evaluate(var, param);
                      fillWithBinAndValue(idx, res);
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

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( JetResolutionObject ){
  PAYLOAD_INSPECTOR_CLASS( JetResolutionVsEta );
  PAYLOAD_INSPECTOR_CLASS( JetResolutionVsPt );
}

}  // namespace

