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
 *         1d histogram of JetResolution in Eta of 1 IOV 
 *
   *******************************************************/

  // inherit from one of the predefined plot class: Histogram1D

  class JetResolutionEta : public cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV> {
    public:
      static const int MIN_ETA = -5.5;
      static const int MAX_ETA =  5.5;

    JetResolutionEta() : cond::payloadInspector::Histogram1D<JetResolutionObject, SINGLE_IOV>( "Jet Energy Resolution", "#eta", 50, MIN_ETA, MAX_ETA, "Resolution"){
      Base::setSingleIov( true );
    }

    bool fill()override{
      auto tag = PlotBase::getTag<0>();
        for (auto const& iov : tag.iovs) {
          std::shared_ptr<JetResolutionObject> payload = Base::fetchPayload(std::get<1>(iov));
          if (payload.get()) {
//            const std::vector<Binning>& bins = payload->getDefinition().getBins();
//            edm::LogWarning("JetResObj_PI") << "Bins size: " << payload->getDefinition().getBins().size() << "\n";
//            edm::LogWarning("JetResObj_PI") << "No. of binning variables: " << payload->getDefinition().nBins() << "\n";
//            for (const auto& bin: payload->getDefinition().getBinsName()) {
//              edm::LogWarning("JetResObj_PI") << bin << ", ";
//            }
//
//            edm::LogWarning("JetResObj_PI") << "No. of variables: " << payload->getDefinition().nVariables() << "\n";
//            for (const auto& bin: payload->getDefinition().getVariablesName()) {
//              edm::LogWarning("JetResObj_PI") << bin << ", ";
//            }

            edm::LogWarning("JetResObj_PI") << "Formula: " << payload->getDefinition().getFormulaString() << std::endl;

            edm::LogWarning("JetResObj_PI") << "Records: " << payload->getRecords().size() << "\n";

            for(const auto& record : payload->getRecords()){ 
              // Check Eta
              if(record.getVariablesRange().size()>0 && 
                 payload->getDefinition().getVariableName(0) == "JetPt" &&
                 record.getVariablesRange()[0].is_inside(100.)){
                if(record.getBinsRange().size()>1 &&
                    payload->getDefinition().getBinName(1) == "Rho" &&
                    record.getBinsRange()[1].is_inside(20.)){
                   if(record.getBinsRange().size()>0 &&
                      payload->getDefinition().getBinName(0) == "JetEta"){
                     reco::FormulaEvaluator f(payload->getDefinition().getFormulaString());

                     for(size_t idx = 0; idx < 50; idx++){
                       double x_axis = (idx+0.5)*(MAX_ETA-MIN_ETA)/50.+MIN_ETA;
                       if(record.getBinsRange()[0].is_inside(x_axis)){
                         std::vector<double> var={100.};
                         std::vector<double> param;// = record.getParametersValues();
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
              
          }    // payload
        }      // iovs
        return true;
//        return false;
    }  // fill
      
};  // class

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( JetResolutionObject ){
  PAYLOAD_INSPECTOR_CLASS( JetResolutionEta );
}

}  // namespace

