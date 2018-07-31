#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalRecoParams.h" //or RecoParam.h???

#include "TH2F.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPave.h"
#include "TPaveStats.h"
#include <string>
#include <fstream>

namespace {

  /********************************************
     printing float values of reco paramters
  *********************************************/
  class HcalRecoParamsSummary : public cond::payloadInspector::PlotImage<HcalRecoParams> {
  public:
    HcalRecoParamsSummary() : cond::payloadInspector::PlotImage<HcalRecoParams>("HCAL RecoParam Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalRecoParams> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        
        std::string subDetName;
        std::vector<HcalRecoParam> itemsVec;
        std::pair<std::string,int> depthKey;
        const char* histLabel;
        for(std::pair< std::string, std::vector<HcalRecoParam> > cont : (*payload).getAllContainers()){
            subDetName = std::get<0>(cont);
            itemsVec = std::get<1>(cont);
            //TODO Fill appropriately with all params of interest
            auto exampleVal = itemsVec.front().param1();
        }

        //TODO: Fill accordingly, should be same across all subdetectors, right??? CONFIRM
        //TCanvas* can = new TCanvas("",

        //std::string ImageName(m_imageFileName);
        //objContainer->getCanvasHE()->SaveAs(ImageName.c_str());
        return false;
        } else return false;
    }// fill method
  };
//TODO: Add a Change Summary?

} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalRecoParams){
  PAYLOAD_INSPECTOR_CLASS(HcalRecoParamsSummary);
}
