#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
//#include "DataFormats/EcalDetId/interface/EBDetId.h"
//#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
//#include "Geometry/HcalCommonData/interface/HcalTopologyMode.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h" //or ChannelStatus.h???

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

  class HcalChannelStatusContainer : public HcalObjRepresent::HcalDataContainer<HcalChannelQuality,HcalChannelStatus> {
  public:
    HcalChannelStatusContainer(std::shared_ptr<HcalChannelQuality> payload, unsigned int run) : HcalObjRepresent::HcalDataContainer<HcalChannelQuality,HcalChannelStatus>(payload, run) {}
    float getValue(HcalChannelStatus* chan) override {
      return chan->getValue()/32770;
    }
  };

  /******************************************
     2d plot of ECAL ChannelStatusRatios of 1 IOV
  ******************************************/
  class HcalChannelQualityPlot : public cond::payloadInspector::PlotImage<HcalChannelQuality> {
  public:
    HcalChannelQualityPlot() : cond::payloadInspector::PlotImage<HcalChannelQuality>("HCAL ChannelStatus Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalChannelQuality> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalChannelStatusContainer* objContainer = new HcalChannelStatusContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  class HcalChannelQualityChange : public cond::payloadInspector::PlotImage<HcalChannelQuality> {
  public:
    HcalChannelQualityChange() : cond::payloadInspector::PlotImage<HcalChannelQuality>("HCAL ChannelStatus Ratios - map ") {
      setSingleIov( false );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();
      std::shared_ptr<HcalChannelQuality> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalChannelQuality> payload2 = fetchPayload( std::get<1>(iov2) );
      if(payload1.get() && payload2.get()) {
        HcalChannelStatusContainer* objContainer1 = new HcalChannelStatusContainer(payload1, std::get<0>(iov1));
        HcalChannelStatusContainer* objContainer2 = new HcalChannelStatusContainer(payload2, std::get<0>(iov2));

        objContainer2->Subtract(objContainer1);
//
//        std::map< std::pair< std::string, int >, TH2F* > depths = objContainer1->GetDepths();
//        
//        
//        TODO: How do I display this?
//
//
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalChannelQuality){
  PAYLOAD_INSPECTOR_CLASS(HcalChannelQualityPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalChannelQualityChange);
}
