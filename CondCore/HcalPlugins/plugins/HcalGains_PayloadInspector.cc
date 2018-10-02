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
#include "CondFormats/HcalObjects/interface/HcalGains.h" //or Gain.h???

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

  class HcalGainContainer : public HcalObjRepresent::HcalDataContainer<HcalGains,HcalGain> {
  public:
    HcalGainContainer(std::shared_ptr<HcalGains> payload, unsigned int run) : HcalObjRepresent::HcalDataContainer<HcalGains,HcalGain>(payload, run) {}
    float getValue(HcalGain* gain) override {
      return gain->getValue(0) + gain->getValue(1) + gain->getValue(2) + gain->getValue(3);
    }
  };

  /******************************************
     2d plot of HCAL Gain of 1 IOV
  ******************************************/
  class HcalGainsPlot : public cond::payloadInspector::PlotImage<HcalGains> {
  public:
    HcalGainsPlot() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalGains> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalGainContainer* objContainer = new HcalGainContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL Gain ratios between 2 IOVs
  **********************************************************/
  class HcalGainsRatio : public cond::payloadInspector::PlotImage<HcalGains> {

  public:
    HcalGainsRatio() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalGains> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalGains> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalGainContainer* objContainer1 = new HcalGainContainer(payload1, std::get<0>(iov1));
        HcalGainContainer* objContainer2 = new HcalGainContainer(payload2, std::get<0>(iov2));
 
        objContainer2->Divide(objContainer1);
  

        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;


    }// fill method
  };
  /******************************************
     2d plot of HCAL Gain of 1 IOV, projected along iphi
  ******************************************/
  class HcalGainsPhiPlot : public cond::payloadInspector::PlotImage<HcalGains> {
  public:
    HcalGainsPhiPlot() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalGains> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalGainContainer* objContainer = new HcalGainContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL Gain ratios between 2 IOVs, projected along iphi
  **********************************************************/
  class HcalGainsPhiRatio : public cond::payloadInspector::PlotImage<HcalGains> {

  public:
    HcalGainsPhiRatio() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalGains> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalGains> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalGainContainer* objContainer1 = new HcalGainContainer(payload1, std::get<0>(iov1));
        HcalGainContainer* objContainer2 = new HcalGainContainer(payload2, std::get<0>(iov2));
 
        objContainer2->Divide(objContainer1);
  

        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;


    }// fill method
  };
  /******************************************
     2d plot of HCAL Gain of 1 IOV, projected along ieta
  ******************************************/
  class HcalGainsEtaPlot : public cond::payloadInspector::PlotImage<HcalGains> {
  public:
    HcalGainsEtaPlot() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalGains> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalGainContainer* objContainer = new HcalGainContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL Gain ratios between 2 IOVs
  **********************************************************/
  class HcalGainsEtaRatio : public cond::payloadInspector::PlotImage<HcalGains> {

  public:
    HcalGainsEtaRatio() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalGains> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalGains> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalGainContainer* objContainer1 = new HcalGainContainer(payload1, std::get<0>(iov1));
        HcalGainContainer* objContainer2 = new HcalGainContainer(payload2, std::get<0>(iov2));
 
        objContainer2->Divide(objContainer1);
  

        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;


    }// fill method
  };
} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalGains){
  PAYLOAD_INSPECTOR_CLASS(HcalGainsPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsRatio);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsEtaPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsPhiPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsPhiRatio);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsEtaRatio);
}
