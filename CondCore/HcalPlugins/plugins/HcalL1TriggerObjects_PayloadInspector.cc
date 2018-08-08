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
#include "CondFormats/HcalObjects/interface/HcalL1TriggerObjects.h" //or L1TriggerObject.h???

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

  class HcalL1TriggerObjectContainer : public HcalObjRepresent::HcalDataContainer<HcalL1TriggerObjects,HcalL1TriggerObject> {
  public:
    HcalL1TriggerObjectContainer(std::shared_ptr<HcalL1TriggerObjects> payload, unsigned int run) : HcalObjRepresent::HcalDataContainer<HcalL1TriggerObjects,HcalL1TriggerObject>(payload, run) {}
    float getValue(HcalL1TriggerObject* trig) override {
      return trig->getRespGain();
    }
  };

  /******************************************
     2d plot of HCAL L1TriggerObject of 1 IOV
  ******************************************/
  class HcalL1TriggerObjectsPlot : public cond::payloadInspector::PlotImage<HcalL1TriggerObjects> {
  public:
    HcalL1TriggerObjectsPlot() : cond::payloadInspector::PlotImage<HcalL1TriggerObjects>("HCAL L1TriggerObject Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalL1TriggerObjects> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalL1TriggerObjectContainer* objContainer = new HcalL1TriggerObjectContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL L1TriggerObject difference between 2 IOVs
  **********************************************************/
  class HcalL1TriggerObjectsRatio : public cond::payloadInspector::PlotImage<HcalL1TriggerObjects> {

  public:
    HcalL1TriggerObjectsRatio() : cond::payloadInspector::PlotImage<HcalL1TriggerObjects>("HCAL L1TriggerObject Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalL1TriggerObjects> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalL1TriggerObjects> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalL1TriggerObjectContainer* objContainer1 = new HcalL1TriggerObjectContainer(payload1, std::get<0>(iov1));
        HcalL1TriggerObjectContainer* objContainer2 = new HcalL1TriggerObjectContainer(payload2, std::get<0>(iov2));
 
        objContainer2->Divide(objContainer1);

  

        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;


    }// fill method
  };
  /******************************************
     2d plot of HCAL L1TriggerObject of 1 IOV
  ******************************************/
  class HcalL1TriggerObjectsEtaPlot : public cond::payloadInspector::PlotImage<HcalL1TriggerObjects> {
  public:
    HcalL1TriggerObjectsEtaPlot() : cond::payloadInspector::PlotImage<HcalL1TriggerObjects>("HCAL L1TriggerObject Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalL1TriggerObjects> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalL1TriggerObjectContainer* objContainer = new HcalL1TriggerObjectContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL L1TriggerObject difference between 2 IOVs
  **********************************************************/
  class HcalL1TriggerObjectsEtaRatio : public cond::payloadInspector::PlotImage<HcalL1TriggerObjects> {

  public:
    HcalL1TriggerObjectsEtaRatio() : cond::payloadInspector::PlotImage<HcalL1TriggerObjects>("HCAL L1TriggerObject Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalL1TriggerObjects> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalL1TriggerObjects> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalL1TriggerObjectContainer* objContainer1 = new HcalL1TriggerObjectContainer(payload1, std::get<0>(iov1));
        HcalL1TriggerObjectContainer* objContainer2 = new HcalL1TriggerObjectContainer(payload2, std::get<0>(iov2));
 
        objContainer2->Divide(objContainer1);
  

        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;


    }// fill method
  };
  /******************************************
     2d plot of HCAL L1TriggerObject of 1 IOV
  ******************************************/
  class HcalL1TriggerObjectsPhiPlot : public cond::payloadInspector::PlotImage<HcalL1TriggerObjects> {
  public:
    HcalL1TriggerObjectsPhiPlot() : cond::payloadInspector::PlotImage<HcalL1TriggerObjects>("HCAL L1TriggerObject Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      

      auto iov = iovs.front();
      std::shared_ptr<HcalL1TriggerObjects> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalL1TriggerObjectContainer* objContainer = new HcalL1TriggerObjectContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL L1TriggerObject difference between 2 IOVs
  **********************************************************/
  class HcalL1TriggerObjectsPhiRatio : public cond::payloadInspector::PlotImage<HcalL1TriggerObjects> {

  public:
    HcalL1TriggerObjectsPhiRatio() : cond::payloadInspector::PlotImage<HcalL1TriggerObjects>("HCAL L1TriggerObject Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalL1TriggerObjects> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalL1TriggerObjects> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalL1TriggerObjectContainer* objContainer1 = new HcalL1TriggerObjectContainer(payload1, std::get<0>(iov1));
        HcalL1TriggerObjectContainer* objContainer2 = new HcalL1TriggerObjectContainer(payload2, std::get<0>(iov2));
 
        objContainer2->Divide(objContainer1);
  

        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;


    }// fill method
  };
} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalL1TriggerObjects){
  PAYLOAD_INSPECTOR_CLASS(HcalL1TriggerObjectsPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalL1TriggerObjectsRatio);
  PAYLOAD_INSPECTOR_CLASS(HcalL1TriggerObjectsEtaPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalL1TriggerObjectsEtaRatio);
  PAYLOAD_INSPECTOR_CLASS(HcalL1TriggerObjectsPhiPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalL1TriggerObjectsPhiRatio);
}
