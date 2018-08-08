#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalPedestals.h"

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

  class HcalPedestalContainer : public HcalObjRepresent::HcalDataContainer<HcalPedestals,HcalPedestal> {
  public:
    HcalPedestalContainer(std::shared_ptr<HcalPedestals> payload, unsigned int run) : HcalObjRepresent::HcalDataContainer<HcalPedestals,HcalPedestal>(payload, run) {}
    float getValue(HcalPedestal* ped) override {
      return (ped->getValue(0) + ped->getValue(1) + ped->getValue(2) + ped->getValue(3))/4;
    }
  };

  /******************************************
     2d plot of HCAL Pedestal of 1 IOV
  ******************************************/
  class HcalPedestalsPlot : public cond::payloadInspector::PlotImage<HcalPedestals> {
  public:
    HcalPedestalsPlot() : cond::payloadInspector::PlotImage<HcalPedestals>("HCAL Pedestal Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalPedestals> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalPedestalContainer* objContainer = new HcalPedestalContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL Pedestal difference between 2 IOVs
  **********************************************************/
  class HcalPedestalsDiff : public cond::payloadInspector::PlotImage<HcalPedestals> {

  public:
    HcalPedestalsDiff() : cond::payloadInspector::PlotImage<HcalPedestals>("HCAL Pedestal Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalPedestals> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalPedestals> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalPedestalContainer* objContainer1 = new HcalPedestalContainer(payload1, std::get<0>(iov1));
        HcalPedestalContainer* objContainer2 = new HcalPedestalContainer(payload2, std::get<0>(iov2));
        objContainer2->Subtract(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
  /******************************************
     2d plot of HCAL Pedestal of 1 IOV
  ******************************************/
  class HcalPedestalsEtaPlot : public cond::payloadInspector::PlotImage<HcalPedestals> {
  public:
    HcalPedestalsEtaPlot() : cond::payloadInspector::PlotImage<HcalPedestals>("HCAL Pedestal Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalPedestals> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalPedestalContainer* objContainer = new HcalPedestalContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL Pedestal difference between 2 IOVs
  **********************************************************/
  class HcalPedestalsEtaDiff : public cond::payloadInspector::PlotImage<HcalPedestals> {

  public:
    HcalPedestalsEtaDiff() : cond::payloadInspector::PlotImage<HcalPedestals>("HCAL Pedestal Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalPedestals> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalPedestals> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalPedestalContainer* objContainer1 = new HcalPedestalContainer(payload1, std::get<0>(iov1));
        HcalPedestalContainer* objContainer2 = new HcalPedestalContainer(payload2, std::get<0>(iov2));
        objContainer2->Subtract(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
  /******************************************
     2d plot of HCAL Pedestal of 1 IOV
  ******************************************/
  class HcalPedestalsPhiPlot : public cond::payloadInspector::PlotImage<HcalPedestals> {
  public:
    HcalPedestalsPhiPlot() : cond::payloadInspector::PlotImage<HcalPedestals>("HCAL Pedestal Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalPedestals> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalPedestalContainer* objContainer = new HcalPedestalContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL Pedestal difference between 2 IOVs
  **********************************************************/
  class HcalPedestalsPhiDiff : public cond::payloadInspector::PlotImage<HcalPedestals> {

  public:
    HcalPedestalsPhiDiff() : cond::payloadInspector::PlotImage<HcalPedestals>("HCAL Pedestal Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalPedestals> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalPedestals> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalPedestalContainer* objContainer1 = new HcalPedestalContainer(payload1, std::get<0>(iov1));
        HcalPedestalContainer* objContainer2 = new HcalPedestalContainer(payload2, std::get<0>(iov2));
        objContainer2->Subtract(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalPedestals){
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsDiff);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsPhiPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsEtaPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsEtaDiff);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalsPhiDiff);
}
