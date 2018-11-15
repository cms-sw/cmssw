#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalPedestalWidths.h"

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

  class HcalPedestalWidthContainer : public HcalObjRepresent::HcalDataContainer<HcalPedestalWidths,HcalPedestalWidth> {
  public:
    HcalPedestalWidthContainer(std::shared_ptr<HcalPedestalWidths> payload, unsigned int run) : HcalObjRepresent::HcalDataContainer<HcalPedestalWidths,HcalPedestalWidth>(payload, run) {}
    float getValue(HcalPedestalWidth* ped) override {
      return (ped->getWidth(0) + ped->getWidth(1) + ped->getWidth(2) + ped->getWidth(3))/4;
    }
  };

  /******************************************
     2d plot of HCAL PedestalWidth of 1 IOV
  ******************************************/
  class HcalPedestalWidthsPlot : public cond::payloadInspector::PlotImage<HcalPedestalWidths> {
  public:
    HcalPedestalWidthsPlot() : cond::payloadInspector::PlotImage<HcalPedestalWidths>("HCAL PedestalWidth Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalPedestalWidths> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalPedestalWidthContainer* objContainer = new HcalPedestalWidthContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL PedestalWidth difference between 2 IOVs
  **********************************************************/
  class HcalPedestalWidthsDiff : public cond::payloadInspector::PlotImage<HcalPedestalWidths> {

  public:
    HcalPedestalWidthsDiff() : cond::payloadInspector::PlotImage<HcalPedestalWidths>("HCAL PedestalWidth Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalPedestalWidths> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalPedestalWidths> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalPedestalWidthContainer* objContainer1 = new HcalPedestalWidthContainer(payload1, std::get<0>(iov1));
        HcalPedestalWidthContainer* objContainer2 = new HcalPedestalWidthContainer(payload2, std::get<0>(iov2));
        objContainer2->Subtract(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
  /******************************************
     2d plot of HCAL PedestalWidth of 1 IOV
  ******************************************/
  class HcalPedestalWidthsEtaPlot : public cond::payloadInspector::PlotImage<HcalPedestalWidths> {
  public:
    HcalPedestalWidthsEtaPlot() : cond::payloadInspector::PlotImage<HcalPedestalWidths>("HCAL PedestalWidth Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalPedestalWidths> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalPedestalWidthContainer* objContainer = new HcalPedestalWidthContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL PedestalWidth difference between 2 IOVs
  **********************************************************/
  class HcalPedestalWidthsEtaDiff : public cond::payloadInspector::PlotImage<HcalPedestalWidths> {

  public:
    HcalPedestalWidthsEtaDiff() : cond::payloadInspector::PlotImage<HcalPedestalWidths>("HCAL PedestalWidth Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalPedestalWidths> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalPedestalWidths> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalPedestalWidthContainer* objContainer1 = new HcalPedestalWidthContainer(payload1, std::get<0>(iov1));
        HcalPedestalWidthContainer* objContainer2 = new HcalPedestalWidthContainer(payload2, std::get<0>(iov2));
        objContainer2->Subtract(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("EtaProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
  /******************************************
     2d plot of HCAL PedestalWidth of 1 IOV
  ******************************************/
  class HcalPedestalWidthsPhiPlot : public cond::payloadInspector::PlotImage<HcalPedestalWidths> {
  public:
    HcalPedestalWidthsPhiPlot() : cond::payloadInspector::PlotImage<HcalPedestalWidths>("HCAL PedestalWidth Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalPedestalWidths> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalPedestalWidthContainer* objContainer = new HcalPedestalWidthContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of HCAL PedestalWidth difference between 2 IOVs
  **********************************************************/
  class HcalPedestalWidthsPhiDiff : public cond::payloadInspector::PlotImage<HcalPedestalWidths> {

  public:
    HcalPedestalWidthsPhiDiff() : cond::payloadInspector::PlotImage<HcalPedestalWidths>("HCAL PedestalWidth Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalPedestalWidths> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalPedestalWidths> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalPedestalWidthContainer* objContainer1 = new HcalPedestalWidthContainer(payload1, std::get<0>(iov1));
        HcalPedestalWidthContainer* objContainer2 = new HcalPedestalWidthContainer(payload2, std::get<0>(iov2));
        objContainer2->Subtract(objContainer1);
        std::string ImageName(m_imageFileName);
        objContainer2->getCanvasAll("PhiProfile")->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalPedestalWidths){
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsDiff);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsPhiPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsPhiDiff);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsEtaPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalPedestalWidthsEtaDiff);
}
