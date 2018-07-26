
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "CondCore/HcalPlugins/interface/HcalObjRepresent.h"

// the data format of the condition to be inspected
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"

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

  class HcalRespCorrContainer : public HcalObjRepresent::HcalDataContainer<HcalRespCorrs,HcalRespCorr> {
  public:
    HcalRespCorrContainer(std::shared_ptr<HcalRespCorrs> payload, unsigned int run) : HcalObjRepresent::HcalDataContainer<HcalRespCorrs,HcalRespCorr>(payload, run) {}
    float getValue(HcalRespCorr* rCor) override {
      return rCor->getValue();
    }
  };

  /******************************************
     2d plot of ECAL RespCorrRatios of 1 IOV
  ******************************************/
  class HcalRespCorrsPlotAll : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPlotAll() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of ECAL RespCorrRatios difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsRatioAll : public cond::payloadInspector::PlotImage<HcalRespCorrs> {

  public:
    HcalRespCorrsRatioAll() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer1->Divide(objContainer2);
        std::string ImageName(m_imageFileName);
        objContainer1->getCanvasAll()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
  /******************************************
     2d plot of ECAL RespCorrRatios of 1 IOV
  ******************************************/
  class HcalRespCorrsPlotHBHO : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPlotHBHO() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasHBHO()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of ECAL RespCorrRatios difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsRatioHBHO : public cond::payloadInspector::PlotImage<HcalRespCorrs> {

  public:
    HcalRespCorrsRatioHBHO() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer1->Divide(objContainer2);
        std::string ImageName(m_imageFileName);
        objContainer1->getCanvasHBHO()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
  /******************************************
     2d plot of ECAL RespCorrRatios of 1 IOV
  ******************************************/
  class HcalRespCorrsPlotHE : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPlotHE() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasHE()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of ECAL RespCorrRatios difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsRatioHE : public cond::payloadInspector::PlotImage<HcalRespCorrs> {

  public:
    HcalRespCorrsRatioHE() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer1->Divide(objContainer2);
        std::string ImageName(m_imageFileName);
        objContainer1->getCanvasHE()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
  /******************************************
     2d plot of ECAL RespCorrRatios of 1 IOV
  ******************************************/
  class HcalRespCorrsPlotHF : public cond::payloadInspector::PlotImage<HcalRespCorrs> {
  public:
    HcalRespCorrsPlotHF() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      auto iov = iovs.front();
      std::shared_ptr<HcalRespCorrs> payload = fetchPayload( std::get<1>(iov) );
      if(payload.get()) {
        HcalRespCorrContainer* objContainer = new HcalRespCorrContainer(payload, std::get<0>(iov));
        std::string ImageName(m_imageFileName);
        objContainer->getCanvasHF()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };

  /**********************************************************
     2d plot of ECAL RespCorrRatios difference between 2 IOVs
  **********************************************************/
  class HcalRespCorrsRatioHF : public cond::payloadInspector::PlotImage<HcalRespCorrs> {

  public:
    HcalRespCorrsRatioHF() : cond::payloadInspector::PlotImage<HcalRespCorrs>("HCAL RespCorr Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      auto iov1 = iovs.front();
      auto iov2 = iovs.back();

      std::shared_ptr<HcalRespCorrs> payload1 = fetchPayload( std::get<1>(iov1) );
      std::shared_ptr<HcalRespCorrs> payload2 = fetchPayload( std::get<1>(iov2) );

      if(payload1.get() && payload2.get()) {
        HcalRespCorrContainer* objContainer1 = new HcalRespCorrContainer(payload1, std::get<0>(iov1));
        HcalRespCorrContainer* objContainer2 = new HcalRespCorrContainer(payload2, std::get<0>(iov2));
        objContainer1->Divide(objContainer2);
        std::string ImageName(m_imageFileName);
        objContainer1->getCanvasHF()->SaveAs(ImageName.c_str());
        return true;} else return false;
    }// fill method
  };
} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalRespCorrs){
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPlotAll);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsRatioAll);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPlotHBHO);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsRatioHBHO);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPlotHE);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsRatioHE);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsPlotHF);
  PAYLOAD_INSPECTOR_CLASS(HcalRespCorrsRatioHF);
}
