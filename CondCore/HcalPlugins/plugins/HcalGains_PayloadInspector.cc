#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
//#include "DataFormats/EcalDetId/interface/EBDetId.h"
//#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"


//#include "CondCore/HcalPlugins/plugins/HcalDrawUtils.h"

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


  /******************************************
     2d plot of ECAL GainRatios of 1 IOV
  ******************************************/
  class HcalGainsPlot : public cond::payloadInspector::PlotImage<HcalGains> {
  public:
    HcalGainsPlot() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios - map ") {
      setSingleIov( true );
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{

      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map",1680,1320);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Hcal Gains, IOV %i", -1));
      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };

  /**********************************************************
     2d plot of ECAL GainRatios difference between 2 IOVs
  **********************************************************/
  class HcalGainsDiff : public cond::payloadInspector::PlotImage<HcalGains> {

  public:
    HcalGainsDiff() : cond::payloadInspector::PlotImage<HcalGains>("HCAL Gain Ratios difference") {
      setSingleIov(false);
    }

    bool fill( const std::vector<std::tuple<cond::Time_t,cond::Hash> >& iovs ) override{
      gStyle->SetPalette(1);
      gStyle->SetOptStat(0);      
      TCanvas canvas("CC map","CC map",1680,1320);
      TLatex t1;
      t1.SetNDC();
      t1.SetTextAlign(26);
      t1.SetTextSize(0.05);
      t1.DrawLatex(0.5, 0.96, Form("Hcal Gain Ratios, IOV %i - %i", -1, -1));

      std::string ImageName(m_imageFileName);
      canvas.SaveAs(ImageName.c_str());
      return true;
    }// fill method
  };
} // close namespace

  // Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(HcalGains){
  PAYLOAD_INSPECTOR_CLASS(HcalGainsPlot);
  PAYLOAD_INSPECTOR_CLASS(HcalGainsDiff);
}
