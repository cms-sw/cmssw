#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"
#include "CondCore/EcalPlugins/plugins/EcalFunctionParametersUtils.h"
// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionParameters.h"

#include "TH2F.h" // a 2-D histogram with four bytes per cell (float)
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"//write mathematical equations.
#include "TPave.h"
#include "TPaveStats.h"
#include <string>
#include <fstream>
#include <cmath>

namespace {
/*****************************************
 2d plot of Ecal Cluster Energy Correction Parameters of 1 IOV
 ******************************************/
class EcalClusterEnergyCorrectionParametersPlot: public cond::payloadInspector::PlotImage<EcalClusterEnergyCorrectionParameters> {
public:
  EcalClusterEnergyCorrectionParametersPlot() :
      cond::payloadInspector::PlotImage<EcalClusterEnergyCorrectionParameters>("Ecal Cluster Energy Correction Parameters - map ") {
    setSingleIov(true);
  }

  bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override {

    auto iov = iovs.front(); //get reference to 1st element in the vector iovs
    std::shared_ptr < EcalClusterEnergyCorrectionParameters > payload = fetchPayload(std::get < 1 > (iov)); //std::get<1>(iov) refers to the Hash in the tuple iov
    unsigned int run = std::get < 0 > (iov);  //referes to Time_t in iov.
    TH2F* align;  //pointer to align which is a 2D histogram

    int gridRows;
    int NbColumns;

    if (payload.get()) { //payload is an iov retrieved from payload using hash.
      align =new TH2F("","", 0, 0, 0, 0, 0, 0);
      fillFunctionParamsValues(align, (*payload).params(), "Ecal Cluster Energy Correction Parameters",
        gridRows, NbColumns);
 
    }   // if payload.get()
    else
      return false;


    gStyle->SetPalette(1);
    gStyle->SetOptStat(0);
    TCanvas canvas("CC map", "CC map", 1000, 1000);
    TLatex t1;
    t1.SetNDC();
    t1.SetTextAlign(26);
    t1.SetTextSize(0.04);
    t1.SetTextColor(2);
    t1.DrawLatex(0.5, 0.96,Form("Ecal Cluster Energy Correction Parameters, IOV %i", run));


    TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
    pad->Draw();
    pad->cd();
    align->Draw("TEXT");

    drawTable(gridRows,NbColumns);

    align->GetXaxis()->SetTickLength(0.);
    align->GetXaxis()->SetLabelSize(0.);
    align->GetYaxis()->SetTickLength(0.);
    align->GetYaxis()->SetLabelSize(0.);

    std::string ImageName(m_imageFileName);
    canvas.SaveAs(ImageName.c_str());
    return true;
  }      // fill method


};

}

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalClusterEnergyCorrectionParameters) {
  PAYLOAD_INSPECTOR_CLASS(EcalClusterEnergyCorrectionParametersPlot);
}
