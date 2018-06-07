#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalClusterEnergyCorrectionObjectSpecificParameters.h"

#include "TH2F.h" // a 2-D histogram with four bytes per cell (float)
#include "TCanvas.h"
#include "TLine.h"
#include "TStyle.h"
#include "TLatex.h"//write mathematical equations.
#include "TPave.h"
#include "TPaveStats.h"
#include <string>
#include <fstream>

namespace {
/*****************************************
 2d plot of Ecal Cluster Energy Correction Object Specific Parameters of 1 IOV
 ******************************************/
class EcalClusterEnergyCorrectionObjectSpecificParametersPlot: public cond::payloadInspector::PlotImage<EcalClusterEnergyCorrectionObjectSpecificParameters> {
public:
  EcalClusterEnergyCorrectionObjectSpecificParametersPlot() :
      cond::payloadInspector::PlotImage<EcalClusterEnergyCorrectionObjectSpecificParameters>("Ecal Cluster Energy Correction Object Specific Parameters - map ") {
    setSingleIov(true);
  }

  bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override {
    const int maxInCol=25;

    auto iov = iovs.front(); //get reference to 1st element in the vector iovs
    std::shared_ptr < EcalClusterEnergyCorrectionObjectSpecificParameters > payload = fetchPayload(std::get < 1 > (iov)); //std::get<1>(iov) refers to the Hash in the tuple iov
    unsigned int run = std::get < 0 > (iov);  //referes to Time_t in iov.
    TH2F* align;  //pointer to align which is a 2D histogram

    int NbRows,gridRows;
    int NbColumns;

    if (payload.get()) { //payload is an iov retrieved from payload using hash.
	    EcalFunctionParameters m_params=(*payload).params();
      NbRows = m_params.size()-countEmptyRows(m_params);
            
      gridRows=(NbRows<=maxInCol)?NbRows:maxInCol;
      NbColumns=ceil(1.0*NbRows/maxInCol)+1;

      align =new TH2F("Ecal Cluster Energy Correction Object Specific Parameters","Ecal Function Parameters",
        NbColumns, 0, NbColumns, gridRows, 0, gridRows);

      double row = gridRows - 0.5;
      double column=1.5;
      int cnt=0;
      
      for(int i=0;i<gridRows;i++){
        align->Fill(0.5, gridRows-i-0.5, i+1);        
      }


     for (std::vector<float>::const_iterator it =m_params.begin(); it != m_params.end();it++) {
        if((*it)==0.0f) continue;
        align->Fill(column, row, *it);

        cnt++;
        column=floor(1.0*cnt/maxInCol)+1.5;
        row=(row==0.5?(gridRows-0.5):row-1);
      }
    }   // if payload.get()
    else
      return false;

    gStyle->SetPalette(1);
    gStyle->SetOptStat(0);
    TCanvas canvas("CC map", "CC map", 1000, 1000);
    TLatex t1;
    t1.SetNDC();
    t1.SetTextAlign(26);
    t1.SetTextSize(0.03);
    t1.SetTextColor(2);
    t1.DrawLatex(0.5, 0.96,Form("Ecal Cluster Energy Correction Object Specific Parameters, IOV %i", run));


    TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
    pad->Draw();
    pad->cd();
    align->Draw("TEXT");
    TLine* l = new TLine;
    l->SetLineWidth(1);

    for (int i = 1; i < gridRows; i++) {
      double y = (double) i;
      l = new TLine(0., y, NbColumns, y);
      l->Draw();
    }

    for (int i = 1; i < NbColumns; i++) {
      double x = (double) i;
      double y = (double) gridRows;
      l = new TLine(x, 0., x, y);
      l->Draw();
    }

    align->GetXaxis()->SetTickLength(0.);
    align->GetXaxis()->SetLabelSize(0.);
    align->GetYaxis()->SetTickLength(0.);
    align->GetYaxis()->SetLabelSize(0.);

    std::string ImageName(m_imageFileName);
    canvas.SaveAs(ImageName.c_str());
    return true;
  }      // fill method

  int countEmptyRows(std::vector<float> & vec){

    int cnt=0;
    for(std::vector<float>::const_iterator it=vec.begin();it!=vec.end();it++)
      if((*it)==0.0f)
       cnt++;
      
    return cnt;

  }

};

}

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalClusterEnergyCorrectionObjectSpecificParameters) {
  PAYLOAD_INSPECTOR_CLASS(EcalClusterEnergyCorrectionObjectSpecificParametersPlot);
}
