#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

#include "TH2F.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLine.h"
#include "TLatex.h"

#include <memory>
#include <sstream>

namespace {

/*****************************************
 2d plot of Ecal DCS Tower Status Errors Total of 1 IOV
 ******************************************/
class EcalDCSTowerStatusSummaryPlot: public cond::payloadInspector::PlotImage<EcalDCSTowerStatus> {
public:
  EcalDCSTowerStatusSummaryPlot() :
      cond::payloadInspector::PlotImage<EcalDCSTowerStatus>("Ecal DCS Tower Status Errors Total - map ") {
    setSingleIov(true);
  }

  bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override {

    auto iov = iovs.front(); //get reference to 1st element in the vector iovs
    std::shared_ptr < EcalDCSTowerStatus > payload = fetchPayload(std::get < 1 > (iov)); //std::get<1>(iov) refers to the Hash in the tuple iov
    unsigned int run = std::get < 0 > (iov);  //referes to Time_t in iov.
    TH2F* align;  //pointer to align which is a 2D histogram

    int NbRows=2;
    int NbColumns=8;

    if (payload.get()) { //payload is an iov retrieved from payload using hash.
     

      align =new TH2F("Ecal DCS Tower Status Errors Total","EB/EE    LV    LVNOMINAL    HV    HVNOMINAL    HVEED    HVEEDNOMINAL    TotalItems",
        NbColumns, 0, NbColumns, NbRows, 0, NbRows);

      float ebVals[]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
      float eeVals[]={0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};

      long unsigned int ebTotal=(payload->barrelItems()).size();
      long unsigned int eeTotal=(payload->endcapItems()).size();

      getSummary(payload->barrelItems(),ebVals,ebTotal);
	    getSummary(payload->endcapItems(),eeVals,eeTotal);

      
      double row = NbRows - 0.5;
      
      //EB summary values
      align->Fill(0.5, row, 1);    
      
      for(int i=0;i<6;i++){
        align->Fill(1.5+i, row, ebVals[i]);
      }
      align->Fill(7.5, row, ebTotal);

     
      row--;


      align->Fill(0.5, row, 2);    
      
      for(int i=0;i<6;i++){
        align->Fill(1.5+i, row, eeVals[i]);
      }
      align->Fill(7.5, row, eeTotal);


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
    t1.DrawLatex(0.5, 0.96,Form("Ecal DCSTower Status Errors Total, IOV %i", run));


    TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
    pad->Draw();
    pad->cd();
    align->Draw("TEXT");


    drawTable(NbRows,NbColumns);

    align->GetXaxis()->SetTickLength(0.);
    align->GetXaxis()->SetLabelSize(0.);
    align->GetYaxis()->SetTickLength(0.);
    align->GetYaxis()->SetLabelSize(0.);

    std::string ImageName(m_imageFileName);
    canvas.SaveAs(ImageName.c_str());
    return true;
  }      // fill method

  void getSummary(std::vector<EcalChannelStatusCode> vItems,float vals[],long unsigned int & total){
    unsigned int shift = 0, mask = 1;
    unsigned int statusCode;

  	for(std::vector<EcalChannelStatusCode>::const_iterator iItems = vItems.begin(); iItems != vItems.end(); ++iItems){
      statusCode = iItems->getStatusCode();
      for (shift = 0; shift < 6; ++shift){
        mask = 1 << (shift);
        if (statusCode & mask){
           vals[shift] += 1;
        }
      }
    }

  }


};

} // close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE( EcalDCSTowerStatus){
  PAYLOAD_INSPECTOR_CLASS( EcalDCSTowerStatusSummaryPlot);
}
