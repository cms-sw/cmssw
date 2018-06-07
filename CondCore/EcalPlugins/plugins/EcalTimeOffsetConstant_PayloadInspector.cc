#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"

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

/*******************************************************
 2d plot of Ecal Time Offset Constant of 1 IOV
 *******************************************************/
class EcalTimeOffsetConstantPlot: public cond::payloadInspector::PlotImage<EcalTimeOffsetConstant>{
	public:
		EcalTimeOffsetConstantPlot():
			cond::payloadInspector::PlotImage<EcalTimeOffsetConstant>("Ecal Time Offset Constant - map "){
				setSingleIov(true);
		}

	bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override {
		auto iov=iovs.front();
		std::shared_ptr <EcalTimeOffsetConstant> payload = fetchPayload(std::get<1> (iov));
		unsigned int run=std::get<0> (iov);
		TH2F* align;
		int NbRows;

		if(payload.get()){
			NbRows=1;
			align=new TH2F("Time Offset Constant [ns]","EB          EE",2,0,2,NbRows,0,NbRows);
			EcalTimeOffsetConstant it=(*payload);

			double row = NbRows-0.5;

			align->Fill(0.5,row,it.getEBValue());
			align->Fill(1.5,row,it.getEEValue());
		}else
			return false;

			  gStyle->SetPalette(1);
		    gStyle->SetOptStat(0);
		    TCanvas canvas("CC map", "CC map", 1000, 1000);
		    TLatex t1;
		    t1.SetNDC();
		    t1.SetTextAlign(26);
		    t1.SetTextSize(0.05);
		    t1.SetTextColor(2);
		    t1.DrawLatex(0.5, 0.96,Form("Ecal Time Offset Constant, IOV %i", run));


		    TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
		    pad->Draw();
		    pad->cd();
		    align->Draw("TEXT");

        drawTable(NbRows,2);

		    align->GetXaxis()->SetTickLength(0.);
		    align->GetXaxis()->SetLabelSize(0.);
		    align->GetYaxis()->SetTickLength(0.);
		    align->GetYaxis()->SetLabelSize(0.);

		    std::string ImageName(m_imageFileName);
		    canvas.SaveAs(ImageName.c_str());

			return true;
	}
};


/*******************************************************
 2d plot of Ecal Time Offset Constant difference between 2 IOVs
*******************************************************/

class EcalTimeOffsetConstantDiff: public cond::payloadInspector::PlotImage<EcalTimeOffsetConstant> {

public:
  EcalTimeOffsetConstantDiff() :
      cond::payloadInspector::PlotImage<EcalTimeOffsetConstant>("Ecal Time Offset Constant difference") {
    setSingleIov(false);
  }

  bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override {
  
    unsigned int run[2], irun = 0;
    float val[2]={};
    TH2F* align = new TH2F("", "", 1, 0., 1., 1, 0., 1.); // pseudo creation
    int NbRows = 0;

    for (auto const & iov : iovs) {
      std::shared_ptr < EcalTimeOffsetConstant > payload = fetchPayload(std::get < 1 > (iov));
      run[irun] = std::get < 0 > (iov);

      if (payload.get()) {
        NbRows = 1;

        if (irun == 1)
          align=new TH2F("Ecal Time Offset Constant [ns]","EB          EE",2,0,2,NbRows,0,NbRows);
        

		    EcalTimeOffsetConstant it=(*payload);
        double row = NbRows - 0.5;

        if (irun == 0) {
        	val[0] = it.getEBValue();
          val[1] = it.getEEValue();

        } else {
          align->Fill(0.5,row,it.getEBValue()-val[0]);
			    align->Fill(1.5,row,it.getEEValue()-val[1]);

          row = row - 1.;
        }

      }  //  if payload.get()
      else
        return false;

      irun++;
    }      // loop over IOVs

    gStyle->SetPalette(1);
    gStyle->SetOptStat(0);
    TCanvas canvas("CC map", "CC map", 1000, 1000);
    TLatex t1;
    t1.SetNDC();
    t1.SetTextAlign(26);
    t1.SetTextSize(0.05);
    t1.SetTextColor(2);
    t1.DrawLatex(0.5, 0.96,Form("Ecal Time Offset Constant, IOV %i - %i", run[1],run[0]));

    TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
    pad->Draw();
    pad->cd();
    align->Draw("TEXT");
    
    drawTable(NbRows,2);
    
    align->GetXaxis()->SetTickLength(0.);
    align->GetXaxis()->SetLabelSize(0.);
    align->GetYaxis()->SetTickLength(0.);
    align->GetYaxis()->SetLabelSize(0.);

    std::string ImageName(m_imageFileName);
    canvas.SaveAs(ImageName.c_str());

  	return true;
  }

};

} //close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTimeOffsetConstant) {
  PAYLOAD_INSPECTOR_CLASS(EcalTimeOffsetConstantPlot);
  PAYLOAD_INSPECTOR_CLASS(EcalTimeOffsetConstantDiff);
}