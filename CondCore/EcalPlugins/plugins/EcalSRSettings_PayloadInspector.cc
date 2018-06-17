#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalSRSettings.h"

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
 2d plot of Ecal SR Settings Summary of 1 IOV
 *******************************************************/
class EcalSRSettingsSummaryPlot: public cond::payloadInspector::PlotImage<EcalSRSettings>{
	public:
		EcalSRSettingsSummaryPlot():
			cond::payloadInspector::PlotImage<EcalSRSettings>("Ecal SR Settings Summary - map "){
				setSingleIov(true);
		}

	bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override {
		const int maxInCol=27;

		auto iov=iovs.front();
		std::shared_ptr <EcalSRSettings> payload = fetchPayload(std::get<1> (iov));
		unsigned int run=std::get<0> (iov);

		TH2F* align;

	    int NbRows,gridRows;
	    int NbColumns,offset;

		if(payload.get()){
		  EcalSRSettings ecalSR=(*payload);

		  NbRows = ecalSR.srpLowInterestChannelZS_.size();
	            
	      gridRows=(NbRows<=maxInCol)?NbRows:maxInCol;
	      offset=ceil(1.0*NbRows/maxInCol);
	      NbColumns=offset*2+3;

	      align =new TH2F("Ecal SR Settings Summary","ebDccAdcToGeV    eeDccAdcToGeV    Rows#    srpLowInterestChannelZS    srpHighInterestChannelZS",
	        NbColumns, 0, NbColumns, gridRows, 0, gridRows);

	      double row = gridRows - 0.5;
	      double column=3.5;
	      int cnt=0;
	      
	      align->Fill(0.5,gridRows-0.5,ecalSR.ebDccAdcToGeV_);
	      align->Fill(1.5,gridRows-0.5,ecalSR.eeDccAdcToGeV_);

	      for(int i=0;i<gridRows;i++){
	        align->Fill(2.5, gridRows-i-0.5, i+1);        
	      }


 		  for (std::vector<float>::const_iterator it =ecalSR.srpLowInterestChannelZS_.begin();
	       it != ecalSR.srpLowInterestChannelZS_.end();it++) {

	        align->Fill(column, row, *it);

	        cnt++;
	        column=floor(1.0*cnt/maxInCol)+3.5;
	        row=(row==0.5?(gridRows-0.5):row-1);
	      }

	      row = gridRows - 0.5;
	      column=3.5;
	      cnt=0;

		  for (std::vector<float>::const_iterator it =ecalSR.srpHighInterestChannelZS_.begin();
	       it != ecalSR.srpHighInterestChannelZS_.end();it++) {

	        align->Fill(column+offset, row, *it);

	        cnt++;
	        column=floor(1.0*cnt/maxInCol)+3.5;
	        row=(row==0.5?(gridRows-0.5):row-1);
	      }

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
		t1.DrawLatex(0.5, 0.96,Form("Ecal SRSettings Summary, IOV %i", run));


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
	}
};

}

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalSRSettings) {
  PAYLOAD_INSPECTOR_CLASS(EcalSRSettingsSummaryPlot);
}