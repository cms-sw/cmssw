#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"

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
 2d plot of Ecal TBWeights of 1 IOV
 *******************************************************/
class EcalTBWeightsPlot: public cond::payloadInspector::PlotImage<EcalTBWeights>{
	public:
		EcalTBWeightsPlot():
			cond::payloadInspector::PlotImage<EcalTBWeights>("Ecal TBWeights - map "){
				setSingleIov(true);
		}

	bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override {
		auto iov=iovs.front();
		std::shared_ptr <EcalTBWeights> payload = fetchPayload(std::get<1> (iov));
		unsigned int run=std::get<0> (iov);
		TH2F* align;
		int NbRows;

		if(payload.get()){
			EcalTBWeights::EcalTBWeightMap map=(*payload).getMap();
			NbRows=map.size();
			align=new TH2F("Ecal PTM Temperatures","EcalXtalGroupId          EcalTDCId          EcalWeightSet",
				3,0,3,NbRows,0,NbRows);
		
			double row = NbRows-0.5;

			for (EcalTBWeights::EcalPTMTemperatureMap::const_iterator it =map.begin(); 
				it != map.end();it++) {
				
				//uint32_t mapKey=it->first;
				//float val=it->second;

				//align->Fill(0.5,row,mapKey);
				//align->Fill(1.5,row,val);

				row--;
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
		t1.DrawLatex(0.5, 0.96,Form("Ecal TBWeights, IOV %i", run));


		TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
		pad->Draw();
		pad->cd();
		align->Draw("TEXT");
		TLine* l = new TLine;
		l->SetLineWidth(1);

		for (int i = 1; i < NbRows; i++) {
		  double y = (double) i;
		  l = new TLine(0., y, 3., y);
		  l->Draw();
		}

		for (int i = 1; i < 3; i++) {
		  double x = (double) i;
		  double y = (double) NbRows;
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
	}
};

} //close namespace

// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTBWeights) {
  PAYLOAD_INSPECTOR_CLASS(EcalTBWeightsPlot);
}