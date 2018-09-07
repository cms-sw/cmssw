#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"

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
 2d plot of Ecal TPG Weight Id Map of 1 IOV
 ******************************************/
class EcalTPGWeightIdMapPlot: public cond::payloadInspector::PlotImage<EcalTPGWeightIdMap>{
	public:
		EcalTPGWeightIdMapPlot():
			cond::payloadInspector::PlotImage<EcalTPGWeightIdMap>("Ecal TPG Weight Id Map - map "){
				setSingleIov(true);
		}

		bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
			auto iov=iovs.front();
			std::shared_ptr <EcalTPGWeightIdMap> payload = fetchPayload(std::get<1> (iov));
			unsigned int run=std::get<0> (iov);
			TH2F* align;
			int NbRows;

			if(payload.get()){
				EcalTPGWeightIdMap::EcalTPGWeightMap map=(*payload).getMap();
				NbRows=map.size();


			  align =new TH2F("Ecal TPG Weight Id Map","MapKey          w0                     w1                     w2                       w3                       w4",
			  	6, 0, 6, NbRows, 0, NbRows);

			  double row = NbRows - 0.5;
			  for (EcalTPGWeightIdMap::EcalTPGWeightMapItr it = map.begin();it != map.end();it++) {
				
				uint32_t mapKey=it->first;
         		EcalTPGWeights item=it->second;
         		uint32_t w0,w1,w2,w3,w4;
         		item.getValues(w0,w1,w2,w3,w4);

         		align->Fill(0.5, row, mapKey+1);
				align->Fill(1.5, row, w0);
          		align->Fill(2.5, row, w1);
          		align->Fill(3.5, row, w2);
          		align->Fill(4.5, row, w3);
          		align->Fill(5.5, row, w4);

			    row = row - 1.;
			  }//loop over EcalTPGWeightIdMap rows


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
		    t1.DrawLatex(0.5, 0.96,Form("Ecal TPG Weight Id Map, IOV %i", run));

		    TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
		    pad->Draw();
		    pad->cd();
		    align->Draw("TEXT");
		    
		    drawTable(NbRows, 6);

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
PAYLOAD_INSPECTOR_MODULE(EcalTPGWeightIdMap) {
  PAYLOAD_INSPECTOR_CLASS(EcalTPGWeightIdMapPlot);
}
