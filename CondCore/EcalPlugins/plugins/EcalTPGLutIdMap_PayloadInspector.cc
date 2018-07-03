#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"

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
 2d plot of Ecal TPG LutId Map of 1 IOV
 ******************************************/
class EcalTPGLutIdMapPlot: public cond::payloadInspector::PlotImage<EcalTPGLutIdMap>{
	public:
		EcalTPGLutIdMapPlot():
			cond::payloadInspector::PlotImage<EcalTPGLutIdMap>("Ecal TPG LutId Map - map "){
				setSingleIov(true);
		}

		bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs) override {
			auto iov=iovs.front();
			std::shared_ptr <EcalTPGLutIdMap> payload = fetchPayload(std::get<1> (iov));
			unsigned int run=std::get<0> (iov);
			TH2F* align;
			int NbRows;

			if(payload.get()){
				EcalTPGLutIdMap::EcalTPGLutMap map=(*payload).getMap();
				NbRows=30;


			  align =new TH2F("Ecal TPG LutId Map","EB                                 LutEcalTPGLut                 EE                                 LutEcalTPGLut",
			  	4, 0, 4, NbRows, 0, NbRows);

			  double row = NbRows - 0.5;
			  int columnBase=0;
			  for (EcalTPGLutIdMap::EcalTPGLutMapItr it = map.begin();it != map.end();it++) {

			      EcalTPGLut ecaltpgLut=it->second;
			      uint32_t mapKey=it->first;

			      const unsigned int* lut=(ecaltpgLut.getLut());

			      for(int i=0;i<30;i++){
				      if(i==(NbRows/2-1))
				      	align->Fill(0.5+columnBase, row, mapKey+1);

					  align->Fill(1.5+columnBase, row, *(lut+i));	
			      	  row = row - 1.;
			      }

			      columnBase +=2;
			      row = NbRows - 0.5;
			  }//loop over TPGLutIdMap rows


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
		    t1.DrawLatex(0.5, 0.96,Form("ECAL TPG LutId Map, IOV %i", run));

		    TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
		    pad->Draw();
		    pad->cd();
		    align->Draw("TEXT");
		    TLine* l = new TLine;
		    l->SetLineWidth(1);
		    for (int i = 1; i < NbRows; i++) {
		      double y = (double) i;
		      l = new TLine(1., y, 2., y);
		      l->Draw();
		    }
		    
		    for (int i = 1; i < NbRows; i++) {
		      double y = (double) i;
		      l = new TLine(3., y, 4., y);
		      l->Draw();
		    }

		    for (int i = 1; i < 4; i++) {
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

}
// Register the classes as boost python plugin
PAYLOAD_INSPECTOR_MODULE(EcalTPGLutIdMap) {
  PAYLOAD_INSPECTOR_CLASS(EcalTPGLutIdMapPlot);
}
