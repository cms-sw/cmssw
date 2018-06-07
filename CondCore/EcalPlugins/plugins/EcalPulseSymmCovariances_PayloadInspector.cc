#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "CondCore/EcalPlugins/plugins/EcalDrawUtils.h"

// the data format of the condition to be inspected
#include "CondFormats/EcalObjects/interface/EcalPulseSymmCovariances.h"

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
 2d plot of Ecal Pulse Symm Covariance of 1 IOV
 *******************************************************/
class EcalPulseSymmCovariancesPlot: public cond::payloadInspector::PlotImage<EcalPulseSymmCovariances>{
	public:
		EcalPulseSymmCovariancesPlot():
			cond::payloadInspector::PlotImage<EcalPulseSymmCovariances>("Ecal Pulse Symm Covariances - map "){
				setSingleIov(true);
		}

	bool fill(const std::vector<std::tuple<cond::Time_t, cond::Hash> >& iovs)override {
		auto iov=iovs.front();
		std::shared_ptr <EcalPulseSymmCovariances> payload = fetchPayload(std::get<1> (iov));
		unsigned int run=std::get<0> (iov);
		TH2F* align;
		int NbRows;

		if(payload.get()){
			EcalPulseSymmCovariancesMap item=(*payload);
			
			NbRows=sizeof(item.covval);
			align=new TH2F("Ecal Pulse Symm Covariances","Covariance",1,0,1,NbRows,0,NbRows);
			

			double row = NbRows-0.5;

			for(int i=0;i<NbRows;i--){
				align->Fill(0.5,row,item.covval[i]);
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
		t1.DrawLatex(0.5, 0.96,Form("Ecal Pulse Symm Covariances, IOV %i", run));


		TPad* pad = new TPad("pad", "pad", 0.0, 0.0, 1.0, 0.94);
		pad->Draw();
		pad->cd();
		align->Draw("TEXT");
		TLine* l = new TLine;
		l->SetLineWidth(1);

		for (int i = 1; i < NbRows; i++) {
		  double y = (double) i;
		  l = new TLine(0., y, 2., y);
		  l->Draw();
		}

		for (int i = 1; i < 1; i++) {
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
PAYLOAD_INSPECTOR_MODULE(EcalPulseSymmCovariances) {
  PAYLOAD_INSPECTOR_CLASS(EcalPulseSymmCovariancesPlot);
}