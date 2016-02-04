#include <string>
#include <sstream>
	
#include "TFile.h"
#include "TList.h"
#include "TNtuple.h"
#include "TTree.h"

void compareGeomSerial(){
	
	
	//gStyle->SetOptStat(000000000);
	gStyle->SetOptStat("emr");
	gStyle->SetPadLeftMargin(0.15);
	gStyle->SetPadTopMargin(0.20);
	gStyle->SetTitleFontSize(0.08);
	
	

	TFile fin("<PATH>/comparisonV3_<N>.root");
	
	fin.cd();
	TTree* data = alignTree;
	
	// subdetectors ("sublevel"): PXB (1), PXF, TIB (3), TID, TOB, TEC (6)
	TCut levelCut = "((level == 1) && (sublevel >= 3))";
	
	TCanvas* c = new TCanvas("c", "c", 200, 10, 800, 800);
	c->SetFillColor(0);
	data->SetMarkerStyle(6);
	
	TH2D* hist2d = new TH2D("hist2D", "#Delta r vs. r; #Delta r (cm); r (cm)", 500, 0, 130, 1000, -1.0, 1.0);
	data->Project("hist2D", "r*dphi:r", levelCut);
	
	
	hist2d->Draw();
	TAxis *xaxis = hist2d->GetXaxis();
	TAxis *yaxis = hist2d->GetYaxis();
	xaxis->SetTitle("");
	yaxis->SetTitle("");
	xaxis->CenterTitle(true);
	yaxis->CenterTitle(true);
	xaxis->SetTitleSize(0.06);
	yaxis->SetTitleSize(0.06);
	yaxis->SetTitleOffset(-0.25);
	yaxis->SetLabelSize(.06);
	xaxis->SetLabelSize(.06);
	
	
	c->Print("<PATH>/comparisonV3_<N>.jpg");


	
}
