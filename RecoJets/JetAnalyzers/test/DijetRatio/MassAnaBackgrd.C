# root script for making Djiet Ratio plots from 
# output of dijet ratio analysis.  Used by Manoj Jha.

#include <TH1.h>
#include <TFile.h>
#include <TBranch.h>
#include <TTree.h>
#include <TLeaf.h>

#include <iostream>
using namespace std;

void  scale(TH1* &h1) {

	        Int_t nbin = h1->GetNbinsX();
		TH1* h2 = (TH1F*) h1->Clone();
//		h1->Sumw2();
		for (Int_t i =0; i < nbin; ++i){
			float binWidth = h2->GetBinWidth(i);
			float binContent = h2->GetBinContent(i);
			float value = binContent/binWidth;
			float error = h2->GetBinError(i);
			float errVal = error/binWidth;
			h1->SetBinContent(i, value);
			h1->SetBinError(i, errVal);
		}
}

void Addition(TString hName, TH1F* &h0, TH1F* &h0Var){
	
	//Reading cross-sections
	ifstream in ("datasetBackgrd.txt");
	if (!in){
		cout << "Not able to open the file \n" << endl;
		return 1;
	}

        
	double n1;
	//X-section
	double n2;
	
	vector<double> tried ;  // Vectors for number of events
	vector<double> xSection;
	vector<double> weight;

	tried.clear();
	xSection.clear();
	weight.clear();
	
	while (in){
	in >> n1 >> n2 ;
	tried.push_back(n1);
	xSection.push_back(n2);
	weight.push_back(n2*pow(10.0,9.0)/n1);
	}

	in.close();

	
        double nEvents = 0.;
	double crossSec = 0.;

	//Number of files
	TFile* f[21];
	TString s1 = "QcdBackgrd_";
	TString s2 = ".root";

	TF1 *f1 = new TF1("f1","1.0",0,10);
	
	for (int i =0; i < 21; i++){
		int j = i+1 ;
		TString fileName = "file25/" + s1 + j + s2;
		f[i] = new TFile(fileName);
		double wt = weight[i];
		double wt2 = wt*wt;

	        if (i==0) {
		 h0 = (TH1F*)f[i]->Get(hName)->Clone(fileName + hName + "0");
		 h0Var = (TH1F*)f[i]->Get(hName)->Clone(fileName + hName + "Var");

		h0->Multiply(f1, wt);
		h0Var->Multiply(f1, wt2);
		}
		if (i >=1){
		TH1F* hA0 = (TH1F*)f[i]->Get(hName)->Clone(fileName + "A0");
	
		h0->Add(hA0,wt);
		h0Var->Add(hA0, wt2);
		
		delete hA0; 
		} // i>=1
		
	} //for (i =0 ; i <21; i++)
	
	int nBin = h0->GetNbinsX();
	for (int i = 0; i < nBin; i++){
//		double valueVar1 = h0Var->GetBinError(i);
		double valueVar1 = h0Var->GetBinContent(i);
		double error = TMath::Sqrt(valueVar1);
		h0->SetBinError(i,error);
	}

	h0 = (TH1F*)h0->Clone(hName);	

}//Addition

void f(){
	cout << "program is working fine " << endl;
}

	
TH1F* Ratio(TH1F* h1, TH1F* h2){

	TH1F* ratio = (TH1F*) h1->Clone("ratio");
	ratio->Divide(h2);
	int nbin = h1->GetNbinsX();

	for (Int_t i = 0; i < nbin; i++){
		double sigma1 = h1->GetBinError(i);
		double sigma2 = h2->GetBinError(i);

		double x1 = h1->GetBinContent(i);
		double x2 = h2->GetBinContent(i);
                 
		double r = ratio->GetBinContent(i);
		double error =0. ;
		if (x1 !=0. && x2 !=0.){
	         error = r * TMath::Sqrt(TMath::Power(sigma1/x1, 2.0) + TMath::Power(sigma2/x2, 2.0)); 
		}
		ratio->SetBinError(i, error);
	}

	return ratio;
}//Ratio

TCanvas* fitting(TCanvas* c1, TH1F* ratio, TString jetFlavor, TFile* f){

	char labelCons[150];
	char labelLine[150];
	char labelPoly[150];

	ratio->SetTitle("DiJet Ratio");
	ratio->GetXaxis()->SetTitle("DiJet Mass (in GeV)");
	ratio->GetXaxis()->SetRangeUser(330.,7600.);
        if (!strcmp(jetFlavor,"ratioGen") || !strcmp(jetFlavor,"ratioCalo") || !strcmp(jetFlavor,"ratioCor") )
		ratio->GetYaxis()->SetRangeUser(0.,1.2);
	else 
		ratio->GetYaxis()->SetRangeUser(0.6,1.6);
        if (!strcmp(jetFlavor,"ratioGen") || !strcmp(jetFlavor,"ratioCalo") || !strcmp(jetFlavor,"ratioCor") )
		ratio->GetYaxis()->SetTitle("Ratio=N(|#eta|<0.5)/N(0.5<|#eta|<1)");
	else
		ratio->GetYaxis()->SetTitle("Ratio(CorrectedJets/GenJets)");

	ratio->GetYaxis()->SetTitleOffset(1.3);
	ratio->SetStats(kFALSE);
	ratio->Draw("E1P");
	
	TF1* f1 = new TF1("f1", "pol0");
	f1->SetLineColor(1);
	ratio->Fit("f1");
	TF1* fit = ratio->GetFunction("f1");
	float constChi2 = fit->GetChisquare();
	float constP0 = fit->GetParameter(0);
	float constP0Er = fit->GetParError(0);
	int constNDF = fit->GetNDF();
	float constChi = constChi2/constNDF;
	sprintf(labelCons, "y = %5.3f #pm %5.3f, #chi^{2} = %5.3f, NDF = %d, #chi^{2}/NDF = %5.3f", constP0, constP0Er, constChi2, constNDF, constChi);
	
	
	TF1* f2 = new TF1("f2", "pol1");
	f2->SetLineColor(3);
	ratio->Fit("f2", "+");
	f2->Draw("SAME");
	TF1* fit = ratio->GetFunction("f2");
	float lineChi2 = fit->GetChisquare();
	float lineP0 = fit->GetParameter(0);
	float lineP1 = fit->GetParameter(1);
	int lineNDF = fit->GetNDF();
	float lineChi = lineChi2/lineNDF;
	sprintf(labelLine, "y = %5.3e*x + %5.3f, #chi^{2} = %5.3f, NDF = %d, #chi^{2}/NDF = %5.3f", lineP1, lineP0, lineChi2, lineNDF, lineChi);
	
	TF1* f3 = new TF1("f3", "pol2");
	f3->SetLineColor(4);
	ratio->Fit("f3", "+");
	TF1* fit = ratio->GetFunction("f3");
	float polyChi2 = fit->GetChisquare();
	float polyP0 = fit->GetParameter(0);
	float polyP1 = fit->GetParameter(1);
	float polyP2 = fit->GetParameter(2);
	int polyNDF = fit->GetNDF();
	float polyChi = polyChi2/polyNDF;
	sprintf(labelPoly, "y = %5.3e*x^{2} + %5.3e*x + %5.3f, #chi^{2} = %5.3f, NDF = %d, #chi^{2}/NDF = %5.3f",polyP2, polyP1, polyP0, polyChi2, polyNDF, polyChi);

        if (!strcmp(jetFlavor,"ratioGen"))
	TLegend *leg = new TLegend(0.25,0.7,0.9,0.9, "Generated Jets");
	else if (!strcmp(jetFlavor,"ratioCalo"))
	TLegend *leg = new TLegend(0.25,0.7,0.9,0.9, "Calorimetry Jets");
	else if (!strcmp(jetFlavor,"ratioCor"))
	TLegend *leg = new TLegend(0.25,0.7,0.9,0.9, "Corrected Jets");
	else if (!strcmp(jetFlavor,"R0_CorGen"))
	TLegend *leg = new TLegend(0.25,0.7,0.9,0.9, "|#eta| < 0.5");
	else
	TLegend *leg = new TLegend(0.25,0.7,0.9,0.9, "|#eta| < 1.0");

	leg->AddEntry(f1,labelCons,"L");
	leg->AddEntry(f2,labelLine,"L");
	leg->AddEntry(f3,labelPoly,"L");
	leg->Draw();
	//save ratio histo into the root file
        if (!strcmp(jetFlavor,"ratioGen")){
		TH1F* ratioGen = (TH1F*)ratio->Clone("ratioGen");
		f->cd();
		ratioGen->Write();
	}
        else if (!strcmp(jetFlavor,"ratioCalo")){
		TH1F* ratioCalo = (TH1F*)ratio->Clone("ratioCalo");
		f->cd();
		ratioCalo->Write();
	}
        if (!strcmp(jetFlavor,"ratioCor")){
		TH1F* ratioCor = (TH1F*)ratio->Clone("ratioCor");
		f->cd();
		ratioCor->Write();
	}
	
	c1->Print("c1.ps");

	return c1;
}// fitting

TCanvas* plot(TCanvas* c1, TString id, TFile* f){
	
	c1 = new TCanvas("c1", "c1",3,25,999,799);
	c1->SetLogy(); c1->SetGridx(); c1->SetGridy();
	
	TH1F* hGen;	TH1F* hGenVar;
	TH1F* hCalo;	TH1F* hCaloVar;
	TH1F* hCor;	TH1F* hCorVar;

	TString sGen = "hGen" + id;
	TString sCalo = "hCalo" + id;
	TString sCor = "hCor" + id;
        
	//title 
	TString title = "DiJet Mass Distribution";
	
	Addition(sGen, hGen, hGenVar);
	f->cd();
	hGen->Write();
	Addition(sCalo, hCalo, hCaloVar);
	f->cd();
	hCalo->Write();
	Addition(sCor, hCor, hCorVar);
	f->cd();
	hCor->Write();

	float GenMax = hGen->GetMaximumStored();
	float CaloMax = hCalo->GetMaximumStored();
	float CorMax = hCor->GetMaximumStored();

	float max1 = TMath::Max(GenMax, CaloMax);
	float max = TMath::Max(max1, CorMax);
	float maxY = max + 1000.;
	
	hGen->GetXaxis()->SetTitle("DiJet Mass (in GeV)");
	hGen->GetXaxis()->SetRangeUser(330.,7600.);
//	hGen->GetYaxis()->SetRangeUser(0.,maxY);
	hGen->GetYaxis()->SetTitle("d#sigma/dM_{diJet} (pb/GeV)");
	hGen->GetYaxis()->SetTitleOffset(1.3);
	hGen->SetMarkerStyle(20);
	hGen->SetMarkerColor(1);
	hGen->SetLineColor(1);
	hGen->SetStats(kFALSE);
	scale(hGen);
	hGen->Draw("EP");
	
	hCalo->SetMarkerStyle(21);
	hCalo->SetMarkerColor(2);
	hCalo->SetLineColor(2);
	hCalo->SetStats(kFALSE);
	scale(hCalo);
	hCalo->Draw("EPSAME");
	
	hCor->SetMarkerStyle(29);
	hCor->SetMarkerColor(3);
	hCor->SetLineColor(3);
	hCor->SetStats(kFALSE);
	scale(hCor);
	hCor->Draw("EPSAME");
	
	if (!strcmp(id,"0")){
	TLegend *leg = new TLegend(0.65,0.75,0.9,0.9, "-0.5<#eta_{1}<0.5,  -0.5<#eta_{2}<0.5");
	leg->AddEntry(hGen,"Generated Jets","P");
	leg->AddEntry(hCalo,"Calorimetry Jets","P");
	leg->AddEntry(hCor,"Corrected Jets","P");}

	else if (!strcmp(id,"1")){
	TLegend *leg = new TLegend(0.65,0.75,0.9,0.9, "-1.0<#eta_{1}<1.0,  -1.0<#eta_{2}<1.0");
	leg->AddEntry(hGen,"Generated Jets","P");
	leg->AddEntry(hCalo,"Calorimetry Jets","P");
	leg->AddEntry(hCor,"Corrected Jets","P");}
	
	else if (!strcmp(id,"2")){
	TLegend *leg = new TLegend(0.65,0.75,0.9,0.9, "0.5<#eta_{1}<1.0,   0.5<#eta_{2}<1.0 ");
	leg->AddEntry(hGen,"Generated Jets","P");
	leg->AddEntry(hCalo,"Calorimetry Jets","P");
	leg->AddEntry(hCor,"Corrected Jets","P");}

	else if (!strcmp(id,"3")){
	TLegend *leg = new TLegend(0.65,0.75,0.9,0.9, "-1.0<#eta_{1}<-0.5,  -1.0<#eta_{2}<-0.5");
	leg->AddEntry(hGen,"Generated Jets","P");
	leg->AddEntry(hCalo,"Calorimetry Jets","P");
	leg->AddEntry(hCor,"Corrected Jets","P");}

	else if (!strcmp(id,"4")){
	TLegend *leg = new TLegend(0.65,0.75,0.9,0.9, "0.5<#eta_{1}<1.0,  -1.0<#eta_{2}<-0.5");
	leg->AddEntry(hGen,"Generated Jets","P");
	leg->AddEntry(hCalo,"Calorimetry Jets","P");
	leg->AddEntry(hCor,"Corrected Jets","P");}

	else if (!strcmp(id,"5")){
	TLegend *leg = new TLegend(0.65,0.75,0.9,0.9, "-1.0<#eta_{1}<-0.5,  0.5<#eta_{2}<1.0");
	leg->AddEntry(hGen,"Generated Jets","P");
	leg->AddEntry(hCalo,"Calorimetry Jets","P");
	leg->AddEntry(hCor,"Corrected Jets","P");}

	else  (!strcmp(id,"6")){
	TLegend *leg = new TLegend(0.65,0.75,0.9,0.9, "0.5 < |#eta| < 1.0");
	leg->AddEntry(hGen,"Generated Jets","P");
	leg->AddEntry(hCalo,"Calorimetry Jets","P");
	leg->AddEntry(hCor,"Corrected Jets","P");}
	
	leg->Draw();
	c1->Print("c1.ps");
	
        
	if (!strcmp(id, "6")){		
	
		TH1F* h0Gen;	TH1F* h0GenVar;
		TH1F* h0Calo;	TH1F* h0CaloVar;
		TH1F* h0Cor;	TH1F* h0CorVar;
	
		TString sGen = "hGen0";
		TString sCalo = "hCalo0";
		TString sCor = "hCor0";
        
		//title 
		TString ratioTitle = "Dijet Ratio";
	
		Addition(sGen, h0Gen, h0GenVar);
		Addition(sCalo, h0Calo, h0CaloVar);
		Addition(sCor, h0Cor, h0CorVar);
		
		//scaling 
		scale(h0Gen);
		scale(h0Calo);
		scale(h0Cor);
		
		//Ratio
		TCanvas *c1 = new TCanvas("c1", "c1",3,25,999,799);
		TH1F* ratioGen = Ratio(h0Gen,hGen);
		TH1F* ratioCalo = Ratio(h0Calo,hCalo);
		TH1F* ratioCor = Ratio(h0Cor,hCor);

	

	cout << " Ratio = " << ratioCor->GetMean(2) << "+/-" << ratioCor->GetMeanError(2) << endl; 

	ratioGen->SetTitle("DiJet Ratio");
	ratioGen->GetXaxis()->SetTitle("DiJet Mass (in GeV)");
	ratioGen->GetXaxis()->SetRangeUser(330.,7600.);
	ratioGen->GetYaxis()->SetRangeUser(0.,1.2);
	ratioGen->GetYaxis()->SetTitle("Ratio=N(|#eta|<0.5)/N(0.5<|#eta|<1)");
	ratioGen->GetYaxis()->SetTitleOffset(1.3);
	ratioGen->SetMarkerStyle(20);
	ratioGen->SetMarkerColor(1);
	ratioGen->SetStats(kFALSE);
	ratioGen->Draw("E1P");


	ratioCalo->SetMarkerStyle(21);
	ratioCalo->SetMarkerColor(2);
	ratioCalo->SetStats(kFALSE);
	ratioCalo->Draw("E1PSAME");
	
	ratioCor->SetMarkerStyle(29);
	ratioCor->SetMarkerColor(3);
	ratioCor->SetStats(kFALSE);
	ratioCor->Draw("E1PSAME");
	
	TLegend *leg = new TLegend(0.65,0.75,0.9,0.9, "Ratio");
	leg->AddEntry(ratioGen,"Generated Jets","P");
	leg->AddEntry(ratioCalo,"Calorimetry Jets","P");
	leg->AddEntry(ratioCor,"Corrected Jets","P");
	leg->Draw();
	
	c1->Print("c1.ps");

	c1 = fitting(c1, ratioGen, "ratioGen", f);
	c1 = fitting(c1, ratioCalo,"ratioCalo", f);
	c1 = fitting(c1, ratioCor, "ratioCor", f);
  	
	

}// id ==6
	
	if (!strcmp(id, "0")){
		
		TH1F* h1Gen;	TH1F* h1GenVar;
		TH1F* h1Calo;	TH1F* h1CaloVar;
		TH1F* h1Cor;	TH1F* h1CorVar;
	
		TString sGen = "hGen1";
		TString sCalo = "hCalo1";
		TString sCor = "hCor1";
        
		//title 
		Addition(sGen, h1Gen, h1GenVar);
		Addition(sCalo, h1Calo, h1CaloVar);
		Addition(sCor, h1Cor, h1CorVar);
		
		//scaling 
		scale(h1Gen);
		scale(h1Calo);
		scale(h1Cor);

		// ration CorJets/GenJets
		TCanvas *c1 = new TCanvas("c1", "c1",3,25,999,799);
		TH1F* R0_CorGen = Ratio(hCor,hGen);
		TH1F* R1_CorGen = Ratio(h1Cor,h1Gen);
	
		R0_CorGen->SetTitle("DiJet Ratio");
		R0_CorGen->GetXaxis()->SetTitle("DiJet Mass (in GeV)");
		R0_CorGen->GetXaxis()->SetRangeUser(330.,7600.);
		R0_CorGen->GetYaxis()->SetRangeUser(0.6,1.6);
		R0_CorGen->GetYaxis()->SetTitle("Ratio (CorrectedJet/GenJet)");
		R0_CorGen->GetYaxis()->SetTitleOffset(1.3);
		R0_CorGen->SetMarkerStyle(20);
		R0_CorGen->SetMarkerColor(1);
		R0_CorGen->SetLineColor(1);
		R0_CorGen->SetStats(kFALSE);
		R0_CorGen->Draw("E1P");
	
		R1_CorGen->SetMarkerStyle(21);
		R1_CorGen->SetMarkerColor(2);
		R1_CorGen->SetLineColor(2);
		R1_CorGen->SetStats(kFALSE);
		R1_CorGen->Draw("E1PSAME");
	
		TLegend *leg = new TLegend(0.65,0.75,0.9,0.9, "Ratio: CorrectedJets/GenJets");
		leg->AddEntry(R0_CorGen,"|#eta| < 0.5","P");
		leg->AddEntry(R1_CorGen,"|#eta| < 1.0","P");
		leg->Draw();
		c1->Print("c1.ps");
        	c1 = fitting(c1, R0_CorGen, "R0_CorGen", f);
         	c1 = fitting(c1, R1_CorGen, "R1_CorGen", f);
	} //id == 0
	
	return c1;
	delete hGen, hCalo, hCor;
	delete hGenVar, hCaloVar, hCorVar;

}//plot

int MassAnaBackgrd(){
	
	TCanvas *c1 = new TCanvas("c1", "c1",3,25,999,799);
	c1->SetLogy(); c1->SetGridx(); c1->SetGridy();
	gSystem->Exec("rm -f backgrd.root");
	TFile* f = new TFile("backgrd.root", "new");
        
	c1->Print("c1.ps(");
	
	c1 = plot(c1, "0", f);
	c1 = plot(c1, "1", f);
	c1 = plot(c1, "2", f);
	c1 = plot(c1, "3", f);
	c1 = plot(c1, "4", f);
	c1 = plot(c1, "5", f);
	c1 = plot(c1, "6", f);
	
	c1->Print("c1.ps)");
	f->Close();

//	delete c1;
	return 0;
}

