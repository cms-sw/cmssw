// Producing suite of plots for diagnostic purposes in CombinedLimit
// Designed to work with mlfit.root file produced with MaxLikelihoodFit

// ROOT includes

#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TH1F.h"
#include "TAxis.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TObjArray.h"
#include "TBranch.h"
#include "TGraph.h"
#include "TLatex.h"
#include "TF1.h"
#include "TH2D.h"
#include "TLegend.h"

// RooFit includes
#include "RooRealVar.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooWorkspace.h"
#include "RooAbsReal.h"
#include "RooFitResult.h"
#include "RooDataSet.h"
#include "RooStats/ModelConfig.h"


// standard includes
#include <iostream>

std::map<std::string, std::pair<double,double> > prevals_;
std::map<std::string, std::pair<double,double> > bfvals_;
std::map<std::string, std::pair<double,double> > bfvals_sb_;

bool doPull(false);
bool doLH(false);

RooAbsReal *nll;
RooWorkspace *w;
RooStats::ModelConfig *mc_s;

// For LH Plots
int npoints = 15;
int nsigma  = 3;

TGraph *graphLH(std::string nuisname, double err ,std::string whichfit){

	w->loadSnapshot(whichfit.c_str()); // SetTo BestFit values as start

	// Get The parameter we want 
	RooRealVar *nuis =(RooRealVar*) w->var(nuisname.c_str());
	double bf = nuis->getVal();
	double nll_0=nll->getVal();


	TGraph *gr = new TGraph(2*npoints+1);
	for (int i=-1*npoints;i<=npoints;i++){
		nuis->setVal(bf+err*( ((float)i)*nsigma/npoints));
		double nll_v = nll->getVal();
		gr->SetPoint(i+npoints,nuis->getVal(),nll_v-nll_0);
	}

	gr->SetTitle("");
	gr->GetYaxis()->SetTitle("NLL - obs data");
	gr->GetYaxis()->SetTitleOffset(1.2);
	gr->GetXaxis()->SetTitle(nuisname.c_str());
	gr->SetLineColor(4);
	gr->SetLineWidth(2);
	gr->SetMarkerStyle(21);
	gr->SetMarkerSize(0.6);
	
	return gr;
	

}

// grab the initial parameters and errors for making pull distributions:
// Take these from a fit file to the data themselves 
void fillInitialParams(RooArgSet *args, std::map<std::string, std::pair<double,double> > &vals){
	
	 TIterator* iter(args->createIterator());
         for (TObject *a = iter->Next(); a != 0; a = iter->Next()) {
                 RooRealVar *rrv = dynamic_cast<RooRealVar *>(a);      
                 std::string name = rrv->GetName();
		 std::pair<double,double> valE(rrv->getVal(),rrv->getError());
		 vals.insert( std::pair<std::string,std::pair<double ,double> > (name,valE)) ;
	 }
	
}

bool findNuisancePre(std::string name){

	std::map<std::string, std::pair<double, double> >::iterator it=prevals_.find(name);
	if (it!=prevals_.end()) return true;
	else return false;
}


void plotTree(TTree *tree_, std::string whichfit, std::string selectString){

	// Create a map for plotting the pullsummaries:
	std::map < const char*, std::pair <double,double> > pullSummaryMap;
	int nPulls=0;

	TObjArray *l_branches = tree_->GetListOfBranches();
	int nBranches = l_branches->GetEntries();

	TCanvas *c = new TCanvas("c","",960,800);

	std::string treename = tree_->GetName();
	c->SaveAs(Form("%s.pdf[",treename.c_str()));

	for (int iobj=0;iobj<nBranches;iobj++){

		TBranch *br =(TBranch*) l_branches->At(iobj);

		// Draw the normal histogram
		const char* name = br->GetName();
		bool fitPull=false;
		bool plotLH=false;

		TGraph *gr=0;
		double p_mean =0;
		double p_err  =0;

		int nToysInTree = tree_->GetEntries();
		if (doPull && findNuisancePre(name)){
			
			p_mean = bfvals_[name].first;	// toy constrainits thrown about best fit to data
			p_err  = prevals_[name].second; // uncertainties taken from card

			const char* drawInput = Form("(%s-%f)/%f",name,p_mean,p_err);
			tree_->Draw(Form("%s>>%s",drawInput,name),"");
			tree_->Draw(Form("%s>>%s_fail",drawInput,name),selectString.c_str(),"same");
			fitPull = true;
			if (doLH) {
			  gr = graphLH(name,p_err,whichfit);
			  plotLH=true;
			}
			
		}

		else{
			tree_->Draw(Form("%s>>%s",name,name),"");
			tree_->Draw(Form("%s>>%s_fail",name,name),"mu<0","same");
		}
		
		TH1F* bH  = (TH1F*) gROOT->FindObject(Form("%s",name))->Clone();
		TH1F* bHf = (TH1F*) gROOT->FindObject(Form("%s_fail",name))->Clone();
		bHf->SetLineColor(2);
		bH->GetXaxis()->SetTitle(bH->GetTitle());
		bH->GetYaxis()->SetTitle(Form("no toys (%d total)",nToysInTree));
		bH->GetYaxis()->SetTitleOffset(1.32);
		
		bH->SetTitle("");	

		if (fitPull) bH->Fit("gaus");
	
		c->Clear();
		TPad pad1("t1","",0.01,0.02,0.59,0.98);
		TPad pad2("t2","",0.59,0.04,0.98,0.62);
		TPad pad3("t3","",0.59,0.64,0.98,0.90);

		pad1.SetNumber(1); pad2.SetNumber(2); pad3.SetNumber(3);
		pad1.Draw(); pad2.Draw();pad3.Draw();
		pad2.SetGrid(true);

		c->cd(1); bH->Draw(); bHf->Draw("same");
		TLatex *titletext = new TLatex();titletext->SetNDC();titletext->SetTextSize(0.04); titletext->DrawLatex(0.1,0.95,name);
		TLegend *legend = new TLegend(0.6,0.8,0.9,0.89);
		legend->SetFillColor(0);
		legend->AddEntry(bH,"All Toys","L");
		legend->AddEntry(bHf,selectString.c_str(),"L");
		legend->Draw();

		if (doPull && plotLH) {
			c->cd(2); gr->Draw("ALP");
		}
		if (fitPull){
			c->cd(3);
			TLatex *tlatex = new TLatex(); tlatex->SetNDC(); tlatex->SetTextSize(0.12);
			tlatex->DrawLatex(0.15,0.75,Form("Mean    : %.3f #pm %.3f",bH->GetFunction("gaus")->GetParameter(1),bH->GetFunction("gaus")->GetParError(1)));
			tlatex->DrawLatex(0.15,0.60,Form("Sigma   : %.3f #pm %.3f",bH->GetFunction("gaus")->GetParameter(2),bH->GetFunction("gaus")->GetParError(2)));
			tlatex->DrawLatex(0.15,0.35,Form("Pre-fit : %.3f ",prevals_[name].first));
			tlatex->DrawLatex(0.15,0.2,Form("Best-fit (B)  : %.3f ",p_mean));
			tlatex->DrawLatex(0.15,0.05,Form("Best-fit (S+B): %.3f ",bfvals_sb_[name].first));
			
			pullSummaryMap[name]=std::make_pair<double,double>(bH->GetFunction("gaus")->GetParameter(1),bH->GetFunction("gaus")->GetParameter(2));
			nPulls++;

		}

		c->SaveAs(Form("%s.pdf",treename.c_str()));
	}
	
	if (doPull && nPulls>0){
	   
	    int nRemainingPulls = nPulls;
	    TCanvas *hc = new TCanvas("hc","",3000,2000); hc->SetGrid(0);
	    std::map < const char*, std::pair <double,double> >::iterator pull_it = pullSummaryMap.begin();
	    std::map < const char*, std::pair <double,double> >::iterator pull_end = pullSummaryMap.end();

	    while (nRemainingPulls > 0){

		int nThisPulls = min(15,nRemainingPulls);

		TH1F pullSummaryHist("pullSummary","",nThisPulls,0,nThisPulls);
		for (int pi=1;pull_it!=pull_end && pi<=nThisPulls ;pull_it++,pi++){
			pullSummaryHist.GetXaxis()->SetBinLabel(pi,(*pull_it).first);
			pullSummaryHist.SetBinContent(pi,((*pull_it).second).first);
			pullSummaryHist.SetBinError(pi,((*pull_it).second).second);
			nRemainingPulls--;
		}		

		pullSummaryHist.SetMarkerStyle(21);pullSummaryHist.SetMarkerSize(1.5);pullSummaryHist.SetMarkerColor(2);pullSummaryHist.SetLabelSize(0.018);
		pullSummaryHist.GetYaxis()->SetRangeUser(-3,3);pullSummaryHist.GetYaxis()->SetTitle("pull summary");pullSummaryHist.Draw("E1");
		hc->SaveAs(Form("%s.pdf",treename.c_str()));
	   }

	    delete hc;
	}

	c->SaveAs(Form("%s.pdf]",treename.c_str()));

	delete c;
	return;


}

void plotParamtersFromToys(std::string inputFile, std::string dataFits="", std::string workspace="", std::string selectString="mu<0"){

	// Some Global preferences
	gSystem->Load("$CMSSW_BASE/lib/$SCRAM_ARCH/libHiggsAnalysisCombinedLimit.so");
	gROOT->SetBatch(true);
	gStyle->SetOptFit(0);
	gStyle->SetOptStat(0);
	gStyle->SetPalette(1,0);

	TFile *fi_ = TFile::Open(inputFile.c_str());
	TFile *fd_=0;
	TFile *fw_=0;

	if (dataFits!=""){
		std::cout << "Getting fit to data from "<< dataFits <<std::endl;
		doPull = true;
		fd_ = TFile::Open(dataFits.c_str());

		// Toys are thrown from best fit to data (background only/mu=0) 
		RooFitResult *bestfit=(RooFitResult*)fd_->Get("fit_b");
		RooArgSet fitargs = bestfit->floatParsFinal();

		RooFitResult *bestfit_s=(RooFitResult*)fd_->Get("fit_s");
		RooArgSet fitargs_s = bestfit_s->floatParsFinal();
		// These are essentially the nuisances in the card (note, from toys file, they will be randomized)
		// so need to use the data fit.
		RooArgSet *prefitargs = (RooArgSet*)fd_->Get("nuisances_prefit");

		fillInitialParams(prefitargs,prevals_);
		fillInitialParams(&fitargs,bfvals_);
		fillInitialParams(&fitargs_s,bfvals_sb_);

	   	if (workspace != ""){
			std::cout << "Getting the workspace from "<< workspace << std::endl;
			fw_ =  TFile::Open(workspace.c_str());
			w   = (RooWorkspace*) fw_->Get("w");
			RooDataSet *data = (RooDataSet*) w->data("data_obs");
			mc_s = (RooStats::ModelConfig*)w->genobj("ModelConfig");
			std::cout << "make nll"<<std::endl;
			nll = mc_s->GetPdf()->createNLL(
				*data,RooFit::Constrain(*mc_s->GetNuisanceParameters())
				,RooFit::Extended(mc_s->GetPdf()->canBeExtended()));

			
			// grab r (mu) from workspace to set to 0 for bonly fit since it wasnt floating 
			RooRealVar *r = w->var("r"); r->setVal(0);fitargs.add(*r);
			
			w->saveSnapshot("bestfitparams",fitargs,true);	
			w->saveSnapshot("bestfitparams_sb",fitargs_s,true);	
			doLH=true;
			std::cout << "Workspace OK!"<<std::endl;
			
	        }
	}

	
	// b and s+b trees
	TTree *tree_b  = (TTree*) fi_->Get("tree_fit_b");
	TTree *tree_sb = (TTree*) fi_->Get("tree_fit_sb");

	// create a plot for each branch (one per nuisance/global obs param)
	plotTree(tree_b,"bestfitparams",selectString);		// LH plot will be centered around B-only fit to data
	plotTree(tree_sb,"bestfitparams_sb",selectString);	// LH plot will be centered around S+B fit to data
	
	fi_->Close();
	if (doPull) fd_->Close();
	if (doLH) fw_->Close();
	
}

