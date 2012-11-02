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
#include "TLine.h"
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

// For LH Plots, n-sigma along x axis
int npoints = 15;
int nsigma  = 3;
// Label size for Pull Summary
double pullLabelSize = 0.028;
int maxPullsPerPlot = 10;

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
	gr->GetYaxis()->SetTitleOffset(1.1);
	gr->GetXaxis()->SetTitleSize(0.05);
	gr->GetYaxis()->SetTitleSize(0.05);
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

	gStyle->SetPadTopMargin(0.01);

	TCanvas *c = new TCanvas("c","",960,800);

	std::string treename = tree_->GetName();
	c->SaveAs(Form("%s.pdf[",treename.c_str()));
	// File to store plots in 
	TFile *fOut = new TFile(Form("%s.root",treename.c_str()),"RECREATE");

	for (int iobj=0;iobj<nBranches;iobj++){

		TBranch *br =(TBranch*) l_branches->At(iobj);

		// Draw the normal histogram
		const char* name = br->GetName();
		bool fitPull=false;
		bool fitPullf=false;

		bool plotLH=false;

		TGraph *gr=NULL;
		double p_mean =0;
		double p_err  =0;

		int nToysInTree = tree_->GetEntries();
		// Find out if paramter is fitted value or constraint term
		bool isFitted = findNuisancePre(name);
		if (doPull && isFitted){
			
			p_mean = bfvals_[name].first;	// toy constrainits thrown about best fit to data
			p_err  = prevals_[name].second; // uncertainties taken from card
			std::cout << "******* "<< name << " *******"<<std::endl;
			std::cout << p_mean <<  " " << p_err << std::endl;
			std::cout << "******************************" <<std::endl;

			const char* drawInput = Form("(%s-%f)/%f",name,p_mean,p_err);
			tree_->Draw(Form("%s>>%s",drawInput,name),"");
			tree_->Draw(Form("%s>>%s_fail",drawInput,name),selectString.c_str(),"same");
			fitPull  = true;
			fitPullf = true;
			if (doLH) {
			  gr = graphLH(name,p_err,whichfit);
			  if (gr) plotLH=true;
			}
			
		}

		else{
			tree_->Draw(Form("%s>>%s",name,name),"");
			tree_->Draw(Form("%s>>%s_fail",name,name),selectString.c_str(),"same");
		}
		

		TH1F* bH  = (TH1F*) gROOT->FindObject(Form("%s",name))->Clone();
		TH1F* bHf = (TH1F*) gROOT->FindObject(Form("%s_fail",name))->Clone();
		bHf->SetLineColor(2);
		bH->GetXaxis()->SetTitle(bH->GetTitle());
		bH->GetYaxis()->SetTitle(Form("no toys (%d total)",nToysInTree));
		bH->GetYaxis()->SetTitleOffset(1.05);
		bH->GetXaxis()->SetTitleOffset(0.9);
		bH->GetYaxis()->SetTitleSize(0.05);
		bH->GetXaxis()->SetTitleSize(0.05);
		if (isFitted) {bH->GetXaxis()->SetTitle(Form("(%s-#theta_{B})/#sigma_{#theta}",name));}
		else {bH->GetXaxis()->SetTitle(Form("%s",name));}
		
		bH->SetTitle("");	

		if ( bH->Integral() <=0 )  fitPull = false;
		if (fitPull) {bH->Fit("gaus"); bH->GetFunction("gaus")->SetLineColor(4);}
		
		if ( bHf->Integral() <=0 )  fitPullf = false;
		if (fitPullf) {bHf->Fit("gaus"); bHf->GetFunction("gaus")->SetLineColor(2);}

		c->Clear();
		//TPad pad1("t1","",0.01,0.02,0.59,0.98);
		// Pad 1 sizes depend on the parameter type ...
		double pad1_x1,pad1_x2,pad1_y1,pad1_y2;
		if ( !isFitted ) {
			 pad1_x1 = 0.01; 
			 pad1_x2 = 0.98; 
			 pad1_y1 = 0.045; 
			 pad1_y2 = 0.98; 
		} else {
			 pad1_x1 = 0.01; 
			 pad1_x2 = 0.59; 
			 pad1_y1 = 0.56; 
			 pad1_y2 = 0.98; 
		}
		
		TPad pad1("t1","",pad1_x1,pad1_y1,pad1_x2,pad1_y2);
		TPad pad1a("t1a","",0.01,0.045,0.59,0.522);
		TPad pad2("t2","",0.59,0.04,0.98,0.62);
		TPad pad3("t3","",0.55,0.64,0.96,0.95);

		pad1.SetNumber(1); pad2.SetNumber(2); pad3.SetNumber(3); pad1a.SetNumber(4);

		if ( isFitted ) {pad1a.Draw();pad2.Draw();pad3.Draw();}

		pad1.Draw();
		pad2.SetGrid(true);


		TLatex *titletext = new TLatex();titletext->SetNDC();

		if ( isFitted ){
			c->cd(4); 
			tree_->Draw(Form("%s:%s_In>>%s_%s_2d",name,name,name,tree_->GetName()),""); 
			//TH2D *h2d_corr = (TH2D*)gROOT->FindObject(Form("%s_2d",name));
			//h2d_corr->SetMarkerColor(4);
			//h2d_corr->SetTitle("");
			//h2d_corr->GetXaxis()->SetTitle(Form("%s_In",name));
			//h2d_corr->GetYaxis()->SetTitle(Form("%s",name));
			titletext->SetTextAlign(11);
			titletext->SetTextSize(0.05);
			titletext->DrawLatex(0.05,0.02,Form("%s_In",name));
			titletext->SetTextAngle(90);
			titletext->DrawLatex(0.04,0.06,Form("%s",name));
			titletext->SetTextAngle(0);
		}

		
		c->cd(1); bH->Draw(); bHf->Draw("same");
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
			double gap;
			TLatex *tlatex = new TLatex(); tlatex->SetNDC(); 
			if (fitPullf) {tlatex->SetTextSize(0.09); gap=0.12;}
			else  {tlatex->SetTextSize(0.11);gap=0.14;}

			tlatex->SetTextColor(4);
			tlatex->DrawLatex(0.11,0.80,Form("Mean    : %.3f #pm %.3f",bH->GetFunction("gaus")->GetParameter(1),bH->GetFunction("gaus")->GetParError(1)));
			tlatex->DrawLatex(0.11,0.80-gap,Form("Sigma   : %.3f #pm %.3f",bH->GetFunction("gaus")->GetParameter(2),bH->GetFunction("gaus")->GetParError(2)));

			if (fitPullf){ 
				tlatex->SetTextColor(2);
				tlatex->DrawLatex(0.11,0.60,Form("Mean    : %.3f #pm %.3f",bHf->GetFunction("gaus")->GetParameter(1),bHf->GetFunction("gaus")->GetParError(1)));
				tlatex->DrawLatex(0.11,0.60-gap,Form("Sigma   : %.3f #pm %.3f",bHf->GetFunction("gaus")->GetParameter(2),bHf->GetFunction("gaus")->GetParError(2)));
			}

			tlatex->SetTextSize(0.10);
			tlatex->SetTextColor(1);
					
			tlatex->DrawLatex(0.11,0.33,Form("Pre-fit #pm #sigma_{#theta}: %.3f #pm %.3f",prevals_[name].first, p_err));
			tlatex->DrawLatex(0.11,0.18,Form("Best-fit (#theta_{B})  : %.3f ",p_mean));
			tlatex->DrawLatex(0.11,0.03,Form("Best-fit (#theta_{S+B}): %.3f ",bfvals_sb_[name].first));
			
			pullSummaryMap[name]=std::make_pair<double,double>(bH->GetFunction("gaus")->GetParameter(1),bH->GetFunction("gaus")->GetParameter(2));
			nPulls++;

		}

		double titleSize = isFitted ? 0.1 : 0.028;
		titletext->SetTextSize(titleSize);titletext->SetTextAlign(21); titletext->DrawLatex(0.55,0.92,name);
		c->SaveAs(Form("%s.pdf",treename.c_str()));
		fOut->WriteObject(c,Form("%s_%s",treename.c_str(),name));
		//c->SaveAs(Form("%s_%s.pdf",treename.c_str(),name));
	}
	
	if (doPull && nPulls>0){
	  
	    std::cout << "Generating Pull Summaries" <<std::endl; 
	    int nRemainingPulls = nPulls;
	    TCanvas *hc = new TCanvas("hc","",3000,2000); hc->SetGrid(0);
	    std::map < const char*, std::pair <double,double> >::iterator pull_it = pullSummaryMap.begin();
	    std::map < const char*, std::pair <double,double> >::iterator pull_end = pullSummaryMap.end();

	    int pullPlots = 1;
	    while (nRemainingPulls > 0){

		int nThisPulls = min(maxPullsPerPlot,nRemainingPulls);

		TH1F pullSummaryHist("pullSummary","",nThisPulls,0,nThisPulls);
		for (int pi=1;pull_it!=pull_end && pi<=nThisPulls ;pull_it++,pi++){
			pullSummaryHist.GetXaxis()->SetBinLabel(pi,(*pull_it).first);
			pullSummaryHist.SetBinContent(pi,((*pull_it).second).first);
			pullSummaryHist.SetBinError(pi,((*pull_it).second).second);
			nRemainingPulls--;
		}		

		pullSummaryHist.SetMarkerStyle(21);pullSummaryHist.SetMarkerSize(1.5);pullSummaryHist.SetMarkerColor(2);pullSummaryHist.SetLabelSize(pullLabelSize);
		pullSummaryHist.GetYaxis()->SetRangeUser(-3,3);pullSummaryHist.GetYaxis()->SetTitle("pull summary (n#sigma)");pullSummaryHist.Draw("E1");
		hc->SaveAs(Form("%s.pdf",treename.c_str()));
		fOut->WriteObject(hc,Form("comb_pulls_%s_%d",treename.c_str(),pullPlots));
	//	hc->SaveAs(Form("comb_pulls_%s_%d.pdf",treename.c_str(),pullPlots));
		pullPlots++;
	   }

	    delete hc;
	}

	c->SaveAs(Form("%s.pdf]",treename.c_str()));
	fOut->Close();
	delete c;
	return;


}

void plotLLRdistribution(TTree *tree_, TFile *fdata_){

	TCanvas *c = new TCanvas("llr","",960,800);
	tree_->Draw("0*nll_nll0>>htemp_llr(100,0,4)","mu<0");
	tree_->Draw("-2*nll_nll0>>htemp_llr_2(100,0,4)","mu>=0");
	TH1F *ht0  = (TH1F*) gROOT->FindObject("htemp_llr");
	TH1F *htmu = (TH1F*) gROOT->FindObject("htemp_llr_2");
	ht0->Add(htmu);
	ht0->GetXaxis()->SetRangeUser(0,4);
	ht0->SetFillColor(kGray+3); ht0->SetLineColor(1);
	ht0->GetXaxis()->SetTitle("-2 #times llr");
	ht0->SetTitle("");
	ht0->Scale(1./ht0->Integral());
	ht0->GetYaxis()->SetTitleOffset(1.2); ht0->GetYaxis()->SetTitle("probability");
	ht0->Draw();

	if (fdata_!=0) {
		TTree* trdata_ = (TTree*)fdata_->Get("tree_fit_sb");	
		double res; trdata_->SetBranchAddress("nll_nll0",&res);
		trdata_->GetEntry(0);
		double q0_obs = -2*res;
		TLine* ldata = new TLine(q0_obs,0,q0_obs,ht0->GetMaximum());
		ldata->SetLineColor(2);
		ldata->SetLineWidth(2);	ldata->Draw();
		std::cout << "-2xllr observed = " << q0_obs << std::endl;
	}
	c->SetLogy();
	c->SaveAs("llrdist.pdf");

}

void plotParamtersFromToys(std::string inputFile, std::string dataFits="", std::string workspace="", std::string selectString="mu>0"){

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
	// will also create a pull summary if datafit is available.
	plotTree(tree_b,"bestfitparams",selectString);		// LH plot will be centered around B-only fit to data
	plotTree(tree_sb,"bestfitparams_sb",selectString);	// LH plot will be centered around S+B fit to data

	// Produce plot pf q_mu distribution in toys (where q_mu is not modified for upper limits only)
	// The s+b tree has this info
	plotLLRdistribution(tree_sb,fd_);
	
	fi_->Close();
	if (doPull) fd_->Close();
	if (doLH) fw_->Close();
	
	
}

