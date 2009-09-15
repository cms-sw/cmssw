#include "PhysicsTools/TagAndProbe/interface/TPRooSimultaneousFitter.hh"

// RooFit - clean headers only
#include <RooRealVar.h>
#include <RooAddPdf.h>
#include <RooSimultaneous.h>
#include <RooStringVar.h>
#include <RooFitResult.h>
#include <RooDataSet.h>
#include <RooCategory.h>
#include <RooCatType.h>
#include <RooPlot.h>
#include <RooTreeData.h>
#include <RooArgSet.h>
#include <RooDataHist.h>
#include <RooArgList.h>
#include <RooFitResult.h>
#include <RooNLLVar.h>
#include <RooGlobalFunc.h> 
#include <RooChi2Var.h>
#include <RooMinuit.h>
#include <RooCmdArg.h>
#include <RooCmdConfig.h>

// ROOT
#include <TROOT.h>  // gROOT
#include <TChain.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>
#include <TFile.h>
#include <TGraphAsymmErrors.h>
#include <TStyle.h>
#include <TMath.h> // Gamma
#include <TFile.h>


#include <sstream>
#include <string>

TPRooSimultaneousFitter::TPRooSimultaneousFitter(){}



TPRooSimultaneousFitter::~TPRooSimultaneousFitter(){}



void TPRooSimultaneousFitter::configure(RooRealVar &ResonanceMass, TTree *fitTree,
					const std::string &bvar1, const std::vector< double >& bins1, const int bin1,
					const std::string &bvar2, const std::vector< double >& bins2, const int bin2,
					std::vector<double> &efficiency, std::vector<double> &numSignal,
					std::vector<double> &numBkgPass ,std::vector<double> &numBkgFail){
  

  rooMass_.reset(&ResonanceMass);
  
  // The binning variables
  std::string bunits = "GeV";
  double lowEdge1 = bins1[bin1];
  double highEdge1 = bins1[bin1+1];
  if( bvar1 == "Eta" || bvar1 == "Phi" ) bunits  = "";
  
  // Var1
  rooVar1_.reset( new RooRealVar(bvar1.c_str(),bvar1.c_str(),lowEdge1,highEdge1,bunits.c_str()));
  
  bunits = "GeV";
  double lowEdge2 = bins2[bin2];
  double highEdge2 = bins2[bin2+1];
  if( bvar2 == "Eta" || bvar2 == "Phi" ) bunits = "";
  
  // Var2
  rooVar2_.reset(new  RooRealVar(bvar2.c_str(),bvar2.c_str(),lowEdge2,highEdge2,bunits.c_str()));

  rooBin1_= bin1;
  rooBin2_= bin2;
  
  // The weighting
  RooRealVar Weight("Weight","Weight",1.0);
  
  // Make the category variable that defines the two fits,
  // namely whether the probe passes or fails the eff criteria.
  
  RooCategory ProbePass("ProbePass","sample");
  ProbePass.defineType("pass",1);
  ProbePass.defineType("fail",0);  
  
  
  rooData_.reset(new RooDataSet("fitData","fitData",fitTree,RooArgSet(ProbePass,*rooMass_,*rooVar1_,*rooVar2_,Weight)));

  //  rooData_ = new RooDataSet("fitData","fitData",fitTree,RooArgSet(ProbePass,*rooMass_,*rooVar1_,*rooVar2_,Weight));
  rooData_->setWeightVar("Weight");
  roobData_.reset(new RooDataHist("bdata","Binned Data", RooArgList(*rooMass_,ProbePass),*rooData_));
		  
		  
		  
  //  std::stringstream roofitstream;
  //#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,0)
  //  rooData_->defaultStream(&roofitstream);
  //#else
  //  rooData_->defaultPrintStream(&roofitstream);
  //#endif
  //  rooData_->get()->Print();
  //  roofitstream.str(std::string());
  
  
  
  
  rooEfficiency_.reset( new RooRealVar("efficiency","efficiency",efficiency[0]));
  if( efficiency.size() == 3 )
    {
      rooEfficiency_->setRange(efficiency[1],efficiency[2]);
      rooEfficiency_->setConstant(false);
    }
  
  rooNumSignal_.reset( new RooRealVar("numSignal","numSignal",numSignal[0]));
  
  if( numSignal.size() == 3 )
    {
      rooNumSignal_->setRange(numSignal[1],numSignal[2]);
      rooNumSignal_->setConstant(false);
    }
  
  
  rooNumBkgPass_.reset( new RooRealVar("numBkgPass","numBkgPass",numBkgPass[0]));
  
  if( numBkgPass.size() == 3 )
    {
      rooNumBkgPass_->setRange(numBkgPass[1],numBkgPass[2]);
      rooNumBkgPass_->setConstant(false);
    }
  
  
  rooNumBkgFail_.reset( new RooRealVar("numBkgFail","numBkgFail",numBkgFail[0]));
  if( numBkgFail.size() == 3 )
    {
      rooNumBkgFail_->setRange(numBkgFail[1],numBkgFail[2]);
      rooNumBkgFail_->setConstant(false);
    }
  
  //   rooResultsRootFile_ =new RooStringVar("TPresultsRootFile","ROOT file with persisted fit results","") ;
}








void TPRooSimultaneousFitter::createTotalPDF(RooAddPdf *signalShapePdf,RooAddPdf *bkgShapePdf){

  RooFormulaVar numSigPass("numSigPass","numSignal*efficiency", 
			   RooArgList(*rooNumSignal_,*rooEfficiency_) );
  RooFormulaVar numSigFail("numSigFail","numSignal*(1.0 - efficiency)", 
			   RooArgList(*rooNumSignal_,*rooEfficiency_) );
  
  RooArgList componentspass(*signalShapePdf, *bkgShapePdf);
  RooArgList yieldspass(numSigPass, *rooNumBkgPass_);

  RooAddPdf *signalShapeFailPdf;

  signalShapeFailPdf = signalShapePdf;

  RooArgList componentsfail(*signalShapeFailPdf,*bkgShapePdf);
  RooArgList yieldsfail(numSigFail, *rooNumBkgFail_);
  
  RooAddPdf sumpass("sumpass","fixed extended sum pdf",componentspass,yieldspass);
  RooAddPdf sumfail("sumfail","fixed extended sum pdf",componentsfail, yieldsfail);

  // The total simultaneous fit ...

   RooCategory ProbePass("ProbePass","sample");
   ProbePass.defineType("pass",1);
   ProbePass.defineType("fail",0);  

   rooTotalPDF_.reset(new RooSimultaneous("totalPdf","totalPdf",ProbePass));

   ProbePass.setLabel("pass");
   
   rooTotalPDF_->addPdf(sumpass,ProbePass.getLabel());

   //#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,0)
   //   rooTotalPDF_->defaultStream(&roofitstream;)
   //#else
   //     rooTotalPDF_->defaultPrintStream(&roofitstream);
   //#endif
   //   rooTotalPDF_->Print();
   ProbePass.setLabel("fail");
   rooTotalPDF_->addPdf(sumfail,ProbePass.getLabel());
   // rooTotalPDF_->Print();


}




RooFitResult* TPRooSimultaneousFitter::performFit(bool UnBinnedFit,int& npassResult, int& nfailResult){
   
   // Count the number of passing and failing probes in the region
   // making sure we have enough to fit ...

 std::ostringstream passCond;
   passCond.str(std::string());
   passCond << "(ProbePass==1) && (Mass<" << rooMass_->getMax()<< ") && (Mass>" << rooMass_->getMin()
            << ") && (" << rooVar1_->GetName() << "<" << rooVar1_->getMax() << ") && (" << rooVar1_->GetName() << ">"
            << rooVar1_->getMin() << ") && (" << rooVar2_->GetName() << "<" << rooVar2_->getMax() << ") && ("
            << rooVar2_->GetName() << ">" << rooVar2_->getMin() << ")";
   std::ostringstream failCond;
   failCond.str(std::string());
   failCond << "(ProbePass==0) && (Mass<" << rooMass_->getMax() << ") && (Mass>" << rooMass_->getMin()
            << ") && (" << rooVar1_->GetName() << "<" << rooVar1_->getMax() << ") && (" << rooVar1_->GetName() << ">"
            << rooVar1_->getMin() << ") && (" << rooVar2_->GetName() << "<" << rooVar2_->getMax() << ") && ("
            << rooVar2_->GetName() << ">" << rooVar2_->getMin() << ")";

   npassResult = static_cast<int>(rooData_->sumEntries(passCond.str().c_str()));
   nfailResult = static_cast<int>(rooData_->sumEntries(failCond.str().c_str()));

   RooAbsCategoryLValue& simCat = (RooAbsCategoryLValue&) rooTotalPDF_->indexCat();

   TList* dsetList = const_cast<RooAbsData*>((RooAbsData*)&rooData_)->split(simCat);
   RooCatType* type;
   TIterator* catIter = simCat.typeIterator();
   while( (type=(RooCatType*)catIter->Next()) )
   {
      // Retrieve the PDF for this simCat state
      RooAbsPdf* pdf =  rooTotalPDF_->getPdf(type->GetName());
      RooAbsData* dset = (RooAbsData*) dsetList->FindObject(type->GetName());
     
      if (pdf && dset && dset->numEntries() != 0.0) 
      {               
	if( strcmp(type->GetName(),"pass") == 0 ) 
	  {
	    npassResult = dset->numEntries(); 
	  }
	 else if( strcmp(type->GetName(),"fail") == 0 ) 
	 {
	    nfailResult = dset->numEntries();
	 }
      }
   }
   
   // Return if there's nothing to fit
   if( npassResult==0 && nfailResult==0 )  return (RooFitResult*)0;;

   // Remember to set eff val when writing the wraper functions

   RooFitResult *fitResult = 0;

   if( UnBinnedFit)
     {
    RooNLLVar nll("nll","nll",*rooTotalPDF_,*rooData_,kTRUE);
    RooMinuit m(nll);
    m.setErrorLevel(0.5);
    m.setStrategy(2);
    m.hesse();
    m.migrad();
    m.hesse();
    m.minos();
    fitResult = m.save();

     }
 else 
   {

     RooChi2Var chi2("chi2","chi2",*rooTotalPDF_,*roobData_,RooFit::DataError(RooDataHist::SumW2),RooFit::Extended(kTRUE));

     RooMinuit m(chi2);
     m.setErrorLevel(0.5); // <<< HERE
     m.setStrategy(2);
      m.hesse();
      m.migrad();
      m.hesse();
      m.minos();
      fitResult = m.save();

   }

    return fitResult;


}




void TPRooSimultaneousFitter::persistFitresultsToRoot(RooFitResult* FitResults){


   //   TFile f(rooResultsRootFile_->getVal(),"recreate") ;
   if (FitResults) FitResults->Write("TPFitResults");
   //   f.Close();
}




RooArgSet TPRooSimultaneousFitter::readFitresultsFromRoot(char* filename){


 TFile f(filename,"read") ;
  if ( ! f.IsOpen() ) {
    cout << "TPRooSimultaneousFitter::readFitresultsFromRoot() : Error : Can't read this file." << endl;
    exit(-1);
  }

  RooArgSet parsBack;
 
  RooFitResult* tpres = (RooFitResult*)f.Get("TPFitResult");
  if(tpres){
    parsBack.add(tpres->floatParsFinal());
    parsBack.add(tpres->constPars());
  } else {
    cout<< "Failed to retrieve fit results" <<endl;

  }
  f.Close();

  return parsBack;
}



void TPRooSimultaneousFitter::saveCanvasTPResults(TFile *outputfile   ,char* filename,
						  RooAddPdf *signalShapePdf,RooAddPdf *bkgShapePdf,
						  int& npassResult, int& nfailResult,const bool is2D){


  using namespace RooFit;
 
  if (!outputfile)
    TFile f(filename,"recreate") ;

  RooDataHist::ErrorType fitError = RooDataHist::SumW2;

   int font_num = 42;
   double font_size = 0.05;

   TStyle fitStyle("fitStyle","Style for Fit Plots");
   fitStyle.Reset("Plain");
   fitStyle.SetFillColor(10);
   fitStyle.SetTitleFillColor(10);
   fitStyle.SetTitleStyle(0000);
   fitStyle.SetStatColor(10);
   fitStyle.SetErrorX(0);
   fitStyle.SetEndErrorSize(10);
   fitStyle.SetPadBorderMode(0);
   fitStyle.SetFrameBorderMode(0);

   fitStyle.SetTitleFont(font_num);
   fitStyle.SetTitleFontSize(font_size);
   fitStyle.SetTitleFont(font_num, "XYZ");
   fitStyle.SetTitleSize(font_size, "XYZ");
   fitStyle.SetTitleXOffset(0.9);
   fitStyle.SetTitleYOffset(1.05);
   fitStyle.SetLabelFont(font_num, "XYZ");
   fitStyle.SetLabelOffset(0.007, "XYZ");
   fitStyle.SetLabelSize(font_size, "XYZ");
   fitStyle.cd();

   std::ostringstream oss1;
   oss1 << rooBin1_;
   std::ostringstream oss2;
   oss2 << rooBin2_;

   std::ostringstream oss3;
   oss3 << rooVar1_->GetName();
   std::ostringstream oss4;
   oss4 << rooVar2_->GetName();


   std::string cname = "fit_canvas_"+ oss3.str() + "_" + oss1.str() + "_" + oss4.str() + "_" + oss2.str();
   if( !is2D ) cname = "fit_canvas_" + oss3.str() + "_" + oss1.str();

   TCanvas c(cname.c_str(),"Sum over Modes, Signal Region",1000,1500);


   c.Divide(1,2);
   c.cd(1);
   c.SetFillColor(10);

   TPad *lhs = (TPad*)gPad;
   lhs->Divide(2,1);
   lhs->cd(1);



   RooCategory ProbePass("ProbePass","sample");
   ProbePass.defineType("pass",1);
   ProbePass.defineType("fail",0);  

   RooPlot* frame1 = rooMass_->frame();
   frame1->SetTitle("Passing Tag-Probes");
   frame1->SetName("pass");
   rooData_->plotOn(frame1,Cut("ProbePass==1"),RooFit::DataError(fitError));
   ProbePass.setLabel("pass");
   if( npassResult > 0 )
   {
      rooTotalPDF_->plotOn(frame1,Slice(ProbePass),Components(*bkgShapePdf),
      LineColor(kRed),ProjWData(*rooMass_,*rooData_));
      rooTotalPDF_->plotOn(frame1,Slice(ProbePass),ProjWData(*rooMass_,*rooData_),Precision(1e-5));
   }
   frame1->Draw("e0");

   lhs->cd(2);
   RooPlot* frame2 = rooMass_->frame();
   frame2->SetTitle("Failing Tag-Probes");
   frame2->SetName("fail");
   rooData_->plotOn(frame2,Cut("ProbePass==0"),RooFit::DataError(fitError));
   ProbePass.setLabel("fail");
   if( nfailResult > 0 )
   {
      rooTotalPDF_->plotOn(frame2,Slice(ProbePass),Components(*bkgShapePdf),
      LineColor(kRed),ProjWData(*rooMass_,*rooData_));
      rooTotalPDF_->plotOn(frame2,Slice(ProbePass),ProjWData(*rooMass_,*rooData_),Precision(1e-5));
   }
   frame2->Draw("e0");

   c.cd(2);
   RooPlot* frame3 = rooMass_->frame();
   frame3->SetTitle("All Tag-Probes");
   frame3->SetName("total");
   rooData_->plotOn(frame3,RooFit::DataError(fitError));
   rooTotalPDF_->plotOn(frame3,Components(*bkgShapePdf),
   LineColor(kRed),ProjWData(*rooMass_,*rooData_));
   rooTotalPDF_->plotOn(frame3,ProjWData(*rooMass_,*rooData_),Precision(1e-5));
   rooTotalPDF_->paramOn(frame3);
   frame3->Draw("e0");

   outputfile->cd();
   c.Write();
   outputfile->Close();
}


