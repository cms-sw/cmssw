// -*- C++ -*-
//
// Package:    TagProbeEDMAnalysis
// Class:      TagProbeEDMAnalysis
// 
/**\class TagProbeEDMAnalysis TagProbeEDMAnalysis.cc PhysicsTools/TagProbeEDMAnalysis/src/TagProbeEDMAnalysis.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  "Nadia Adam"
//         Created:  Sun Apr 20 10:35:25 CDT 2008
//
// Kalanand Mishra: July 1, 2008 
// Added a configurable option "useRecoVarsForTruthMatchedCands" 
// (default == 'false') to use reconstructed or detector values 
// (instead of MC generated values) for var1 and var2 when doing MC truth efficiencies.
//
//

#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/TagAndProbe/interface/TagProbeEDMAnalysis.h"
#include "PhysicsTools/TagAndProbe/interface/RooCMSShapePdf.h"


// ROOT headers

#include <TArrow.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TGaxis.h>
#include <TGraphAsymmErrors.h>
#include <TIterator.h>
#include <TLatex.h>
#include <TString.h>
#include <TStyle.h>

// RooFit headers

#include <RooAbsData.h>
#include <RooAddPdf.h>
#include <RooBifurGauss.h>
#include <RooBreitWigner.h>
#include <RooCategory.h>
#include <RooCatType.h>
#include <RooCBShape.h>
#include <RooChi2Var.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooFitResult.h>
#include <RooGenericPdf.h>
#include <RooGlobalFunc.h>
#include <RooLandau.h>
#include <RooMinuit.h>
#include <RooNLLVar.h>
#include <RooPlot.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooTreeData.h>
#include <RooVoigtian.h>

using namespace std;
using namespace edm;
using namespace RooFit;

TagProbeEDMAnalysis::TagProbeEDMAnalysis(const edm::ParameterSet& iConfig)
{
   // TagProbeEDMAnalysis variables
   vector<string>       dEmptyStringVec;
   vector<unsigned int> dEmptyUIntVec;
   vector<double>       dEmptyDoubleVec;
   quantities_      = iConfig.getUntrackedParameter< vector<string> >("quantities",dEmptyStringVec); 
   conditions_      = iConfig.getUntrackedParameter< vector<string> >("conditions",dEmptyStringVec); 
   outputFileNames_ = iConfig.getUntrackedParameter< vector<string> >("outputFileNames",dEmptyStringVec);
   XBins_           = iConfig.getUntrackedParameter< vector<unsigned int> >("XBins",dEmptyUIntVec);
   XMin_            = iConfig.getUntrackedParameter< vector<double> >("XMin",dEmptyDoubleVec);
   XMax_            = iConfig.getUntrackedParameter< vector<double> >("XMax",dEmptyDoubleVec);
   logY_            = iConfig.getUntrackedParameter< vector<unsigned int> >("logY",dEmptyUIntVec);
   
   // Efficiency input variables
   tagProbeType_   = iConfig.getUntrackedParameter< int >("TagProbeType",0);

   calcEffsSB_     = iConfig.getUntrackedParameter< bool >("CalculateEffSideBand",false);
   calcEffsFitter_ = iConfig.getUntrackedParameter< bool >("CalculateEffFitter",false);
   calcEffsTruth_  = iConfig.getUntrackedParameter< bool >("CalculateEffTruth",false);
   
   truthParentId_  = iConfig.getUntrackedParameter< int >("MCTruthParentId",23);

   // Type of fit
   unbinnedFit_    = iConfig.getUntrackedParameter< bool >("UnbinnedFit",false);
   do2DFit_        = iConfig.getUntrackedParameter< bool >("Do2DFit",false);

   massNbins_      = iConfig.getUntrackedParameter< int >("NumBinsMass",20);
   massLow_        = iConfig.getUntrackedParameter< double >("MassLow",0.0);
   massHigh_       = iConfig.getUntrackedParameter< double >("MassHigh",100.0);

   // The efficiency binning variables (default pt and eta)
   vector< double > dBins;
   var1Name_       = iConfig.getUntrackedParameter< string >("NameVar1","pt");
   var1Nbins_      = iConfig.getUntrackedParameter< int >("NumBinsVar1",20);
   var1Low_        = iConfig.getUntrackedParameter< double >("Var1Low",0.0);
   var1High_       = iConfig.getUntrackedParameter< double >("Var1High",100.0);
   var1Bins_       = iConfig.getUntrackedParameter< vector<double> >("Var1BinBoundaries",dBins);

   var2Name_       = iConfig.getUntrackedParameter< string >("NameVar2","eta");
   var2Nbins_      = iConfig.getUntrackedParameter< int >("NumBinsVar2",20);
   var2Low_        = iConfig.getUntrackedParameter< double >("Var2Low",-2.4);
   var2High_       = iConfig.getUntrackedParameter< double >("Var2High",2.4);
   var2Bins_       = iConfig.getUntrackedParameter< vector<double> >("Var2BinBoundaries",dBins);


   // If want to use reconstructed or detector values (instead of MC generated values) 
   // of var1 and var2 when doing MC truth efficiencies (i.e., when "calcEffsTruth==true").
   useRecoVarsForTruthMatchedCands_ = 
     iConfig.getUntrackedParameter< bool >("useRecoVarsForTruthMatchedCands",false);


   // Check that the names of the variables are okay ...
   if( !( var1Name_ == "pt" || var1Name_ == "p"   || var1Name_ == "px" ||
	  var1Name_ == "py" || var1Name_ == "pz"  || var1Name_ == "e"  ||
	  var1Name_ == "et" || var1Name_ == "eta" || var1Name_ == "phi" || 
	  var1Name_ == "ptDet" || var1Name_ == "pDet"   || var1Name_ == "pxDet" ||
	  var1Name_ == "pyDet" || var1Name_ == "pzDet"  || var1Name_ == "eDet"  ||
	  var1Name_ == "etDet" || var1Name_ == "etaDet" || var1Name_ == "phiDet") )
   {
      LogWarning("TagAndProbe") << "Warning: Var1 name invalid, setting var1 name to pt!";
      var1Name_ = "pt";
   }

   if( !( var2Name_ == "pt" || var2Name_ == "p"   || var2Name_ == "px" ||
	  var2Name_ == "py" || var2Name_ == "pz"  || var2Name_ == "e"  ||
	  var2Name_ == "et" || var2Name_ == "eta" || var2Name_ == "phi" ||
	  var2Name_ == "ptDet" || var2Name_ == "pDet"   || var2Name_ == "pxDet" ||
	  var2Name_ == "pyDet" || var2Name_ == "pzDet"  || var2Name_ == "eDet"  ||
	  var2Name_ == "etDet" || var2Name_ == "etaDet" || var2Name_ == "phiDet") )
   {
      LogWarning("TagAndProbe") << "Warning: Var2 name invalid, setting var2 name to eta!";
      var2Name_ = "eta";
   }

   // Make the uppercase names ...
   var1NameUp_ = var1Name_;
   var1NameUp_.at(0) -= 32;

   var2NameUp_ = var2Name_;
   var2NameUp_.at(0) -= 32;

   // Set up the bins for the eff histograms ...
   if( var1Bins_.size() == 0 ) 
   {
      // User didn't set bin boundaries, so use even binning
      double bwidth = (var1High_-var1Low_)/(double)var1Nbins_;
      for( int i=0; i<=var1Nbins_; ++i )
      {
	 double low_edge = var1Low_+(double)i*bwidth;
	 var1Bins_.push_back(low_edge);
      }
   }
   if( var2Bins_.size() == 0 ) 
   {
      // User didn't set bin boundaries, so use even binning
      double bwidth = (var2High_-var2Low_)/(double)var2Nbins_;
      for( int i=0; i<=var2Nbins_; ++i )
      {
	 double low_edge = var2Low_+(double)i*bwidth;
	 var2Bins_.push_back(low_edge);
      }
   }


   // SBS
   SBSPeak_     = iConfig.getUntrackedParameter< double >("SBSPeak",90);
   SBSStanDev_  = iConfig.getUntrackedParameter< double >("SBSStanDev",2);

   // Fitter variables
   vector<double> dSigM;
   dSigM.push_back(91.1876);
   dSigM.push_back(85.0);
   dSigM.push_back(95.0);
   signalMean_     = iConfig.getUntrackedParameter< vector<double> >("SignalMean",dSigM);
   vector<double> dSigW;
   dSigW.push_back(2.3);
   dSigW.push_back(1.0);
   dSigW.push_back(4.0);
   signalWidth_     = iConfig.getUntrackedParameter< vector<double> >("SignalWidth",dSigW);
   vector<double> dSigS;
   dSigS.push_back(1.5);
   dSigS.push_back(0.0);
   dSigS.push_back(4.0);
   signalSigma_     = iConfig.getUntrackedParameter< vector<double> >("SignalSigma",dSigS);
   vector<double> dSigWL;
   dSigWL.push_back(3.0);
   dSigWL.push_back(1.0);
   dSigWL.push_back(10.0);
   signalWidthL_    = iConfig.getUntrackedParameter< vector<double> >("SignalWidthL",dSigWL);
   vector<double> dSigWR;
   dSigWR.push_back(0.52);
   dSigWR.push_back(0.0);
   dSigWR.push_back(2.0);
   signalWidthR_    = iConfig.getUntrackedParameter< vector<double> >("SignalWidthR",dSigWR);
   
   vector<double> dBGF;
   dBGF.push_back(0.87);
   dBGF.push_back(0.0);
   dBGF.push_back(1.0);
   bifurGaussFrac_  = iConfig.getUntrackedParameter< vector<double> >("BifurGaussFrac",dBGF);

   vector<double> dBAl;
   dBAl.push_back(63.0);
   bkgAlpha_        = iConfig.getUntrackedParameter< vector<double> >("BkgAlpha",dBAl);
   vector<double> dBBt;
   dBBt.push_back(0.001);
   bkgBeta_         = iConfig.getUntrackedParameter< vector<double> >("BkgBeta",dBBt);
   vector<double> dBPk;
   dBPk.push_back(91.1876);
   bkgPeak_         = iConfig.getUntrackedParameter< vector<double> >("BkgPeak",dBPk);
   vector<double> dBGam;
   dBGam.push_back(0.08);
   dBGam.push_back(0.0);
   dBGam.push_back(1.0);
   bkgGamma_        = iConfig.getUntrackedParameter< vector<double> >("BkgGamma",dBGam);

   vector<double> dEff;
   dEff.push_back(0.98);
   dEff.push_back(0.0);
   dEff.push_back(1.1);
   efficiency_      = iConfig.getUntrackedParameter< vector<double> >("Efficiency",dEff);
   vector<double> dNSig;
   dNSig.push_back(1000.0);
   dNSig.push_back(-10.0);
   dNSig.push_back(1000000.0);
   numSignal_       = iConfig.getUntrackedParameter< vector<double> >("NumSignal",dNSig);
   vector<double> dNBPs;
   dNBPs.push_back(1000.0);
   dNBPs.push_back(-10.0);
   dNBPs.push_back(1000000.0);
   numBkgPass_      = iConfig.getUntrackedParameter< vector<double> >("NumBkgPass",dNBPs);
   vector<double> dNBFl;
   dNBFl.push_back(1000.0);
   dNBFl.push_back(-10.0);
   dNBFl.push_back(1000000.0);
   numBkgFail_      = iConfig.getUntrackedParameter< vector<double> >("NumBkgFail",dNBFl);

   // Get the mode of operation variables
   mode_        = iConfig.getUntrackedParameter< string >("Mode","Normal");
   fitFileName_ = iConfig.getUntrackedParameter< string >("FitFileName","fitfile.root");

   vector< string > dReadFiles;
   readFiles_   = iConfig.getUntrackedParameter< vector<string> >("ReadFromFiles",dReadFiles);
   
   // Allocate space for the simple plot histograms
   numQuantities_ = quantities_.size();
   Histograms_ = new TH1F[numQuantities_];

   for(unsigned int i = 0; i < numQuantities_; i++)
   {
      Histograms_[i] =  TH1F(("h"+quantities_[i]).c_str(), "", XBins_[i], XMin_[i], XMax_[i]);
   }

   // Verify correct use of input variables for making simple plots 
   doAnalyze_ = true;
   if( numQuantities_ == 0 )
   {
      doAnalyze_ = false;
   }
   else if(outputFileNames_.size() != numQuantities_){
      doAnalyze_ = false;
      cout << "outputFileNames is not the same size as quantities" << endl;    
   }else if(conditions_.size() != numQuantities_){
      doAnalyze_ = false;
      cout << "conditions is not the same size as quantities" << endl;    
   }else if(XBins_.size() != numQuantities_){
      doAnalyze_ = false;
      cout << "XBins is not the same size as quantities" << endl;    
   }else if(XMax_.size() != numQuantities_){
      doAnalyze_ = false;
      cout << "XMax is not the same size as quantities" << endl;    
   }else if(XMin_.size() != numQuantities_){
      doAnalyze_ = false;
      cout << "XMin is not the same size as quantities" << endl;    
   }else if(logY_.size() != numQuantities_){
      doAnalyze_ = false;
      cout << "logY is not the same size as quantities" << endl;    
   }   


   // Make the simple fit tree
   if( mode_ != "Read" )
   {
      string fmode = "RECREATE";
      outRootFile_ = new TFile(fitFileName_.c_str(),fmode.c_str());
      outRootFile_->cd();

      fitTree_ = new TTree("fitter_tree","Tree For Fitting",1);
      fitTree_->Branch("ProbePass",         &ProbePass_,"ProbePass/I");
      fitTree_->Branch("Mass",              &Mass_,     "Mass/D");
      fitTree_->Branch("Weight",            &Weight_,   "Weight/D");
      fitTree_->Branch(var1NameUp_.c_str(), &Var1_,     (var1NameUp_+"/D").c_str());
      fitTree_->Branch(var2NameUp_.c_str(), &Var2_,     (var2NameUp_+"/D").c_str());   
   }

   // MC Truth Histograms
   var1Pass_ = new TH1F("hvar1pass","Var1 Pass",var1Bins_.size()-1,&var1Bins_[0]);
   var1All_  = new TH1F("hvar1all","Var1 All",var1Bins_.size()-1,&var1Bins_[0]);
   
   var2Pass_ = new TH1F("hvar2pass","Var2 Pass",var2Bins_.size()-1,&var2Bins_[0]);
   var2All_  = new TH1F("hvar2all","Var2 All",var2Bins_.size()-1,&var2Bins_[0]);
   
   var1var2Pass_ = new TH2F("hvar1var2pass","Var1:Var2 Pass",var1Bins_.size()-1,&var1Bins_[0],
			    var2Bins_.size()-1,&var2Bins_[0]);
   var1var2All_  = new TH2F("hvar1var2all","Var1:Var2 All",var1Bins_.size()-1,&var1Bins_[0],
			    var2Bins_.size()-1,&var2Bins_[0]);

}

TagProbeEDMAnalysis::~TagProbeEDMAnalysis()
{
   // Clean up
//    if (fitTree_)
//    {
//       if( calcEffsFitter_ || calcEffsSB_ ) delete fitTree_->GetCurrentFile();
//       fitTree_ = 0;
//    }

   if( var1Pass_  ) delete var1Pass_;
   if( var1All_   ) delete var1All_;
   if( var2Pass_ ) delete var2Pass_;
   if( var2All_  ) delete var2All_;

   if(Histograms_)
   {
      delete [] Histograms_; 
      Histograms_ = 0;
   }
}

// ------------ method called to for each event  ------------
void
TagProbeEDMAnalysis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // Safety check .. if mode = read, the user should use an empty source.
   if( mode_ == "Read" ) return;

   // Fill the fit tree if we are fitting or doing SB subtraction
   if( calcEffsSB_ || calcEffsFitter_ || mode_ == "Write" )
   {
      // Get the TP variables to fill the fitter tree ...
      Handle< vector<int> > tp_type;
      if ( !iEvent.getByLabel("TPEdm","TPtype",tp_type) ) {
         cout << "No TPtype in Tree!" << endl; 
      }

      Handle< vector<int> > tp_true;
      if ( !iEvent.getByLabel("TPEdm","TPtrue",tp_true) ) {
         cout << "No TPtrue in Tree!" << endl; 
      }
      
      Handle< vector<int> > tp_ppass;
      if ( !iEvent.getByLabel("TPEdm","TPppass",tp_ppass) ) {
         cout << "No TPppass in Tree!" << endl; 
      }
      
      Handle< vector<float> > tp_mass;
      if ( !iEvent.getByLabel("TPEdm","TPmass",tp_mass) ) {
         cout << "No TPmass in Tree!" << endl; 
      }

      Handle< vector<float> > tp_probe_var1;
      if ( !iEvent.getByLabel("TPEdm",("TPProbe"+var1Name_).c_str(),tp_probe_var1) ) {
         cout << "No TPProbe"+var1Name_+" in Tree!" << endl; 
      }
      
      Handle< vector<float> > tp_probe_var2;
      if ( !iEvent.getByLabel("TPEdm",("TPProbe"+var2Name_).c_str(),tp_probe_var2) ) {
         cout << "No TPProbe"+var2Name_+" in Tree!" << endl; 
      }
      
      outRootFile_->cd();

      if( tp_type.isValid() )
      {
	 int nrTP = tp_type->size();
	 for( int i=0; i<nrTP; ++i )
	 {
	    if( (*tp_type)[i] != tagProbeType_ ) continue;
	 
	    Weight_ = 1.0;
	    ProbePass_ = (*tp_ppass)[i];
	    Mass_ = (double)(*tp_mass)[i];
	    Var1_ = (double)(*tp_probe_var1)[i];
	    Var2_ = (double)(*tp_probe_var2)[i];
	 
	    fitTree_->Fill();
	 }
      }      
   }

   // Fill the MC truth information if required
   if( calcEffsTruth_ || mode_ == "Write" )
   {
      // Get the Cnd variable for the MC truth ...
      Handle< vector<int> > cnd_type;
      if ( !iEvent.getByLabel("TPEdm","Cndtype",cnd_type) ) {
         cout << "No Cndtype in Tree!" << endl; 
      }

      Handle< vector<int> > cnd_tag;
      if ( !iEvent.getByLabel("TPEdm","Cndtag",cnd_tag) ) {
         cout << "No Cndtag in Tree!" << endl; 
      }

      Handle< vector<int> > cnd_aprobe;
      if ( !iEvent.getByLabel("TPEdm","Cndaprobe",cnd_aprobe) ) {
         cout << "No Cndaprobe in Tree!" << endl; 
      }

      Handle< vector<int> > cnd_pprobe;
      if ( !iEvent.getByLabel("TPEdm","Cndpprobe",cnd_pprobe) ) {
         cout << "No Cndpprobe in Tree!" << endl; 
      }

      Handle< vector<int> > cnd_moid;
      if ( !iEvent.getByLabel("TPEdm","Cndmoid",cnd_moid) ) {
         cout << "No Cndmoid in Tree!" << endl; 
      }

      Handle< vector<int> > cnd_gmid;
      if ( !iEvent.getByLabel("TPEdm","Cndgmid",cnd_gmid) ) {
         cout << "No Cndgmid in Tree!" << endl; 
      }



      std::string truthVar1 = var1Name_.c_str();
      std::string truthVar2 = var2Name_.c_str();

      if(!useRecoVarsForTruthMatchedCands_) {
	truthVar1.insert(0, "Cnd");
	truthVar2.insert(0, "Cnd");
      }


      Handle< vector<float> > cnd_var1;
      if ( !iEvent.getByLabel("TPEdm",truthVar1.c_str(),cnd_var1) ) {
         cout << "No Cnd"+var1Name_+" in Tree!" << endl; 
      }

      Handle< vector<float> > cnd_var2;
      if ( !iEvent.getByLabel("TPEdm",truthVar2.c_str(),cnd_var2) ) {
         cout << "No Cnd"+var2Name_+" in Tree!" << endl; 
      }

      if( cnd_type.isValid() )
      {
	 int nCnd = (int)cnd_type->size();
	 for( int i=0; i<nCnd; ++i )
	 {
	    if( (*cnd_type)[i] != tagProbeType_ ) continue;
	    
	    if( truthParentId_ != 0 && !( fabs((*cnd_gmid)[i]) == truthParentId_ ) ) continue;
	    
	    bool inVar1Range = false;
	    if( (*cnd_var1)[i] > var1Bins_[0] &&
		(*cnd_var1)[i] < var1Bins_[var1Bins_.size()-1] )
	       inVar1Range = true;
	    bool inVar2Range = false;
	    if( (*cnd_var2)[i] > var2Bins_[0] &&
		(*cnd_var2)[i] < var2Bins_[var2Bins_.size()-1] )
	       inVar2Range = true;

	    if( (*cnd_aprobe)[i] == 1 && (*cnd_pprobe)[i] == 1 )
	    {
	       if( inVar2Range ) var1Pass_->Fill((*cnd_var1)[i]);
	       if( inVar1Range ) var2Pass_->Fill((*cnd_var2)[i]);
	       var1var2Pass_->Fill((*cnd_var1)[i],(*cnd_var2)[i]);
	    }
	    if( (*cnd_aprobe)[i] == 1 )
	    {
	       if( inVar2Range ) var1All_->Fill((*cnd_var1)[i]);
	       if( inVar1Range ) var2All_->Fill((*cnd_var2)[i]);
	       var1var2All_->Fill((*cnd_var1)[i],(*cnd_var2)[i]);
	    }
	 }
      }
   }


   // Do plot making etc ...
   if( doAnalyze_ )
   {
      for(unsigned int i = 0; i < numQuantities_; i++)
      {
	 // int
	 if( quantities_[i].find("n") == 0 || 
	     quantities_[i] == "Run"       || 
	     quantities_[i] == "Event" )
	 {
	    Handle< int > h;
	    if ( !iEvent.getByLabel("TPEdm",quantities_[i].c_str(),h) ) { 
	       cout << "No "+quantities_[i]+" in Tree!" << endl; 
            }
	    if( h.isValid() ) Histograms_[i].Fill(*h);
	 }
	 // vector<int>
	 else if( quantities_[i].find("type") != string::npos ||
		  quantities_[i].find("true") != string::npos || 
		  quantities_[i].find("ppass") != string::npos || 
		  quantities_[i].find("l1") != string::npos || 
		  quantities_[i].find("hlt") != string::npos || 
		  quantities_[i].find("tag") != string::npos || 
		  quantities_[i].find("aprobe") != string::npos || 
		  quantities_[i].find("pprobe") != string::npos || 
		  quantities_[i].find("moid") != string::npos || 
		  quantities_[i].find("gmid") != string::npos || 
		  quantities_[i].find("pid") != string::npos || 
		  quantities_[i].find("bc") != string::npos )
	 {
	    Handle< vector<int> > h;
	    if ( !iEvent.getByLabel("TPEdm",quantities_[i].c_str(),h) ) {
	        cout << "No "+quantities_[i]+" in Tree!" << endl; 
            }
	    if( h.isValid() ) for( int n=0; n<(int)h->size(); ++n ) Histograms_[i].Fill((*h)[n]);
	 }
	 // vector< float >
	 else
	 {
	    Handle< vector<float> > h;
	    if ( !iEvent.getByLabel("TPEdm",quantities_[i].c_str(),h) ) {
	        cout << "No "+quantities_[i]+" in Tree!" << endl; 
            }
	    if( h.isValid() ) for( int n=0; n<(int)h->size(); ++n ) Histograms_[i].Fill((*h)[n]);
	 }
      }
   }
}

// ********* Save the user requested histograms ******** //
int TagProbeEDMAnalysis::SaveHistogram(TH1F& Histo, std::string outFileName, Int_t LogY = 0)
{
  
   TCanvas* c1 = new TCanvas("c1","c1",700,500);
   c1->GetPad(0)->SetTicks(1,1);
   c1->SetLogy(LogY);
  
   Histo.Draw();
  
   c1->SaveAs(outFileName.c_str());
  
   delete c1;

   return 0;
}
// ***************************************************** //

// ****************** Zll Eff Side band subtraction *************
void TagProbeEDMAnalysis::ZllEffSBS( string &fileName, string &bvar, vector< double > bins,
				     string &bvar2, double bvar2Lo, double bvar2Hi )
{

   outRootFile_->cd();
   fitTree_->SetDirectory(outRootFile_);
  
   //return;
   cout << "***** Here in Zll sideband subtraction ******" << endl;
   cout << "Number of entries " << fitTree_->GetEntries() << endl;
   
   string hname = "sbs_eff_" + bvar;
   string htitle = "SBS Efficiency vs " + bvar;

   stringstream condition;
   stringstream histoName;
   stringstream histoTitle;;

   int bnbins = bins.size()-1;
   cout << "There are " << bnbins << " bins " << endl;
   TH1F effhist(hname.c_str(),htitle.c_str(),bnbins,&bins[0]);
   for( int bin=0; bin<bnbins; ++bin ) cout << "Bin low edge " << effhist.GetBinLowEdge(bin+1) << endl;

   TH1F* PassProbes;
   TH1F* FailProbes;

   TH1F* SBSPassProbes;
   TH1F* SBSFailProbes;

   const int XBinsSBS = massNbins_;
   double XMinSBS = massLow_;
   double XMaxSBS = massHigh_;


   if( XMinSBS > ( SBSPeak_ - 13*SBSStanDev_ ) )
     XMinSBS = SBSPeak_ - 13*SBSStanDev_;

   if( XMaxSBS < ( SBSPeak_ + 13*SBSStanDev_ ) )
     XMaxSBS = SBSPeak_ + 13*SBSStanDev_;

   for( int bin=0; bin<bnbins; ++bin )
   {
 
      // The binning variable
      string bunits = "";
      double lowEdge = bins[bin];
      double highEdge = bins[bin+1];
      if( bvar == "Pt" ) bunits = "GeV";

      // Passing Probes
      condition.str(std::string());
      condition  << "((ProbePass == 1) && ( " << bvar << " > " <<  lowEdge << " ) && ( " 
		 << bvar << " < " << highEdge << " ) && ( " << bvar2 << " > " << bvar2Lo
		 << " ) && ( " << bvar2 << " < " << bvar2Hi <<"))*Weight";
      std::cout << "Pass condition ( " << bvar << " ): " << condition.str() << std::endl;
      histoName.str(std::string());
      histoName << "sbs_pass_" << bvar << "_" << bin;
      histoTitle.str(std::string());
      histoTitle << "Passing Probes - " << lowEdge << " < " << bvar << " < " << highEdge;
      PassProbes = new TH1F(histoName.str().c_str(), histoTitle.str().c_str(), XBinsSBS, XMinSBS, XMaxSBS); 
      PassProbes->Sumw2();
      PassProbes->SetDirectory(outRootFile_);
      fitTree_->Draw(("Mass >> " + histoName.str()).c_str(), condition.str().c_str() );

      // Failing Probes
      condition.str(std::string());
      condition  << "((ProbePass == 0) && ( " << bvar << " > " <<  lowEdge << " ) && ( " 
		 << bvar << " < " << highEdge << " ) && ( " << bvar2 << " > " << bvar2Lo
		 << " ) && ( " << bvar2 << " < " << bvar2Hi <<"))*Weight";
      std::cout << "Fail condition ( " << bvar << " ): " << condition.str() << std::endl;
      histoName.str(std::string());
      histoName << "sbs_fail_" <<  bvar << "_" << bin;
      histoTitle.str(std::string());
      histoTitle << "Failing Probes - " << lowEdge << " < " << bvar << " < " << highEdge;
      FailProbes = new TH1F(histoName.str().c_str(), histoTitle.str().c_str(), XBinsSBS, XMinSBS, XMaxSBS); 
      FailProbes->Sumw2();
      FailProbes->SetDirectory(outRootFile_);
      fitTree_->Draw(("Mass >> " + histoName.str()).c_str(), condition.str().c_str());

      // SBS Passing  Probes
      histoName.str(std::string());
      histoName << "sbs_pass_subtracted_" << bvar << "_" << bin;
      histoTitle.str(std::string());
      histoTitle << "Passing Probes SBS - "  << lowEdge << " < " << bvar << " < " << highEdge;
      SBSPassProbes = new TH1F(histoName.str().c_str(), histoTitle.str().c_str(), XBinsSBS, XMinSBS, XMaxSBS); 
      SBSPassProbes->Sumw2();

      // SBS Failing Probes
      histoName.str(std::string());
      histoName << "sbs_fail_subtracted_" << bvar << "_" << bin; 
      histoTitle.str(std::string());
      histoTitle << "Failing Probes SBS - "  << lowEdge << " < " << bvar << " < " << highEdge;
      SBSFailProbes = new TH1F(histoName.str().c_str(), histoTitle.str().c_str(), XBinsSBS, XMinSBS, XMaxSBS); 
      SBSFailProbes->Sumw2();

      // Perform side band subtraction
      SideBandSubtraction(*PassProbes, *SBSPassProbes, SBSPeak_, SBSStanDev_);
      SideBandSubtraction(*FailProbes, *SBSFailProbes, SBSPeak_, SBSStanDev_);

      // Count the number of passing and failing probes in the region
      cout << "About to count the number of events" << endl;
      double npassR = SBSPassProbes->Integral("width");
      double nfailR = SBSFailProbes->Integral("width");

      if((npassR + nfailR) != 0){
	Double_t eff = npassR/(npassR + nfailR);
	Double_t effErr = sqrt(npassR * nfailR / (npassR + nfailR))/(npassR + nfailR);

	cout << "Num pass " << npassR << endl;
	cout << "Num fail " << nfailR << endl;
	cout << "Eff " << eff << endl;
	cout << "Eff error " << effErr << endl;

	// Fill the efficiency hist
	effhist.SetBinContent(bin+1,eff);
	effhist.SetBinError(bin+1,effErr);
      }else {
	cout << " no probes " << endl;
      }

      // ********** Make and save Canvas for the plots ********** //
      outRootFile_->cd();

      PassProbes->Write();
      FailProbes->Write();
      SBSPassProbes->Write();
      SBSFailProbes->Write();
   }
   
   outRootFile_->cd();
   effhist.Write();

   return;

}

// ****************** Zll Eff Side band subtraction *************
void TagProbeEDMAnalysis::ZllEffSBS2D( string &fileName, string &bvar1, vector< double > bins1,
				       string &bvar2, vector< double > bins2 )
{

   outRootFile_->cd();
   fitTree_->SetDirectory(outRootFile_);
  
   //return;
   cout << "***** Here in Zll sideband subtraction 2D ******" << endl;
   cout << "Number of entries " << fitTree_->GetEntries() << endl;
   
   string hname = "sbs_eff_" + bvar1 + "_" + bvar2;
   string htitle = "SBS Efficiency: " + bvar1 + " vs " + bvar2;

   stringstream condition;
   stringstream histoName;
   stringstream histoTitle;;

   int bnbins1 = bins1.size()-1;
   int bnbins2 = bins2.size()-1;
   cout << "There are " << bnbins1 << " bins for var1 " << endl;

   TH2F effhist(hname.c_str(),htitle.c_str(),bnbins1,&bins1[0],bnbins2,&bins2[0]);

   TH1F* PassProbes;
   TH1F* FailProbes;

   TH1F* SBSPassProbes;
   TH1F* SBSFailProbes;

   const int XBinsSBS = massNbins_;
   const double XMinSBS = massLow_;
   const double XMaxSBS = massHigh_;

   for( int bin1=0; bin1<bnbins1; ++bin1 )
   {
      double lowEdge1 = bins1[bin1];
      double highEdge1 = bins1[bin1+1];

      for( int bin2=0; bin2<bnbins2; ++bin2 )
      {
	 // The binning variables
	 double lowEdge2 = bins2[bin2];
	 double highEdge2 = bins2[bin2+1];

	 // Passing Probes
	 condition.str(std::string());
	 condition  << "((ProbePass == 1) && ( " << bvar1 << " > " <<  lowEdge1 << " ) && ( " 
		    << bvar1 << " < " << highEdge1 << " ) && ( " << bvar2 << " > " 
		    << lowEdge2 << " ) && ( " << bvar2 << " < " << highEdge2 
		    << "))*Weight";

	 std::cout << "Pass condition ( " << bvar1 << ":" << bvar2 << " ): " 
		   << condition.str() << std::endl;
	 histoName.str(std::string());
	 histoName << "sbs_pass_" << bvar1 << "_" << bin1 << "_" << bvar2 << "_" << bin2;
	 histoTitle.str(std::string());
	 histoTitle << "Passing Probes - " << lowEdge1 << " < " << bvar1 << " < " << highEdge1
		    << " & " << lowEdge2 << " < " << bvar2 << " < " << highEdge2;
	 PassProbes = new TH1F(histoName.str().c_str(), histoTitle.str().c_str(), 
			       XBinsSBS, XMinSBS, XMaxSBS); 
	 PassProbes->Sumw2();
	 PassProbes->SetDirectory(outRootFile_);
	 fitTree_->Draw(("Mass >> " + histoName.str()).c_str(), condition.str().c_str() );

	 // Failing Probes
	 condition.str(std::string());
	 condition  << "((ProbePass == 0) && ( " << bvar1 << " > " <<  lowEdge1 << " ) && ( " 
		    << bvar1 << " < " << highEdge1 << " ) && ( " << bvar2 << " > " 
		    << lowEdge2 << " ) && ( " << bvar2 << " < " << highEdge2 
		    << "))*Weight";
	 std::cout << "Fail condition ( " << bvar1 << ":" << bvar2 << " ): " 
		   << condition.str() << std::endl;
	 histoName.str(std::string());
	 histoName << "sbs_fail_" << bvar1 << "_" << bin1 << "_" << bvar2 << "_" << bin2;
	 histoTitle.str(std::string());
	 histoTitle << "Failing Probes - " << lowEdge1 << " < " << bvar1 << " < " << highEdge1
		    << " & " << lowEdge2 << " < " << bvar2 << " < " << highEdge2;
	 FailProbes = new TH1F(histoName.str().c_str(), histoTitle.str().c_str(), 
			       XBinsSBS, XMinSBS, XMaxSBS); 
	 FailProbes->Sumw2();
	 FailProbes->SetDirectory(outRootFile_);
	 fitTree_->Draw(("Mass >> " + histoName.str()).c_str(), condition.str().c_str());

	 // SBS Passing  Probes
	 histoName.str(std::string());
	 histoName << "sbs_pass_subtracted_" << bvar1 << "_" << bin1 << "_" << bvar2 << "_" << bin2;
	 histoTitle.str(std::string());
	 histoTitle << "Passing Probes SBS - " << lowEdge1 << " < " << bvar1 << " < " << highEdge1
		    << " & " << lowEdge2 << " < " << bvar2 << " < " << highEdge2;
	 SBSPassProbes = new TH1F(histoName.str().c_str(), histoTitle.str().c_str(), 
				  XBinsSBS, XMinSBS, XMaxSBS); 
	 SBSPassProbes->Sumw2();

	 // SBS Failing Probes
	 histoName.str(std::string());
	 histoName << "sbs_fail_subtracted_" << bvar1 << "_" << bin1 << "_" << bvar2 << "_" << bin2;
	 histoTitle.str(std::string());
	 histoTitle << "Failing Probes SBS - " << lowEdge1 << " < " << bvar1 << " < " << highEdge1
		    << " & " << lowEdge2 << " < " << bvar2 << " < " << highEdge2;
	 SBSFailProbes = new TH1F(histoName.str().c_str(), histoTitle.str().c_str(), 
				  XBinsSBS, XMinSBS, XMaxSBS); 
	 SBSFailProbes->Sumw2();

	 // Perform side band subtraction
	 SideBandSubtraction(*PassProbes, *SBSPassProbes, SBSPeak_, SBSStanDev_);
	 SideBandSubtraction(*FailProbes, *SBSFailProbes, SBSPeak_, SBSStanDev_);

	 // Count the number of passing and failing probes in the region
	 cout << "About to count the number of events" << endl;
	 double npassR = SBSPassProbes->Integral("width");
	 double nfailR = SBSFailProbes->Integral("width");

	 if((npassR + nfailR) != 0){
	    Double_t eff = npassR/(npassR + nfailR);
	    Double_t effErr = sqrt(npassR * nfailR / (npassR + nfailR))/(npassR + nfailR);

	    cout << "Num pass " << npassR << endl;
	    cout << "Num fail " << nfailR << endl;
	    cout << "Eff " << eff << endl;
	    cout << "Eff error " << effErr << endl;

	    // Fill the efficiency hist
	    effhist.SetBinContent(bin1+1,bin2+1,eff);
	    effhist.SetBinError(bin1+1,bin2+1,effErr);
	 }else {
	    cout << " no probes " << endl;
	 }

	 // ********** Make and save Canvas for the plots ********** //
	 outRootFile_->cd();

	 PassProbes->Write();
	 FailProbes->Write();
	 SBSPassProbes->Write();
	 SBSFailProbes->Write();
      }
   }
   
   outRootFile_->cd();
   effhist.Write();

   return;

}

// ********* Do sideband subtraction on the requested histogram ********* //
void TagProbeEDMAnalysis::SideBandSubtraction( const TH1F& Total, TH1F& Result,
                                               Double_t Peak, Double_t SD)
{
   // Total Means signal plus background

   const Double_t BinWidth  = Total.GetXaxis()->GetBinWidth(1);
   const Int_t nbins = Total.GetNbinsX();
   const Double_t xmin = Total.GetXaxis()->GetXmin();

   const Int_t PeakBin = (Int_t)((Peak - xmin)/BinWidth + 1); // Peak
   const Int_t SDBin = (Int_t)(SD/BinWidth); // Standard deviation
   const Int_t I = 3*SDBin; // Interval
   const Int_t D = 10*SDBin;  // Distance from peak

   const Double_t IntegralRight = Total.Integral(PeakBin + D, PeakBin + D + I);
   const Double_t IntegralLeft = Total.Integral(PeakBin - D - I, PeakBin - D);

   double SubValue = 0.0;
   double NewValue = 0.0;

   const double Slope     = (IntegralRight - IntegralLeft)/(double)((2*D + I )*(I+1));
   const double Intercept = IntegralLeft/(double)(I+1) - ((double)PeakBin - (double)D - (double)I/2.0)*Slope;

   for(Int_t bin = 1; bin < (nbins + 1); bin++){
      SubValue = Slope*bin + Intercept;
      if(SubValue < 0)
         SubValue = 0;

      NewValue = Total.GetBinContent(bin)-SubValue;
      if(NewValue > 0){
         Result.SetBinContent(bin, NewValue);
      }
   }
   Result.SetEntries(Result.Integral());
}
// ********************************************************************** //

// ********** Z -> l+l- Fitter ********** //
void TagProbeEDMAnalysis::ZllEffFitter( string &fileName, string &bvar, vector< double > bins,
					string &bvar2, double bvar2Lo, double bvar2Hi )
{
   outRootFile_->cd();
   TChain *nFitTree = new TChain();
   nFitTree->Add((fileName+"/fitter_tree").c_str());

   //return;
   cout << "Here in Zll fitter" << endl;
   
   string hname = "fit_eff_" + bvar;
   string htitle = "Efficiency vs " + bvar;
   int bnbins = bins.size()-1;
   cout << "The number of bins is " << bnbins << endl;
   TH1F effhist(hname.c_str(),htitle.c_str(),bnbins,&bins[0]);

   for( int bin=0; bin<bnbins; ++bin )
   {

      // The fit variable - lepton invariant mass
      RooRealVar Mass("Mass","Invariant Di-Lepton Mass", massLow_, massHigh_, "GeV/c^{2}");
      Mass.setBins(massNbins_);

      // The binning variable
      string bunits = "GeV";
      double lowEdge = bins[bin];
      double highEdge = bins[bin+1];
      if( bvar == "Eta" || bvar == "Phi" ) bunits = "";
      RooRealVar Var1(bvar.c_str(),bvar.c_str(),lowEdge,highEdge,bunits.c_str());

      bunits = "GeV";
      if( bvar2 == "Eta" || bvar2 == "Phi" ) bunits = "";
      RooRealVar Var2(bvar2.c_str(),bvar2.c_str(),bvar2Lo,bvar2Hi,bunits.c_str());

      // The weighting
      RooRealVar Weight("Weight","Weight",0.0,10000.0);

      // Make the category variable that defines the two fits,
      // namely whether the probe passes or fails the eff criteria.
      RooCategory ProbePass("ProbePass","sample");
      ProbePass.defineType("pass",1);
      ProbePass.defineType("fail",0);  

      cout << "Made fit variables" << endl;

      // Add the TTree as our data set ... with the weight in case 
      // we are using chained MC
      //RooDataSet* data = new RooDataSet("fitData","fitData",(TTree*)fitTree_->Clone(),
      //				RooArgSet(ProbePass,Mass,Var1,Weight),"","");
      // Above command doesn't work in root 5.18 (lovely) so we have this
      // silly workaround with TChain for now
      RooDataSet* data = new RooDataSet("fitData","fitData",(TTree*)nFitTree,
					RooArgSet(ProbePass,Mass,Var1,Var2,Weight));
      //RooDataSet* data = new RooDataSet("fitData",fileName.c_str(),"fitter_tree",
      //				RooArgSet(ProbePass,Mass,Var1,Weight));


      //data->get()->Print();
      data->setWeightVar("Weight");
      data->get()->Print();

      cout << "Made dataset" << endl;

      RooDataHist *bdata = new RooDataHist("bdata","Binned Data",
					   RooArgList(Mass,ProbePass),*data);
 
      // ********** Construct signal shape PDF ********** //

      // Signal PDF variables
      RooRealVar signalMean("signalMean","signalMean",signalMean_[0]);
      RooRealVar signalWidth("signalWidth","signalWidth",signalWidth_[0]);
      RooRealVar signalSigma("signalSigma","signalSigma",signalSigma_[0]);
      RooRealVar signalWidthL("signalWidthL","signalWidthL",signalWidthL_[0]);
      RooRealVar signalWidthR("signalWidthR","signalWidthR",signalWidthR_[0]);

      // If the user has set a range, make the variable float
      if( signalMean_.size() == 3 )
      {
	 signalMean.setRange(signalMean_[1],signalMean_[2]);
	 signalMean.setConstant(false);
      }
      if( signalWidth_.size() == 3 )
      {
	 signalWidth.setRange(signalWidth_[1],signalWidth_[2]);
	 signalWidth.setConstant(false);
      }
      if( signalSigma_.size() == 3 )
      {
	 signalSigma.setRange(signalSigma_[1],signalSigma_[2]);
	 signalSigma.setConstant(false);
      }
      if( signalWidthL_.size() == 3 )
      {
	 signalWidthL.setRange(signalWidthL_[1],signalWidthL_[2]);
	 signalWidthL.setConstant(false);
      }
      if( signalWidthR_.size() == 3 )
      {
	 signalWidthR.setRange(signalWidthR_[1],signalWidthR_[2]);
	 signalWidthR.setConstant(false);
      }
  
//       // Voigtian
         RooVoigtian signalVoigtPdf("signalVoigtPdf", "signalVoigtPdf", 
 				 Mass, signalMean, signalWidth, signalSigma);

//       // Bifurcated Gaussian
       RooBifurGauss signalGaussBifurPdf("signalGaussBifurPdf", "signalGaussBifurPdf", 
 					Mass, signalMean, signalWidthL, signalWidthR);

      // Bifurcated Gaussian fraction
      RooRealVar bifurGaussFrac("bifurGaussFrac","bifurGaussFrac",bifurGaussFrac_[0]);
      if( bifurGaussFrac_.size() == 3 )
      {
	 bifurGaussFrac.setRange(bifurGaussFrac_[1],bifurGaussFrac_[2]);
	 bifurGaussFrac.setConstant(false);
      } 

      // The total signal PDF
       RooAddPdf  signalShapePdf("signalShapePdf", "signalShapePdf",
 				 signalVoigtPdf,signalGaussBifurPdf,bifurGaussFrac);


      // ********** Construct background shape PDF ********** //

      // Background PDF variables
      RooRealVar bkgAlpha("bkgAlpha","bkgAlpha",bkgAlpha_[0]);
      RooRealVar bkgBeta("bkgBeta","bkgBeta",bkgBeta_[0]);
      RooRealVar bkgGamma("bkgGamma","bkgGamma",bkgGamma_[0]);
      RooRealVar bkgPeak("bkgPeak","bkgPeak",bkgPeak_[0]);

      // If the user has specified a range, let the bkg shape 
      // variables float in the fit
      if( bkgAlpha_.size() == 3 )
      {
	 bkgAlpha.setRange(bkgAlpha_[1],bkgAlpha_[2]);
	 bkgAlpha.setConstant(false);
      }
      if( bkgBeta_.size() == 3 )
      {
	 bkgBeta.setRange(bkgBeta_[1],bkgBeta_[2]);
	 bkgBeta.setConstant(false);
      }
      if( bkgGamma_.size() == 3 )
      {
	 bkgGamma.setRange(bkgGamma_[1],bkgGamma_[2]);
	 bkgGamma.setConstant(false);
      }
      if( bkgPeak_.size() == 3 )
      {
	 bkgPeak.setRange(bkgPeak_[1],bkgPeak_[2]);
	 bkgPeak.setConstant(false);
      }

      // CMS Background shape
      RooCMSShapePdf bkgShapePdf("bkgShapePdf","bkgShapePdf", 
				 Mass,bkgAlpha,bkgBeta,bkgGamma,bkgPeak);

      // Now define some efficiency/yield variables  
      RooRealVar efficiency("efficiency","efficiency",efficiency_[0]);
      RooRealVar numSignal("numSignal","numSignal",numSignal_[0]);
      RooRealVar numBkgPass("numBkgPass","numBkgPass",numBkgPass_[0]);
      RooRealVar numBkgFail("numBkgFail","numBkgFail",numBkgFail_[0]);

      // If ranges are specifed these are floating variables
      if( efficiency_.size() == 3 )
      {
	 efficiency.setRange(efficiency_[1],efficiency_[2]);
	 efficiency.setConstant(false);
      }
      if( numSignal_.size() == 3 )
      {
	 numSignal.setRange(numSignal_[1],numSignal_[2]);
	 numSignal.setConstant(false);
      }
      if( numBkgPass_.size() == 3 )
      {
	 numBkgPass.setRange(numBkgPass_[1],numBkgPass_[2]);
	 numBkgPass.setConstant(false);
      }
      if( numBkgFail_.size() == 3 )
      {
	 numBkgFail.setRange(numBkgFail_[1],numBkgFail_[2]);
	 numBkgFail.setConstant(false);
      }
      

      RooFormulaVar numSigPass("numSigPass","numSignal*efficiency", 
			       RooArgList(numSignal,efficiency) );
      RooFormulaVar numSigFail("numSigFail","numSignal*(1.0 - efficiency)", 
			       RooArgList(numSignal,efficiency) );

      RooArgList componentspass(signalShapePdf,bkgShapePdf);
      RooArgList yieldspass(numSigPass, numBkgPass);
      RooArgList componentsfail(signalShapePdf,bkgShapePdf);
      RooArgList yieldsfail(numSigFail, numBkgFail);	  

      RooAddPdf sumpass("sumpass","fixed extended sum pdf",componentspass,yieldspass);
      RooAddPdf sumfail("sumfail","fixed extended sum pdf",componentsfail, yieldsfail);
  
      // The total simultaneous fit ...
      RooSimultaneous totalPdf("totalPdf","totalPdf",ProbePass);
      ProbePass.setLabel("pass");
      totalPdf.addPdf(sumpass,ProbePass.getLabel());
      totalPdf.Print();
      ProbePass.setLabel("fail");
      totalPdf.addPdf(sumfail,ProbePass.getLabel());
      totalPdf.Print();

      // Count the number of passing and failing probes in the region
      // making sure we have enough to fit ...
      cout << "About to count the number of events" << endl;
      int npassR = (int)data->sumEntries("ProbePass==1");
      int nfailR = (int)data->sumEntries("ProbePass==0");
      cout << "Num pass " << npassR << endl;
      cout << "Num fail " << nfailR << endl;

      RooAbsCategoryLValue& simCat = (RooAbsCategoryLValue&) totalPdf.indexCat();
   
      TList* dsetList = const_cast<RooAbsData*>((RooAbsData*)data)->split(simCat);
      RooCatType* type;
      TIterator* catIter = simCat.typeIterator();
      while( (type=(RooCatType*)catIter->Next()) )
      {
	 // Retrieve the PDF for this simCat state
	 RooAbsPdf* pdf =  totalPdf.getPdf(type->GetName());
	 RooAbsData* dset = (RooAbsData*) dsetList->FindObject(type->GetName());

	 if (pdf && dset && dset->numEntries(kTRUE)!=0.) 
	 {               
	    cout << "GOF Entries " << dset->numEntries() << " " 
		 << type->GetName() << std::endl;
	    if( (string)type->GetName() == "pass" ) 
	    {
	       npassR = dset->numEntries(); 
	       cout << "Pass " << npassR << endl; 
	    }
	    else if( (string)type->GetName() == "fail" ) 
	    {
	       nfailR = dset->numEntries();
	       cout << "Fail " << nfailR << endl; 
	    }
	 }
      }
      // End the pass fail counting.

      // Return if there's nothing to fit
      if( npassR==0 && nfailR==0 ) return;

      cout << "**** About to start the fitter ****" << endl;

      // ********* Do the Actual Fit ********** //  
      RooFitResult *fitResult = 0;

      // The user chooses between binnned/unbinned fitting
      if( unbinnedFit_ )
      {
	 cout << "Starting unbinned fit using LL." << endl;
	 RooNLLVar nll("nll","nll",totalPdf,*data,kTRUE);
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
	 cout << "Starting binned fit using Chi2." << endl;
	 RooChi2Var chi2("chi2","chi2",totalPdf,*bdata,DataError(RooAbsData::SumW2),Extended(kTRUE));
	 RooMinuit m(chi2);
	 m.setErrorLevel(0.5); // <<< HERE
	 m.setStrategy(2);
	 m.hesse();
	 m.migrad();
	 m.hesse();
	 m.minos();
	 fitResult = m.save();
      }

      fitResult->Print("v");

      std::cout << "Signal yield: " << numSignal.getVal() << " +- "
		<< numSignal.getError() << " + " << numSignal.getAsymErrorHi()
		<<" - "<< numSignal.getAsymErrorLo() << std::endl;
      std::cout << "Efficiency: "<< efficiency.getVal() << " +- "
		<< efficiency.getError() << " + " << efficiency.getAsymErrorHi()
		<<" + "<< efficiency.getAsymErrorLo() << std::endl;

      // Fill the efficiency hist
      effhist.SetBinContent(bin+1,efficiency.getVal());
      effhist.SetBinError(bin+1,efficiency.getError());

      // ********** Make and save Canvas for the plots ********** //
      //outRootFile_->cd();

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

      ostringstream oss;
      oss << bin;
      string cname = "fit_canvas_" + bvar + "_" + oss.str();
      TCanvas *c = new TCanvas(cname.c_str(),"Sum over Modes, Signal Region",1500,2500);
      c->Divide(1,2);
      c->cd(1);
      c->SetFillColor(10);

      TPad *lhs = (TPad*)gPad;
      lhs->Divide(2,1);
      lhs->cd(1);

      RooPlot* frame1 = Mass.frame();
      frame1->SetTitle("Passing Tag-Probes");
      frame1->SetName("pass");
      data->plotOn(frame1,Cut("ProbePass==1"));
      ProbePass.setLabel("pass");
      totalPdf.plotOn(frame1,Slice(ProbePass),Components(bkgShapePdf),
		      LineColor(kRed),ProjWData(Mass,*data));
      totalPdf.plotOn(frame1,Slice(ProbePass),ProjWData(Mass,*data),Precision(1e-5));
      frame1->Draw("e0");

      lhs->cd(2);
      RooPlot* frame2 = Mass.frame();
      frame2->SetTitle("Failing Tag-Probes");
      frame2->SetName("fail");
      data->plotOn(frame2,Cut("ProbePass==0"));
      ProbePass.setLabel("fail");
      totalPdf.plotOn(frame2,Slice(ProbePass),Components(bkgShapePdf),
		      LineColor(kRed),ProjWData(Mass,*data));
      totalPdf.plotOn(frame2,Slice(ProbePass),ProjWData(Mass,*data),Precision(1e-5));
      frame2->Draw("e0");

      c->cd(2);
      RooPlot* frame3 = Mass.frame();
      frame3->SetTitle("All Tag-Probes");
      frame3->SetName("total");
      data->plotOn(frame3);
      totalPdf.plotOn(frame3,Components(bkgShapePdf),
		      LineColor(kRed),ProjWData(Mass,*data));
      totalPdf.plotOn(frame3,ProjWData(Mass,*data),Precision(1e-5));
      totalPdf.paramOn(frame3);
      frame3->Draw("e0");

      outRootFile_->cd();
		
      c->Write();

      std::cout << "Finished with fitter - fit results saved to " << fileName << std::endl;

      delete data;
      delete bdata;
   }

   outRootFile_->cd();
   effhist.Write();

   return;
}
// ************************************** //

// ********** Z -> l+l- Fitter ********** //
void TagProbeEDMAnalysis::ZllEffFitter2D( string &fileName, string &bvar1, vector< double > bins1,
					  string &bvar2, vector< double > bins2 )
{
   outRootFile_->cd();
   TChain *nFitTree = new TChain();
   nFitTree->Add((fileName+"/fitter_tree").c_str());

   //return;
   cout << "Here in Zll fitter" << endl;
   
   string hname = "fit_eff_" + bvar1 + "_" + bvar2;
   string htitle = "Efficiency: " + bvar1 + " vs " + bvar2;
   int bnbins1 = bins1.size()-1;
   int bnbins2 = bins2.size()-1;
   cout << "The number of bins is " << bnbins1 << ":" << bnbins2 << endl;
   TH2F effhist(hname.c_str(),htitle.c_str(),bnbins1,&bins1[0],bnbins2,&bins2[0]);

   for( int bin1=0; bin1<bnbins1; ++bin1 )
   {
      for( int bin2=0; bin2<bnbins2; ++bin2 )
      {
	 // The fit variable - lepton invariant mass
	 RooRealVar Mass("Mass","Invariant Di-Lepton Mass", massLow_, massHigh_, "GeV/c^{2}");
	 Mass.setBins(massNbins_);

	 // The binning variables
	 string bunits = "GeV";
	 double lowEdge1 = bins1[bin1];
	 double highEdge1 = bins1[bin1+1];
	 if( bvar1 == "Eta" || bvar1 == "Phi" ) bunits = "";
	 RooRealVar Var1(bvar1.c_str(),bvar1.c_str(),lowEdge1,highEdge1,bunits.c_str());

	 bunits = "GeV";
	 double lowEdge2 = bins2[bin2];
	 double highEdge2 = bins2[bin2+1];
	 if( bvar2 == "Eta" || bvar2 == "Phi" ) bunits = "";
	 RooRealVar Var2(bvar2.c_str(),bvar2.c_str(),lowEdge2,highEdge2,bunits.c_str());

	 // The weighting
	 RooRealVar Weight("Weight","Weight",0.0,10000.0);

	 // Make the category variable that defines the two fits,
	 // namely whether the probe passes or fails the eff criteria.
	 RooCategory ProbePass("ProbePass","sample");
	 ProbePass.defineType("pass",1);
	 ProbePass.defineType("fail",0);  

	 cout << "Made fit variables" << endl;

	 // Add the TTree as our data set ... with the weight in case 
	 // we are using chained MC
	 //RooDataSet* data = new RooDataSet("fitData","fitData",fitTree,
	 //				RooArgSet(ProbePass,Mass,Pt,Weight),"","");
	 // Above command doesn't work in root 5.18 (lovely) so we have this
	 // silly workaround with TChain for now
	 RooDataSet* data = new RooDataSet("fitData","fitData",(TTree*)nFitTree,
					   RooArgSet(ProbePass,Mass,Var1,Var2,Weight));


	 //data->get()->Print();
	 data->setWeightVar("Weight");
	 data->get()->Print();

	 cout << "Made dataset" << endl;

	 RooDataHist *bdata = new RooDataHist("bdata","Binned Data",
					      RooArgList(Mass,ProbePass),*data);
 
	 // ********** Construct signal shape PDF ********** //

	 // Signal PDF variables
	 RooRealVar signalMean("signalMean","signalMean",signalMean_[0]);
	 RooRealVar signalWidth("signalWidth","signalWidth",signalWidth_[0]);
	 RooRealVar signalSigma("signalSigma","signalSigma",signalSigma_[0]);
	 RooRealVar signalWidthL("signalWidthL","signalWidthL",signalWidthL_[0]);
	 RooRealVar signalWidthR("signalWidthR","signalWidthR",signalWidthR_[0]);

	 // If the user has set a range, make the variable float
	 if( signalMean_.size() == 3 )
	 {
	    signalMean.setRange(signalMean_[1],signalMean_[2]);
	    signalMean.setConstant(false);
	 }
	 if( signalWidth_.size() == 3 )
	 {
	    signalWidth.setRange(signalWidth_[1],signalWidth_[2]);
	    signalWidth.setConstant(false);
	 }
	 if( signalSigma_.size() == 3 )
	 {
	    signalSigma.setRange(signalSigma_[1],signalSigma_[2]);
	    signalSigma.setConstant(false);
	 }
	 if( signalWidthL_.size() == 3 )
	 {
	    signalWidthL.setRange(signalWidthL_[1],signalWidthL_[2]);
	    signalWidthL.setConstant(false);
	 }
	 if( signalWidthR_.size() == 3 )
	 {
	    signalWidthR.setRange(signalWidthR_[1],signalWidthR_[2]);
	    signalWidthR.setConstant(false);
	 }
  
	 // Voigtian
	 RooVoigtian signalVoigtPdf("signalVoigtPdf", "signalVoigtPdf", 
				    Mass, signalMean, signalWidth, signalSigma);

	 // Bifurcated Gaussian
	 RooBifurGauss signalGaussBifurPdf("signalGaussBifurPdf", "signalGaussBifurPdf", 
					   Mass, signalMean, signalWidthL, signalWidthR);

	 // Bifurcated Gaussian fraction
	 RooRealVar bifurGaussFrac("bifurGaussFrac","bifurGaussFrac",bifurGaussFrac_[0]);
	 if( bifurGaussFrac_.size() == 3 )
	 {
	    bifurGaussFrac.setRange(bifurGaussFrac_[1],bifurGaussFrac_[2]);
	    bifurGaussFrac.setConstant(false);
	 } 

	 // The total signal PDF
	 RooAddPdf  signalShapePdf("signalShapePdf", "signalShapePdf",
				   signalVoigtPdf,signalGaussBifurPdf,bifurGaussFrac);

	 // ********** Construct background shape PDF ********** //

	 // Background PDF variables
	 RooRealVar bkgAlpha("bkgAlpha","bkgAlpha",bkgAlpha_[0]);
	 RooRealVar bkgBeta("bkgBeta","bkgBeta",bkgBeta_[0]);
	 RooRealVar bkgGamma("bkgGamma","bkgGamma",bkgGamma_[0]);
	 RooRealVar bkgPeak("bkgPeak","bkgPeak",bkgPeak_[0]);

	 // If the user has specified a range, let the bkg shape 
	 // variables float in the fit
	 if( bkgAlpha_.size() == 3 )
	 {
	    bkgAlpha.setRange(bkgAlpha_[1],bkgAlpha_[2]);
	    bkgAlpha.setConstant(false);
	 }
	 if( bkgBeta_.size() == 3 )
	 {
	    bkgBeta.setRange(bkgBeta_[1],bkgBeta_[2]);
	    bkgBeta.setConstant(false);
	 }
	 if( bkgGamma_.size() == 3 )
	 {
	    bkgGamma.setRange(bkgGamma_[1],bkgGamma_[2]);
	    bkgGamma.setConstant(false);
	 }
	 if( bkgPeak_.size() == 3 )
	 {
	    bkgPeak.setRange(bkgPeak_[1],bkgPeak_[2]);
	    bkgPeak.setConstant(false);
	 }

	 // CMS Background shape
	 RooCMSShapePdf bkgShapePdf("bkgShapePdf","bkgShapePdf", 
				    Mass,bkgAlpha,bkgBeta,bkgGamma,bkgPeak);

	 // Now define some efficiency/yield variables  
	 RooRealVar efficiency("efficiency","efficiency",efficiency_[0]);
	 RooRealVar numSignal("numSignal","numSignal",numSignal_[0]);
	 RooRealVar numBkgPass("numBkgPass","numBkgPass",numBkgPass_[0]);
	 RooRealVar numBkgFail("numBkgFail","numBkgFail",numBkgFail_[0]);

	 // If ranges are specifed these are floating variables
	 if( efficiency_.size() == 3 )
	 {
	    efficiency.setRange(efficiency_[1],efficiency_[2]);
	    efficiency.setConstant(false);
	 }
	 if( numSignal_.size() == 3 )
	 {
	    numSignal.setRange(numSignal_[1],numSignal_[2]);
	    numSignal.setConstant(false);
	 }
	 if( numBkgPass_.size() == 3 )
	 {
	    numBkgPass.setRange(numBkgPass_[1],numBkgPass_[2]);
	    numBkgPass.setConstant(false);
	 }
	 if( numBkgFail_.size() == 3 )
	 {
	    numBkgFail.setRange(numBkgFail_[1],numBkgFail_[2]);
	    numBkgFail.setConstant(false);
	 }
      

	 RooFormulaVar numSigPass("numSigPass","numSignal*efficiency", 
				  RooArgList(numSignal,efficiency) );
	 RooFormulaVar numSigFail("numSigFail","numSignal*(1.0 - efficiency)", 
				  RooArgList(numSignal,efficiency) );

	 RooArgList componentspass(signalShapePdf,bkgShapePdf);
	 RooArgList yieldspass(numSigPass, numBkgPass);
	 RooArgList componentsfail(signalShapePdf,bkgShapePdf);
	 RooArgList yieldsfail(numSigFail, numBkgFail);	  

	 RooAddPdf sumpass("sumpass","fixed extended sum pdf",componentspass,yieldspass);
	 RooAddPdf sumfail("sumfail","fixed extended sum pdf",componentsfail, yieldsfail);
  
	 // The total simultaneous fit ...
	 RooSimultaneous totalPdf("totalPdf","totalPdf",ProbePass);
	 ProbePass.setLabel("pass");
	 totalPdf.addPdf(sumpass,ProbePass.getLabel());
	 totalPdf.Print();
	 ProbePass.setLabel("fail");
	 totalPdf.addPdf(sumfail,ProbePass.getLabel());
	 totalPdf.Print();

	 // Count the number of passing and failing probes in the region
	 // making sure we have enough to fit ...
	 cout << "About to count the number of events" << endl;
	 int npassR = (int)data->sumEntries("ProbePass==1");
	 int nfailR = (int)data->sumEntries("ProbePass==0");
	 cout << "Num pass " << npassR << endl;
	 cout << "Num fail " << nfailR << endl;

	 RooAbsCategoryLValue& simCat = (RooAbsCategoryLValue&) totalPdf.indexCat();
   
	 TList* dsetList = const_cast<RooAbsData*>((RooAbsData*)data)->split(simCat);
	 RooCatType* type;
	 TIterator* catIter = simCat.typeIterator();
	 while( (type=(RooCatType*)catIter->Next()) )
	 {
	    // Retrieve the PDF for this simCat state
	    RooAbsPdf* pdf =  totalPdf.getPdf(type->GetName());
	    RooAbsData* dset = (RooAbsData*) dsetList->FindObject(type->GetName());

	    if (pdf && dset && dset->numEntries(kTRUE)!=0.) 
	    {               
	       cout << "GOF Entries " << dset->numEntries() << " " 
		    << type->GetName() << std::endl;
	       if( (string)type->GetName() == "pass" ) 
	       {
		  npassR = dset->numEntries(); 
		  cout << "Pass " << npassR << endl; 
	       }
	       else if( (string)type->GetName() == "fail" ) 
	       {
		  nfailR = dset->numEntries();
		  cout << "Fail " << nfailR << endl; 
	       }
	    }
	 }
	 // End the pass fail counting.

	 // Return if there's nothing to fit
	 if( npassR==0 && nfailR==0 ) return;

	 cout << "**** About to start the fitter ****" << endl;

	 // ********* Do the Actual Fit ********** //  
	 RooFitResult *fitResult = 0;

	 // The user chooses between binnned/unbinned fitting
	 if( unbinnedFit_ )
	 {
	    cout << "Starting unbinned fit using LL." << endl;
	    RooNLLVar nll("nll","nll",totalPdf,*data,kTRUE);
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
	    cout << "Starting binned fit using Chi2." << endl;
	    RooChi2Var chi2("chi2","chi2",totalPdf,*bdata,DataError(RooAbsData::SumW2),Extended(kTRUE));
	    RooMinuit m(chi2);
	    m.setErrorLevel(0.5); // <<< HERE
	    m.setStrategy(2);
	    m.hesse();
	    m.migrad();
	    m.hesse();
	    m.minos();
	    fitResult = m.save();
	 }

	 fitResult->Print("v");

	 std::cout << "Signal yield: " << numSignal.getVal() << " +- "
		   << numSignal.getError() << " + " << numSignal.getAsymErrorHi()
		   <<" - "<< numSignal.getAsymErrorLo() << std::endl;
	 std::cout << "Efficiency: "<< efficiency.getVal() << " +- "
		   << efficiency.getError() << " + " << efficiency.getAsymErrorHi()
		   <<" + "<< efficiency.getAsymErrorLo() << std::endl;

	 // Fill the efficiency hist
	 effhist.SetBinContent(bin1+1,bin2+1,efficiency.getVal());
	 effhist.SetBinError(bin1+1,bin2+1,efficiency.getError());

	 // ********** Make and save Canvas for the plots ********** //
	 //outRootFile_->cd();

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

	 ostringstream oss1;
	 oss1 << bin1;
	 ostringstream oss2;
	 oss2 << bin2;
	 string cname = "fit_canvas_" + bvar1 + "_" + oss1.str() + "_" + bvar2 + "_" + oss2.str();
	 TCanvas *c = new TCanvas(cname.c_str(),"Sum over Modes, Signal Region",1000,1500);
	 c->Divide(1,2);
	 c->cd(1);
	 c->SetFillColor(10);

	 TPad *lhs = (TPad*)gPad;
	 lhs->Divide(2,1);
	 lhs->cd(1);

	 RooPlot* frame1 = Mass.frame();
	 frame1->SetTitle("Passing Tag-Probes");
	 frame1->SetName("pass");
	 data->plotOn(frame1,Cut("ProbePass==1"));
	 ProbePass.setLabel("pass");
	 totalPdf.plotOn(frame1,Slice(ProbePass),ProjWData(Mass,*data));
	 totalPdf.plotOn(frame1,Slice(ProbePass),Components(bkgShapePdf),
			 LineStyle(kDashed),ProjWData(Mass,*data));
	 frame1->Draw("e0");
	 //outRootFile_->cd();

	 lhs->cd(2);
	 RooPlot* frame2 = Mass.frame();
	 frame2->SetTitle("Failing Tag-Probes");
	 frame2->SetName("fail");
	 data->plotOn(frame2,Cut("ProbePass==0"));
	 ProbePass.setLabel("fail");
	 totalPdf.plotOn(frame2,Slice(ProbePass),ProjWData(Mass,*data));
	 totalPdf.plotOn(frame2,Slice(ProbePass),Components(bkgShapePdf),
			 LineStyle(kDashed),ProjWData(Mass,*data));
	 frame2->Draw("e0");
	 //outRootFile_->cd();

	 c->cd(2);
	 RooPlot* frame3 = Mass.frame();
	 frame3->SetTitle("All Tag-Probes");
	 frame3->SetName("total");
	 data->plotOn(frame3);
	 totalPdf.plotOn(frame3,ProjWData(Mass,*data));
	 totalPdf.plotOn(frame3,Components(bkgShapePdf),
			 LineStyle(kDashed),ProjWData(Mass,*data));
	 totalPdf.paramOn(frame3);
	 frame3->Draw("e0");
	 outRootFile_->cd();
		
	 //outRootFile_->cd();
	 c->Write();

	 std::cout << "Finished with fitter - fit results saved to " << fileName << std::endl;

	 delete data;
	 delete bdata;
      }

   }

   outRootFile_->cd();
   effhist.Write();

   return;
}
// ************************************** //


// ********** Get the true efficiency from this TTree ********** //
void TagProbeEDMAnalysis::ZllEffMCTruth()
{
   // Loop over the number of different types of 
   // efficiency measurement in the input tree
   // Make a simple tree for fitting, and then
   // call the fitter.
   cout << "Here in MC truth" << endl;

   outRootFile_->cd();
   cout << "Writing MC Truth Eff hists!" << endl; 

   string hname = "truth_eff_"+var1NameUp_;
   string htitle = "Efficiency vs "+var1NameUp_;
   TH1F var1effhist(hname.c_str(),htitle.c_str(),var1Bins_.size()-1,&var1Bins_[0]);
   var1effhist.Sumw2();
   var1effhist.Divide(var1Pass_,var1All_,1.0,1.0,"B");
   var1effhist.Write();

   outRootFile_->cd();
   hname = "truth_eff_"+var2NameUp_;
   htitle = "Efficiency vs "+var2NameUp_;
   TH1F var2effhist(hname.c_str(),htitle.c_str(),var2Bins_.size()-1,&var2Bins_[0]);
   var2effhist.Sumw2();
   var2effhist.Divide(var2Pass_,var2All_,1.0,1.0,"B");
   var2effhist.Write();

   return;
}
// ******************************************************** //

// ********** Get the true 2D efficiency from this TTree ********** //
void TagProbeEDMAnalysis::ZllEffMCTruth2D()
{
   // Loop over the number of different types of 
   // efficiency measurement in the input tree
   // Make a simple tree for fitting, and then
   // call the fitter.
   cout << "Here in MC truth" << endl;

   outRootFile_->cd();
   cout << "Writing MC Truth Eff hists!" << endl; 

   string hname = "truth_eff_"+var1NameUp_+"_"+var2NameUp_;
   string htitle = "Efficiency: "+var1NameUp_+" vs "+var2NameUp_;
   TH2F var1var2effhist(hname.c_str(),htitle.c_str(),var1Bins_.size()-1,&var1Bins_[0],
		     var2Bins_.size()-1,&var2Bins_[0]);
   var1var2effhist.Sumw2();
   var1var2effhist.Divide(var1var2Pass_,var1var2All_,1.0,1.0,"B");
   var1var2effhist.Write();

   return;
}
// ******************************************************** //

// ********** Get the efficiency from this TTree ********** //
void TagProbeEDMAnalysis::CalculateEfficiencies()
{

   if( calcEffsTruth_ ) 
   {
      ZllEffMCTruth();

      // 2D MC Truth
      if( do2DFit_ ) ZllEffMCTruth2D();
   }

   if( calcEffsFitter_ || calcEffsSB_ )
   {
      cout << "Entries in fit tree ... " << fitTree_->GetEntries() << endl;
      fitTree_->Write();

      cout << "There are " << var1Bins_.size()-1 << " " << var1NameUp_ << " bins." << endl;
      int nbins1 = var1Bins_.size()-1;
      int nbins2 = var2Bins_.size()-1;

      if( calcEffsFitter_ )
      {
	 // We have filled the simple tree ... call the fitter
	 ZllEffFitter( fitFileName_, var1NameUp_, var1Bins_, var2NameUp_, var2Bins_[0], var2Bins_[nbins2] );
	 ZllEffFitter( fitFileName_, var2NameUp_, var2Bins_, var1NameUp_, var1Bins_[0], var1Bins_[nbins1] );

	 // 2D Fit
	 if( do2DFit_ )
	 {
	    ZllEffFitter2D( fitFileName_, var1NameUp_, var1Bins_, var2NameUp_, var2Bins_ );
	 }
      }

      if( calcEffsSB_ )
      {
	 // We have filled the simple tree ... call side band subtraction
	 ZllEffSBS(  fitFileName_, var1NameUp_, var1Bins_, var2NameUp_, var2Bins_[0], var2Bins_[nbins2] );
	 ZllEffSBS(  fitFileName_, var2NameUp_, var2Bins_, var1NameUp_, var1Bins_[0], var1Bins_[nbins1] );

	 // 2D SBS
	 if( do2DFit_ )
	 {
	    ZllEffSBS2D( fitFileName_, var1NameUp_, var1Bins_, var2NameUp_, var2Bins_ );
	 }
      }
   }

   return;
}
// ******************************************************** //


// ------------ method called once each job just before starting event loop  ------------
void 
TagProbeEDMAnalysis::beginJob(const edm::EventSetup&)
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
TagProbeEDMAnalysis::endJob() 
{
   // Check for the various modes ...
   if( mode_ == "Write" )
   {
      // All we need to do is write out the truth histograms and fitTree
      outRootFile_->cd();

      cout << "Fit tree has " << fitTree_->GetEntries() << " entries." << endl;
      //fitTree_->SetDirectory(outRootFile_);
      fitTree_->Write();

      //var1Pass_->SetDirectory(outRootFile_);
      var1Pass_->Write();
      //var1All_->SetDirectory(outRootFile_);
      var1All_->Write();  
      
      //var2Pass_->SetDirectory(outRootFile_);
      var2Pass_->Write();
      //var2All_->SetDirectory(outRootFile_);
      var2All_->Write();  
      
      //var1var2Pass_->SetDirectory(outRootFile_);
      var1var2Pass_->Write();
      //var1var2All_->SetDirectory(outRootFile_);
      var1var2All_->Write();

      outRootFile_->Close();
      cout << "Closed ROOT file and returning!" << endl;

      return;
   }

   if( mode_ == "Normal" )
   {
      // Save plots ...
      for( int i=0; i<(int)quantities_.size(); ++i )
      {
	 SaveHistogram(Histograms_[i], outputFileNames_[i], logY_[i]);
      }
 
      // Calculate the efficiencies etc ...
      outRootFile_->cd();
      CalculateEfficiencies();  
      outRootFile_->Close();

      return;
   }

   cout << "Here in endjob " << readFiles_.size() << endl;
   if( mode_ == "Read" && readFiles_.size() > 0 )
   {
      cout << "Here in end job: Num files = " << readFiles_.size() << endl;

      // For the fittree chain the files together, then merge the
      // trees from the chain into the fitTree_ ...
      TChain fChain("fitter_tree");
      for( int iFile=0; iFile<(int)readFiles_.size(); ++iFile )
      {
	 cout << "fChain adding: " << readFiles_[iFile].c_str() << endl;
	 fChain.Add(readFiles_[iFile].c_str());
      }
      cout << "Added all files: Num Entries = " << fChain.GetEntries() << endl;

      // Now merge the trees into the output file ...
      fChain.Merge(fitFileName_.c_str());

      // Get the private tree ...
      TFile f(fitFileName_.c_str(),"update");
      fitTree_ = (TTree*)f.Get("fitter_tree");
      cout << "Read mode: Fit tree total entries " << fitTree_->GetEntries() << endl;

      // Now read in the MC truth histograms and add the results
      for( int iFile=0; iFile<(int)readFiles_.size(); ++iFile )
      {
	 TFile inputFile(readFiles_[iFile].c_str());

	 var1Pass_->Add( (TH1F*)inputFile.Get("hvar1pass") );
	 var1All_->Add( (TH1F*)inputFile.Get("hvar1all") );  
	 
	 var2Pass_->Add( (TH1F*)inputFile.Get("hvar2pass") );
	 var2All_->Add( (TH1F*)inputFile.Get("hvar2all") );  
	 
	 var1var2Pass_->Add( (TH2F*)inputFile.Get("hvar1var2pass") );
	 var1var2All_->Add( (TH2F*)inputFile.Get("hvar1var2all") );
      }
      
      // Now call for and calculate the efficiencies as normal
      // Set the file pointer
      outRootFile_ = &f;
      outRootFile_->cd();
      CalculateEfficiencies();  
      outRootFile_->Close();

      return;
   }

}

//define this as a plug-in
DEFINE_FWK_MODULE( TagProbeEDMAnalysis );

