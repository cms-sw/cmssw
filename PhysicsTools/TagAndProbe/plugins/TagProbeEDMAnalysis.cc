// -*- C++ -*-
//
// Package:    TagProbeEDMAnalysis
// Class:      TagProbeEDMAnalysis
// 
/**\class TagProbeEDMAnalysis TagProbeEDMAnalysis.cc PhysicsTools/TagProbeEDMAnalysis/src/TagProbeEDMAnalysis.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*///
// Original Author:  "Nadia Adam"
//         Created:  Sun Apr 20 10:35:25 CDT 2008
//

#include "PhysicsTools/TagAndProbe/interface/TagProbeEDMAnalysis.h"

// TP Utilities
#include "PhysicsTools/TagAndProbe/interface/FCIntervals.hh"
#include "PhysicsTools/TagAndProbe/interface/EffTableLoader.h"
#include "PhysicsTools/TagAndProbe/interface/SideBandSubtraction.hh"
#include "PhysicsTools/TagAndProbe/interface/TPRooSimultaneousFitter.hh"

// Line Shapes
#include "PhysicsTools/TagAndProbe/interface/ZLineShape.hh"
#include "PhysicsTools/TagAndProbe/interface/CBLineShape.hh"
#include "PhysicsTools/TagAndProbe/interface/GaussianLineShape.hh"
#include "PhysicsTools/TagAndProbe/interface/PolynomialLineShape.hh"
#include "PhysicsTools/TagAndProbe/interface/CMSBkgLineShape.hh"

// CMSSW
#include "FWCore/Framework/interface/MakerMacros.h"  // DEFINE_FWK_MODULE
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h" // edm::ParameterSet

// RooFit - clean headers only
#include <RooDataSet.h>
#include <RooCategory.h>
#include <RooCatType.h>
#include <RooFitResult.h>
#include <RooMinuit.h>
#include <RooPlot.h>
#include <RooRealVar.h>
#include <RooTreeData.h>

// ROOT
#include <TROOT.h>  // gROOT
#include <TChain.h>
#include <TCanvas.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TGraphAsymmErrors.h>
#include <TStyle.h>
#include <TMath.h> // Gamma

// STL
#include <vector>
#include <string>

TagProbeEDMAnalysis::TagProbeEDMAnalysis (const edm::ParameterSet& iConfig): 
  effBinsFromTxt_(0),SBS_(0),
  zLineShape_(0), cbLineShape_(0), gaussLineShape_(0),
  polyBkgLineShape_(0),cmsBkgLineShape_(0), signalShapePdf_(0),
  var1Pass_(0), var1All_(0),
  var2Pass_(0), var2All_(0),
  var1var2Pass_(0), var1var2All_(0) {

  // Efficiency input variables
  tagProbeType_   = iConfig.getUntrackedParameter< int >("TagProbeType",0);
  
  calcEffsSB_     = iConfig.getUntrackedParameter< bool >("CalculateEffSideBand",false);
  calcEffsFitter_ = iConfig.getUntrackedParameter< bool >("CalculateEffFitter",false);
  calcEffsTruth_  = iConfig.getUntrackedParameter< bool >("CalculateEffTruth",false);
  
  truthParentId_  = iConfig.getUntrackedParameter< int >("MCTruthParentId");
  
  // Type of fit
  unbinnedFit_    = iConfig.getUntrackedParameter< bool >("UnbinnedFit",false);
  do2DFit_        = iConfig.getUntrackedParameter< bool >("Do2DFit",false);
  
  // This gives the option to use bins from a file. 
  doTextDefinedBins_ = iConfig.getUntrackedParameter< bool >("DoBinsFromTxt",false);
  if (doTextDefinedBins_) {
    textBinsFile_ = iConfig.getUntrackedParameter< std::string >("EffBinsFile",
								 "EffBinsFile.txt");
    effBinsFromTxt_ = new EffTableLoader(textBinsFile_);
    var1Bins_.clear();
    unsigned int nBins = effBinsFromTxt_->size();
    for (unsigned int iBin=0; iBin<=nBins; ++iBin) {
      var1Bins_.push_back(iBin);
    }
  }
  
  // If want to use reconstructed or detector values (instead of MC generated values) 
  // of var1 and var2 when doing MC truth efficiencies (i.e., when "calcEffsTruth==true").
  useRecoVarsForTruthMatchedCands_ = 
    iConfig.getUntrackedParameter< bool >("useRecoVarsForTruthMatchedCands",true);
  
  CheckEfficiencyVariables ();
  
  var1NameUp_ = var1Name_;
  std::toupper(var1NameUp_.at(0));
  
  var2NameUp_ = var2Name_;
  std::toupper(var2NameUp_.at(0));
  
  // Set up the bins for the eff histograms ...   
  if (var1Bins_.size() == 0) {
    // User didn't set bin boundaries, so use even binning
    double bwidth = (var1High_-var1Low_)/static_cast<double>(var1Nbins_);
    for (unsigned int iBin=0; iBin<=var1Nbins_; ++iBin) {
      double low_edge = var1Low_+ static_cast<double>(iBin)*bwidth;
      var1Bins_.push_back(low_edge);
    }
  }

  if (var2Bins_.size() == 0) {
    // User didn't set bin boundaries, so use even binning
    double bwidth = (var2High_-var2Low_)/static_cast<double>(var2Nbins_);
    for (unsigned int iBin=0; iBin<=var2Nbins_; ++iBin) {
      double low_edge = var2Low_+static_cast<double>(iBin)*bwidth;
      var2Bins_.push_back(low_edge);
    }
  }

  const edm::ParameterSet dummyPSet;

  // Fitter variables
  massNbins_      = iConfig.getUntrackedParameter< int >("NumBinsMass");
  massLow_        = iConfig.getUntrackedParameter< double >("MassLow");
  massHigh_       = iConfig.getUntrackedParameter< double >("MassHigh");
  
  rooMass_ = new RooRealVar("Mass",
			    "Invariant Di-Lepton Mass", 
			    massLow_, 
			    massHigh_, 
			    "GeV/c^{2}");
  rooMass_->setBins(massNbins_);
  
  edm::ParameterSet ZLineShapePSet
    = iConfig.getUntrackedParameter< edm::ParameterSet >("ZLineShape", dummyPSet);
  if (!ZLineShapePSet.empty()) {
    zLineShape_ = new ZLineShape(ZLineShapePSet, rooMass_);
  }
  
  edm::ParameterSet CBLineShapePSet 
    = iConfig.getUntrackedParameter< edm::ParameterSet >("CBLineShape",dummyPSet);
  if (!CBLineShapePSet.empty()) {
    cbLineShape_ = new CBLineShape(CBLineShapePSet, rooMass_);
  }
  
  edm::ParameterSet GaussLineShapePSet 
    = iConfig.getUntrackedParameter< edm::ParameterSet >("GaussLineShape",dummyPSet);
  if (!GaussLineShapePSet.empty()) {
    gaussLineShape_ = new GaussianLineShape(GaussLineShapePSet, rooMass_);
  }
  
  edm::ParameterSet PolyLineShapePSet 
    = iConfig.getUntrackedParameter< edm::ParameterSet >("PolynomialLineShape",dummyPSet);
  if (!PolyLineShapePSet.empty()) {
    polyBkgLineShape_ = new PolynomialLineShape(PolyLineShapePSet, rooMass_);
  }
  
  edm::ParameterSet CMSBkgLineShapePSet 
    = iConfig.getUntrackedParameter< edm::ParameterSet >("CMSBkgLineShape",dummyPSet);
  if (!CMSBkgLineShapePSet.empty()) {
    cmsBkgLineShape_ = new CMSBkgLineShape(CMSBkgLineShapePSet, rooMass_);
  }
  
  // Should not have defaults!
  efficiency_ = iConfig.getUntrackedParameter< std::vector<double> >("Efficiency");
  numSignal_  = iConfig.getUntrackedParameter< std::vector<double> >("NumSignal");
  numBkgPass_ = iConfig.getUntrackedParameter< std::vector<double> >("NumBkgPass");
  numBkgFail_ = iConfig.getUntrackedParameter< std::vector<double> >("NumBkgFail");
  
  // Get the mode of operation variables
  mode_        = iConfig.getUntrackedParameter< std::string >("Mode","Normal");
  fitFileName_ = iConfig.getUntrackedParameter< std::string >("FitFileName","fitfile.root");
  
  std::vector< std::string > dReadFiles;
  readFiles_ = iConfig.getUntrackedParameter< std::vector<std::string> >("ReadFromFiles",dReadFiles);
  
  inweight_ = iConfig.getUntrackedParameter< double >("Weight",1.0); 

  std::vector< double > dBins;
  var1Name_  = iConfig.getUntrackedParameter< std::string >("NameVar1","pt");
  var1Bins_  = iConfig.getUntrackedParameter< std::vector<double> >("Var1BinBoundaries",dBins);
  if (var1Bins_.size() == 0) {  // bins are not specified, bins must be equally spaced.
    var1Nbins_ = iConfig.getUntrackedParameter< unsigned int >("NumBinsVar1");
    var1Low_   = iConfig.getUntrackedParameter< double >("Var1Low");
    var1High_  = iConfig.getUntrackedParameter< double >("Var1High");
  }
  
  var2Name_  = iConfig.getUntrackedParameter< std::string >("NameVar2","eta");
  var2Bins_  = iConfig.getUntrackedParameter< std::vector<double> >("Var2BinBoundaries",dBins);
  if (var2Bins_.size() == 0) {  // bins are not specified, bins must be equally spaced.
    var2Nbins_ = iConfig.getUntrackedParameter< unsigned int >("NumBinsVar2");
    var2Low_   = iConfig.getUntrackedParameter< double >("Var2Low");
    var2High_  = iConfig.getUntrackedParameter< double >("Var2High");
  }

  CreateFitTree();
  InitializeMCHistograms();
}

void TagProbeEDMAnalysis::CheckEfficiencyVariables () {

  // Check that the names of the variables are okay ...
  if (!(var1Name_ == "pt" || var1Name_ == "p"   || var1Name_ == "px" ||
	var1Name_ == "py" || var1Name_ == "pz"  || var1Name_ == "e"  ||
	var1Name_ == "et" || var1Name_ == "eta" || var1Name_ == "phi" || 
	var1Name_ == "ptDet" || var1Name_ == "pDet"   || var1Name_ == "pxDet" ||
	var1Name_ == "pyDet" || var1Name_ == "pzDet"  || var1Name_ == "eDet"  ||
	var1Name_ == "etDet" || var1Name_ == "etaDet" || var1Name_ == "phiDet" ||
	var1Name_ == "jetDeltaR" || var1Name_ == "totJets")) {
    edm::LogWarning("TagAndProbe") << "Warning: Var1 name invalid, setting var1 name to pt!";
    var1Name_ = "pt";
  }
  
  if (!(var2Name_ == "pt" || var2Name_ == "p"   || var2Name_ == "px" ||
	var2Name_ == "py" || var2Name_ == "pz"  || var2Name_ == "e"  ||
	var2Name_ == "et" || var2Name_ == "eta" || var2Name_ == "phi" ||
	var2Name_ == "ptDet" || var2Name_ == "pDet"   || var2Name_ == "pxDet" ||
	var2Name_ == "pyDet" || var2Name_ == "pzDet"  || var2Name_ == "eDet"  ||
	var2Name_ == "etDet" || var2Name_ == "etaDet" || var2Name_ == "phiDet" ||
	var2Name_ == "jetDeltaR" || var2Name_ == "totJets")) {
    edm::LogWarning("TagAndProbe") << "Warning: Var2 name invalid, setting var2 name to eta!";
    var2Name_ = "eta";
  }
}

void TagProbeEDMAnalysis::InitializeMCHistograms(){

   // MC Truth Histograms
   var1Pass_ = fs->make<TH1F>("hvar1pass","Var1 Pass",
			      var1Bins_.size()-1,&var1Bins_[0]);
   var1All_  = fs->make<TH1F>("hvar1all","Var1 All",
			      var1Bins_.size()-1,&var1Bins_[0]);
   
   var2Pass_ = fs->make<TH1F>("hvar2pass","Var2 Pass",
			      var2Bins_.size()-1,&var2Bins_[0]);
   var2All_  = fs->make<TH1F>("hvar2all","Var2 All",
			      var2Bins_.size()-1,&var2Bins_[0]);
   
   var1var2Pass_ = fs->make<TH2F>("hvar1var2pass","Var1:Var2 Pass",
				  var1Bins_.size()-1,&var1Bins_[0],
				  var2Bins_.size()-1,&var2Bins_[0]);
   var1var2All_  = fs->make<TH2F>("hvar1var2all","Var1:Var2 All",
				  var1Bins_.size()-1,&var1Bins_[0],
				  var2Bins_.size()-1,&var2Bins_[0]);
}

void TagProbeEDMAnalysis::ReadMCHistograms(){

  // Now read in the MC truth histograms and add the results
  for (std::vector< std::string >::iterator iFile = readFiles_.begin();
       iFile != readFiles_.end(); ++iFile) {
    TFile inputFile(iFile->c_str());
    
    var1Pass_->Add( static_cast<TH1F*>(inputFile.Get("hvar1pass")) );
    var1All_->Add( static_cast<TH1F*>(inputFile.Get("hvar1all")) );  
    
    var2Pass_->Add( static_cast<TH1F*>(inputFile.Get("hvar2pass")) );
    var2All_->Add( static_cast<TH1F*>(inputFile.Get("hvar2all")) );  
    
    var1var2Pass_->Add( static_cast<TH2F*>(inputFile.Get("hvar1var2pass")) );
    var1var2All_->Add( static_cast<TH1F*>(inputFile.Get("hvar1var2all")) );
  }
}

void TagProbeEDMAnalysis::WriteMCHistograms() {
  var1Pass_->Write();
  var1All_->Write();  
  
  var2Pass_->Write();
  var2All_->Write();  
  
  var1var2Pass_->Write();
  var1var2All_->Write();
}

void TagProbeEDMAnalysis::CreateFitTree() {

  if (mode_ != "Read") {
//     std::string fmode = "RECREATE";
//     outRootFile_ = new TFile(fitFileName_.c_str(),fmode.c_str());
//     outRootFile_->cd();
    
    fitTree_ = fs->make<TTree>("fitter_tree","Tree For Fitting",1);
    fitTree_->Branch("ProbePass",         &ProbePass_,"ProbePass/I");
    fitTree_->Branch("Mass",              &Mass_,     "Mass/D");
    fitTree_->Branch("Weight",            &Weight_,   "Weight/D");
    fitTree_->Branch(var1NameUp_.c_str(), &Var1_,     
		     (var1NameUp_+"/D").c_str());
    fitTree_->Branch(var2NameUp_.c_str(), &Var2_,     
		     (var2NameUp_+"/D").c_str());   
  }
}

void TagProbeEDMAnalysis::FillFitTree(const edm::Event& iEvent){
  
  edm::Handle< std::vector<int> > tp_type;
  if ( !iEvent.getByLabel("TPEdm","TPtype",tp_type) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No TPtype in Tree!"; 
  }
  
  edm::Handle< std::vector<int> > tp_true;
  if ( !iEvent.getByLabel("TPEdm","TPtrue",tp_true) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No TPtrue in Tree!"; 
  }
  
  edm::Handle< std::vector<int> > tp_ppass;
  if ( !iEvent.getByLabel("TPEdm","TPppass",tp_ppass) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No TPppass in Tree!"; 
  }
  
  edm::Handle< std::vector<float> > tp_mass;
  if ( !iEvent.getByLabel("TPEdm","TPmass",tp_mass) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No TPmass in Tree!"; 
  }
  
  edm::Handle< std::vector<float> > tp_probe_var1;
  if ( !iEvent.getByLabel("TPEdm",("TPProbe"+var1Name_).c_str(),tp_probe_var1) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No TPProbe"+var1Name_+" in Tree!"; 
  }
  
  edm::Handle< std::vector<float> > tp_probe_var2;
  if ( !iEvent.getByLabel("TPEdm",("TPProbe"+var2Name_).c_str(),tp_probe_var2) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No TPProbe"+var2Name_+" in Tree!"; 
  }
  
  //  outRootFile_->cd();
  
  if( tp_type.isValid() ) {
    unsigned int nrTP = tp_type->size();
    for (unsigned int iTP=0; iTP<nrTP; ++iTP) {
      if( (*tp_type)[iTP] != tagProbeType_ ) continue;
      
      Weight_ = inweight_;
      ProbePass_ = (*tp_ppass)[iTP];
      // Do these values really need to be doubles?
      Mass_ = static_cast<double>((*tp_mass)[iTP]);
      Var1_ = static_cast<double>((*tp_probe_var1)[iTP]);
      Var2_ = static_cast<double>((*tp_probe_var2)[iTP]);
      fitTree_->Fill();
    }
  }      
}


void TagProbeEDMAnalysis::CreateMCTree() {

   if (mode_ != "Read") {
//       std::string fmode = "RECREATE";
//       outRootFile_ = new TFile(fitFileName_.c_str(),fmode.c_str());
//       outRootFile_->cd();

      mcTree_ = new TTree("fitter_tree","Tree For Fitting",1);
      mcTree_->Branch("ProbePass",         &ProbePass_,"ProbePass/I");
      mcTree_->Branch("Mass",              &Mass_,     "Mass/D");
      mcTree_->Branch("Weight",            &Weight_,   "Weight/D");
      mcTree_->Branch(("MC"+var1NameUp_).c_str(), &MCVar1_,     
		       (var1NameUp_+"/D").c_str());
      mcTree_->Branch(("MC"+var2NameUp_).c_str(), &MCVar2_,     
		       (var2NameUp_+"/D").c_str());   

   }
}

void TagProbeEDMAnalysis::FillMCTree(const edm::Event& iEvent){
 
  // Get the Cnd variable for the MC truth ...
  edm::Handle< std::vector<int> > cnd_type;
  if ( !iEvent.getByLabel("TPEdm","Cndtype",cnd_type) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndtype in Tree!"; 
  }
  
  edm::Handle< std::vector<int> > cnd_tag;
  if ( !iEvent.getByLabel("TPEdm","Cndtag",cnd_tag) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndtag in Tree!"; 
  }
  
  edm::Handle< std::vector<int> > cnd_aprobe;
  if ( !iEvent.getByLabel("TPEdm","Cndaprobe",cnd_aprobe) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndaprobe in Tree!"; 
  }
  
  edm::Handle< std::vector<int> > cnd_pprobe;
  if ( !iEvent.getByLabel("TPEdm","Cndpprobe",cnd_pprobe) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndpprobe in Tree!"; 
  }
  
  edm::Handle< std::vector<int> > cnd_moid;
  if ( !iEvent.getByLabel("TPEdm","Cndmoid",cnd_moid) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndmoid in Tree!"; 
  }
  
  edm::Handle< std::vector<int> > cnd_gmid;
  if ( !iEvent.getByLabel("TPEdm","Cndgmid",cnd_gmid) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndgmid in Tree!"; 
  }
  
  std::string truthVar1 = var1Name_.c_str();
  std::string truthVar2 = var2Name_.c_str();
  
  if(!useRecoVarsForTruthMatchedCands_) {
    truthVar1.insert(0, "Cnd");
    truthVar2.insert(0, "Cnd");
  } else {
    truthVar1.insert(0, "Cndr");
    truthVar2.insert(0, "Cndr");
  }
  
  edm::Handle< std::vector<float> > cnd_var1;
  if ( !iEvent.getByLabel("TPEdm",truthVar1.c_str(),cnd_var1) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cnd"+var1Name_+" in Tree!"; 
  }
  
  edm::Handle< std::vector<float> > cnd_var2;
  if ( !iEvent.getByLabel("TPEdm",truthVar2.c_str(),cnd_var2) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cnd"+var2Name_+" in Tree!"; 
  }
  
  edm::Handle< std::vector<float> > cnd_rpx;
  if ( !iEvent.getByLabel("TPEdm","Cndrpx",cnd_rpx) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndrpx in Tree!"; 
  }
  
  edm::Handle< std::vector<float> > cnd_rpy;
  if ( !iEvent.getByLabel("TPEdm","Cndrpy",cnd_rpy) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndrpy in Tree!"; 
  }
  
  edm::Handle< std::vector<float> > cnd_rpz;
  if ( !iEvent.getByLabel("TPEdm","Cndrpz",cnd_rpz) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndrpz in Tree!"; 
  }
  
  edm::Handle< std::vector<float> > cnd_re;
  if ( !iEvent.getByLabel("TPEdm","Cndre",cnd_re) ) {
    edm::LogInfo("TagProbeEDMAnalysis") << "No Cndre in Tree!"; 
  }
  
  if (cnd_type.isValid()) {
    int nTag = 0;
    int nMatch = 0;
    float px = 0;
    float py = 0;
    float pz = 0;
    float e = 0;
    unsigned int nCnd = cnd_type->size();
    for (unsigned int iCnd=0; iCnd<nCnd; ++iCnd) {
      if( (*cnd_type)[iCnd] != tagProbeType_ ) continue;
      if( truthParentId_ != 0 &&
	  !( fabs((*cnd_gmid)[iCnd]) == truthParentId_ || fabs((*cnd_moid)[iCnd]) == truthParentId_ ) ) continue;
      
      if( (*cnd_tag)[iCnd] == 1 ) ++nTag;
      if( (*cnd_tag)[iCnd] == 1 || (*cnd_aprobe)[iCnd] == 1) {
	++nMatch;
	px += (*cnd_rpx)[iCnd];
	py += (*cnd_rpy)[iCnd];
	pz += (*cnd_rpz)[iCnd];
	e += (*cnd_re)[iCnd];
      }
    }
    
    if( nTag >= 1 && nMatch == 2) {
      float invMass = sqrt(e*e - px*px - py*py - pz*pz);
      if (invMass > massLow_ && invMass < massHigh_) {
	for (unsigned int iCnd=0; iCnd<nCnd; ++iCnd)  {
	  if ((*cnd_type)[iCnd] != tagProbeType_) continue;
	  
	  if ( !(truthParentId_ == 0 || 
		 fabs((*cnd_gmid)[iCnd]) == truthParentId_ || 
		 fabs((*cnd_moid)[iCnd]) == truthParentId_ ) ) continue;
	  
	  // If there is only one tag, only count the probe
	  bool FillHists = true;
	  if( nTag==1 && (*cnd_tag)[iCnd]==1 ) FillHists = false;
	  
	  bool inVar1Range = false;
	  if( (*cnd_var1)[iCnd] > var1Bins_[0] &&
	      (*cnd_var1)[iCnd] < var1Bins_[var1Bins_.size()-1] )
	    inVar1Range = true;
	  bool inVar2Range = false;
	  if( (*cnd_var2)[iCnd] > var2Bins_[0] &&
	      (*cnd_var2)[iCnd] < var2Bins_[var2Bins_.size()-1] )
	    inVar2Range = true;
	  
	  if ((*cnd_aprobe)[iCnd] == 1 && (*cnd_pprobe)[iCnd] == 1) {
	    if (inVar2Range) var1Pass_->Fill((*cnd_var1)[iCnd]);
	    if (inVar1Range) var2Pass_->Fill((*cnd_var2)[iCnd]);
	    var1var2Pass_->Fill((*cnd_var1)[iCnd],(*cnd_var2)[iCnd]);
	    
	    if (FillHists) {
	      if( inVar2Range ) var1Pass_->Fill((*cnd_var1)[iCnd]);
	      if( inVar1Range ) var2Pass_->Fill((*cnd_var2)[iCnd]);
	      var1var2Pass_->Fill((*cnd_var1)[iCnd],(*cnd_var2)[iCnd]);
	    }
	  }
	  
	  if ((*cnd_aprobe)[iCnd] == 1) {
	    if (inVar2Range) var1All_->Fill((*cnd_var1)[iCnd]);
	    if (inVar1Range) var2All_->Fill((*cnd_var2)[iCnd]);
	    var1var2All_->Fill((*cnd_var1)[iCnd],(*cnd_var2)[iCnd]);
	    
	    if (FillHists) {
	      if( inVar2Range ) var1All_->Fill((*cnd_var1)[iCnd]);
	      if( inVar1Range ) var2All_->Fill((*cnd_var2)[iCnd]);
	      var1var2All_->Fill((*cnd_var1)[iCnd],(*cnd_var2)[iCnd]);
	    }
	  }
	}
      }
    }
  }
  
  //  outRootFile_->cd();
    
      //      MCVar1_ = static_cast<double>((*cnd_type) );
      //MCVar2_ = static_cast<double>( );
      //MCMass_ = static_cast<double>( );

}

// ------------ method called to for each event  ------------

void TagProbeEDMAnalysis::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup) {

   // Safety check .. if mode = read, the user should use an empty source.
   if (mode_ == "Read") return;

   // Fill the fit tree if we are fitting or doing SB subtraction
   if( calcEffsSB_ || calcEffsFitter_ || mode_ == "Write" )
   {
      // Get the TP variables to fill the fitter tree ...
     FillFitTree(iEvent);
   }

   // Fill the MC truth information if required
   if (calcEffsTruth_ || mode_ == "Write") {
     FillMCTree(iEvent);
   }
}

// ****************** TP Eff Side band subtraction *************
void TagProbeEDMAnalysis::TPEffSBS (std::string &fileName, std::string &bvar, 
				    std::vector< double > bins, std::string &bvar2, 
				    double bvar2Lo, double bvar2Hi ) {
  
  if (bins.size() < 2) return;
  unsigned int bnbins = bins.size() - 1;
  
  //  outRootFile_->cd();
  //fitTree_->SetDirectory(outRootFile_);
  
  edm::LogInfo("TagProbeEDMAnalysis") << "***** Here in TP sideband subtraction ******";
  edm::LogInfo("TagProbeEDMAnalysis") << "Number of entries " << fitTree_->GetEntries();
  
  // Here I will just change the names if we are using 1D regions by reading 
  // in the efficiencies from a file.  
  std::string bvard = bvar;
  if ( doTextDefinedBins_ ) bvard = bvar + "_" + bvar2;
  
  std::string hdname = "sbs_den_" + bvard; 
  std::string hdtitle = "SBS Denominator vs " + bvard; 
  TH1F* denhist = fs->make<TH1F>(hdname.c_str(), hdtitle.c_str(), bnbins, &bins[0]);
  
  std::ostringstream histoName;
  std::ostringstream histoTitle;;
  
  std::ostringstream condition;
  condition << "(" << bvar2 << ">" << bvar2Lo << ") && (" << bvar2 << "<" << bvar2Hi << ")";
  const std::string bvar2Cond = condition.str();
  
  TGraphAsymmErrors* effhist = fs->make<TGraphAsymmErrors>(bnbins);
  std::string hname = "sbs_eff_" + bvard;
  std::string htitle = "SBS Efficiency vs " + bvard;
  effhist->SetNameTitle(hname.c_str(), htitle.c_str());
  
  for (unsigned int iBin=0; iBin<bnbins; ++iBin ) {
    // Limit the scope of these variables and speed up by declaring them static.
    static const std::string passCond = "(ProbePass==1)";
    static const std::string failCond = "(ProbePass==0)";
    static std::ostringstream bvar1Cond;
    static double lowEdge;
    static double highEdge;
    static std::string bunits = "";

    if( bvar == "Pt" ) bunits = "GeV";
    if ( doTextDefinedBins_ ) {
      bunits = "";
      std::vector<std::pair<float, float> > bininfo = effBinsFromTxt_->GetCellInfo(iBin);
      lowEdge  = bininfo[0].first;
      highEdge = bininfo[0].second;
      bvar2Lo  = bininfo[1].first;
      bvar2Hi  = bininfo[1].second;
      edm::LogInfo("TagProbeEDMAnalysis") << " Bin " << iBin << ", lowEdge " << lowEdge << 
	", highEdge " << highEdge << ", bvar2Lo " << bvar2Lo << ", bvar2Hi " << bvar2Hi << std::endl;
    }

    lowEdge = bins[iBin];
    highEdge = bins[iBin+1];
    bvar1Cond.str(std::string());
    bvar1Cond << bvar << ">" << lowEdge << " && " << bvar << "<" << highEdge;

    // The binning variable
    ///////////// HERE I NEED TO CHANGE THESE VALUES
    // Print out the pass/fail condition
    std::ostringstream DisplayCondition;
    DisplayCondition.str(std::string());
    DisplayCondition << "(ProbePass==1(0) && " << bvar1Cond << " && " << bvar2Cond <<")*Weight";
    edm::LogInfo("TagProbeEDMAnalysis") << "Pass(Fail) condition[" << bvar<< "]: " << 
      DisplayCondition.str();
    
    // Passing Probes
    condition.str(std::string());
    condition  << "(" << passCond << " && " << bvar1Cond << " && " << bvar2Cond <<")*Weight";
    histoName.str(std::string());
    histoName << "sbs_pass_" << bvard << "_" << iBin;
    histoTitle.str(std::string());
    histoTitle << "Passing Probes - " << lowEdge << "<" << bvar << 
      "<" << highEdge;
    TH1F* PassProbes = fs->make<TH1F>(histoName.str().c_str(), 
				      histoTitle.str().c_str(), massNbins_, 
				      massLow_, massHigh_); 
    PassProbes->Sumw2();
    //    PassProbes.SetDirectory(outRootFile_);
    fitTree_->Draw(("Mass >> " + histoName.str()).c_str(), 
		   condition.str().c_str() );
    
    // Failing Probes
    condition.str(std::string());
    condition  << "(" << failCond << " && " << bvar1Cond << " && " << bvar2Cond <<")*Weight";
    histoName.str(std::string());
    histoName << "sbs_fail_" <<  bvard << "_" << iBin;
    histoTitle.str(std::string());
    histoTitle << "Failing Probes - " << lowEdge << "<" << bvar << 
      "<" << highEdge;
    TH1F* FailProbes = fs->make<TH1F>(histoName.str().c_str(), 
				      histoTitle.str().c_str(), 
				      massNbins_, massLow_, massHigh_); 
    FailProbes->Sumw2();
    //    FailProbes->SetDirectory(outRootFile_);
    fitTree_->Draw(("Mass >> " + histoName.str()).c_str(), 
		   condition.str().c_str());
    
    // SBS Passing  Probes
    histoName.str(std::string());
    histoName << "sbs_pass_subtracted_" << bvard << "_" << iBin;
    histoTitle.str(std::string());
    histoTitle << "Passing Probes SBS - "  << lowEdge << "<" << 
      bvar << "<" << highEdge;
    TH1F* SBSPassProbes = fs->make<TH1F>(histoName.str().c_str(), 
					 histoTitle.str().c_str(), 
					 massNbins_, massLow_, massHigh_); 
    SBSPassProbes->Sumw2();
    
    // SBS Failing Probes
    histoName.str(std::string());
    histoName << "sbs_fail_subtracted_" << bvard << "_" << iBin; 
    histoTitle.str(std::string());
    histoTitle << "Failing Probes SBS - "  << lowEdge << "<" << 
      bvar << "<" << highEdge;
    TH1F* SBSFailProbes = fs->make<TH1F>(histoName.str().c_str(), 
			histoTitle.str().c_str(), 
			massNbins_, massLow_, massHigh_); 
    SBSFailProbes->Sumw2();
    
    // Perform side band subtraction
    SBS_->Subtract(*PassProbes, *SBSPassProbes);
    SBS_->Subtract(*FailProbes, *SBSFailProbes);
    
    // Count the number of passing and failing probes in the region
    double npassR = SBSPassProbes->Integral("");
    double nfailR = SBSFailProbes->Integral("");
    
    if((npassR + nfailR) != 0){
      double eff, effErrHi, effErrLo;      
      FCIntervals::Efficiency (npassR, npassR + nfailR, eff, effErrHi, effErrLo);
      
      edm::LogInfo("TagProbeEDMAnalysis") << "Num pass " << npassR << ",  Num fail " << 
	nfailR << ". Eff " << eff << " + " << effErrHi << " - " << effErrLo;
      
      // Fill the efficiency hist
      effhist->SetPoint (iBin, (highEdge + lowEdge)/2.0, eff );
      effhist->SetPointEXhigh (iBin, (highEdge - lowEdge)/2.0 );
      effhist->SetPointEXlow (iBin, (highEdge - lowEdge)/2.0 );
      effhist->SetPointEYhigh (iBin, effErrHi );
      effhist->SetPointEYlow (iBin, effErrLo );
      
      //Fill the denominator hist
      denhist->SetBinContent(iBin+1,npassR+nfailR);
      denhist->SetBinError(iBin+1,pow(npassR+nfailR,0.5));
      
    }else {
      edm::LogInfo("TagProbeEDMAnalysis") << " no probes ";
    }
    
    // ********** Make and save Canvas for the plots ********** //
    //    outRootFile_->cd();
    
    PassProbes->Write();
    FailProbes->Write();
    SBSPassProbes->Write();
    SBSFailProbes->Write();
  }
  
  //  outRootFile_->cd();
  effhist->Write();
  denhist->Write();
}


// ****************** TP Eff Side band subtraction *************
void TagProbeEDMAnalysis::TPEffSBS2D( std::string &fileName, std::string &bvar1, 
				       std::vector< double > bins1,
				       std::string &bvar2, std::vector< double > bins2 )
{

  //  outRootFile_->cd();
  //  fitTree_->SetDirectory(outRootFile_);
  
  //return;
  edm::LogInfo("TagProbeEDMAnalysis") << "***** Here in TP sideband subtraction 2D ******";
  edm::LogInfo("TagProbeEDMAnalysis") << "Number of entries " << fitTree_->GetEntries();
   
  std::string hname = "sbs_eff_" + bvar1 + "_" + bvar2;
  std::string htitle = "SBS Efficiency: " + bvar1 + " vs " + bvar2;
  
  std::string hdname = "sbs_den_" + bvar1 + "_" + bvar2; 
  std::string hdtitle = "SBS Denominator vs " + bvar1 + " vs " + bvar2; 
  
  std::stringstream condition;
  std::stringstream histoName;
  std::stringstream histoTitle;;
  
   int bnbins1 = bins1.size()-1;
   int bnbins2 = bins2.size()-1;
   edm::LogInfo("TagProbeEDMAnalysis") << "There are " << bnbins1 << " bins for var1 ";

   TH2F effhist(hname.c_str(),htitle.c_str(),bnbins1,&bins1[0],
		bnbins2,&bins2[0]);
   TH2F denhist(hdname.c_str(),hdtitle.c_str(),bnbins1,&bins1[0],
		bnbins2,&bins2[0]);

   TH1F* PassProbes(0);
   TH1F* FailProbes(0);

   TH1F* SBSPassProbes(0);
   TH1F* SBSFailProbes(0);

   for (int iBin1=0; iBin1<bnbins1; ++iBin1){
      double lowEdge1 = bins1[iBin1];
      double highEdge1 = bins1[iBin1+1];
      condition.str(std::string());
      condition << "(" << bvar1 << ">" <<  lowEdge1
			 << ") && (" << bvar1 << "<" << highEdge1 << ")";
      std::string bvar1Cond = condition.str();

      for( int bin2=0; bin2<bnbins2; ++bin2 )
      {
	 // The binning variables
	 double lowEdge2 = bins2[bin2];
	 double highEdge2 = bins2[bin2+1];
	 condition.str(std::string());
	 condition << "(" << bvar2 << ">" <<  lowEdge2
		   << ") && (" << bvar2 << "<" << highEdge2 << ")";
	 std::string bvar2Cond = condition.str();

	 // Print out the pass/fail condition
	 std::stringstream DisplayCondition;
	 DisplayCondition.str(std::string());
	 DisplayCondition << "(ProbePass==1(0) && " << bvar1Cond << " && " << bvar2Cond <<")*Weight";
	 edm::LogInfo("TagProbeEDMAnalysis") << "Pass(Fail) condition[" << bvar1 << ":" << 
	   bvar2 << "]: " << DisplayCondition.str() <<  std::endl;
	 
	 // Passing Probes
	 condition.str(std::string());
	 condition  << "(ProbePass==1 && " << bvar1Cond << 
	   " && " << bvar1Cond << ")*Weight";

	 histoName.str(std::string());
	 histoName << "sbs_pass_" << bvar1 << "_" << iBin1 << 
	   "_" << bvar2 << "_" << bin2;
	 histoTitle.str(std::string());
	 histoTitle << "Passing Probes - " << lowEdge1 << 
	   "<" << bvar1 << "<" << highEdge1 << " & " << lowEdge2 << 
	   "<" << bvar2 << "<" << highEdge2;
	 PassProbes = new TH1F(histoName.str().c_str(), 
			       histoTitle.str().c_str(), 
			       massNbins_, massLow_, massHigh_); 
	 PassProbes->Sumw2();
	 //	 PassProbes->SetDirectory(outRootFile_);
	 fitTree_->Draw(("Mass >> " + histoName.str()).c_str(), 
			condition.str().c_str() );

	 // Failing Probes
	 condition.str(std::string());
	 condition  << "(ProbePass==0 && " << bvar1Cond << 
	   " && " << bvar2Cond << ")*Weight";

	 histoName.str(std::string());
	 histoName << "sbs_fail_" << bvar1 << "_" << iBin1 << 
	   "_" << bvar2 << "_" << bin2;
	 histoTitle.str(std::string());
	 histoTitle << "Failing Probes - " << lowEdge1 << 
	   "<" << bvar1 << "<" << highEdge1 << " & " << lowEdge2 << 
	   "<" << bvar2 << "<" << highEdge2;
	 FailProbes = new TH1F(histoName.str().c_str(), 
			       histoTitle.str().c_str(), 
			       massNbins_, massLow_, massHigh_); 
	 FailProbes->Sumw2();
	 //	 FailProbes->SetDirectory(outRootFile_);
	 fitTree_->Draw(("Mass >> " + histoName.str()).c_str(), 
			condition.str().c_str());

	 // SBS Passing  Probes
	 histoName.str(std::string());
	 histoName << "sbs_pass_subtracted_" << bvar1 << 
	   "_" << iBin1 << "_" << bvar2 << "_" << bin2;
	 histoTitle.str(std::string());
	 histoTitle << "Passing Probes SBS - " << lowEdge1 << 
	   "<" << bvar1 << "<" << highEdge1 << " & " << lowEdge2 << 
	   "<" << bvar2 << "<" << highEdge2;
	 SBSPassProbes = new TH1F(histoName.str().c_str(), 
				  histoTitle.str().c_str(), 
				  massNbins_, massLow_, massHigh_); 
	 SBSPassProbes->Sumw2();

	 // SBS Failing Probes
	 histoName.str(std::string());
	 histoName << "sbs_fail_subtracted_" << bvar1 << "_" << 
	   iBin1 << "_" << bvar2 << "_" << bin2;
	 histoTitle.str(std::string());
	 histoTitle << "Failing Probes SBS - " << lowEdge1 << "<" << 
	   bvar1 << "<" << highEdge1 << " & " << lowEdge2 << "<" << 
	   bvar2 << "<" << highEdge2;
	 SBSFailProbes = new TH1F(histoName.str().c_str(), 
				  histoTitle.str().c_str(), 
				  massNbins_, massLow_, massHigh_); 
	 SBSFailProbes->Sumw2();

	 // Perform side band subtraction
	 SBS_->Subtract(*PassProbes, *SBSPassProbes);
	 SBS_->Subtract(*FailProbes, *SBSFailProbes);

	 // Count the number of passing and failing probes in the region
	 double npassR = SBSPassProbes->Integral("");
	 double nfailR = SBSFailProbes->Integral("");

	 if((npassR + nfailR) != 0){
	   double eff, effErrHi, effErrLo;      
	   FCIntervals::Efficiency (npassR, npassR + nfailR, eff, effErrHi, effErrLo);
	   
	   edm::LogInfo("TagProbeEDMAnalysis") << "Num pass " << npassR << ",  Num fail " << 
	     nfailR << ". Eff " << eff << " + " << effErrHi << " - " << effErrLo;
	   
	    // Fill the efficiency hist
	    effhist.SetBinContent( iBin1+1, bin2+1, eff);
	    effhist.SetBinError( iBin1+1, bin2+1, ((effErrLo > effErrHi)? effErrLo : effErrHi));

            denhist.SetBinContent(iBin1+1,bin2+1,(npassR + nfailR));
	    denhist.SetBinError(iBin1+1,bin2+1,pow(npassR + nfailR,0.5));
	      
	 }else {
	    edm::LogInfo("TagProbeEDMAnalysis") << " no probes ";
	 }

	 // ********** Make and save Canvas for the plots ********** //
	 //	 outRootFile_->cd();

	 PassProbes->Write();
	 FailProbes->Write();
	 SBSPassProbes->Write();
	 SBSFailProbes->Write();
	 edm::LogInfo("TagProbeEDMAnalysis") << "Wrote probes.";
      }
   }
   
   //   outRootFile_->cd();
   effhist.Write();
   denhist.Write();
   
   edm::LogInfo("TagProbeEDMAnalysis") << "Wrote eff hist!";

   if (PassProbes) {
     delete PassProbes;
     PassProbes = 0;
   }
   if (FailProbes) {
     delete FailProbes;
     FailProbes = 0;
   }
   if(SBSPassProbes) {
     delete SBSPassProbes;
     SBSPassProbes = 0;
   }
   if (SBSFailProbes) {
     delete SBSFailProbes;
     SBSFailProbes = 0;
   }
}





// ********* Do sideband subtraction on the requested histogram ********* //
// ********************************************************************** //



// ********** Z -> l+l- Fitter ********** //
void TagProbeEDMAnalysis::TPEffFitter( std::string &fileName, std::string &bvar, 
					std::vector< double > bins, std::string &bvar2, 
					double bvar2Lo, double bvar2Hi )
{
   edm::LogInfo("TagProbeEDMAnalysis") << "Here in TP fitter";
    
   //   outRootFile_->cd();
   //   fitTree_ = (TTree*)outRootFile_->Get("fitter_tree");

   const unsigned int bnbins = bins.size()-1;
   edm::LogInfo("TagProbeEDMAnalysis") << "TPEffFitter: The number of bins is " << bnbins;

   std::vector< double > bins2;
   bins2.push_back(bvar2Lo);
   bins2.push_back(bvar2Hi);

   TGraphAsymmErrors* effhist = fs->make<TGraphAsymmErrors>(bnbins);
   std::string hname = "fit_eff_" + bvar;
   std::string htitle = "Efficiency vs " + bvar;
   effhist->SetNameTitle(hname.c_str(), htitle.c_str());

   TGraph* chi2hist = fs->make<TGraph>(bnbins);
   TGraph* qualityhist = fs->make<TGraph>(bnbins);
   hname = "fit_chi2_" + bvar;
   htitle = "Chi^{2} vs " + bvar;
   chi2hist->SetNameTitle(hname.c_str(), htitle.c_str());
   hname = "fit_quality_" + bvar;
   htitle = "Quality vs " + bvar;
   qualityhist->SetNameTitle(hname.c_str(), htitle.c_str());
    
   double eff, hierr, loerr, xval, xerr, chi2Val, quality;
   for (unsigned int iBin=0; iBin<bnbins; ++iBin ) 
   {
     
     performFit( bvar, bins, iBin, bvar2, bins2, 0, eff, loerr, hierr, chi2Val, quality, false );
     
     xval = (bins[iBin + 1] + bins[iBin])/2;
     xerr = (bins[iBin + 1] - bins[iBin])/2;
     
     effhist->SetPoint( iBin, xval, eff );
     effhist->SetPointEXhigh( iBin, xerr);
     effhist->SetPointEXlow( iBin, xerr);
     effhist->SetPointEYhigh( iBin, hierr );
     effhist->SetPointEYlow( iBin, loerr );
     
     chi2hist->SetPoint(iBin, xval, chi2Val);
     qualityhist->SetPoint(iBin, xval, quality);
   }
   
   //   outRootFile_->cd();
   effhist->Write();
   chi2hist->Write();
   qualityhist->Write();
   
   return;
}


// ********** Z -> l+l- Fitter ********** //
void TagProbeEDMAnalysis::TPEffFitter2D (
const std::string &fileName, std::string &bvar1, std::vector< double > &bins1,
const std::string &bvar2, std::vector<double> &bins2 )
{

  //   outRootFile_->cd();
  //fitTree_->SetDirectory(outRootFile_);

   edm::LogInfo("TagProbeEDMAnalysis") << "Here in TP fitter 2D";
   const unsigned int bnbins1 = bins1.size()-1;
   const unsigned int bnbins2 = bins2.size()-1;
   edm::LogVerbatim("TagAndProbe|Fitter") << "TPEffFitter2D: The number of bins is " << bnbins1 << ":" << bnbins2;

   std::string hname = "fit_eff_" + bvar1 + "_" + bvar2;
   std::string htitle = "Efficiency: " + bvar1 + " vs " + bvar2;
   TH2F* effhist = fs->make<TH2F>(hname.c_str(),htitle.c_str(),bnbins1,&bins1[0],bnbins2,&bins2[0]);
   hname = "fit_chi2_" + bvar1 + "_" + bvar2;
   htitle = "Chi^{2}: " + bvar1 + " vs " + bvar2;
   TH2F* chi2hist = fs->make<TH2F>(hname.c_str(),htitle.c_str(),bnbins1,&bins1[0],bnbins2,&bins2[0]);
   hname = "fit_quality_" + bvar1 + "_" + bvar2;
   htitle = "Quality: " + bvar1 + " vs " + bvar2;
   TH2F* qualityhist= fs->make<TH2F>(hname.c_str(),htitle.c_str(),bnbins1,&bins1[0],bnbins2,&bins2[0]);

   for( unsigned int bin1=0; bin1<bnbins1; ++bin1 )
   {
      for( unsigned int bin2=0; bin2<bnbins2; ++bin2 )
      {
        double eff, loerr, hierr, chi2Val, quality;

        performFit(bvar1, bins1, bin1, bvar2, bins2, bin2, eff, loerr, hierr, chi2Val, quality, true);
        // Fill the efficiency hist
        effhist->SetBinContent(bin1+1,bin2+1,eff);
        effhist->SetBinError(bin1+1,bin2+1, (loerr > hierr)? loerr : hierr);

        // Fill chi2 histogram
        chi2hist->SetBinContent( bin1+1, bin2+1, chi2Val);
        qualityhist->SetBinContent( bin1+1, bin2+1, quality);
      }
   }

   //   outRootFile_->cd();
   effhist->Write();
   chi2hist->Write();
   qualityhist->Write();
}
// ************************************** //


// ***** Function to return the signal Pdf depending on the users choice of fit func ******* //
void TagProbeEDMAnalysis::makeSignalPdf( )
{

  
   if (cbLineShape_) {
     cbLineShape_->CreatePDF(signalShapePdf_);
     signalShapeFailPdf_  = signalShapePdf_;
   } else if (zLineShape_) {
     zLineShape_->CreatePDF(signalShapePdf_);
     signalShapeFailPdf_  = signalShapePdf_;
   } else if (gaussLineShape_) {
     gaussLineShape_->CreatePDF(signalShapePdf_);
     signalShapeFailPdf_  = signalShapePdf_;
   } else {
     edm::LogError("TagAndProbe") << "No signal PDF specified";
     exit(1);
   }
}

// ***** Function to return the background Pdf depending on the users choice of fit func ******* //
void TagProbeEDMAnalysis::makeBkgPdf( )
{
   if (polyBkgLineShape_) {
     polyBkgLineShape_->CreatePDF(bkgShapePdf_);
   } else if(cmsBkgLineShape_) {
     cmsBkgLineShape_->CreatePDF(bkgShapePdf_);
   } else {
     edm::LogError("TagAndProbe") << "No signal PDF specified";
     exit(1);
   }
}

// ********** Get the true efficiency from this TTree ********** //
void TagProbeEDMAnalysis::TPEffMCTruth()
{
   // Loop over the number of different types of 
   // efficiency measurement in the input tree
   // Make a simple tree for fitting, and then
   // call the fitter.
   edm::LogInfo("TagProbeEDMAnalysis") << "Here in MC truth";

   //   outRootFile_->cd();
   edm::LogInfo("TagProbeEDMAnalysis") << "Writing MC Truth Eff hists!"; 

   std::string hname = "truth_eff_"+var1NameUp_;
   std::string htitle = "Efficiency vs "+var1NameUp_;
   TGraphAsymmErrors var1effhist(var1Pass_,var1All_,"");
   var1effhist.SetNameTitle(hname.c_str(), htitle.c_str());
   var1effhist.GetXaxis()->SetName(var1NameUp_.c_str());
   var1effhist.GetYaxis()->SetName("Efficiency");
   var1effhist.Write();

   hname = "truth_eff_"+var2NameUp_;
   htitle = "Efficiency vs "+var2NameUp_;
   TGraphAsymmErrors var2effhist(var2Pass_,var2All_,"");
   var2effhist.SetNameTitle(hname.c_str(), htitle.c_str());
   var2effhist.GetXaxis()->SetName(var2NameUp_.c_str());
   var2effhist.GetYaxis()->SetName("Efficiency");
   var2effhist.Write();

   return;
}
// ******************************************************** //


// ********** Get the true 2D efficiency from this TTree ********** //
void TagProbeEDMAnalysis::TPEffMCTruth2D()
{
   // Loop over the number of different types of 
   // efficiency measurement in the input tree
   // Make a simple tree for fitting, and then
   // call the fitter.
   edm::LogInfo("TagProbeEDMAnalysis") << "Here in MC truth";

   //   outRootFile_->cd();
   edm::LogInfo("TagProbeEDMAnalysis") << "Writing MC Truth Eff hists!"; 

   std::string hname = "truth_eff_"+var1NameUp_+"_"+var2NameUp_;
   std::string htitle = "Efficiency: "+var1NameUp_+" vs "+var2NameUp_;
   TH2F var1var2effhist(hname.c_str(),htitle.c_str(),var1Bins_.size()-1,&var1Bins_[0],
		     var2Bins_.size()-1,&var2Bins_[0]);
   var1var2effhist.Sumw2();
   var1var2effhist.Divide(var1var2Pass_,var1var2All_,1.0,1.0,"B");
   var1var2effhist.Write();

   hname = "truth_eff_bias_"+var1NameUp_+"_"+var2NameUp_;
   htitle = "Efficiency: "+var1NameUp_+" vs "+var2NameUp_;
   TH2F var1var2biaseffhist(hname.c_str(),htitle.c_str(),var1Bins_.size()-1,&var1Bins_[0],
		            var2Bins_.size()-1,&var2Bins_[0]);
   var1var2biaseffhist.Sumw2();
   var1var2biaseffhist.Divide(var1var2Pass_,var1var2All_,1.0,1.0,"B");
   var1var2biaseffhist.Write();

   return;
}
// ******************************************************** //


// ********** Get the efficiency from this TTree ********** //
void TagProbeEDMAnalysis::CalculateEfficiencies() {
  
  if (calcEffsTruth_) {
      TPEffMCTruth();
      // 2D MC Truth
      if( do2DFit_ ) TPEffMCTruth2D();
   }

   if( calcEffsFitter_ || calcEffsSB_ )
   {
      edm::LogInfo("TagProbeEDMAnalysis") << "Entries in fit tree ... " << fitTree_->GetEntries();
      fitTree_->Write();

      edm::LogInfo("TagProbeEDMAnalysis") << "There are " << var1Bins_.size()-1 << " " << var1NameUp_ << " bins.";
      const unsigned int nbins1 = var1Bins_.size()-1;
      const unsigned int nbins2 = var2Bins_.size()-1;

      if (calcEffsFitter_) {
	 // We have filled the simple tree ... call the fitter
	 TPEffFitter( fitFileName_, var1NameUp_, var1Bins_, var2NameUp_, var2Bins_[0], var2Bins_[nbins2] );
	 TPEffFitter( fitFileName_, var2NameUp_, var2Bins_, var1NameUp_, var1Bins_[0], var1Bins_[nbins1] );

	 // 2D Fit
	 if( do2DFit_ )
	 {
	    TPEffFitter2D( fitFileName_, var1NameUp_, var1Bins_, var2NameUp_, var2Bins_ );
	 }
      }

      if (calcEffsSB_) {
	 // We have filled the simple tree ... call side band subtraction
	 TPEffSBS(  fitFileName_, var1NameUp_, var1Bins_, var2NameUp_, var2Bins_[0], var2Bins_[nbins2] );
	 if (!doTextDefinedBins_) TPEffSBS(  fitFileName_, var2NameUp_, var2Bins_, var1NameUp_, var1Bins_[0], var1Bins_[nbins1] );

	 // 2D SBS
	 if (do2DFit_) {
	    TPEffSBS2D( fitFileName_, var1NameUp_, var1Bins_, var2NameUp_, var2Bins_ );
	 }
      }
   }

   return;
}
// ******************************************************** //

// ------------ method called once each job just after ending the event loop  ------------
void 
TagProbeEDMAnalysis::endJob() 
{
   //return;

   // Check for the various modes ...
   if( mode_ == "Write" )
   {
      // All we need to do is write out the truth histograms and fitTree
     //      outRootFile_->cd();

      edm::LogInfo("TagProbeEDMAnalysis") << "Fit tree has " << fitTree_->GetEntries() << " entries.";
      fitTree_->Write();

      WriteMCHistograms();

      //      outRootFile_->Close();
      edm::LogInfo("TagProbeEDMAnalysis") << "Closed ROOT file and returning!";

      return;
   }

   if( mode_ == "Normal" )
   {
     // Calculate the efficiencies etc ...
     //      outRootFile_->cd();
     CalculateEfficiencies();  
     // outRootFile_->Close();

      return;
   }

   edm::LogInfo("TagProbeEDMAnalysis") << "Here in endjob " << readFiles_.size();
   if( mode_ == "Read" && readFiles_.size() > 0 )
   {
      edm::LogInfo("TagProbeEDMAnalysis") << "Here in end job: Num files = " << readFiles_.size();

      // For the fittree chain the files together, then merge the
      // trees from the chain into the fitTree_ ...
      TChain fChain("fitter_tree");
      for(std::vector<std::string>::iterator iFile=readFiles_.begin(); iFile!=readFiles_.end(); ++iFile )
      {
	 edm::LogInfo("TagProbeEDMAnalysis") << "fChain adding: " << iFile->c_str();
	 fChain.Add(iFile->c_str());
      }      edm::LogInfo("TagProbeEDMAnalysis") << "Added all files: Num Entries = " << fChain.GetEntries();

      // Now merge the trees into the output file ...
      fChain.Merge(fitFileName_.c_str());

      // Get the private tree ...
      //      TFile f(fitFileName_.c_str(),"update");
      //      fitTree_ = (TTree*)f.Get("fitter_tree");
      //      edm::LogInfo("TagProbeEDMAnalysis") << "Read mode: Fit tree total entries " << fitTree_->GetEntries();

      ReadMCHistograms();
      
      // Now call for and calculate the efficiencies as normal
      // Set the file pointer
      //      outRootFile_ = &f;
      //      outRootFile_->cd();
      CalculateEfficiencies();  
      edm::LogInfo("TagProbeEDMAnalysis") << "Done calculating efficiencies!";
      //      outRootFile_->Close();

      return;
   }
}

//******************************* Clean up and destructors ***********************

TagProbeEDMAnalysis::~TagProbeEDMAnalysis() {

  CleanUpMCHistograms();
//   if (outRootFile_) {
//     outRootFile_->Close();
//     if( mode_ != "Read" ) delete outRootFile_;
//   }
  cleanFitVariables();
}

void TagProbeEDMAnalysis::CleanUpMCHistograms() {

  if (var1Pass_) {
    delete var1Pass_;
    var1Pass_ = 0;
  }
  if (var1All_) {
    delete var1All_;
    var1All_ = 0;
  }
  if (var2Pass_) {
    delete var2Pass_;
    var2Pass_ = 0;
  }
  if (var2All_) {
    delete var2All_;
    var2All_ = 0;
  }  
}

/****************************************************************
WARNING: The following headers pollute the namespace by calling
"using namespace std" in Minuit2/stackAllocator.h. The exact path
is not clear.
****************************************************************/
#include <RooAddPdf.h> 
#include <RooChi2Var.h>
#include <RooNLLVar.h>
#include <RooSimultaneous.h>
#include <RooDataHist.h>
#include <RooGlobalFunc.h> // DataError, Extended, ...

void TagProbeEDMAnalysis::cleanFitVariables()
{
  
  if (rooMass_) {
    delete rooMass_;
    rooMass_ = 0;
  }
  if (signalShapePdf_) {
    delete signalShapePdf_;
    signalShapePdf_ = 0;
  }
  if (bkgShapePdf_) {
    delete bkgShapePdf_;
    bkgShapePdf_ = 0;
  }
}

// ****************** Function to perform the efficiency fit ************ //
void TagProbeEDMAnalysis::performFit(const std::string &bvar1, const std::vector< double >& bins1, const int bin1,
				const std::string &bvar2, const std::vector< double >& bins2, const int bin2,
				double &eff, double &loerr, double &hierr, double &chi2Val, 
				double& quality, const bool is2D){

  using namespace RooFit;

   // The fit variable - lepton invariant mass
   RooRealVar Mass = *rooMass_;

   // The binning variables
   std::string bunits = "GeV";
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
   RooRealVar Weight("Weight","Weight",1.0);
  
   // Make the category variable that defines the two fits,
   // namely whether the probe passes or fails the eff criteria.
   RooCategory ProbePass("ProbePass","sample");
   ProbePass.defineType("pass",1);
   ProbePass.defineType("fail",0);  

   gROOT->cd();

   RooDataSet* data = new RooDataSet("fitData","fitData",fitTree_,
   RooArgSet(ProbePass,Mass,Var1,Var2,Weight));

   data->setWeightVar("Weight");

   std::stringstream roofitstream;
#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,0)
   data->defaultStream(&roofitstream);
#else
   data->defaultPrintStream(&roofitstream);
#endif
   data->get()->Print();
   edm::LogInfo("RooFit") << roofitstream.str();
   roofitstream.str(std::string());

   edm::LogInfo("TagProbeEDMAnalysis") << "Made dataset";
   RooDataHist *bdata = new RooDataHist("bdata","Binned Data",
   RooArgList(Mass,ProbePass),*data);
   edm::LogInfo("TagProbeEDMAnalysis") << "Made binned data: Weighted = " << bdata->isWeighted();

   // ********** Construct signal shape PDF ********** //

   makeSignalPdf();

   // ********** Construct background shape PDF ********** //

   makeBkgPdf();

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

   RooArgList componentspass(*signalShapePdf_, *bkgShapePdf_);
   RooArgList yieldspass(numSigPass, numBkgPass);
   RooArgList componentsfail(*signalShapeFailPdf_,*bkgShapePdf_);
   RooArgList yieldsfail(numSigFail, numBkgFail);	  

   RooAddPdf sumpass("sumpass","fixed extended sum pdf",componentspass,yieldspass);
   RooAddPdf sumfail("sumfail","fixed extended sum pdf",componentsfail, yieldsfail);

   // The total simultaneous fit ...
   RooSimultaneous totalPdf("totalPdf","totalPdf",ProbePass);
   ProbePass.setLabel("pass");
   totalPdf.addPdf(sumpass,ProbePass.getLabel());
#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,0)
   totalPdf.defaultStream(&roofitstream;)
#else
   totalPdf.defaultPrintStream(&roofitstream);
#endif
   totalPdf.Print();
   ProbePass.setLabel("fail");
   totalPdf.addPdf(sumfail,ProbePass.getLabel());
   totalPdf.Print();
   edm::LogInfo("RooFit") << roofitstream.str();
   roofitstream.str(std::string(""));

   // Count the number of passing and failing probes in the region
   // making sure we have enough to fit ...
   edm::LogInfo("TagProbeEDMAnalysis") << "About to count the number of events";
   std::ostringstream passCond;
   passCond.str(std::string());
   passCond << "(ProbePass==1) && (Mass<" << massHigh_ << ") && (Mass>" << massLow_
            << ") && (" << bvar1 << "<" << highEdge1 << ") && (" << bvar1 << ">"
            << lowEdge1 << ") && (" << bvar2 << "<" << highEdge2 << ") && ("
            << bvar2 << ">" << lowEdge2 << ")";
   edm::LogVerbatim("TagAndProbe|Fitter") << passCond.str();
   std::ostringstream failCond;
   failCond.str(std::string());
   failCond << "(ProbePass==0) && (Mass<" << massHigh_ << ") && (Mass>" << massLow_
            << ") && (" << bvar1 << "<" << highEdge1 << ") && (" << bvar1 << ">"
            << lowEdge1 << ") && (" << bvar2 << "<" << highEdge2 << ") && ("
            << bvar2 << ">" << lowEdge2 << ")";
   edm::LogVerbatim("TagAndProbe|Fitter") << failCond.str();
   int npassR = static_cast<int>(data->sumEntries(passCond.str().c_str()));
   int nfailR = static_cast<int>(data->sumEntries(failCond.str().c_str()));
   edm::LogInfo("TagProbeEDMAnalysis") << "Num pass " << npassR;
   edm::LogInfo("TagProbeEDMAnalysis") << "Num fail " << nfailR;

   RooAbsCategoryLValue& simCat = (RooAbsCategoryLValue&) totalPdf.indexCat();

   TList* dsetList = const_cast<RooAbsData*>((RooAbsData*)data)->split(simCat);
   RooCatType* type;
   TIterator* catIter = simCat.typeIterator();
   while( (type=(RooCatType*)catIter->Next()) )
   {
      // Retrieve the PDF for this simCat state
      RooAbsPdf* pdf =  totalPdf.getPdf(type->GetName());
      RooAbsData* dset = (RooAbsData*) dsetList->FindObject(type->GetName());
     
      if (pdf && dset && dset->sumEntries() != 0.0) 
      {               
	 edm::LogInfo("TagProbeEDMAnalysis") << "GOF Entries " << dset->numEntries() << " " 
	      << type->GetName() << std::endl;
	 if( strcmp(type->GetName(),"pass") == 0 ) 
	 {
	    npassR = dset->numEntries(); 
	    edm::LogInfo("TagProbeEDMAnalysis") << "Pass " << npassR; 
	 }
	 else if( strcmp(type->GetName(),"fail") == 0 ) 
	 {
	    nfailR = dset->numEntries();
	    edm::LogInfo("TagProbeEDMAnalysis") << "Fail " << nfailR; 
	 }
      }
   }
   
   // Return if there's nothing to fit
   if( npassR==0 && nfailR==0 ) return;

   if( npassR==0 )
   {
      efficiency.setVal(0.0);
      efficiency.setConstant(true);
      numBkgPass.setVal(0.0);
      numBkgPass.setConstant(true);
   }
   else if( nfailR==0 )
   {
      efficiency.setVal(1.0);
      efficiency.setConstant(true);
      numBkgFail.setVal(0.0);
      numBkgFail.setConstant(true);
   }

   edm::LogInfo("TagProbeEDMAnalysis") << "**** About to start the fitter ****";

   // ********* Do the Actual Fit ********** //  
   RooFitResult *fitResult = 0;
   RooAbsData::ErrorType fitError = RooAbsData::SumW2;
   //RooAbsData::ErrorType fitError = RooAbsData::Poisson;

   // The user chooses between binnned/unbinned fitting
   if( unbinnedFit_ )
   {
      edm::LogInfo("TagProbeEDMAnalysis") << "Starting unbinned fit using LL.";
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
      edm::LogInfo("TagProbeEDMAnalysis") << "Starting binned fit using Chi2.";
      RooChi2Var chi2("chi2","chi2",totalPdf,*bdata,DataError(fitError),Extended(kTRUE));
      RooMinuit m(chi2);
      m.setErrorLevel(0.5); // <<< HERE
      m.setStrategy(2);
      m.hesse();
      m.migrad();
      m.hesse();
      m.minos();
      fitResult = m.save();
      chi2Val = chi2.getVal();
      double numParams = fitResult->floatParsFinal().getSize();
      double dof = 2*massNbins_ - numParams;

      quality = 1 - TMath::Gamma(0.5*dof, chi2Val);
      chi2Val /= dof;
   }

#if ROOT_VERSION_CODE <= ROOT_VERSION(5,19,0)
   fitResult->defaultStream(&roofitstream;)
#else
   fitResult->defaultPrintStream(&roofitstream);
#endif
   fitResult->Print("v");
   edm::LogInfo("RooFit") << roofitstream.str();
   roofitstream.str(std::string(""));

   eff = efficiency.getVal();

   if( efficiency.hasAsymError() ){
     hierr = efficiency.getAsymErrorHi();
     // RooFit returns a negative number.  TGraphAsymmErrors expects a positive number.
     loerr = -efficiency.getAsymErrorLo();
   } else {
     hierr = efficiency.getError();
     loerr = efficiency.getError();
   }

   edm::LogInfo("TagProbeEDMAnalysis") << "Signal yield: " << numSignal.getVal() << " +- "
	     << numSignal.getError() << " + " << numSignal.getAsymErrorHi()
	     <<" - "<< numSignal.getAsymErrorLo() << std::endl;
   edm::LogInfo("TagProbeEDMAnalysis") << "Efficiency: "<< efficiency.getVal() << " +- "
	     << efficiency.getError() << " + " << efficiency.getAsymErrorHi()
	     <<" - "<< efficiency.getAsymErrorLo() << std::endl;

   // ********** Make and save Canvas for the plots ********** //

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
   oss1 << bin1;
   std::ostringstream oss2;
   oss2 << bin2;
   std::string cname = "fit_canvas_" + bvar1 + "_" + oss1.str() + "_" + bvar2 + "_" + oss2.str();
   if( !is2D ) cname = "fit_canvas_" + bvar1 + "_" + oss1.str();
   TCanvas *fitCanvas = fs->make<TCanvas>(cname.c_str(),"Sum over Modes, Signal Region",1000,1500);
   fitCanvas->Divide(1,2);
   fitCanvas->cd(1);
   fitCanvas->SetFillColor(10);

   TPad *lhs = (TPad*)gPad;
   lhs->Divide(2,1);
   lhs->cd(1);

   RooPlot* frame1 = Mass.frame();
   frame1->SetTitle("Passing Tag-Probes");
   frame1->SetName("pass");
   data->plotOn(frame1,Cut("ProbePass==1"),DataError(fitError));
   ProbePass.setLabel("pass");
   if( npassR > 0 )
   {
      totalPdf.plotOn(frame1,Slice(ProbePass),Components(*bkgShapePdf_),
      LineColor(kRed),ProjWData(Mass,*data));
      totalPdf.plotOn(frame1,Slice(ProbePass),ProjWData(Mass,*data),Precision(1e-5));
   }
   frame1->Draw("e0");

   lhs->cd(2);
   RooPlot* frame2 = Mass.frame();
   frame2->SetTitle("Failing Tag-Probes");
   frame2->SetName("fail");
   data->plotOn(frame2,Cut("ProbePass==0"),DataError(fitError));
   ProbePass.setLabel("fail");
   if( nfailR > 0 )
   {
      totalPdf.plotOn(frame2,Slice(ProbePass),Components(*bkgShapePdf_),
      LineColor(kRed),ProjWData(Mass,*data));
      totalPdf.plotOn(frame2,Slice(ProbePass),ProjWData(Mass,*data),Precision(1e-5));
   }
   frame2->Draw("e0");

   fitCanvas->cd(2);
   RooPlot* frame3 = Mass.frame();
   frame3->SetTitle("All Tag-Probes");
   frame3->SetName("total");
   data->plotOn(frame3,DataError(fitError));
   totalPdf.plotOn(frame3,Components(*bkgShapePdf_),
   LineColor(kRed),ProjWData(Mass,*data));
   totalPdf.plotOn(frame3,ProjWData(Mass,*data),Precision(1e-5));
   totalPdf.paramOn(frame3);
   frame3->Draw("e0");

   //   outRootFile_->cd();
   fitCanvas->Write();

   edm::LogInfo("TagProbeEDMAnalysis") << "Finished with fitter - fit results saved to " << fitFileName_.c_str() << std::endl;

   if(data) delete data;
   if(bdata) delete bdata;
}



//define this as a plug-in
DEFINE_FWK_MODULE( TagProbeEDMAnalysis );
