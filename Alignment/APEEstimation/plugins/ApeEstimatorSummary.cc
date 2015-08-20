// -*- C++ -*-
//
// Package:    ApeEstimatorSummary
// Class:      ApeEstimatorSummary
// 
/**\class ApeEstimatorSummary ApeEstimatorSummary.cc Alignment/APEEstimation/src/ApeEstimatorSummary.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Johannes Hauk,6 2-039,+41227673512,
//         Created:  Mon Oct 11 13:44:03 CEST 2010
//         Modified by: Christian Schomakers (RWTH Aachen)
// $Id: ApeEstimatorSummary.cc,v 1.14 2012/01/26 00:06:33 hauk Exp $
//
//


// system include files
#include <memory>
#include <fstream>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "CLHEP/Matrix/SymMatrix.h"


//#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"

//...............
#include "Alignment/APEEstimation/interface/TrackerSectorStruct.h"


#include "TH1.h"
#include "TString.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TF1.h"
#include "TMath.h"
//
// class declaration
//

class ApeEstimatorSummary : public edm::EDAnalyzer {
   public:
      explicit ApeEstimatorSummary(const edm::ParameterSet&);
      ~ApeEstimatorSummary();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      void openInputFile();
      void getTrackerSectorStructs();
      void bookHists();
      std::vector<double> residualErrorBinning();
      void writeHists();
      
      void calculateApe();
      
      enum ApeWeight{wInvalid, wUnity, wEntries, wEntriesOverSigmaX2};
      
      // ----------member data ---------------------------
      const edm::ParameterSet parameterSet_;
      
      bool firstEvent=true;
      
      
      TFile* inputFile_;
      
      std::map<unsigned int, TrackerSectorStruct> m_tkSector_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ApeEstimatorSummary::ApeEstimatorSummary(const edm::ParameterSet& iConfig):
parameterSet_(iConfig),
inputFile_(0)
{
}


ApeEstimatorSummary::~ApeEstimatorSummary()
{
}


//
// member functions
//


void
ApeEstimatorSummary::openInputFile(){
  const std::string inputFileName(parameterSet_.getParameter<std::string>("InputFile"));
  std::ifstream inputFileStream;
  // Check if baseline file exists
  inputFileStream.open(inputFileName.c_str());
  if(inputFileStream.is_open()){
    inputFile_ = new TFile(inputFileName.c_str(),"READ");
  }
  if(inputFile_){
    edm::LogInfo("CalculateAPE")<<"Input file from loop over tracks and corresponding hits sucessfully opened";
  }
  else{
    edm::LogError("CalculateAPE")<<"There is NO input file\n"
       <<"...APE calculation stopped. Please check path of input file name in config:\n"
       <<"\t"<<inputFileName;
    throw edm::Exception( edm::errors::Configuration,
                          "Bad input file name");
  }
}



void
ApeEstimatorSummary::getTrackerSectorStructs(){
  // At present first of the plugins registered in TFileService needs to be the one containing the normalized residual histos per sector per error bin
  TString pluginName(inputFile_->GetListOfKeys()->At(0)->GetName());
  
  pluginName += "/";
  TDirectory *sectorDir(0), *intervalDir(0);
  bool sectorBool(true);
  for(unsigned int iSector(1);sectorBool;++iSector){
    std::stringstream sectorName, fullSectorName;
    sectorName << "Sector_" << iSector << "/";
    fullSectorName << pluginName << sectorName.str();
    TString fullName(fullSectorName.str().c_str());
    inputFile_->cd();
    sectorDir = (TDirectory*)inputFile_->TDirectory::GetDirectory(fullName);
    if(sectorDir){
      TrackerSectorStruct tkSector;
      sectorDir->GetObject("z_name;1", tkSector.Name);
      tkSector.name = tkSector.Name->GetTitle();
      bool intervalBool(true);
      for(unsigned int iInterval(1);intervalBool;++iInterval){
        std::stringstream intervalName, fullIntervalName;
	intervalName << "Interval_" << iInterval <<"/";
	fullIntervalName << fullSectorName.str() << intervalName.str();
	fullName = fullIntervalName.str().c_str();
	intervalDir = (TDirectory*)inputFile_->TDirectory::GetDirectory(fullName);
	if(intervalDir){
	  intervalDir->GetObject("h_sigmaX;1", tkSector.m_binnedHists[iInterval]["sigmaX"]);
	  intervalDir->GetObject("h_norResX;1", tkSector.m_binnedHists[iInterval]["norResX"]);
	  intervalDir->GetObject("h_sigmaY;1", tkSector.m_binnedHists[iInterval]["sigmaY"]);
	  intervalDir->GetObject("h_norResY;1", tkSector.m_binnedHists[iInterval]["norResY"]);
	  if(tkSector.m_binnedHists[iInterval]["sigmaY"] && tkSector.m_binnedHists[iInterval]["norResY"]){
	    tkSector.isPixel = true;
	  }
	}
	else{
	  intervalBool = false;
	  if(iSector==1)edm::LogInfo("CalculateAPE")<<"There are "<<iInterval-1<<" intervals per sector defined in input file";
	}
      }
      TDirectory *resultsDir(0);
      std::stringstream fullResultName;
      fullResultName << fullSectorName.str() << "Results/";
      fullName = fullResultName.str().c_str();
      resultsDir = (TDirectory*)inputFile_->TDirectory::GetDirectory(fullName);
      if(resultsDir){
        resultsDir->GetObject("h_entriesX;1", tkSector.EntriesX);
	if(tkSector.isPixel)resultsDir->GetObject("h_entriesY;1", tkSector.EntriesY);
	TTree* rawIdTree(0);
	resultsDir->GetObject("rawIdTree", rawIdTree);
	unsigned int rawId(0);
	rawIdTree->SetBranchAddress("RawId", &rawId);
	for(Int_t entry=0; entry<rawIdTree->GetEntries(); ++entry){
	  rawIdTree->GetEntry(entry);
	  // command "hadd" adds entries in TTree, so rawId are existing as often as number of files are added
	  bool alreadyAdded(false);
	  for(std::vector<unsigned int>::const_iterator i_rawId = tkSector.v_rawId.begin(); i_rawId != tkSector.v_rawId.end(); ++i_rawId){
	    if(rawId==*i_rawId)alreadyAdded = true;
	  }
	  if(alreadyAdded)break;
	  tkSector.v_rawId.push_back(rawId);
	}
      }
      m_tkSector_[iSector] = tkSector;
    }
    else{
      sectorBool = false;
      edm::LogInfo("CalculateAPE")<<"There are "<<iSector-1<<" sectors defined in input file";
    }
  }
}



void
ApeEstimatorSummary::bookHists(){
  const std::vector<double> v_binX(this->residualErrorBinning());
  for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector=m_tkSector_.begin(); i_sector!=m_tkSector_.end(); ++i_sector){
    (*i_sector).second.WeightX         = new TH1F("h_weightX","relative weight w_{x}/w_{tot,x};#sigma_{x}  [#mum];w_{x}/w_{tot,x}",v_binX.size()-1,&(v_binX[0]));
    (*i_sector).second.MeanX           = new TH1F("h_meanX","residual mean <r_{x}/#sigma_{r,x}>;#sigma_{x}  [#mum];<r_{x}/#sigma_{r,x}>",v_binX.size()-1,&(v_binX[0]));
    (*i_sector).second.RmsX            = new TH1F("h_rmsX","residual rms RMS(r_{x}/#sigma_{r,x});#sigma_{x}  [#mum];RMS(r_{x}/#sigma_{r,x})",v_binX.size()-1,&(v_binX[0]));
    (*i_sector).second.FitMeanX1       = new TH1F("h_fitMeanX1","fitted residual mean #mu_{x};#sigma_{x}  [#mum];#mu_{x}",v_binX.size()-1,&(v_binX[0]));
    (*i_sector).second.ResidualWidthX1 = new TH1F("h_residualWidthX1","residual width #Delta_{x};#sigma_{x}  [#mum];#Delta_{x}",v_binX.size()-1,&(v_binX[0]));
    (*i_sector).second.CorrectionX1    = new TH1F("h_correctionX1","correction to APE_{x};#sigma_{x}  [#mum];#Delta#sigma_{align,x}  [#mum]",v_binX.size()-1,&(v_binX[0]));
    (*i_sector).second.FitMeanX2       = new TH1F("h_fitMeanX2","fitted residual mean #mu_{x};#sigma_{x}  [#mum];#mu_{x}",v_binX.size()-1,&(v_binX[0]));
    (*i_sector).second.ResidualWidthX2 = new TH1F("h_residualWidthX2","residual width #Delta_{x};#sigma_{x}  [#mum];#Delta_{x}",v_binX.size()-1,&(v_binX[0]));
    (*i_sector).second.CorrectionX2    = new TH1F("h_correctionX2","correction to APE_{x};#sigma_{x}  [#mum];#Delta#sigma_{align,x}  [#mum]",v_binX.size()-1,&(v_binX[0]));
  
    if((*i_sector).second.isPixel){
      (*i_sector).second.WeightY         = new TH1F("h_weightY","relative weight w_{y}/w_{tot,y};#sigma_{y}  [#mum];w_{y}/w_{tot,y}",v_binX.size()-1,&(v_binX[0]));
      (*i_sector).second.MeanY           = new TH1F("h_meanY","residual mean <r_{y}/#sigma_{r,y}>;#sigma_{y}  [#mum];<r_{y}/#sigma_{r,y}>",v_binX.size()-1,&(v_binX[0]));
      (*i_sector).second.RmsY            = new TH1F("h_rmsY","residual rms RMS(r_{y}/#sigma_{r,y});#sigma_{y}  [#mum];RMS(r_{y}/#sigma_{r,y})",v_binX.size()-1,&(v_binX[0]));
      (*i_sector).second.FitMeanY1       = new TH1F("h_fitMeanY1","fitted residual mean #mu_{y};#sigma_{y}  [#mum];#mu_{y}",v_binX.size()-1,&(v_binX[0]));
      (*i_sector).second.ResidualWidthY1 = new TH1F("h_residualWidthY1","residual width #Delta_{y};#sigma_{y}  [#mum];#Delta_{y}",v_binX.size()-1,&(v_binX[0]));
      (*i_sector).second.CorrectionY1    = new TH1F("h_correctionY1","correction to APE_{y};#sigma_{y}  [#mum];#Delta#sigma_{align,y}  [#mum]",v_binX.size()-1,&(v_binX[0]));
      (*i_sector).second.FitMeanY2       = new TH1F("h_fitMeanY2","fitted residual mean #mu_{y};#sigma_{y}  [#mum];#mu_{y}",v_binX.size()-1,&(v_binX[0]));
      (*i_sector).second.ResidualWidthY2 = new TH1F("h_residualWidthY2","residual width #Delta_{y};#sigma_{y}  [#mum];#Delta_{y}",v_binX.size()-1,&(v_binX[0]));
      (*i_sector).second.CorrectionY2    = new TH1F("h_correctionY2","correction to APE_{y};#sigma_{y}  [#mum];#Delta#sigma_{align,y}  [#mum]",v_binX.size()-1,&(v_binX[0]));
    }
  }
}



std::vector<double>
ApeEstimatorSummary::residualErrorBinning(){
  std::vector<double> v_binX;
  TH1* EntriesX(m_tkSector_[1].EntriesX);
  for(int iBin=1; iBin<=EntriesX->GetNbinsX()+1; ++iBin){
    v_binX.push_back(EntriesX->GetBinLowEdge(iBin));
  }
  return v_binX;
}


void
ApeEstimatorSummary::writeHists(){
  TFile* resultsFile = new TFile(parameterSet_.getParameter<std::string>("ResultsFile").c_str(), "RECREATE");
  TDirectory* baseDir = resultsFile->mkdir("ApeEstimatorSummary");
  for(std::map<unsigned int,TrackerSectorStruct>::const_iterator i_sector=m_tkSector_.begin(); i_sector!=m_tkSector_.end(); ++i_sector){
    std::stringstream dirName;
    dirName<<"Sector_" << (*i_sector).first;
    TDirectory* dir = baseDir->mkdir(dirName.str().c_str());
    dir->cd();
    
    (*i_sector).second.Name->Write();
    
    (*i_sector).second.WeightX->Write();
    (*i_sector).second.MeanX->Write();
    (*i_sector).second.RmsX->Write();	   
    (*i_sector).second.FitMeanX1->Write();	   
    (*i_sector).second.ResidualWidthX1 ->Write();
    (*i_sector).second.CorrectionX1->Write();   
    (*i_sector).second.FitMeanX2->Write();	   
    (*i_sector).second.ResidualWidthX2->Write();
    (*i_sector).second.CorrectionX2->Write();
    
    if((*i_sector).second.isPixel){
      (*i_sector).second.WeightY->Write();
      (*i_sector).second.MeanY->Write();
      (*i_sector).second.RmsY->Write();	   
      (*i_sector).second.FitMeanY1->Write();	   
      (*i_sector).second.ResidualWidthY1 ->Write();
      (*i_sector).second.CorrectionY1->Write();   
      (*i_sector).second.FitMeanY2->Write();	   
      (*i_sector).second.ResidualWidthY2->Write();
      (*i_sector).second.CorrectionY2->Write();
    }
  }
  resultsFile->Close();
}



void
ApeEstimatorSummary::calculateApe(){
   // Set baseline or calculate APE value?
   const bool setBaseline(parameterSet_.getParameter<bool>("setBaseline"));
      
   // Read in baseline file for calculation of APE value (if not setting baseline)
   // Has same format as iterationFile
   const std::string baselineFileName(parameterSet_.getParameter<std::string>("BaselineFile"));
   TFile* baselineFile(0);
   TTree* baselineTreeX(0);
   TTree* baselineTreeY(0);
   TTree* sectorNameBaselineTree(0);
   if(!setBaseline){
     std::ifstream baselineFileStream;
     // Check if baseline file exists
     baselineFileStream.open(baselineFileName.c_str());
     if(baselineFileStream.is_open()){
       baselineFileStream.close();
       baselineFile = new TFile(baselineFileName.c_str(),"READ");
     }
     if(baselineFile){
       edm::LogInfo("CalculateAPE")<<"Baseline file for APE values sucessfully opened";
       baselineFile->GetObject("iterTreeX;1",baselineTreeX);
       baselineFile->GetObject("iterTreeY;1",baselineTreeY);
       baselineFile->GetObject("nameTree;1",sectorNameBaselineTree);
     }
     else{
       edm::LogWarning("CalculateAPE")<<"There is NO baseline file for APE values, so normalized residual width =1 for ideal conditions is assumed";
     }
   }
   
   // Set up root file for iterations on APE value (or for setting baseline in setBaseline mode)
   const std::string iterationFileName(setBaseline ? baselineFileName : parameterSet_.getParameter<std::string>("IterationFile"));
   // For iterations, updates are needed to not overwrite the iterations before
   TFile* iterationFile = new TFile(iterationFileName.c_str(),setBaseline ? "RECREATE" : "UPDATE");
   
   
   // Set up TTree for iterative APE values on first pass (first iteration) or read from file (further iterations)
   TTree* iterationTreeX(0);
   TTree* iterationTreeY(0);
   iterationFile->GetObject("iterTreeX;1",iterationTreeX);
   iterationFile->GetObject("iterTreeY;1",iterationTreeY);
   // The same for TTree containing the names of the sectors (no additional check, since always handled exactly as iterationTree)
   TTree* sectorNameTree(0);
   iterationFile->GetObject("nameTree;1",sectorNameTree);
   
   bool firstIter(false);
   if(!iterationTreeX){ // should be always true in setBaseline mode, since file is recreated
     firstIter = true;
     if(!setBaseline){
       iterationTreeX = new TTree("iterTreeX","Tree for APE x values of all iterations");
       iterationTreeY = new TTree("iterTreeY","Tree for APE y values of all iterations");
       edm::LogInfo("CalculateAPE")<<"First APE iteration (number 0.), create iteration file with TTree";
       sectorNameTree = new TTree("nameTree","Tree with names of sectors");
     }
     else{
       iterationTreeX = new TTree("iterTreeX","Tree for baseline x values of normalized residual width");
       iterationTreeY = new TTree("iterTreeY","Tree for baseline y values of normalized residual width");
       edm::LogInfo("CalculateAPE")<<"Set baseline, create baseline file with TTree";
       sectorNameTree = new TTree("nameTree","Tree with names of sectors");
     }
   }
   else{
     const unsigned int iteration(iterationTreeX->GetEntries());
     edm::LogWarning("CalculateAPE")<<"NOT the first APE iteration (number 0.) but the "<<iteration<<". one, is this wanted or forgot to delete old iteration file with TTree?";
   }
   
   
   // Assign the information stored in the trees to arrays
   double a_apeSectorX[16589];
   double a_apeSectorY[16589];
   double a_baselineSectorX[16589];
   double a_baselineSectorY[16589];
   
   std::string* a_sectorName[16589];
   std::string* a_sectorBaselineName[16589];
   std::map<unsigned int, TrackerSectorStruct>::const_iterator i_sector;
   for(i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
     const unsigned int iSector(i_sector->first);
     const bool pixelSector(i_sector->second.isPixel);
     a_apeSectorX[iSector] = 99.;
     a_apeSectorY[iSector] = 99.;
     a_baselineSectorX[iSector] = -99.;
     a_baselineSectorY[iSector] = -99.;
     
     a_sectorName[iSector] = 0;
     a_sectorBaselineName[iSector] = 0;
     std::stringstream ss_sector, ss_sectorSuffixed;
     ss_sector << "Ape_Sector_" << iSector;
     if(!setBaseline && baselineTreeX){
       baselineTreeX->SetBranchAddress(ss_sector.str().c_str(), &a_baselineSectorX[iSector]);
       baselineTreeX->GetEntry(0);
       if(pixelSector){
       baselineTreeY->SetBranchAddress(ss_sector.str().c_str(), &a_baselineSectorY[iSector]);
       baselineTreeY->GetEntry(0);
       }
       sectorNameBaselineTree->SetBranchAddress(ss_sector.str().c_str(), &a_sectorBaselineName[iSector]);
       sectorNameBaselineTree->GetEntry(0);
     }
     else{
       // Set default ideal normalized residual width to 1
       a_baselineSectorX[iSector] = 1.;
       a_baselineSectorY[iSector] = 1.;
     }
     if(firstIter){ // should be always true in setBaseline mode, since file is recreated  
       ss_sectorSuffixed << ss_sector.str() << "/D";
       iterationTreeX->Branch(ss_sector.str().c_str(), &a_apeSectorX[iSector], ss_sectorSuffixed.str().c_str());
       if(pixelSector){
       iterationTreeY->Branch(ss_sector.str().c_str(), &a_apeSectorY[iSector], ss_sectorSuffixed.str().c_str());
       }
       sectorNameTree->Branch(ss_sector.str().c_str(), &a_sectorName[iSector], 32000, 00);
     }
     else{
       iterationTreeX->SetBranchAddress(ss_sector.str().c_str(), &a_apeSectorX[iSector]);
       iterationTreeX->GetEntry(iterationTreeX->GetEntries()-1);
       if(pixelSector){
       iterationTreeY->SetBranchAddress(ss_sector.str().c_str(), &a_apeSectorY[iSector]);
       iterationTreeY->GetEntry(iterationTreeY->GetEntries()-1);
       }
       sectorNameTree->SetBranchAddress(ss_sector.str().c_str(), &a_sectorName[iSector]);
       sectorNameTree->GetEntry(0);
     }
   }
   
   
   // Check whether sector definitions are identical with the ones of previous iterations and with the ones in baseline file
   for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
     const std::string& name(i_sector->second.name);
     if(!firstIter){
       const std::string& nameLastIter(*a_sectorName[(*i_sector).first]);
       if(name!=nameLastIter){
         edm::LogError("CalculateAPE")<<"Inconsistent sector definition in iterationFile for sector "<<i_sector->first<<",\n"
                                      <<"Recent iteration has name \""<<name<<"\", while previous had \""<<nameLastIter<<"\"\n"
				      <<"...APE calculation stopped. Please check sector definitions in config!\n";
         return;
       }
     }
     else a_sectorName[(*i_sector).first] = new std::string(name);
     if(!setBaseline && baselineFile){
       const std::string& nameBaseline(*a_sectorBaselineName[(*i_sector).first]);
       if(name!=nameBaseline){
         edm::LogError("CalculateAPE")<<"Inconsistent sector definition in baselineFile for sector "<<i_sector->first<<",\n"
                                      <<"Recent iteration has name \""<<name<<"\", while baseline had \""<<nameBaseline<<"\"\n"
				      <<"...APE calculation stopped. Please check sector definitions in config!\n";
         return;
       }
     }
   }
   
   
   // Set up text file for writing out APE values per module
   std::ofstream apeOutputFile;
   if(!setBaseline){
     const std::string apeOutputFileName(parameterSet_.getParameter<std::string>("ApeOutputFile"));
     apeOutputFile.open(apeOutputFileName.c_str());
     if(apeOutputFile.is_open()){
       edm::LogInfo("CalculateAPE")<<"Text file for writing APE values successfully opened";
     }else{
       edm::LogError("CalculateAPE")<<"Text file for writing APE values NOT opened,\n"
                                    <<"...APE calculation stopped. Please check path of text file name in config:\n"
                                    <<"\t"<<apeOutputFileName;
       return;
     }
   }
   
   
   
   
   // Loop over sectors for calculating APE
   const std::string apeWeightName(parameterSet_.getParameter<std::string>("apeWeight"));
   ApeWeight apeWeight(wInvalid);
   if(apeWeightName=="unity") apeWeight = wUnity;
   else if(apeWeightName=="entries")apeWeight = wEntries;
   else if(apeWeightName=="entriesOverSigmaX2")apeWeight = wEntriesOverSigmaX2;
   if(apeWeight==wInvalid){
     edm::LogError("CalculateAPE")<<"Invalid parameter 'apeWeight' in cfg file: \""<<apeWeightName
                                  <<"\"\nimplemented apeWeights are \"unity\", \"entries\", \"entriesOverSigmaX2\""
				  <<"\n...APE calculation stopped.";
     return;
   }
   const double minHitsPerInterval(parameterSet_.getParameter<double>("minHitsPerInterval"));
   const double sigmaFactorFit(parameterSet_.getParameter<double>("sigmaFactorFit"));
   const double correctionScaling(parameterSet_.getParameter<double>("correctionScaling"));
   const bool smoothIteration(parameterSet_.getParameter<bool>("smoothIteration"));
   const double smoothFraction(parameterSet_.getParameter<double>("smoothFraction"));
   if(smoothFraction<=0. || smoothFraction>1.){
     edm::LogError("CalculateAPE")<<"Incorrect parameter in cfg file,"
                                  <<"\nsmoothFraction has to be in [0,1], but is "<<smoothFraction
				  <<"\n...APE calculation stopped.";
     return;
   }
   for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
     
     typedef std::pair<double,double> Error2AndResidualWidth2PerBin;
     typedef std::pair<double,Error2AndResidualWidth2PerBin> WeightAndResultsPerBin;
     std::vector<WeightAndResultsPerBin> v_weightAndResultsPerBinX;
     std::vector<WeightAndResultsPerBin> v_weightAndResultsPerBinY;
     
     
     double baselineWidthX2(a_baselineSectorX[(*i_sector).first]);
     double baselineWidthY2(a_baselineSectorY[(*i_sector).first]);
     
     
     // Loop over residual error bins to calculate APE for every bin
     for(std::map<unsigned int, std::map<std::string,TH1*> >::const_iterator i_errBins = (*i_sector).second.m_binnedHists.begin();
         i_errBins != (*i_sector).second.m_binnedHists.end(); ++i_errBins){
       std::map<std::string,TH1*> mHists = (*i_errBins).second;
       
       double entriesX = mHists["sigmaX"]->GetEntries();
       double meanSigmaX = mHists["sigmaX"]->GetMean(); 
       
       // Fitting Parameters
       double xMin      = mHists["norResX"]->GetXaxis()->GetXmin();
       double xMax      = mHists["norResX"]->GetXaxis()->GetXmax();
       double integralX = mHists["norResX"]->Integral();
       double meanX     = mHists["norResX"]->GetMean();
       double rmsX      = mHists["norResX"]->GetRMS();
       double maximumX  = mHists["norResX"]->GetBinContent(mHists["norResX"]->GetMaximumBin());
       
       
       // First Gaus Fit
       TF1 funcX_1("mygausX", "gaus", xMin, xMax);
       funcX_1.SetParameters(maximumX, meanX, rmsX);
       TString fitOpt("ILERQ"); //TString fitOpt("IMR"); ("IRLL"); ("IRQ");
       Int_t fitResultX(0);
       if(integralX>minHitsPerInterval){
         if(mHists["norResX"]->Fit(&funcX_1, fitOpt)){
           edm::LogWarning("CalculateAPE")<<"Fit1 did not work: "<<mHists["norResX"]->Fit(&funcX_1, fitOpt);
           continue;
         }
         fitResultX = mHists["norResX"]->Fit(&funcX_1, fitOpt);
         std::cout<<"FitResult1\t"<<fitResultX<<"\n";
       }
       Double_t meanX_1  = funcX_1.GetParameter(1);
       Double_t sigmaX_1 = funcX_1.GetParameter(2);
       
       
       // Second gaus fit
       TF1 funcX_2("mygausX2","gaus",meanX_1 - sigmaFactorFit*TMath::Abs(sigmaX_1), meanX_1 + sigmaFactorFit*TMath::Abs(sigmaX_1));
       funcX_2.SetParameters(funcX_1.GetParameter(0),meanX_1,sigmaX_1);
       if(integralX>minHitsPerInterval){
         if(mHists["norResX"]->Fit(&funcX_2, fitOpt)){
           edm::LogWarning("CalculateAPE")<<"Fit2 did not work for x : "<<mHists["norResX"]->Fit(&funcX_2, fitOpt);
	   continue;
         }
         fitResultX = mHists["norResX"]->Fit(&funcX_2, fitOpt);
       }
       Double_t meanX_2  = funcX_2.GetParameter(1);
       Double_t sigmaX_2 = funcX_2.GetParameter(2);
    
       
       // Now the same for y coordinate
       double entriesY(0.);
       double meanSigmaY(0.);
       if((*i_sector).second.isPixel){
         entriesY = mHists["sigmaY"]->GetEntries();
         meanSigmaY = mHists["sigmaY"]->GetMean();
       }
       
       Double_t meanY_1(0.);
       Double_t sigmaY_1(0.);
       Double_t meanY_2(0.);
       Double_t sigmaY_2(0.);
       double meanY(0.);
       double rmsY(0.);
       if((*i_sector).second.isPixel){
         // Fitting Parameters
         double yMin = mHists["norResY"]->GetXaxis()->GetXmin();
         double yMax = mHists["norResY"]->GetXaxis()->GetXmax();
         double integralY = mHists["norResY"]->Integral();
         meanY = mHists["norResY"]->GetMean();
         rmsY = mHists["norResY"]->GetRMS();
         double maximumY = mHists["norResY"]->GetBinContent(mHists["norResY"]->GetMaximumBin());
	 
	 // First Gaus Fit
         TF1 funcY_1("mygausY", "gaus", yMin, yMax);
         funcY_1.SetParameters(maximumY, meanY, rmsY);
         Int_t fitResultY(0);
         if(integralY>minHitsPerInterval){
           if(mHists["norResY"]->Fit(&funcY_1, fitOpt)){
             edm::LogWarning("CalculateAPE")<<"Fit1 did not work: "<<mHists["norResY"]->Fit(&funcY_1, fitOpt);
             continue;
           }
            fitResultY = mHists["norResY"]->Fit(&funcY_1, fitOpt);
	    std::cout<<"FitResult2\t"<<fitResultY<<"\n";
         }
         meanY_1  = funcY_1.GetParameter(1);
	 sigmaY_1 = funcY_1.GetParameter(2);
	 
	 // Second gaus fit
         TF1 funcY_2("mygausY2","gaus",meanY_1 - sigmaFactorFit*TMath::Abs(sigmaY_1), meanY_1 + sigmaFactorFit*TMath::Abs(sigmaY_1));
         funcY_2.SetParameters(funcY_1.GetParameter(0),meanY_1,sigmaY_1);
         if(integralY>minHitsPerInterval){
           if(mHists["norResY"]->Fit(&funcY_2, fitOpt)){
             edm::LogWarning("CalculateAPE")<<"Fit2 did not work for y : "<<mHists["norResY"]->Fit(&funcY_2, fitOpt);
	     continue;
           }
            fitResultY = mHists["norResY"]->Fit(&funcY_2, fitOpt);
         }
         meanY_2  = funcY_2.GetParameter(1);
         sigmaY_2 = funcY_2.GetParameter(2);
       }
       
       
       
       // Fill histograms
       double fitMeanX_1(meanX_1), fitMeanX_2(meanX_2);
       double residualWidthX_1(sigmaX_1), residualWidthX_2(sigmaX_2);
       double fitMeanY_1(meanY_1), fitMeanY_2(meanY_2);
       double residualWidthY_1(sigmaY_1), residualWidthY_2(sigmaY_2);
       
       double correctionX2_1(-0.0010), correctionX2_2(-0.0010);
       double correctionY2_1(-0.0010), correctionY2_2(-0.0010);
       correctionX2_1 = meanSigmaX*meanSigmaX*(residualWidthX_1*residualWidthX_1 -baselineWidthX2);
       correctionX2_2 = meanSigmaX*meanSigmaX*(residualWidthX_2*residualWidthX_2 -baselineWidthX2);
       if((*i_sector).second.isPixel){
         correctionY2_1 = meanSigmaY*meanSigmaY*(residualWidthY_1*residualWidthY_1 -baselineWidthY2);
         correctionY2_2 = meanSigmaY*meanSigmaY*(residualWidthY_2*residualWidthY_2 -baselineWidthY2);
       }
       
       double correctionX_1 = correctionX2_1>=0. ? std::sqrt(correctionX2_1) : -std::sqrt(-correctionX2_1);
       double correctionX_2 = correctionX2_2>=0. ? std::sqrt(correctionX2_2) : -std::sqrt(-correctionX2_2);
       double correctionY_1 = correctionY2_1>=0. ? std::sqrt(correctionY2_1) : -std::sqrt(-correctionY2_1);
       double correctionY_2 = correctionY2_2>=0. ? std::sqrt(correctionY2_2) : -std::sqrt(-correctionY2_2);
       // Meanwhile, this got very bad default values, or? (negative corrections allowed)
       if(isnan(correctionX_1))correctionX_1 = -0.0010;
       if(isnan(correctionX_2))correctionX_2 = -0.0010;
       if(isnan(correctionY_1))correctionY_1 = -0.0010;
       if(isnan(correctionY_2))correctionY_2 = -0.0010;
       
       if(entriesX<minHitsPerInterval){
         meanX = 0.; rmsX = -0.0010;
         fitMeanX_1 = 0.; correctionX_1 = residualWidthX_1 = -0.0010;
         fitMeanX_2 = 0.; correctionX_2 = residualWidthX_2 = -0.0010;
       }
       
       if(entriesY<minHitsPerInterval){
	 meanY = 0.; rmsY = -0.0010;
         fitMeanY_1 = 0.; correctionY_1 = residualWidthY_1 = -0.0010;
         fitMeanY_2 = 0.; correctionY_2 = residualWidthY_2 = -0.0010;
       }
       
       (*i_sector).second.MeanX         ->SetBinContent((*i_errBins).first,meanX);
       (*i_sector).second.RmsX          ->SetBinContent((*i_errBins).first,rmsX);
       
       (*i_sector).second.FitMeanX1      ->SetBinContent((*i_errBins).first,fitMeanX_1);
       (*i_sector).second.ResidualWidthX1->SetBinContent((*i_errBins).first,residualWidthX_1);
       (*i_sector).second.CorrectionX1   ->SetBinContent((*i_errBins).first,correctionX_1*10000.);
       
       (*i_sector).second.FitMeanX2      ->SetBinContent((*i_errBins).first,fitMeanX_2);
       (*i_sector).second.ResidualWidthX2->SetBinContent((*i_errBins).first,residualWidthX_2);
       (*i_sector).second.CorrectionX2   ->SetBinContent((*i_errBins).first,correctionX_2*10000.);
       
       if((*i_sector).second.isPixel){
         (*i_sector).second.MeanY         ->SetBinContent((*i_errBins).first,meanY);
         (*i_sector).second.RmsY          ->SetBinContent((*i_errBins).first,rmsY);
         
         (*i_sector).second.FitMeanY1      ->SetBinContent((*i_errBins).first,fitMeanY_1);
         (*i_sector).second.ResidualWidthY1->SetBinContent((*i_errBins).first,residualWidthY_1);
         (*i_sector).second.CorrectionY1   ->SetBinContent((*i_errBins).first,correctionY_1*10000.);
         
         (*i_sector).second.FitMeanY2      ->SetBinContent((*i_errBins).first,fitMeanY_2);
         (*i_sector).second.ResidualWidthY2->SetBinContent((*i_errBins).first,residualWidthY_2);
         (*i_sector).second.CorrectionY2   ->SetBinContent((*i_errBins).first,correctionY_2*10000.);
       }
       
       
       // Use result for bin only when entries>=minHitsPerInterval
       if(entriesX<minHitsPerInterval && entriesY<minHitsPerInterval)continue;
       
       double weightX(0.);
       double weightY(0.);
       if(apeWeight==wUnity){
         weightX = 1.;
	 weightY = 1.;
       }
       else if(apeWeight==wEntries){
         weightX = entriesX;
	 weightY = entriesY;
       }
       else if(apeWeight==wEntriesOverSigmaX2){
         weightX = entriesX/(meanSigmaX*meanSigmaX);
	 weightY = entriesY/(meanSigmaY*meanSigmaY);
       }
       
       const Error2AndResidualWidth2PerBin error2AndResidualWidth2PerBinX(meanSigmaX*meanSigmaX, residualWidthX_1*residualWidthX_1);
       const WeightAndResultsPerBin weightAndResultsPerBinX(weightX, error2AndResidualWidth2PerBinX);
       if(!(entriesX<minHitsPerInterval)){
         //Fill absolute weights
	 (*i_sector).second.WeightX->SetBinContent((*i_errBins).first,weightX);
	 v_weightAndResultsPerBinX.push_back(weightAndResultsPerBinX);
       }
       
       const Error2AndResidualWidth2PerBin error2AndResidualWidth2PerBinY(meanSigmaY*meanSigmaY, residualWidthY_1*residualWidthY_1);
       const WeightAndResultsPerBin weightAndResultsPerBinY(weightY, error2AndResidualWidth2PerBinY);
       if(!(entriesY<minHitsPerInterval)){
         //Fill absolute weights
	 (*i_sector).second.WeightY->SetBinContent((*i_errBins).first,weightY);
	 v_weightAndResultsPerBinY.push_back(weightAndResultsPerBinY);
       }
     }
     
     
     // Do the final calculations
     
     if(v_weightAndResultsPerBinX.size()==0){
       edm::LogError("CalculateAPE")<<"NO error interval of sector "<<(*i_sector).first<<" has a valid x APE calculated,\n...so cannot set APE";
       continue;
     }
     
     if((*i_sector).second.isPixel && v_weightAndResultsPerBinY.size()==0){
       edm::LogError("CalculateAPE")<<"NO error interval of sector "<<(*i_sector).first<<" has a valid y APE calculated,\n...so cannot set APE";
       continue;
     }
     
     double correctionX2(999.);
     double correctionY2(999.);
     
     // Get sum of all weights
     double weightSumX(0.);
     std::vector<WeightAndResultsPerBin>::const_iterator i_apeBin;
     for(i_apeBin=v_weightAndResultsPerBinX.begin(); i_apeBin!=v_weightAndResultsPerBinX.end(); ++i_apeBin){
       weightSumX += i_apeBin->first;
     }
     (*i_sector).second.WeightX->Scale(1/weightSumX);
     double weightSumY(0.);
     if((*i_sector).second.isPixel){
       std::vector<WeightAndResultsPerBin>::const_iterator i_apeBin;
       for(i_apeBin=v_weightAndResultsPerBinY.begin(); i_apeBin!=v_weightAndResultsPerBinY.end(); ++i_apeBin){
         weightSumY += i_apeBin->first;
       }
       (*i_sector).second.WeightY->Scale(1/weightSumY);
     }
     
     
     if(!setBaseline){
       // Calculate weighted mean
       bool firstIntervalX(true);
       for(i_apeBin=v_weightAndResultsPerBinX.begin(); i_apeBin!=v_weightAndResultsPerBinX.end(); ++i_apeBin){
         if(firstIntervalX){
	   correctionX2 = i_apeBin->first*i_apeBin->second.first*(i_apeBin->second.second - baselineWidthX2);
	   firstIntervalX = false;
	 }
	 else{
	   correctionX2 += i_apeBin->first*i_apeBin->second.first*(i_apeBin->second.second - baselineWidthX2);
	 }
       }
       correctionX2 = correctionX2/weightSumX;
     }
     else{
       double numeratorX(0.), denominatorX(0.);
       std::vector<WeightAndResultsPerBin>::const_iterator i_apeBin;
       for(i_apeBin=v_weightAndResultsPerBinX.begin(); i_apeBin!=v_weightAndResultsPerBinX.end(); ++i_apeBin){
         numeratorX += i_apeBin->first*i_apeBin->second.first*i_apeBin->second.second;
	 denominatorX += i_apeBin->first*i_apeBin->second.first;
       }
       correctionX2 = numeratorX/denominatorX;
     }
     
     if((*i_sector).second.isPixel){
       if(!setBaseline){
         // Calculate weighted mean
         bool firstIntervalY(true);
         for(i_apeBin=v_weightAndResultsPerBinY.begin(); i_apeBin!=v_weightAndResultsPerBinY.end(); ++i_apeBin){
           if(firstIntervalY){
	     correctionY2 = i_apeBin->first*i_apeBin->second.first*(i_apeBin->second.second - baselineWidthY2);
	     firstIntervalY = false;
	   }
	   else{
	     correctionY2 += i_apeBin->first*i_apeBin->second.first*(i_apeBin->second.second - baselineWidthY2);
	   }
         }
         correctionY2 = correctionY2/weightSumY;
       }
       else{
         double numeratorY(0.), denominatorY(0.);
         std::vector<WeightAndResultsPerBin>::const_iterator i_apeBin;
         for(i_apeBin=v_weightAndResultsPerBinY.begin(); i_apeBin!=v_weightAndResultsPerBinY.end(); ++i_apeBin){
           numeratorY += i_apeBin->first*i_apeBin->second.first*i_apeBin->second.second;
	   denominatorY += i_apeBin->first*i_apeBin->second.first;
         }
         correctionY2 = numeratorY/denominatorY;
       }
     }
     
     
     
     if(!setBaseline){
       // Calculate updated squared APE of current iteration
       double apeX2(999.);
       double apeY2(999.);
       
       // old APE value from last iteration
       if(firstIter){
         apeX2 = 0.;
	 apeY2 = 0.;
       }
       else{
         apeX2 = a_apeSectorX[(*i_sector).first];
	 apeY2 = a_apeSectorY[(*i_sector).first];
       }
       const double apeX2old = apeX2;
       const double apeY2old = apeY2;
       
       // scale APE Correction with value given in cfg (not if smoothed iteration is used)
       edm::LogInfo("CalculateAPE")<<"Unscaled correction x for sector "<<(*i_sector).first<<" is "<<(correctionX2>0. ? +1. : -1.)*std::sqrt(std::fabs(correctionX2));
       if(!smoothIteration || firstIter)correctionX2 = correctionX2*correctionScaling*correctionScaling;
       if((*i_sector).second.isPixel){
         edm::LogInfo("CalculateAPE")<<"Unscaled correction y for sector "<<(*i_sector).first<<" is "<<(correctionY2>0. ? +1. : -1.)*std::sqrt(std::fabs(correctionY2));
         if(!smoothIteration || firstIter)correctionY2 = correctionY2*correctionScaling*correctionScaling;
       }
       
       // new APE value
       // smooth iteration or not?
       if(apeX2 + correctionX2 < 0.) correctionX2 = -apeX2;
       if(apeY2 + correctionY2 < 0.) correctionY2 = -apeY2;
       const double apeX2new(apeX2old + correctionX2);
       const double apeY2new(apeY2old + correctionY2);
       if(!smoothIteration || firstIter){
         apeX2 = apeX2new;
	 apeY2 = apeY2new;
       }
       else{
         apeX2 = std::pow(smoothFraction*std::sqrt(apeX2new) + (1-smoothFraction)*std::sqrt(apeX2old), 2);
	 apeY2 = std::pow(smoothFraction*std::sqrt(apeY2new) + (1-smoothFraction)*std::sqrt(apeY2old), 2);
       }
       if(apeX2<0. || apeY2<0.)edm::LogError("CalculateAPE")<<"\n\n\tBad APE, but why???\n\n\n";
       a_apeSectorX[(*i_sector).first] = apeX2;
       a_apeSectorY[(*i_sector).first] = apeY2;
       
       // Set the calculated APE spherical for all modules of strip sectors
       const double apeX(std::sqrt(apeX2));
       const double apeY(std::sqrt(apeY2));
       const double apeZ(std::sqrt(0.5*(apeX2+apeY2)));
       std::vector<unsigned int>::const_iterator i_rawId;
       for(i_rawId = (*i_sector).second.v_rawId.begin(); i_rawId != (*i_sector).second.v_rawId.end(); ++i_rawId){
         if((*i_sector).second.isPixel){
	   apeOutputFile<<*i_rawId<<" "<<std::fixed<<std::setprecision(5)<<apeX<<" "<<apeY<<" "<<apeZ<<"\n";
	 }
	 else{
	   apeOutputFile<<*i_rawId<<" "<<std::fixed<<std::setprecision(5)<<apeX<<" "<<apeX<<" "<<apeX<<"\n";
	 }
       }
     }
     // In setBaseline mode, just fill estimated mean value of residual width
     else{
       a_apeSectorX[(*i_sector).first] = correctionX2;
       a_apeSectorY[(*i_sector).first] = correctionY2;
     }
   }
   if(!setBaseline)apeOutputFile.close();
   
   iterationTreeX->Fill();
   iterationTreeX->Write("iterTreeX", TObject::kOverwrite);  // TObject::kOverwrite needed to not produce another iterTreeX;2
   iterationTreeY->Fill();
   iterationTreeY->Write("iterTreeY", TObject::kOverwrite);  // TObject::kOverwrite needed to not produce another iterTreeY;2
   if(firstIter){
     sectorNameTree->Fill();
     sectorNameTree->Write("nameTree");
   }
   iterationFile->Close();
   
   if(baselineFile)baselineFile->Close();
}



// ------------ method called to for each event  ------------
void
ApeEstimatorSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // Load APEs from the GT and write them to root files similar to the ones from calculateAPE()
   
   if(firstEvent){
	   // Set baseline or calculate APE value?
	   const bool setBaseline(parameterSet_.getParameter<bool>("setBaseline"));
	   
	   edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
	   iSetup.get<TrackerAlignmentErrorExtendedRcd>().get(alignmentErrors);
	   
	   // Read in baseline file for calculation of APE value (if not setting baseline)
	   // Has same format as iterationFile
	   const std::string baselineFileName(parameterSet_.getParameter<std::string>("BaselineFile"));
	   TFile* baselineFile(0);
	   TTree* sectorNameBaselineTree(0);
	   if(!setBaseline){
	     std::ifstream baselineFileStream;
	     // Check if baseline file exists
	     baselineFileStream.open(baselineFileName.c_str());
	     if(baselineFileStream.is_open()){
	       baselineFileStream.close();
	       baselineFile = new TFile(baselineFileName.c_str(),"READ");
	     }
	     if(baselineFile){
	       edm::LogInfo("CalculateAPE")<<"Baseline file for APE values sucessfully opened";
	       baselineFile->GetObject("nameTree;1",sectorNameBaselineTree);
	     }
	     else{
	       edm::LogWarning("CalculateAPE")<<"There is NO baseline file for APE values, so normalized residual width =1 for ideal conditions is assumed";
	     }
	   }
	   
	   // Set up root file for default APE values
	   const std::string defaultFileName(parameterSet_.getParameter<std::string>("DefaultFile"));
	   TFile* defaultFile = new TFile(defaultFileName.c_str(),"RECREATE");
	   
	   // Naming in the root files has to be iterTreeX to be consistent for the plotting tool
	   TTree* defaultTreeX(0);
	   TTree* defaultTreeY(0);
	   defaultFile->GetObject("iterTreeX;1",defaultTreeX);
	   defaultFile->GetObject("iterTreeY;1",defaultTreeY);
	   // The same for TTree containing the names of the sectors (no additional check, since always handled exactly as defaultTree)
	   TTree* sectorNameTree(0);
	   defaultFile->GetObject("nameTree;1",sectorNameTree);
	   
	   bool firstIter(false);
	   if(!defaultTreeX){ // should be always true in setBaseline mode, since file is recreated
	     firstIter = true;
	     defaultTreeX = new TTree("iterTreeX","Tree for default APE x values from GT");
	     defaultTreeY = new TTree("iterTreeY","Tree for default APE y values from GT");
	     edm::LogInfo("CalculateAPE")<<"First APE iteration (number 0.), create default file with TTree";
	     sectorNameTree = new TTree("nameTree","Tree with names of sectors");
	   }
	   else{
	     edm::LogWarning("CalculateAPE")<<"NOT the first APE iteration (number 0.), is this wanted or forgot to delete old iteration file with TTree?";
	   }
	     
	   // Assign the information stored in the trees to arrays
	   double a_defaultSectorX[16589];
	   double a_defaultSectorY[16589];
	   
	   std::string* a_sectorName[16589];
	   std::string* a_sectorBaselineName[16589];
	   std::map<unsigned int, TrackerSectorStruct>::const_iterator i_sector;
	   for(i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
	     const unsigned int iSector(i_sector->first);
	     const bool pixelSector(i_sector->second.isPixel);
	     
	     a_defaultSectorX[iSector] = -99.;
	     a_defaultSectorY[iSector] = -99.;
	     
	     a_sectorName[iSector] = 0;
	     a_sectorBaselineName[iSector] = 0;
	     std::stringstream ss_sector, ss_sectorSuffixed;
	     ss_sector << "Ape_Sector_" << iSector;
	     if(!setBaseline && sectorNameBaselineTree){
	       sectorNameBaselineTree->SetBranchAddress(ss_sector.str().c_str(), &a_sectorBaselineName[iSector]);
	       sectorNameBaselineTree->GetEntry(0);
	     }

	     if(firstIter){ // should be always true in setBaseline mode, since file is recreated  
	       ss_sectorSuffixed << ss_sector.str() << "/D";
	       defaultTreeX->Branch(ss_sector.str().c_str(), &a_defaultSectorX[iSector], ss_sectorSuffixed.str().c_str());
	       if(pixelSector){
	       defaultTreeY->Branch(ss_sector.str().c_str(), &a_defaultSectorY[iSector], ss_sectorSuffixed.str().c_str());
	       }
	       sectorNameTree->Branch(ss_sector.str().c_str(), &a_sectorName[iSector], 32000, 00);
	     }
	     else{
	       defaultTreeX->SetBranchAddress(ss_sector.str().c_str(), &a_defaultSectorX[iSector]);
	       defaultTreeX->GetEntry(0);
	       if(pixelSector){
	       defaultTreeY->SetBranchAddress(ss_sector.str().c_str(), &a_defaultSectorY[iSector]);
	       defaultTreeY->GetEntry(0);
	       }
	       sectorNameTree->SetBranchAddress(ss_sector.str().c_str(), &a_sectorName[iSector]);
	       sectorNameTree->GetEntry(0);
	     }
	   }
	   
	   
	   // Check whether sector definitions are identical with the ones of previous iterations and with the ones in baseline file
	   for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
	     const std::string& name(i_sector->second.name);
	     if(!firstIter){
	       const std::string& nameLastIter(*a_sectorName[(*i_sector).first]);
	       if(name!=nameLastIter){
	         edm::LogError("CalculateAPE")<<"Inconsistent sector definition in iterationFile for sector "<<i_sector->first<<",\n"
	                                      <<"Recent iteration has name \""<<name<<"\", while previous had \""<<nameLastIter<<"\"\n"
					      <<"...APE calculation stopped. Please check sector definitions in config!\n";
	         return;
	       }
	     }
	     else a_sectorName[(*i_sector).first] = new std::string(name);
	     if(!setBaseline && baselineFile){
	       const std::string& nameBaseline(*a_sectorBaselineName[(*i_sector).first]);
	       if(name!=nameBaseline){
	         edm::LogError("CalculateAPE")<<"Inconsistent sector definition in baselineFile for sector "<<i_sector->first<<",\n"
	                                      <<"Recent iteration has name \""<<name<<"\", while baseline had \""<<nameBaseline<<"\"\n"
					      <<"...APE calculation stopped. Please check sector definitions in config!\n";
	         return;
	       }
	     }
	   }
	   
	   
	   // Loop over sectors for calculating getting default APE
	   
	   for(std::map<unsigned int,TrackerSectorStruct>::iterator i_sector = m_tkSector_.begin(); i_sector != m_tkSector_.end(); ++i_sector){
	     
	     double defaultApeX(0.);
	     double defaultApeY(0.);
	     unsigned int nModules(0);
	     std::vector<unsigned int>::const_iterator i_rawId;
	     for(i_rawId = (*i_sector).second.v_rawId.begin(); i_rawId != (*i_sector).second.v_rawId.end(); ++i_rawId){
			 std::vector<AlignTransformErrorExtended> alignErrors = alignmentErrors->m_alignError;
			 for(std::vector<AlignTransformErrorExtended>::const_iterator i_alignError = alignErrors.begin(); i_alignError != alignErrors.end(); ++i_alignError){
				 if(*i_rawId ==  i_alignError->rawId()){
					 CLHEP::HepSymMatrix errMatrix = i_alignError->matrix();
					 defaultApeX += errMatrix[0][0];
					 defaultApeY += errMatrix[1][1];
					 nModules++;
				 }
			 }
		 }
		 a_defaultSectorX[(*i_sector).first] = defaultApeX/nModules;
		 a_defaultSectorY[(*i_sector).first] = defaultApeY/nModules;
		 
	   }
	   sectorNameTree->Fill();
	   sectorNameTree->Write("nameTree");
	   defaultTreeX->Fill();
	   defaultTreeX->Write("iterTreeX");
	   defaultTreeY->Fill();
	   defaultTreeY->Write("iterTreeY");
	   
	   defaultFile->Close();
	   if(baselineFile)baselineFile->Close();
	   
	   
	   firstEvent = false;   
   }
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
ApeEstimatorSummary::beginJob()
{
  this->openInputFile();
  
  this->getTrackerSectorStructs();
  
  this->bookHists();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ApeEstimatorSummary::endJob() {
  
  this->calculateApe();
  
  this->writeHists();
  
  
}

//define this as a plug-in
DEFINE_FWK_MODULE(ApeEstimatorSummary);
