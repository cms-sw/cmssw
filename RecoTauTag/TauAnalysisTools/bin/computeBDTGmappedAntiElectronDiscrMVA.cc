
/** \executable computeBDTGmappedAntiElectronDiscrMVA
 *
 * Map MVA output for different categories into unique "final" (category independent) number.
 *
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: computeBDTGmappedAntiElectronDiscrMVA.cc,v 1.1 2012/03/06 17:34:42 veelken Exp $
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/FWLite/interface/TFileService.h"
#include "DataFormats/FWLite/interface/InputSource.h"
#include "DataFormats/FWLite/interface/OutputFiles.h"

#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TObjArray.h>
#include <TBranch.h>
#include <TString.h>
#include <TMath.h>
#include <TBenchmark.h>

#include <iostream>
#include <string>
#include <vector>
#include <assert.h>

struct workingPointEntryType
{
  double targetSignalEfficiency_;
  double minPt_;
  double maxPt_;
  typedef std::map<Int_t, double> wpMap; // key = category
  wpMap cuts_;
  double S_;
  double B_;
  double SoverB_;
};

std::vector<workingPointEntryType> readWorkingPoints(const std::string& wpFileName, const std::string& wpTreeName, const std::vector<int>& categories)
{
  TFile* wpFile = new TFile(wpFileName.data());
  if ( !wpFile )
    throw cms::Exception("computeBDTGmappedAntiElectronDiscrMVA") 
      << " Failed to open File = " << wpFileName << " !!\n";
  
  TTree* wpTree = dynamic_cast<TTree*>(wpFile->Get(wpTreeName.data()));
  if ( !wpTree ) 
    throw cms::Exception("computeBDTGmappedAntiElectronDiscrMVA") 
      << " Failed to lood Tree = " << wpTreeName << " from File = " << wpFileName << " !!\n";

  Float_t targetSignalEfficiency;
  wpTree->SetBranchAddress("targetSignalEfficiency", &targetSignalEfficiency);
  
  Float_t minPt;
  wpTree->SetBranchAddress("minPt", &minPt);
  Float_t maxPt;
  wpTree->SetBranchAddress("maxPt", &maxPt);

  std::map<Int_t, Float_t> cuts; // key = category
  for ( std::vector<int>::const_iterator category = categories.begin();
	category != categories.end(); ++category ) {
    std::string branchName = Form("cutCategory%i", *category);
    wpTree->SetBranchAddress(branchName.data(), &cuts[*category]);
  }
    
  Float_t S;
  wpTree->SetBranchAddress("S", &S);
  Float_t B;
  wpTree->SetBranchAddress("B", &B);
  Float_t SoverB;
  wpTree->SetBranchAddress("SoverB", &SoverB);

  std::vector<workingPointEntryType> workingPoints;

  int numWorkingPoints = wpTree->GetEntries();
  for ( int iWorkingPoint = 0; iWorkingPoint < numWorkingPoints; ++iWorkingPoint ) {
    wpTree->GetEntry(iWorkingPoint);

    workingPointEntryType workingPoint;

    std::cout << "targetSignalEfficiency = " << targetSignalEfficiency << ":" << std::endl;
    workingPoint.targetSignalEfficiency_ = targetSignalEfficiency;

    workingPoint.minPt_ = minPt;
    workingPoint.maxPt_ = maxPt;

    for ( std::vector<int>::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      std::cout << " category #" << (*category) << ": cut = " << cuts[*category] << std::endl;
      workingPoint.cuts_[*category] = cuts[*category];
    }

    std::cout << "S = " << S << ", B = " << B << " --> S/B = " << (S/B) << std::endl;
    std::cout << std::endl;
    workingPoint.S_ = S;
    workingPoint.B_ = B;
    workingPoint.SoverB_ = SoverB;

    workingPoints.push_back(workingPoint);
  }

  delete wpTree;
  delete wpFile;

  return workingPoints;
}

struct branchEntryType
{
   branchEntryType()
     : inputValueF_(0.),
       inputValueI_(0),
       inputValueL_(0),
       outputValueF_(0.),
       outputValueI_(0),
       outputValueL_(0)
  {}
  ~branchEntryType() {}
  void copyInputToOutput()
  {
    outputValueF_ = inputValueF_;
    outputValueI_ = inputValueI_;
    outputValueL_ = inputValueL_;
  }
  std::string branchName_;
  enum { kInt_t, kFloat_t, kLong_t };
  int branchType_;
  Float_t inputValueF_;
  Int_t inputValueI_;
  ULong64_t inputValueL_;
  Float_t outputValueF_;
  Int_t outputValueI_;
  ULong64_t outputValueL_;
};

typedef std::vector<std::string> vstring;
typedef std::vector<int> vint;

int main(int argc, char* argv[]) 
{
//--- parse command-line arguments
  if ( argc < 2 ) {
    std::cout << "Usage: " << argv[0] << " [parameters.py]" << std::endl;
    return 0;
  }

  std::cout << "<computeBDTGmappedAntiElectronDiscrMVA>:" << std::endl;

//--- keep track of time it takes the macro to execute
  TBenchmark clock;
  clock.Start("computeBDTGmappedAntiElectronDiscrMVA");

//--- read python configuration parameters
  if ( !edm::readPSetsFrom(argv[1])->existsAs<edm::ParameterSet>("process") ) 
    throw cms::Exception("computeBDTGmappedAntiElectronDiscrMVA") 
      << "No ParameterSet 'process' found in configuration file = " << argv[1] << " !!\n";

  edm::ParameterSet cfg = edm::readPSetsFrom(argv[1])->getParameter<edm::ParameterSet>("process");

  edm::ParameterSet cfgComputeBDTGmappedAntiElectronDiscrMVA = cfg.getParameter<edm::ParameterSet>("computeBDTGmappedAntiElectronDiscrMVA");
  
  std::string inputTreeName = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<std::string>("inputTreeName");
  std::string outputTreeName = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<std::string>("outputTreeName");

  std::string branchName_mvaOutput = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<std::string>("branchName_mvaOutput");
  std::string branchName_categoryIdx = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<std::string>("branchName_categoryIdx");
  std::string branchName_tauPt = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<std::string>("branchName_tauPt");
  std::string branchName_logTauPt = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<std::string>("branchName_logTauPt");

  std::vector<int> categories = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<vint>("categories");
  
  std::string wpFileName = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<std::string>("wpFileName");
  std::string wpTreeName = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<std::string>("wpTreeName");
  std::vector<workingPointEntryType> workingPoints = readWorkingPoints(wpFileName, wpTreeName, categories);

  fwlite::InputSource inputFiles(cfg); 
  int maxEvents = inputFiles.maxEvents();
  std::cout << " maxEvents = " << maxEvents << std::endl;
  unsigned reportEvery = inputFiles.reportAfter();

  std::string outputFileName = cfgComputeBDTGmappedAntiElectronDiscrMVA.getParameter<std::string>("outputFileName");
  std::cout << " outputFileName = " << outputFileName << std::endl;

  TChain* inputTree = new TChain(inputTreeName.data());
  for ( vstring::const_iterator inputFileName = inputFiles.files().begin();
	inputFileName != inputFiles.files().end(); ++inputFileName ) {
    std::cout << "input Tree: adding file = " << (*inputFileName) << std::endl;
    inputTree->AddFile(inputFileName->data());
  }
  
  if ( !(inputTree->GetListOfFiles()->GetEntries() >= 1) ) {
    throw cms::Exception("computeBDTGmappedAntiElectronDiscrMVA") 
      << "Failed to identify input Tree !!\n";
  }

  std::cout << "input Tree contains " << inputTree->GetEntries() << " Entries in " << inputTree->GetListOfFiles()->GetEntries() << " files." << std::endl;

  std::vector<branchEntryType*> branches_to_copy;
  branchEntryType* branch_mvaOutput   = 0;
  branchEntryType* branch_categoryIdx = 0;
  branchEntryType* branch_tauPt       = 0;
  branchEntryType* branch_logTauPt    = 0;

  TObjArray* branches = inputTree->GetListOfBranches();
  int numBranches = branches->GetEntries();
  for ( int iBranch = 0; iBranch < numBranches; ++iBranch ) {
    TBranch* branch = dynamic_cast<TBranch*>(branches->At(iBranch));
    assert(branch);
    std::string branchName = branch->GetName();
    // CV: skip copying branches which do not exist in training trees of all categories
    if ( !(branchName == "classID" || branchName == branchName_mvaOutput || branchName == "weight" || branchName == branchName_categoryIdx || branchName.find("Tau_Pt") != std::string::npos) ) continue;
    std::string branchType_string = TString(branch->GetTitle()).ReplaceAll(Form("%s/", branchName.data()), "").Data();
    //std::cout << "branch #" << iBranch << ": name = " << branchName << ", type = " << branchType_string << std::endl;
    int branchType = -1;
    if      ( branchType_string == "I" ) branchType = branchEntryType::kInt_t;
    else if ( branchType_string == "F" ) branchType = branchEntryType::kFloat_t;
    else if ( branchType_string == "l" ) branchType = branchEntryType::kLong_t;
    else {
      std::cerr << "<computeBDTGmappedAntiElectronDiscrMVA>:" << std::endl;
      std::cerr << " Branch type = " << branchType_string << " not supported --> Branch = " << branchName << " will NOT be copied to outputTree !!" << std::endl;
      continue;
    }
    branchEntryType* branch_to_copy = new branchEntryType();
    branch_to_copy->branchName_ = branchName;
    branch_to_copy->branchType_ = branchType;
    branches_to_copy.push_back(branch_to_copy);
    if ( branchName == branchName_mvaOutput   ) branch_mvaOutput   = branch_to_copy;
    if ( branchName == branchName_categoryIdx ) branch_categoryIdx = branch_to_copy;
    if ( branchName == branchName_tauPt       ) branch_tauPt       = branch_to_copy;
    if ( branchName == branchName_logTauPt    ) branch_logTauPt    = branch_to_copy;
  }
  if ( !(branch_mvaOutput && branch_categoryIdx) ) 
    throw cms::Exception("computeBDTGmappedAntiElectronDiscrMVA") 
      << "Failed to find Branches '" << branchName_mvaOutput << "' and '" << branchName_categoryIdx << "' in input Tree !!\n";
  if ( !(branch_tauPt || branch_logTauPt) )
    throw cms::Exception("computeBDTGmappedAntiElectronDiscrMVA") 
      << "Failed to find either one of the Branches '" << branchName_tauPt << "' and '" << branchName_logTauPt << "' in input Tree !!\n";
  
  for ( std::vector<branchEntryType*>::iterator branch = branches_to_copy.begin();
        branch != branches_to_copy.end(); ++branch ) {
    if ( (*branch)->branchType_ == branchEntryType::kInt_t ) {
      std::cout << "copying branch = " << (*branch)->branchName_ << ", type = Int_t" << std::endl;
      inputTree->SetBranchAddress((*branch)->branchName_.data(), &(*branch)->inputValueI_);
    } else if ( (*branch)->branchType_ == branchEntryType::kLong_t ) {
      std::cout << "copying branch = " << (*branch)->branchName_ << ", type = ULong64_t" << std::endl;
      inputTree->SetBranchAddress((*branch)->branchName_.data(), &(*branch)->inputValueL_);
    } else if ( (*branch)->branchType_ == branchEntryType::kFloat_t ) {
      std::cout << "copying branch = " << (*branch)->branchName_ << ", type = Float_t" << std::endl;
      inputTree->SetBranchAddress((*branch)->branchName_.data(), &(*branch)->inputValueF_);
    } else assert(0);
  }

  TFile* outputFile = new TFile(outputFileName.data(), "RECREATE");
  TTree* outputTree = new TTree(outputTreeName.data(), outputTreeName.data());

  for ( std::vector<branchEntryType*>::iterator branch = branches_to_copy.begin();
        branch != branches_to_copy.end(); ++branch ) {
    if ( (*branch)->branchType_ == branchEntryType::kInt_t ) {
      outputTree->Branch((*branch)->branchName_.data(), &(*branch)->outputValueI_, Form("%s/I", (*branch)->branchName_.data()));
    } else if ( (*branch)->branchType_ == branchEntryType::kLong_t ) {
      outputTree->Branch((*branch)->branchName_.data(), &(*branch)->outputValueL_, Form("%s/l", (*branch)->branchName_.data()));
    } else if ( (*branch)->branchType_ == branchEntryType::kFloat_t ) {
      outputTree->Branch((*branch)->branchName_.data(), &(*branch)->outputValueF_, Form("%s/F", (*branch)->branchName_.data()));    
    } else assert(0);
  }

  Float_t mvaOutput_mapped;
  outputTree->Branch("BDTGmapped", &mvaOutput_mapped, "BDTGmapped/F");    

  int numEntries = inputTree->GetEntries();
  for ( int iEntry = 0; iEntry < numEntries && (maxEvents == -1 || iEntry < maxEvents); ++iEntry ) {
    if ( iEntry > 0 && (iEntry % reportEvery) == 0 ) {
      std::cout << "processing Entry " << iEntry << std::endl;
    }
    
    inputTree->GetEntry(iEntry);

    for ( std::vector<branchEntryType*>::iterator branch = branches_to_copy.begin();
          branch != branches_to_copy.end(); ++branch ) {
      (*branch)->copyInputToOutput();
    }

    Float_t mvaOutput = branch_mvaOutput->inputValueF_;
    Int_t categoryIdx = TMath::Nint(branch_categoryIdx->inputValueF_); // CV: TMVA stores Tau_Category branch in floating-point format !!
    
    double tauPt_value;
    if      ( branch_logTauPt ) tauPt_value = TMath::Exp(branch_logTauPt->inputValueF_);
    else if ( branch_tauPt    ) tauPt_value = branch_tauPt->inputValueF_;
    else assert(0);

    double SoverBmax = 0.;
    for ( vint::const_iterator category = categories.begin();
	  category != categories.end(); ++category ) {
      if ( categoryIdx & (1<<(*category)) ) {
	for ( std::vector<workingPointEntryType>::iterator workingPoint = workingPoints.begin();
	      workingPoint != workingPoints.end(); ++workingPoint ) {
	  if ( tauPt_value > workingPoint->minPt_ && tauPt_value < workingPoint->maxPt_ &&
	       mvaOutput > workingPoint->cuts_[*category] && workingPoint->SoverB_ > SoverBmax ) SoverBmax = workingPoint->SoverB_;
	}
      }
    }

    mvaOutput_mapped = 2.*(SoverBmax/(SoverBmax + 1.)) - 1.;

    //std::cout << "Entry #" << iEntry << ": categoryIdx = " << categoryIdx << ", mvaOutput = " << mvaOutput << " --> (S/B)max = " << SoverBmax << ", mvaOutput(mapped) = " << mvaOutput_mapped << std::endl;
	
    outputTree->Fill();
  }

  std::cout << "--> " << outputTree->GetEntries() << " Entries processed." << std::endl;

  std::cout << "output Tree:" << std::endl;
  //outputTree->Print();
  //outputTree->Scan("*", "", "", 20, 0);

  std::cout << "writing output Tree to file = " << outputFileName << "." << std::endl;
  outputFile->cd();
  outputTree->Write();

  delete outputFile;

  delete inputTree;

  clock.Show("computeBDTGmappedAntiElectronDiscrMVA");

  return 0;
}
