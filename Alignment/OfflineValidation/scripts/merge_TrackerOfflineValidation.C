/*

NOTE: This macro is based on the hadd macro provided by root.
The merging of a TTree is different from what hadd does to merge trees.
The Tree is once read out for entries which are the same in all input
variables - like global position variables for each module - and the values
are copied into a new Ttree.
The mean and RMS values for the different residuals are taken from the merged 
histograms and filled into the tree after the merging procedure.

File names can be given in two ways: 
1)
Provide a comma separated list of filenames as argument to hadd.
2)
Use default empty argument. You will be asked for common file name 'prefix'
and the number of files to be merged.
File names will then be generated as 
<prefix>1.root, <prefix>2.root, <prefix>3.root, etc.
until the number specified.

NOTE:
Includes from CMSSW release as needed below need CMSSW environment as
you can get by these lines (add them to your rootlogon.C):

#include "TSystem.h"
if (gSystem->Getenv("CMSSW_RELEASE_BASE") != '\0') {
  printf("\nLoading CMSSW FWLite...\n");
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
}

*/

// This line works only if we have a CMSSW environment...
#include "Alignment/OfflineValidation/interface/TkOffTreeVariables.h"

#include "TROOT.h"
#include <string>
#include <map>
#include <iostream>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TKey.h"
#include "TF1.h"
#include "TString.h"
#include "TObjString.h"
#include "TMath.h"
// global variables:
bool copiedTree_ = false;
std::map<unsigned int, TkOffTreeVariables> map_;
std::map<unsigned int, TString> idPathMap_;

// methods
void MergeRootfile( TDirectory *target, TList *sourcelist);
void RewriteTree( TDirectory *target,TTree *tree);
std::pair<float,float> FitResiduals(TH1 *h, float meantmp, float rmstmp);
float getMedian(const TH1 *histo);
//////////////////////////////////////////////////////////////////////////
// master method
//////////////////////////////////////////////////////////////////////////
void hadd(const char *filesSeparatedByKommaOrEmpty = "", const char * outputFile = "") {
//void merge_TrackerOfflineValidation(const char *filesSeparatedByKommaOrEmpty = "") {

  TString fileNames(filesSeparatedByKommaOrEmpty);
  TObjArray *names = 0;

  if (fileNames.Length() == 0) {
    // nothing specified, so read in
    TString inputFileName="";
    Int_t nEvt = 0;
    //get file name
    std::cout<<"Type in general file name (ending is added automatically)."<<std::endl;
    std::cout<<"i.e. Validation_CRUZET_data as name, code adds 1,2,3 + .root"<<std::endl;
    std::cin>>inputFileName;
    
    std::cout << "Type number of files to merge." << std::endl;
    std::cin>>nEvt;

    names = new TObjArray;
    names->SetOwner();
    for (Int_t i = 1; i != nEvt+1 ; ++i){
      TString fileName = inputFileName;
      names->Add(new TObjString((fileName += i) += ".root"));
    }
  } else {
    // decode file name from komma separated list
    names = fileNames.Tokenize(","); // is already owner
  }

  int iFilesUsed=0, iFilesSkipped=0;
  const TString keyName = "TrackerOfflineValidationStandalone";
  TList *FileList = new TList();
  for (Int_t iFile = 0; iFile < names->GetEntriesFast(); ++iFile) {
    TFile *file = TFile::Open(names->At(iFile)->GetName());
    if (file) {
      // check if file is ok and contains data
      if (file->FindKey(keyName)) {
	FileList->Add(file);
	std::cout << names->At(iFile)->GetName() << std::endl;
	iFilesUsed++;
      }
      else {
	std::cout << names->At(iFile)->GetName() 
		  << " --- does not contain data, skipping file " << std::endl;
	iFilesSkipped++;
      }
    } else {
      cout << "File " << names->At(iFile)->GetName() << " does not exist!" << endl;
      delete names; names = 0;
      return;
    }
  }
  delete names;

  TString outputFileString;

  if (strlen(outputFile)!=0) 
    outputFileString = TString("$OUTPUTDIR/")+ TString(outputFile);
  else
    outputFileString = "$OUTPUTDIR/merge_output.root";
  
  TFile *Target = TFile::Open( outputFileString, "RECREATE" );
  MergeRootfile( Target, FileList);
  std::cout << "Finished merging of histograms." << std::endl;

  Target->cd( keyName );
  TTree *tree = new TTree("TkOffVal","TkOffVal");
  RewriteTree(Target,tree);
  tree->Write();
  std::cout << std::endl;
  std::cout << "Read " << iFilesUsed << " files, " << iFilesSkipped << " files skipped" << std::endl;
  if (iFilesSkipped>0) 
    std::cout << " (maybe because there was no data in those files?)" << std::endl;
  std::cout << "Written Tree, now saving hists..." << std::endl;

  //  Target->SaveSelf(kTRUE);
  std::cout << "Closing file " << Target->GetName() << std::endl;

  // There is a bug in this file(?) and the script gets
  // stuck in the line "delete Target"
  // without the followin 
  std::cout << endl; 
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << "---  Because of a bug in the code, ROOT gets stuck when  --------" << std::endl;
  std::cout << "---  closing the file.  In that case, please do the following ---" << std::endl;
  std::cout << "--- 1) hit CTRL-Z -----------------------------------------------" << std::endl;
  std::cout << "--- 2) check the ID number or the process root with ps---------- " << std::endl;
  std::cout << "--- 3) kill the root process ------------------------------------" << std::endl;
  std::cout << "--- 4) continue plotting with fg (only a few minutes left)------ " << std::endl;
  std::cout << "-- ------------------------------------------------------------- " << std::endl;

 
  //  gROOT->GetListOfFiles()->Remove(Target);

  // tested, does not work
  //  Target->cd();
  Target->Close();
  //delete Target;

  // The abort() command is ugly, but much quicker than the clean return()
  // Use of return() can take 90 minutes, while abort() takes 10 minutes
  // (merging 20 jobs with 1M events in total)
  abort();

  std::cout << "Now returning from merge_TrackerOfflineValidation.C" << std::endl;
  return;
} 

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
void MergeRootfile( TDirectory *target, TList *sourcelist) {

 
  TString path( (char*)strstr( target->GetPath(), ":" ) );
  path.Remove( 0, 2 );
  //  TString tmp;
  TFile *first_source = (TFile*)sourcelist->First();
  first_source->cd( path );
  TDirectory *current_sourcedir = gDirectory;
  //gain time, do not add the objects in the list in memory
  Bool_t status = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE);

  // loop over all keys in this directory

  TIter nextkey( current_sourcedir->GetListOfKeys() );
  TKey *key, *oldkey=0;
  while ( (key = (TKey*)nextkey())) {

    //keep only the highest cycle number for each key
    if (oldkey && !strcmp(oldkey->GetName(),key->GetName())) continue;

    // read object from first source file
    first_source->cd( path );
    TObject *obj = key->ReadObj();

    if ( obj->IsA()->InheritsFrom( "TH1" ) ) {
      // descendant of TH1 -> merge it

      // But first note the path for this moduleId!
      const TString histName(obj->GetName());
      if (histName.Contains("module_")){
	// GF: fragile! have to look after 'module_'!
	const TString tmp = histName(histName.Last('_')+1, 9);
	unsigned int moduleId = tmp.Atoi(); // check for valid ID???
	idPathMap_[moduleId] = path + '/'; // only for first hist...?
      }
      
      TH1 *h1 = (TH1*)obj;

      // loop over all source files and add the content of the
      // correspondant histogram to the one pointed to by "h1"
      TFile *nextsource = (TFile*)sourcelist->After( first_source );
      while ( nextsource ) {
        
        // make sure we are at the correct directory level by cd'ing to path
        nextsource->cd( path );
        TKey *key2 = (TKey*)gDirectory->GetListOfKeys()->FindObject(h1->GetName());
        if (key2) {
	  TH1 *h2 = (TH1*)key2->ReadObj();
	  h1->Add( h2 );
	  delete h2;
        }

        nextsource = (TFile*)sourcelist->After( nextsource );
      }
    }
    else if ( obj->IsA()->InheritsFrom( "TTree" ) ) {
      if (!copiedTree_ && TString("TkOffVal") == obj->GetName()) {
	// get tree structure and 'const' entries for each module once 
	// ('constant' means not effected by merging)
	TTree* tree =(TTree*)obj;

	TkOffTreeVariables *treeMem = 0; // ROOT will initilise
	tree->SetBranchAddress("TkOffTreeVariables", &treeMem);
	
	std::cout << "Copy info from first TTree." << std::endl;
	Long64_t nentries = tree->GetEntriesFast();
	for (Long64_t jentry = 0; jentry < nentries; ++jentry) {
	  if (jentry%1000==0)
	    std::cout << "Copy entry " << jentry << " / " << nentries << std::endl;
	  tree->GetEntry(jentry);
	  // Erase everything not common:
	  treeMem->clearMergeAffectedPart();
	  map_[treeMem->moduleId] = *treeMem;
	}
	std::cout << "Done copy info from first TTree." << std::endl;
	copiedTree_ = true;
      }
    } else if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      // it's a subdirectory
      //      cout << "Found subdirectory " << obj->GetName() << endl;

      // create a new subdir of same name and title in the target file
      target->cd();
      TDirectory *newdir = target->mkdir( obj->GetName(), obj->GetTitle() );

      // newdir is now the starting point of another round of merging
      // newdir still knows its depth within the target file via
      // GetPath(), so we can still figure out where we are in the recursion
      MergeRootfile( newdir, sourcelist );

    } else {

      // object is of no type that we know or can handle
      cout << "Unknown object type, name: " 
           << obj->GetName() << " title: " << obj->GetTitle() << endl;
    }

    // now write the merged histogram (which is "in" obj) to the target file
    // note that this will just store obj in the current directory level,
    // which is not persistent until the complete directory itself is stored
    // by "target->Write()" below
    if ( obj ) {
      target->cd();

      if(obj->IsA()->InheritsFrom( "TTree" ))cout<<"TTree will be merged separately"<<endl;
      else
	obj->Write( key->GetName() );
     
    }

  } // while ( ( TKey *key = (TKey*)nextkey() ) )

  // save modifications to target file
  target->SaveSelf(kTRUE);
  TH1::AddDirectory(status);

  return;
}

//////////////////////////////////////////////////////////////////////////////////
void RewriteTree( TDirectory *target, TTree *tree)

{ 
  TkOffTreeVariables *treeVar = new TkOffTreeVariables;
  tree->Branch("TkOffTreeVariables", &treeVar);
  
  // As above: gain time, do not add the objects in the list in memory.
  // In addition avoids saving hists twice...
  const Bool_t status = TH1::AddDirectoryStatus();
  TH1::AddDirectory(kFALSE);
  
  std::cout << "Create merged TTree:" << std::endl;
  TH1 *h = 0;
  unsigned int counter = 0;
  for (std::map<unsigned int,TkOffTreeVariables>::const_iterator it = map_.begin(), 
	 itEnd = map_.end(); it != itEnd; ++it, ++counter) {
    treeVar->clear(); 
    // first get 'constant' values from map ('constant' means not effected by merging)
    (*treeVar) = it->second; // (includes module id)
    
    // now path name:
    const TString &path = idPathMap_[treeVar->moduleId]; 
    
    // Read out mean and rms values from merged histograms,
    // fit for fitted values etc.
    
    //////////////////////////////////////////////////////////////
    // xPrime
    //////////////////////////////////////////////////////////////
    target->GetObject(path + treeVar->histNameX, h);
    if (h) {
      treeVar->entries = static_cast<UInt_t>(h->GetEntries());   //entries are const for all histos
      treeVar->meanX = h->GetMean(); //get mean value from histogram 
      treeVar->rmsX = h->GetRMS();   //get RMS value from histogram
      
      int numberOfBins = h->GetNbinsX();
      treeVar->numberOfUnderflows = h->GetBinContent(0);
      treeVar->numberOfOverflows = h->GetBinContent(numberOfBins+1);
      treeVar->numberOfOutliers =  h->GetBinContent(0)+h->GetBinContent(numberOfBins+1);
      
      const std::pair<float,float> meanSigma = FitResiduals(h, h->GetMean(), h->GetRMS());
      treeVar->fitMeanX = meanSigma.first;
      treeVar->fitSigmaX= meanSigma.second;
      treeVar->medianX = getMedian(h);
      delete h; h = 0;
    } else {
      std::cout << "Module " << treeVar->moduleId << " without hist X: " 
      		<< path << treeVar->histNameX << std::endl;
    }

    ////////////////////////////////////////////////////////////////
    // normalized xPrime
    ////////////////////////////////////////////////////////////////
    target->GetObject(path + treeVar->histNameNormX, h);
    if (h) {
      treeVar->meanNormX = h->GetMean(); //get mean value from histogram 
      treeVar->rmsNormX = h->GetRMS();   //get RMS value from histogram
	
      const std::pair<float,float> meanSigma = FitResiduals(h, h->GetMean(), h->GetRMS());
      treeVar->fitMeanNormX  = meanSigma.first; // get mean value from histogram
      treeVar->fitSigmaNormX = meanSigma.second; // get sigma value from histogram
      
      double stats[20];
      h->GetStats(stats);
      // 	treeVar->chi2PerDofX = stats[3]/(stats[0]-1); // GF: why -1?
      if (stats[0]) treeVar->chi2PerDofX = stats[3]/stats[0];
      delete h; h = 0;
    } else {
      std::cout << "Module " << treeVar->moduleId << " without hist normX: " 
		<< path << treeVar->histNameNormX << std::endl;
    }
    
    ////////////////////////////////////////////////////////////////
    // local x if present
    ////////////////////////////////////////////////////////////////
    if (treeVar->histNameLocalX.size()) {
      target->GetObject(path + treeVar->histNameLocalX, h);
      if (h) {
	treeVar->meanLocalX = h->GetMean();      //get mean value from histogram
	treeVar->rmsLocalX = h->GetRMS();        //get RMS value from histogram
	
	delete h; h = 0;
      } else {
	std::cout << "Module " << treeVar->moduleId << " without hist local X: " 
		  << path << treeVar->histNameLocalX << std::endl;
      }
    }
      
    ////////////////////////////////////////////////////////////////
    // local normalized x if present
    ////////////////////////////////////////////////////////////////
    if (treeVar->histNameNormLocalX.size()) {
      target->GetObject(path + treeVar->histNameNormLocalX, h);
      if (h) {
	treeVar->meanNormLocalX = h->GetMean();//get mean value from histogram
	treeVar->rmsNormLocalX = h->GetRMS();  //get RMS value from histogram
	delete h; h = 0;
      } else {
	std::cout << "Module " << treeVar->moduleId << " without hist normLocal X" 
		  << path << treeVar->histNameNormLocalX << std::endl;
      }
    }

    ////////////////////////////////////////////////////////////////
    // yPrime if existing (pixel and, if configured, for strip)
    ////////////////////////////////////////////////////////////////
    if (treeVar->histNameY.size()) {
      target->GetObject(path + treeVar->histNameY, h);
      if (h) {
	treeVar->meanY = h->GetMean();      //get mean value from histogram
	treeVar->rmsY = h->GetRMS();        //get RMS value from histogram

	const std::pair<float,float> meanSigma = FitResiduals(h, h->GetMean(), h->GetRMS());
	treeVar->fitMeanY = meanSigma.first;
	treeVar->fitSigmaY= meanSigma.second;
	treeVar->medianY = getMedian(h);
	delete h; h = 0;
      } else {
	std::cout << "Module " << treeVar->moduleId << " without hist Y " 
		  << path << treeVar->histNameY << std::endl;
      }
    }

    ////////////////////////////////////////////////////////////////
    // normalized yPrime
    ////////////////////////////////////////////////////////////////
    if (treeVar->histNameNormY.size()) {
      target->GetObject(path + treeVar->histNameNormY, h);
      if (h) {
	treeVar->meanNormY = h->GetMean(); //get mean value from histogram
	treeVar->rmsNormY = h->GetRMS();   //get RMS value from histogram

	const std::pair<float,float> meanSigma = FitResiduals(h, h->GetMean(), h->GetRMS());
	treeVar->fitMeanNormY  = meanSigma.first; // get mean value from histogram
	treeVar->fitSigmaNormY = meanSigma.second; // get sigma value from histogram

	double stats[20];
	h->GetStats(stats);
	if (stats[0]) treeVar->chi2PerDofY = stats[3]/stats[0];
	delete h; h = 0;
      } else {
	std::cout << "Module " << treeVar->moduleId << " without hist norm Y " 
		  << path << treeVar->histNameNormY << std::endl;
      }
    }
     
    tree->Fill();
  } // end loop on modules

  TH1::AddDirectory(status); // back to where we were before
}

///////////////////////////////////////////////////////////////////////////
std::pair<float,float> FitResiduals(TH1 *h,float meantmp,float rmstmp)
{
  std::pair<float,float> fitResult(9999., 9999.);
  if (!h || h->GetEntries() < 20) return fitResult;

  // first fit: two RMS around mean
  TF1 func("tmp", "gaus", meantmp - 2.*rmstmp, meantmp + 2.*rmstmp); 
  if (0 == h->Fit(&func,"QNR")) { // N: do not blow up file by storing fit!
    float mean  = func.GetParameter(1);
    float sigma = func.GetParameter(2);

    // second fit: three sigma of first fit around mean of first fit
    func.SetRange(mean - 3.*sigma, mean + 3.*sigma);
    // I: integral gives more correct results if binning is too wide (slow)
    // L: Likelihood can treat empty bins correctly (if hist not weighted...)
    if (0 == h->Fit(&func, "QNRL")) { // I")) {
      fitResult.first = func.GetParameter(1);
      fitResult.second = func.GetParameter(2);
    }
  }

  return fitResult;
}

float getMedian(const TH1 *histo)
{

  float median = 999;
  int nbins = histo->GetNbinsX();

 //extract median from histogram
   
   double *x = new double[nbins];
  double *y = new double[nbins];
  for (int j = 0; j < nbins; j++) {
    x[j] = histo->GetBinCenter(j+1);
    y[j] = histo->GetBinContent(j+1);
  }
  median = TMath::Median(nbins, x, y);
  

  delete[] x; x = 0;
  delete[] y; y = 0;

  return median;

}
