/*

  NOTE: This macro is based on the hadd macro provided by root.
  The merging of a TTree is different from what hadd does to merge trees.
  The Tree is once read out for entries which are the same in all input
  variables - like global position variables for each module - and the values
  are copied into a new Ttree.
  The mean and RMS values for the different residuals are taken from the merged 
  histograms and filled into the tree after the merging procedure.

  To use the tool, the file name should have the same name+i where i is a number 
  starting at 1. The tool automatically counts up the filenumber to the limit 
  set by the user.

 */


#include <string.h>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TKey.h"
#include "TString.h"
#include <map>
#include "Riostream.h"




TString histPathName;
bool copiedTree = false;
Bool_t moduleLevelHistsTransient = false;
Bool_t lCoorHistOn =false;

struct TreeVariables{
  TreeVariables(): meanLocalX_(), meanNormLocalX_(), meanX_(), meanNormX_(),
		   rmsLocalX_(), rmsNormLocalX_(), rmsX_(), rmsNormX_(), 
		   posR_(), posPhi_(), 
		   posX_(), posY_(), posZ_(),
		   entries_(), moduleId_(), subDetId_(),
		   layer_(), side_(), rod_(),ring_(), 
		   petal_(),blade_(), panel_(), outerInner_(),
		   isDoubleSide_(), histNameLocalX_(), histNameNormLocalX_(), histNameX_(), histNameNormX_(),
		   histPathLocalX_(), histPathNormLocalX_(), histPathX_(), histPathNormX_(){} 
  Float_t meanLocalX_, meanNormLocalX_, meanX_, meanNormX_,    //mean value read out from modul histograms
          rmsLocalX_, rmsNormLocalX_, rmsX_,  rmsNormX_,      //rms value read out from modul histograms
          posR_, posPhi_, posEta_,                     //global coordiantes    
          posX_, posY_, posZ_;               //global coordiantes 
  
  UInt_t   entries_,                         //number of entries for each modul
           moduleId_, subDetId_,             //modul Id = detId and subdetector Id
           layer_, side_, rod_, 
           ring_, petal_, 
           blade_, panel_, outerInner_;      //innerOuter = orientation of modules on mounting unit
  Bool_t isDoubleSide_;
  std::string histNameLocalX_, histNameNormLocalX_, histNameX_, histNameNormX_;
  TString  histPathLocalX_, histPathNormLocalX_, histPathX_, histPathNormX_;
};

std::map<unsigned int,TreeVariables> map_;
void MergeRootfile( TDirectory *target, TList *sourcelist );
void RewriteTree( TDirectory *target,TTree *tree,std::map<unsigned int,TreeVariables> map_);
//void RewriteTree(std::map<unsigned int,TreeVariables> map_);

void hadd() {
  
  TList *FileList;
  TFile *Target;
  Target = TFile::Open( "private_CRUZET3Cosmics_MPAlignObj.root", "RECREATE" );
  TString inputFileName="";
  Int_t nEvt;
  Bool_t fileOk_=true;
  //get file name
  std::cout<<"Type in general file name (ending is added automatically)."<<std::endl;
  std::cin>>inputFileName;
  
  if (inputFileName!=""){
    std::cout<<"number of files to merge"<<std::endl;
    std::cin>>nEvt;
 
    
    FileList = new TList();
    for (Int_t i = 1; i != nEvt+1 ; ++i){
      TString fileName=inputFileName;
     
      fileName+=i;
      fileName+=".root";
      
      if(TFile::Open(fileName)) {
	FileList->Add( TFile::Open(fileName) );
	 std::cout<<fileName<<std::endl;
      }else{ 
	cout<<"The file name "<<fileName<<" does not exists!"<<endl;
	cout<<"The filelist could not be load properly."<<endl;
	fileOk_=false;
	break;
      }   
    }
    if (fileOk_){
      MergeRootfile( Target, FileList );
      std::cout<<"finished merging of histograms"<<std::endl;
      Target->cd("TrackerOfflineValidation");
      TTree *tree = new TTree("TkOffVal","TkOffVal");
      RewriteTree(Target,tree,map_);
      tree->Write();
      Target->SaveSelf(kTRUE);
      Target->Close();
    } 
    
  }else cout<<"There is no file name specifiied."<<endl;  
}

void MergeRootfile( TDirectory *target, TList *sourcelist ) {

 
  TString path( (char*)strstr( target->GetPath(), ":" ) );
  path.Remove( 0, 2 );
  TString tmp;
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
      histPathName = path+'/'+obj->GetName();
      
      
      if (histPathName.Contains("module_")){
	Int_t beginChar= histPathName.Last('_')+1;
	tmp =histPathName.operator()(beginChar,9);
	unsigned int moduleId = tmp.Atoi();
	if(histPathName.Contains("h_residuals"))map_[moduleId].histPathLocalX_= histPathName;
	if(histPathName.Contains("h_normresiduals"))map_[moduleId].histPathNormLocalX_= histPathName;
	if(histPathName.Contains("h_xprime_residuals"))map_[moduleId].histPathX_= histPathName;
	if(histPathName.Contains("h_normxprimeresiduals"))map_[moduleId].histPathNormX_= histPathName;
	moduleLevelHistsTransient=true;
      }
      
      //cout << "Merging histogram " << obj->GetName() << endl;
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
      
      if (!copiedTree)	{
	//get tree structure and 'const' entries for each module once ('constant' means not effected by merging)
	TTree* tree =(TTree*)obj;
	TreeVariables treeMem_;
	tree->SetBranchAddress("moduleId",&treeMem_.moduleId_);
	tree->SetBranchAddress("subDetId",&treeMem_.subDetId_);
	tree->SetBranchAddress("posPhi",&treeMem_.posPhi_);
	tree->SetBranchAddress("posEta",&treeMem_.posEta_);
	tree->SetBranchAddress("posR",&treeMem_.posR_);
	tree->SetBranchAddress("posX",&treeMem_.posX_);
	tree->SetBranchAddress("posY",&treeMem_.posY_);
	tree->SetBranchAddress("posZ",&treeMem_.posZ_);
	
	tree->SetBranchAddress("layer",&treeMem_.layer_);
	tree->SetBranchAddress("side",&treeMem_.side_);
	tree->SetBranchAddress("rod",&treeMem_.rod_);
	tree->SetBranchAddress("ring",&treeMem_.ring_);
	tree->SetBranchAddress("petal",&treeMem_.petal_);
	tree->SetBranchAddress("blade",&treeMem_.blade_);
	tree->SetBranchAddress("panel",&treeMem_.panel_);
	tree->SetBranchAddress("outerInner",&treeMem_.outerInner_);
	tree->SetBranchAddress("isDoubleSide",&treeMem_.isDoubleSide_);
	tree->SetBranchAddress("histNameX",&treeMem_.histNameX_);
	tree->SetBranchAddress("histNameNormX",&treeMem_.histNameX_);
	
	//if residuals for local coordiantartes are present in file
	if (tree->GetBranch("histNameLocalX"))
	  {
	    tree->SetBranchAddress("histNameLocalX",&treeMem_.histNameLocalX_);
	    tree->SetBranchAddress("histNameNormLocalX",&treeMem_.histNameNormLocalX_);
	    lCoorHistOn=true;
	  }

	Long64_t nentries = tree->GetEntriesFast();
	for (Long64_t jentry=1; jentry<nentries+1;jentry++) {
	 
	  tree->GetEntry(jentry);

	  map_[treeMem_.moduleId_].subDetId_= treeMem_.subDetId_;
	  map_[treeMem_.moduleId_].posPhi_= treeMem_.posPhi_;
	  map_[treeMem_.moduleId_].posEta_= treeMem_.posEta_;
	  map_[treeMem_.moduleId_].posR_= treeMem_.posR_;
	  map_[treeMem_.moduleId_].posX_= treeMem_.posX_;
	  map_[treeMem_.moduleId_].posY_= treeMem_.posY_;
	  map_[treeMem_.moduleId_].posZ_= treeMem_.posZ_;
	  map_[treeMem_.moduleId_].layer_= treeMem_.layer_;
	  map_[treeMem_.moduleId_].side_= treeMem_.side_;
	  map_[treeMem_.moduleId_].rod_= treeMem_.rod_;
	  map_[treeMem_.moduleId_].ring_= treeMem_.ring_;
	  map_[treeMem_.moduleId_].petal_= treeMem_.petal_;
	  map_[treeMem_.moduleId_].blade_= treeMem_.blade_;
	  map_[treeMem_.moduleId_].panel_= treeMem_.panel_;
	  map_[treeMem_.moduleId_].outerInner_= treeMem_.outerInner_;
	  map_[treeMem_.moduleId_].isDoubleSide_= treeMem_.isDoubleSide_;
	  map_[treeMem_.moduleId_].histNameX_= treeMem_.histNameX_;
	  map_[treeMem_.moduleId_].histNameNormX_= treeMem_.histNameNormX_;
	  
	  //if residuals for local coordinates are present in file
	  if (tree->GetBranch("histNameLocalX"))
	    {
	      map_[treeMem_.moduleId_].histNameLocalX_= treeMem_.histNameLocalX_;
	      map_[treeMem_.moduleId_].histNameNormLocalX_= treeMem_.histNameNormLocalX_;
	    }
	  
	}
      }
    } else if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
      // it's a subdirectory
      
      cout << "Found subdirectory " << obj->GetName() << endl;

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
}

void RewriteTree( TDirectory *target, TTree *tree, std::map<unsigned int,TreeVariables> map_)

{
  if (moduleLevelHistsTransient){
   
    TreeVariables treeVar_;

    tree->Branch("moduleId",&treeVar_.moduleId_,"modulId/i");
    tree->Branch("subDetId",&treeVar_.subDetId_,"subDetId/i");
    tree->Branch("entries",&treeVar_.entries_,"entries/i");
    tree->Branch("meanLocalX",&treeVar_.meanLocalX_,"meanLocalX/F");
    tree->Branch("rmsLocalX",&treeVar_.rmsLocalX_,"rmsLocalX/F");
    tree->Branch("meanNormLocalX",&treeVar_.meanNormLocalX_,"meanNormLocalX/F");
    tree->Branch("rmsNormLocalX",&treeVar_.rmsNormLocalX_,"rmsNormLocalX/F");
    tree->Branch("meanX",&treeVar_.meanX_,"meanX/F");
    tree->Branch("rmsX",&treeVar_.rmsX_,"rmsX/F");
    tree->Branch("meanNormX",&treeVar_.meanNormX_,"meanNormX/F");
    tree->Branch("rmsNormX",&treeVar_.rmsNormX_,"rmsNormX/F");


    tree->Branch("posPhi",&treeVar_.posPhi_,"posPhi/F");
    tree->Branch("posEta",&treeVar_.posEta_,"posEta/F");
    tree->Branch("posR",&treeVar_.posR_,"posR/F");
    tree->Branch("posX",&treeVar_.posX_,"gobalX/F");
    tree->Branch("posY",&treeVar_.posY_,"posY/F");
    tree->Branch("posZ",&treeVar_.posZ_,"posZ/F");
  
    tree->Branch("layer",&treeVar_.layer_,"layer/i");
    tree->Branch("side",&treeVar_.side_,"side/i");
    tree->Branch("rod",&treeVar_.rod_,"rod/i");
    tree->Branch("ring",&treeVar_.ring_,"ring/i");
    tree->Branch("petal",&treeVar_.petal_,"petal/i");
    tree->Branch("blade",&treeVar_.blade_,"blade/i");
    tree->Branch("panel",&treeVar_.panel_,"panel/i");
    tree->Branch("outerInner",&treeVar_.outerInner_,"outerInner/i");
    tree->Branch("isDoubleSide",&treeVar_.isDoubleSide_,"isDoubleSide/O");
    tree->Branch("histNameLocalX",&treeVar_.histNameLocalX_,"histNameLocalX/b");
    tree->Branch("histNameNormLocalX",&treeVar_.histNameNormLocalX_,"histNameNormLocalX/b");
    tree->Branch("histNameX",&treeVar_.histNameX_,"histNameX/b");
    tree->Branch("histNameNormX",&treeVar_.histNameNormX_,"histNameNormX/b");

 
    TH1 *h=0;
    for (std::map<unsigned int,TreeVariables>::const_iterator it = map_.begin(), 
	   itEnd= map_.end(); it != itEnd;++it ) {
  
      //read out mean and rms values from merged histograms

      //in xPrime
      target->GetObject(it->second.histPathX_,h);    //get  histogram path for Xprime-residuum from map
      treeVar_.entries_ = static_cast<UInt_t>(h->GetEntries());   //entries are const for all histos
      treeVar_.meanX_ = h->GetMean(); //get mean value from histogram 
      treeVar_.rmsX_ = h->GetRMS();   //get RMS value from histogram
    
    
      target->GetObject(it->second.histPathNormX_,h);    //get  histogram path for Xprime-residuum from map
      treeVar_.meanNormX_ = h->GetMean(); //get mean value from histogram 
      treeVar_.rmsNormX_ = h->GetRMS();   //get RMS value from histogram

    
      //if present in local coordinates
      if (lCoorHistOn){
	target->GetObject(it->second.histPathLocalX_, h);  //get  histogram path for ordinary residuum from map
	treeVar_.meanLocalX_ = h->GetMean();      //get mean value from histogram
	treeVar_.rmsLocalX_ = h->GetRMS();        //get RMS value from histogram
    
	target->GetObject(it->second.histPathNormLocalX_,h); //get histogram path for normalized residuum from map
	treeVar_.meanNormLocalX_ = h->GetMean();//get mean value from histogram
	treeVar_.rmsNormLocalX_ = h->GetRMS(); //get RMS value from histogram
   
      }
    
      //get 'constant' values from map ('constant' means not effected by merging)
      treeVar_.moduleId_=it->first;             //get module Id 
      treeVar_.subDetId_= it->second.subDetId_;
      treeVar_.posPhi_= it->second.posPhi_;
      treeVar_.posEta_= it->second.posEta_;
      treeVar_.posR_= it->second.posR_;
      treeVar_.posX_= it->second.posX_;
      treeVar_.posY_= it->second.posY_;
      treeVar_.posZ_= it->second.posZ_;
      treeVar_.layer_= it->second.layer_;
      treeVar_.side_= it->second.side_;
      treeVar_.rod_= it->second.rod_;
      treeVar_.ring_= it->second.ring_;
      treeVar_.petal_= it->second.petal_;
      treeVar_.blade_= it->second.blade_;
      treeVar_.panel_= it->second.panel_;
      treeVar_.outerInner_= it->second.outerInner_;
      treeVar_.isDoubleSide_= it->second.isDoubleSide_;

      tree->Fill();
   
    } 
   
    delete h;
  }else cout<<"Warning: Could not merge tree, histograms on module level are not present."<<endl;
  //target->Write();
 

}
