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


#include <string>
#include <utility>
#include "TChain.h"
#include "TFile.h"
#include "TH1.h"
#include "TTree.h"
#include "TKey.h"
#include "TString.h"
#include <map>
#include "Riostream.h"
#include "TF1.h"



TString histPathName;
bool copiedTree_ = false;
Bool_t moduleLevelHistsTransient_ = true;
Bool_t lCoorHistOn_ = false;
Bool_t yCoorHistOn_ = false;
Bool_t useFit_ = false;
struct TreeVariables{
  TreeVariables():  meanLocalX_(), meanNormLocalX_(), meanX_(), meanNormX_(),
		     meanY_(), meanNormY_(),chi2PerDof_(),
		     rmsLocalX_(), rmsNormLocalX_(), rmsX_(), rmsNormX_(), 
		     rmsY_(), rmsNormY_(), sigmaX_(),sigmaNormX_(),
		     fitMeanX_(),  fitSigmaX_(),fitMeanNormX_(),fitSigmaNormX_(),
		     posR_(), posPhi_(), posEta_(),
		     posX_(), posY_(), posZ_(),
		      numberOfUnderflows_(), numberOfOverflows_(),numberOfOutliers_(),
		     entries_(), moduleId_(), subDetId_(),
		     layer_(), side_(), rod_(),ring_(), 
		     petal_(),blade_(), panel_(), outerInner_(),
		     isDoubleSide_(),
		     histNameLocalX_(), histNameNormLocalX_(), histNameX_(), histNameNormX_(), 
                     histNameY_(), histNameNormY_() {} 
   Float_t meanLocalX_, meanNormLocalX_, meanX_,meanNormX_,    //mean value read out from modul histograms
      meanY_,meanNormY_, chi2PerDof_,
      rmsLocalX_, rmsNormLocalX_, rmsX_, rmsNormX_,      //rms value read out from modul histograms
      rmsY_, rmsNormY_,sigmaX_,sigmaNormX_,
      fitMeanX_,  fitSigmaX_,fitMeanNormX_,fitSigmaNormX_,
      posR_, posPhi_, posEta_,                     //global coordiantes    
      posX_, posY_, posZ_,             //global coordiantes 
      numberOfUnderflows_, numberOfOverflows_,numberOfOutliers_;
    UInt_t  entries_, moduleId_, subDetId_,          //number of entries for each modul //modul Id = detId and subdetector Id
      layer_, side_, rod_, 
      ring_, petal_, 
      blade_, panel_, 
      outerInner_; //orientation of modules in TIB:1/2= int/ext string, TID:1/2=back/front ring, TEC 1/2=back/front petal 
    Bool_t isDoubleSide_;
    std::string histNameLocalX_, histNameNormLocalX_, histNameX_, histNameNormX_,
       histNameY_, histNameNormY_;  

  TString  histPathLocalX_, histPathNormLocalX_, histPathX_, histPathNormX_,  histPathY_, histPathNormY_;
};

std::map<unsigned int,TreeVariables> map_;
void MergeRootfile( TDirectory *target, TList *sourcelist );
void RewriteTree( TDirectory *target,TTree *tree,std::map<unsigned int,TreeVariables> map_);
std::pair<float,float>  fitResiduals(const TH1 *h,float meantmp,float rmstmp);
//void RewriteTree(std::map<unsigned int,TreeVariables> map_);

void hadd() {
  
  TList *FileList;
  TFile *Target;
  Target = TFile::Open( "$TMPDIR/ValidationCRAFT_CosmicTF_10hits_2hits2d_p5GeV.root", "RECREATE" );
  TString inputFileName="";
  Int_t nEvt;
  Bool_t fileOk_=true;
  //get file name
  std::cout<<"Type in general file name (ending is added automatically)."<<std::endl;
  std::cout<<"i.e. Validation_CRUZET_data as name, code adds 1,2,3 + .root"<<std::endl;
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
      std::cout<<"try to write out tree"<<std::endl;
      //tree->Write();
      std::cout<<"written out tree"<<std::endl;
      Target->SaveSelf(kTRUE);
      std::cout<<"target saveself"<<std::endl;
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
	if(histPathName.Contains("h_yprime_residuals"))map_[moduleId].histPathY_= histPathName;
	if(histPathName.Contains("h_normyprimeresiduals"))map_[moduleId].histPathNormY_= histPathName;

	moduleLevelHistsTransient_ = false;
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
      
      if (!copiedTree_)	{
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
	
	//if residuals for y coordiantarte are present in file
	if (tree->GetBranch("histNameY"))
	  {
	    tree->SetBranchAddress("histNameY",&treeMem_.histNameY_);
	    tree->SetBranchAddress("histNameNormY",&treeMem_.histNameNormY_);
	    yCoorHistOn_ = true;
	  }
	
	//if residuals for local coordiantartes are present in file
	if (tree->GetBranch("histNameLocalX"))
	  {
	    tree->SetBranchAddress("histNameLocalX",&treeMem_.histNameLocalX_);
	    tree->SetBranchAddress("histNameNormLocalX",&treeMem_.histNameNormLocalX_);
	    lCoorHistOn_ = true;
	  }

	if (tree->GetBranch("fitMeanX"))
	  {
	    useFit_ = true;
	  }


	Long64_t nentries = tree->GetEntriesFast();
	for (Long64_t jentry=0; jentry<nentries;jentry++) {
	 
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
	    map_[treeMem_.moduleId_].histNameLocalX_ = treeMem_.histNameLocalX_;
	    map_[treeMem_.moduleId_].histNameNormLocalX_ = treeMem_.histNameNormLocalX_

	      }
	   
	  //if residuals for y coordinates are present in file
	  if (tree->GetBranch("histNameY"))
	    {
	      map_[treeMem_.moduleId_].histNameY_= treeMem_.histNameY_;
	      map_[treeMem_.moduleId_].histNameNormY_= treeMem_.histNameY_;
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
  
    TreeVariables treeVar_;

    if (!moduleLevelHistsTransient_){
    tree->Branch("moduleId",&treeVar_.moduleId_,"modulId/i");
    tree->Branch("subDetId",&treeVar_.subDetId_,"subDetId/i");
    tree->Branch("entries",&treeVar_.entries_,"entries/i");
  
    tree->Branch("meanX",&treeVar_.meanX_,"meanX/F");
    tree->Branch("rmsX",&treeVar_.rmsX_,"rmsX/F");
    tree->Branch("meanNormX",&treeVar_.meanNormX_,"meanNormX/F");
    tree->Branch("rmsNormX",&treeVar_.rmsNormX_,"rmsNormX/F");
   
    
    if ( lCoorHistOn_ ){
      tree->Branch("meanLocalX",&treeVar_.meanLocalX_,"meanLocalX/F");
      tree->Branch("rmsLocalX",&treeVar_.rmsLocalX_,"rmsLocalX/F");
      tree->Branch("meanNormLocalX",&treeVar_.meanNormLocalX_,"meanNormLocalX/F");
      tree->Branch("rmsNormLocalX",&treeVar_.rmsNormLocalX_,"rmsNormLocalX/F");
      tree->Branch("histNameLocalX",&treeVar_.histNameLocalX_,"histNameLocalX/b");
      tree->Branch("histNameNormLocalX",&treeVar_.histNameNormLocalX_,"histNameNormLocalX/b");
    }

    if ( yCoorHistOn_ ){
      tree->Branch("meanY",&treeVar_.meanY_,"meanY/F");
      tree->Branch("rmsY",&treeVar_.rmsY_,"rmsY/F");
      tree->Branch("meanNormY",&treeVar_.meanNormY_,"meanNormY/F");
      tree->Branch("rmsNormY",&treeVar_.rmsNormY_,"rmsNormY/F");
      tree->Branch("histNameY",&treeVar_.histNameY_,"histNameY/b");
      tree->Branch("histNameNormY",&treeVar_.histNameNormY_,"histNameNormY/b");
    }

    if (useFit_) {
      
      tree->Branch("fitMeanX",&treeVar_.fitMeanX_,"fitMeanX/F"); 
      tree->Branch("fitSigmaX",&treeVar_.fitSigmaX_,"fitSigmaX/F");
      tree->Branch("fitMeanNormX",&treeVar_.fitMeanNormX_,"fitMeanNormX/F"); 
      tree->Branch("fitSigmaNormX",&treeVar_.fitSigmaNormX_,"fitSigmaNormX/F");
    }
    
    tree->Branch("numberOfUnderflows",&treeVar_.numberOfUnderflows_,"numberOfUnderflows/I");
    tree->Branch("numberOfOverflows",&treeVar_.numberOfOverflows_,"numberOfOverflows/I");
    tree->Branch("numberOfOutliers",&treeVar_.numberOfOutliers_,"numberOfOutliers/I");
    tree->Branch("chi2PerDof",&treeVar_.chi2PerDof_,"chi2PerDof/F");
   
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
    
      int numberOfBins=h->GetNbinsX();
      treeVar_.numberOfUnderflows_ = h->GetBinContent(0);
      treeVar_.numberOfOverflows_ = h->GetBinContent(numberOfBins+1);
      treeVar_.numberOfOutliers_ =  h->GetBinContent(0)+h->GetBinContent(numberOfBins+1);
     
      
      if(h->GetEntries()>0){
	double stats[20];
	h->GetStats(stats);
	treeVar_.chi2PerDof_ = stats[3]/(stats[0]-1);
      }
      
      
      if (useFit_) {

	//call fit function which returns mean and sigma from the fit
	//for absolute residuals
	std::pair<float,float> fitResult1 = fitResiduals(h, h->GetMean(), h->GetRMS());
	treeVar_.fitMeanX_=fitResult1.first;
	treeVar_.fitSigmaX_=fitResult1.second;
      }
      
      target->GetObject(it->second.histPathNormX_,h);    //get  histogram path for Xprime-residuum from map
      treeVar_.meanNormX_ = h->GetMean(); //get mean value from histogram 
      treeVar_.rmsNormX_ = h->GetRMS();   //get RMS value from histogram
      
      if (useFit_) {

	//call fit function which returns mean and sigma from the fit
	//for normalized residuals
	std::pair<float,float> fitResult2 = fitResiduals(h, h->GetMean(), h->GetRMS());
	treeVar_.fitMeanNormX_=fitResult2.first;
	treeVar_.fitSigmaNormX_=fitResult2.second;
	
      }
      //if present in local coordinates
      if (lCoorHistOn_){
	target->GetObject(it->second.histPathLocalX_, h);  //get  histogram path for ordinary residuum from map
	treeVar_.meanLocalX_ = h->GetMean();      //get mean value from histogram
	treeVar_.rmsLocalX_ = h->GetRMS();        //get RMS value from histogram

	target->GetObject(it->second.histPathNormLocalX_,h); //get histogram path for normalized residuum from map
	treeVar_.meanNormLocalX_ = h->GetMean();//get mean value from histogram
	treeVar_.rmsNormLocalX_ = h->GetRMS(); //get RMS value from histogram
   
      }

      //if y coordinate is present in the file
      if ( yCoorHistOn_ ){
	target->GetObject(it->second.histPathY_, h);  //get  histogram path for ordinary residuum from map
	treeVar_.meanY_ = h->GetMean();      //get mean value from histogram
	treeVar_.rmsY_ = h->GetRMS();        //get RMS value from histogram
    
	target->GetObject(it->second.histPathNormY_,h); //get histogram path for normalized residuum from map
	treeVar_.meanNormY_ = h->GetMean();//get mean value from histogram
	treeVar_.rmsNormY_ = h->GetRMS(); //get RMS value from histogram
   
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
   cout << "left for loop"<< endl;  
  }//else cout<<"Warning: Could not merge tree, histograms on module level are not present."<<endl;
    //target->Write();
}
 
std::pair<float,float>  fitResiduals(const TH1 *h,float meantmp,float rmstmp)
{
  cout << "enter fit methode"<< endl;
  std::pair<float,float> fitResult;
  
    TH1*hist=0;
    hist = const_cast<TH1*>(h);
    TF1 *ftmp1= new TF1("ftmp1","gaus",meantmp-2*rmstmp, meantmp+2*rmstmp); 
    hist->Fit("ftmp1","QLR");
    float mean = ftmp1->GetParameter(1);
    float sigma = ftmp1->GetParameter(2);
    delete ftmp1;
    TF1 *ftmp2= new TF1("ftmp2","gaus",mean-3*sigma,mean+3*sigma); 
    hist->Fit("ftmp2","Q0LR");
    fitResult.first = ftmp2->GetParameter(1);
    fitResult.second = ftmp2->GetParameter(2);
    delete ftmp2;
    /*
    std::cout << e.what() << std::endl;
    std::cout <<"set values of fit to 9999" << std::endl;
    fitResult.first = 9999.;
    fitResult.second = 9999.;
    */
  return fitResult;
}


