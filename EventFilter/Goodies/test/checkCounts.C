#include <iostream>
#include <sstream>
#include <fstream>

#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>
#include <TBranch.h>
#include <TROOT.h>
#include <TFile.h>

float checkCounts(Int_t run)
{
  TString filename1="microReport";
  filename1+=run;
  filename1+=".root";
  TString filename3="scalers";
  filename3+=run;
  filename3+=".txt";
  std::ofstream off(filename3);

  TFile *f1 = new TFile(filename1,"READONLY");
  f1->cd();
  TTree *t= (TTree*)gROOT->FindObject("microReport");
  TTree *microReport = (TTree*)t->Clone();
  Int_t maxls = Int_t(microReport->GetMaximum("ls"));
  Int_t maxinstance = Int_t(microReport->GetMaximum("instance"));
  std::cout << "max instance " << maxinstance << std::endl;
  Int_t nentries = microReport->GetEntries();
  Int_t indexentries = microReport->BuildIndex("ls","microstates.instance");
  if(indexentries != nentries) 
    std::cout << "ERROR, index entries and total entries differ" << std::endl;

  Int_t instances[1200];
  Int_t sumreported;
  Int_t sumprocessed;
  TBranch *microstates = microReport->GetBranch("microstates");
  TObjArray *leaves = microstates->GetListOfLeaves();
  unsigned int nmicro = leaves->GetSize();
  std::cout << "Number of microstates " << nmicro << std::endl;
  std::vector<std::string> micronames(nmicro);
  Int_t microvector[10000];
  microReport->SetBranchAddress("microstates",microvector);
  for(unsigned int i = 0; i < nmicro; i++){
    micronames[i] = leaves->At(i)->GetTitle();
  }
  Int_t ls;
  microReport->SetBranchAddress("ls",&ls);
  Int_t rate[3];
  //  microReport->SetBranchAddress("rate",rate);
  Int_t cls=0;
  unsigned int totalUpdates = 0;
  unsigned int totalProcessed = 0;
  unsigned int totalIdle=0;
  for( int l = 1; l <= maxls; l++)
    for( int m = 0; m <= maxinstance; m++)
      {
	if(microReport->GetEntryWithIndex(l,m)<0) continue;
	rate[0]=microvector[nmicro-1];
	rate[2]=microvector[nmicro-2];
	rate[1]=microvector[nmicro-3];
	if(cls>ls){
	  std::cout << " ls out of order " << cls << " got " << ls 
		    << " for instance " << rate[2] << std::endl;
	}
	if(cls<ls){
	  if(cls!=0){
	    std::cout << " " << cls 
		      << " total counts " << totalUpdates 
		      << " total processed " 
		      << totalProcessed 
		      << std::endl;
	    Int_t nsubrep =0;
	    for(unsigned int j=0; j<maxinstance; j++){
	      if(instances[j]!=1)
		std::cout << "instance " << j << " reported " 
			  << instances[j] << " times " << std::endl;
	      else
		nsubrep++;
	    }
	    off << cls << ","<<nsubrep << "," << totalProcessed << std::endl; 
	  }
	  for(unsigned int j=0; j<25; j++){
	    sumreported=0;
	    sumprocessed=0;
	  }
	  for(unsigned int j=0; j<1200; j++){
	    instances[j]=0;
	  }
	  cls=ls;totalUpdates=0; totalProcessed=0; 
	  
	}
	
	totalProcessed += rate[0];
	sumprocessed += rate[0];
	instances[rate[2]]+=1;
	
	for(unsigned int j = 0; j < nmicro-3; j++)
	  {
	    totalUpdates+=microvector[j];
	  }
	
    //     std::cout << i << " " << cls 
    // 	      << " total counts " << totalUpdates << std::endl;
      }
  // do the last ls 

  if(cls!=0){
    std::cout << " " << cls 
	      << " total counts " << totalUpdates 
	      << " total processed " 
	      << totalProcessed 
	      << std::endl;
    Int_t nsubrep =0;
    for(unsigned int j=0; j<maxinstance; j++){
      if(instances[j]!=1)
	std::cout << "instance " << j << " reported " 
		  << instances[j] << " times " << std::endl;
      else
	nsubrep++;
    }
    off << cls << ","<<nsubrep << "," << totalProcessed << std::endl; 
  }
    
}
