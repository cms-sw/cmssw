#include <iostream>
#include <sstream>

#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>
#include <TBranch.h>
#include <TROOT.h>
#include <TFile.h>

float analyze(Int_t run)
{
  TString filename1="microReport";
  filename1+=run;
  filename1+=".root";
  TString filename2="summary";
  filename2+=run;
  filename2+=".root";

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
  TFile *f2 = new TFile(filename2,"RECREATE");
  Int_t sumnotidle[25];
  Int_t sumreported[25];
  Int_t sumprocessed[25];
  TH2F *bf[25];
  for(unsigned int i =0; i <25; i++){
    std::ostringstream name;
    name << "bf" << i;
    std::string hn = name.str();
    bf[i] = new TH2F(hn.c_str(),hn.c_str(),1200,0.,1200.,100,0.,1.1);
  }
  TH2F *rt[25];
  for(unsigned int i =0; i <25; i++){
    std::ostringstream name;
    name << "rt" << i;
    std::string hn = name.str();
    rt[i] = new TH2F(hn.c_str(),hn.c_str(),1200,0.,1200.,100,0.,200.);
  }
  TH2F *bfvsrt[25];
  for(unsigned int i =0; i <25; i++){
    std::ostringstream name;
    name << "bfvsrt" << i;
    std::string hn = name.str();
    bfvsrt[i] = new TH2F(hn.c_str(),hn.c_str(),100,0.,200.,100,0.,1.);
  }
  TH2F *et[25];
  for(unsigned int i =0; i <25; i++){
    std::ostringstream name;
    name << "et" << i;
    std::string hn = name.str();
    et[i] = new TH2F(hn.c_str(),hn.c_str(),1200,0.,1200.,500,0.,.2);
  }
  TH1F *set[25];
  for(unsigned int i =0; i <25; i++){
    std::ostringstream name;
    name << "set" << i;
    std::string hn = name.str();
    set[i] = new TH1F(hn.c_str(),hn.c_str(),1200,0.,1200.);
  }
  TH1F *srt[25];
  for(unsigned int i =0; i <25; i++){
    std::ostringstream name;
    name << "srt" << i;
    std::string hn = name.str();
    srt[i] = new TH1F(hn.c_str(),hn.c_str(),1200,0.,1200.);
  }
  TH1F *microhist[5000];
  Int_t microstat[5000];
  TBranch *microstates = microReport->GetBranch("microstates");
  TObjArray *leaves = microstates->GetListOfLeaves();
  unsigned int nmicro = leaves->GetSize();
  std::cout << "Number of microstates " << nmicro << std::endl;
  std::vector<std::string> micronames(nmicro);
  Int_t microvector[10000];
  microReport->SetBranchAddress("microstates",microvector);
  for(unsigned int i = 0; i < nmicro; i++){
    micronames[i] = leaves->At(i)->GetTitle();
    std::cout << i << " name " << micronames[i] << std::endl;
    microhist[i] = new TH1F(micronames[i].c_str(),micronames[i].c_str(),
			    1200,0.,1200.);
    microstat[i] = 0;
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
	if(rate[2]<126)continue;
	//      std::cout << "rate 0 " << rate[0] << " rate 1 " <<  rate[1] 
	//  	      << " rate 2 " << rate[2] << std::endl;

	if(cls<ls){
	  if(cls!=0){
	    std::cout << " " << cls 
		      << " total counts " << totalUpdates 
		      << " total processed " 
		      << totalProcessed 
		      << " totalIdle " << totalIdle 
		      << " fraction " 
		      << float(totalUpdates-totalIdle)/float(totalUpdates) 
		      << " cputime " 
		      << float(totalUpdates-totalIdle)/float(totalUpdates)/totalProcessed*23.4*1000.
		      << std::endl;
	    for(unsigned int j=0; j<25; j++){
	      if(sumreported[j]!=0){
		float esttime = float(sumnotidle[j])/
		  float(sumprocessed[j]);
		std::cout << j << " filling " << cls << " with " 
			  << sumnotidle[j] << " " << sumprocessed[j] 
			  << " " << sumreported[j] << " "
			  << esttime << std::endl;
		float esterror = sqrt(float(sumnotidle[j]))/float(sumprocessed[j])+
		  float(sumnotidle[j])/float(sumprocessed[j])
		  /float(sumprocessed[j])*sqrt(float(sumprocessed[j]));
		set[j]->SetBinContent(cls,esttime);
		set[j]->SetBinError(cls,esterror);
		srt[j]->SetBinContent(cls, float(sumprocessed[j])/23.4);
		srt[j]->SetBinError(cls,sqrt(float(sumprocessed[j]))/23.4);
	      }
	    }
	    for(unsigned int j=0; j<nmicro-3; j++){
	      microhist[j]->SetBinContent(cls,float(microstat[j])
					  /float(sumprocessed[7]));
	    }
	  }
	  for(unsigned int j=0; j<25; j++){
	    sumnotidle[j]=0;
	    sumreported[j]=0;
	    sumprocessed[j]=0;
	  }
	  for(unsigned int j=0; j<nmicro-3; j++){
	    microstat[j]=0;
	  }
	  cls=ls;totalUpdates=0; totalProcessed=0; totalIdle=0;
      
	}
	totalProcessed += rate[0];
	sumprocessed[rate[1]] += rate[0];
	totalIdle += microvector[2];
	rt[rate[1]]->Fill(float(cls),
			  float(rate[0])/23.4);

	Int_t singleUpdates = 0;
	for(unsigned int j = 0; j < nmicro-3; j++)
	  {
	    singleUpdates += microvector[j];
	    totalUpdates+=microvector[j];
	    if(rate[1]==7) microstat[j]+=microvector[j];
	  }
	sumreported[rate[1]]+=singleUpdates;
	sumnotidle[rate[1]]+=singleUpdates-microvector[2];
	float busyfraction = 
	  float(singleUpdates-microvector[2])/float(singleUpdates);
	bf[rate[1]]->Fill(float(cls),
			  busyfraction);
	bfvsrt[rate[1]]->Fill(float(rate[0])/23.4,busyfraction);
	if(rate[0]!=0){
	  float esttime = float(singleUpdates-microvector[2])/
	    float(rate[0]);
	  et[rate[1]]->Fill(float(cls),esttime);
	}

	//     std::cout << i << " " << cls 
	// 	      << " total counts " << totalUpdates << std::endl;
      }

  f2->Write();
}
