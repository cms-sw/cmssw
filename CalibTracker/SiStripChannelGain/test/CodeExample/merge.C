
#include<vector>
#include "TROOT.h"
#include "TFile.h"
#include "TFileMerger.h"
#include "TDirectory.h"
#include "TChain.h"
#include "TObject.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLegend.h"
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TTree.h"
#include "TF1.h"
#include "TPaveText.h"



TObject* GetObjectFromPath(TDirectory* File, const char* Path);
void merge_core(TDirectory* Output, TDirectory* Input);

void merge(char* InputFilePath,char* OutputFilePath)
{
   gROOT->Reset();
   char outname[2048];
   char name[2048];

   unsigned int PackN    = 1;
   unsigned int PackSize = 120;

   for(unsigned int P=0;P<PackN;P++){
        TFileMerger merger(kFALSE);
        if(PackN>1){
        sprintf(outname,"tmp_%s_%i.root",OutputFilePath,P);
        }else{
        sprintf(outname,"%s",OutputFilePath);
        }
        merger.OutputFile(outname);
        merger.PrintFiles("true");        
        merger.SetFastMethod();
        for(unsigned int i=0;i<PackSize;i++){
           sprintf(name,InputFilePath,P*PackSize+i);
           merger.AddFile(name);
        }
        merger.Merge();
   }

   if(PackN>1){
      TFileMerger merger(kFALSE);
      merger.OutputFile(OutputFilePath);
      merger.PrintFiles("true");        
      merger.SetFastMethod();
      for(unsigned int P=0;P<PackN;P++){
           sprintf(outname,"tmp_%s_%i.root",OutputFilePath,P);
           merger.AddFile(outname);
      }
      merger.Merge();

      for(unsigned int P=0;P<PackN;P++){
           sprintf(outname,"tmp_%s_%i.root",OutputFilePath,P);
           remove(outname);
      }
   }
}
        

/*
	TFile* Output = new TFile("out.root","RECREATE");
	TFile* Input;
	for(unsigned int i=0;i<100;i++){
		char name[1024];
		sprintf(name,"file:SingleMu_Discrim_%04i.root",i);
		printf("Oppening %s\n",name);
		Input =  new TFile(name);
		if(Input==NULL || Input->IsZombie() ){printf("### Bug With File %s\n### File will be skipped \n",name); continue;}
		merge_core(Output, Input );
		Input->Close();
		delete Input;		
	}

	Output->Write();

}	Output->Close();
*/




void merge_core(TDirectory* Output, TDirectory* Input)
{ 
   TH1::AddDirectory(kTRUE);

   if(Input==NULL) return;
   TList* input_list = Input->GetListOfKeys();
   if(input_list==NULL){cout <<"LIST IS NULL\n";return;}

   TObject* input_it = input_list->First();  

   while(input_it!=NULL){
	if(input_it->IsFolder()){
//	   printf("Enter in %s\n",input_it->GetName());
	   TDirectory* in_dir_tmp  = (TDirectory*)Input ->Get(input_it->GetName());

	   TDirectory* out_dir_tmp = (TDirectory*)Output->Get(input_it->GetName());
	   if(out_dir_tmp==NULL){
//		printf("Directory do not exist\n");
		out_dir_tmp = Output->mkdir(input_it->GetName());
	   }
	   merge_core(out_dir_tmp, in_dir_tmp);
	}else{
	   TH1* in_hist_tmp  = (TH1*)Input ->Get(input_it->GetName());
	   TH1* out_hist_tmp = (TH1*)Output->Get(input_it->GetName());
	   if(out_hist_tmp==NULL){
//		printf("Creation of a new TH1*\n");
		Output->cd();
		out_hist_tmp = (TH1*)in_hist_tmp->Clone();	
	   }else{
//		printf("Summing Histograms\n");
		out_hist_tmp->Add(in_hist_tmp,1);
	   }
	}
	
	input_it = input_list->After(input_it);
   }

}


TObject* GetObjectFromPath(TDirectory* File, const char* Path)
{
   string str(Path);
   size_t pos = str.find("/"); 

   if(pos < 256){
      string firstPart = str.substr(0,pos);
      string endPart   = str.substr(pos+1,str.length());
      TDirectory* TMP = (TDirectory*)File->Get(firstPart.c_str());
      if(TMP!=NULL)return GetObjectFromPath(TMP,endPart.c_str());
      
      printf("BUG\n");
      return NULL;
   }else{
      return File->Get(Path);
   }

}

