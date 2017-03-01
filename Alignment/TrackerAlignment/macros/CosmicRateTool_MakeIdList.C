#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "TFile.h"
#include "TTree.h"


using namespace std;

struct DATA
{
	long d;
	int f;
};

void Counting(vector<DATA>* data,long x)
{
	
	
	for(unsigned int i=0;i<data->size();i++)
	{

		if( (data->at(i).d)==x)
		{
			data->at(i).f++;
			return;	
		}
	}

	DATA d = {x,1};
	data->push_back(d);
return;	
}

void CosmicRateTool_MakeIdList(const char* fileName)
{
   TString InputFile= Form("%s",fileName); 
   TFile *file = new TFile(InputFile);

   bool IsFileExist;
   IsFileExist = file->IsZombie();
   if(IsFileExist)
   {   
      cout<<endl<<"====================================================================================================="<<endl;
      cout<<fileName << " is not found. Check the file!"<<endl;
      cout<<"====================================================================================================="<<endl<<endl;
      exit (EXIT_FAILURE);
   } 

   TTree *tree;
   tree = (TTree*)file->Get("cosmicRateAnalyzer/Cluster");

   ofstream output("IdList.txt");
   UInt_t Id;
   long x;
   tree->SetBranchAddress("DetID",&Id);

   vector<DATA>* dataIn = new vector<DATA>();
   
   long int nentries = (int)tree->GetEntries();
   cout<<"entries : "<<nentries<<endl;
   for(long int i=0; i < nentries; i++)
   {
   	tree->GetEntry(i);
   	x=Id;
   	Counting(dataIn,x);
   
   	if(i%10000==0)cout<<"At : "<<i<<endl;
//		if (i==10000) break;
   }	

   for(unsigned int j=0;j<dataIn->size();)
   {
	output<<(dataIn->at(j).d)<<" "<<(dataIn->at(j).f)<<endl;
	j++;
   }
   output.close();
}
