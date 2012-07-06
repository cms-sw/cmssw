#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "TString.h"
#include "TTree.h"
#include "TFile.h"
#include "TROOT.h"
#include "TObject.h"

using namespace std;

// define name Tree in calcEntries()

void calcEntries(string  flavour, string category, vector<float> & entries, string dir);

int main(int argc, char **argv){

	string dir = "./";
	if(argc == 2) dir = argv[1];
		
	cout << "reading rootfiles from dir " << dir << endl;

	string flavour[3] = {"B", "C", "DUSG"};
	string cat[3] = {"NoVertex", "PseudoVertex", "RecoVertex"};

	vector<float> entries[9];

	int nIter =0;

	for(int j=0; j<3; j++){//loop on categories
		for(int i =0; i<3; i++){//loop on flavours
			calcEntries(flavour[i], cat[j], entries[nIter], dir);
			//for(int k =0 ; k<entries[nIter].size(); k++) cout<<flavour[i]<<"   "<<cat[j]<<"  "<<entries[nIter][k]<<endl; 
			nIter++;
		}
	}

	for(int j=0; j<3; j++){//loop on categories	
		for(int k=1; k<3; k++){//loop on C and light
			cout<<"***************   "<<cat[j]<<"_B_"<<flavour[k]<<"   ***************"<<endl;
			for(int l = 0; l<15; l++ ){// loop on pt/eta bins defined in xml
				int index = j*3;
				int indexb = k+j*3;
				float bias = (float)((entries[index][l]/(entries[0][l]+entries[3][l]+entries[6][l]))/((entries[indexb][l]/(entries[k][l]+entries[k+3][l]+entries[k+6][l]))));
				cout<<"<bias>"<<bias<<"</bias>"<<endl; 
			}
		}
	}
	
	return 0;
}


void calcEntries(string flavour, string  category,  vector<float> & entries, string dir){	
	string fix = "CombinedSV";

	TFile * f = TFile::Open((dir+fix+category+"_"+flavour+".root").c_str());
     
	f->cd();
	TTree * t =(TTree*)f->Get((fix+category).c_str());

	//definition of pt and eta bins should be the same as in the Train*xml files!!!
	entries.push_back(t->GetEntries("jetPt>15&&jetPt<40&&TMath::Abs(jetEta)<1.2"));
	entries.push_back(t->GetEntries("jetPt>15&&jetPt<40&&TMath::Abs(jetEta)<2.1&&(!(TMath::Abs(jetEta)<1.2))"));
	entries.push_back(t->GetEntries("jetPt>15&&jetPt<40&&(!(TMath::Abs(jetEta)<2.1))"));
	entries.push_back(t->GetEntries("jetPt>40&&jetPt<60&&TMath::Abs(jetEta)<1.2"));
	entries.push_back(t->GetEntries("jetPt>40&&jetPt<60&&TMath::Abs(jetEta)<2.1&&(!(TMath::Abs(jetEta)<1.2))"));
	entries.push_back(t->GetEntries("jetPt>40&&jetPt<60&&(!(TMath::Abs(jetEta)<2.1))"));
	entries.push_back(t->GetEntries("jetPt>60&&jetPt<90&&TMath::Abs(jetEta)<1.2"));
	entries.push_back(t->GetEntries("jetPt>60&&jetPt<90&&TMath::Abs(jetEta)<2.1&&(!(TMath::Abs(jetEta)<1.2))"));
	entries.push_back(t->GetEntries("jetPt>60&&jetPt<90&&(!(TMath::Abs(jetEta)<2.1))"));
	entries.push_back(t->GetEntries("jetPt>90&&jetPt<150&&TMath::Abs(jetEta)<1.2"));
	entries.push_back(t->GetEntries("jetPt>90&&jetPt<150&&TMath::Abs(jetEta)<2.1&&(!(TMath::Abs(jetEta)<1.2))"));
	entries.push_back(t->GetEntries("jetPt>90&&jetPt<150&&(!(TMath::Abs(jetEta)<2.1))"));
	entries.push_back(t->GetEntries("jetPt>150&&jetPt<600&&TMath::Abs(jetEta)<1.2"));
	entries.push_back(t->GetEntries("jetPt>150&&jetPt<600&&TMath::Abs(jetEta)<2.1&&(!(TMath::Abs(jetEta)<1.2))"));
	entries.push_back(t->GetEntries("jetPt>150&&jetPt<600&&(!(TMath::Abs(jetEta)<2.1))"));
    
}



