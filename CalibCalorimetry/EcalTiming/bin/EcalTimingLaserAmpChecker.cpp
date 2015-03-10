#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include "TGraph.h"
#include "TGraphErrors.h"

#include "TFile.h"
#include "TProfile2D.h"

#include<vector>
#include<string>

using namespace std;

Int_t getNumActiveXtals (TProfile2D *tprof);
Double_t getAverageAmplitude(TProfile2D *tprof, Double_t thres);
Double_t getNumAboveThreshold (TProfile2D *tprof, Double_t thres);
TProfile2D* getProfileWithThreshold(TProfile2D *tprof, Double_t thres);

int main(int argc,  char * argv[]){

if(argc < 2){cout<<" Usage: <Input File> "<<endl;
  return -2;}


char *LaserFile = argv[1];

std::cout << " The laser file is " << LaserFile << std::endl;

TFile *lfile = new TFile(LaserFile);

//These are the input profile plots
TProfile2D *ebAmpProfile = (TProfile2D*) lfile->Get("fullAmpProfileEB");
TProfile2D *eepAmpProfile = (TProfile2D*) lfile->Get("fullAmpProfileEEP");
TProfile2D *eemAmpProfile = (TProfile2D*) lfile->Get("fullAmpProfileEEM");

Int_t numXtalsEB=getNumActiveXtals(ebAmpProfile);
Int_t numXtalsEEP=getNumActiveXtals(eepAmpProfile);
Int_t numXtalsEEM=getNumActiveXtals(eemAmpProfile);

std::cout << " Total Number of EB crystals with entries " <<  numXtalsEB << std::endl;
std::cout << " Total Number of EEP crystals with entries " <<  numXtalsEEP << std::endl;
std::cout << " Total Number of EEM crystals with entries " <<  numXtalsEEM << std::endl;


//Ok from the above we know know how many "active" xtals there are, we can now compute some numbers
Double_t ampthres[21];
Double_t numberpresentEB[21],numberpresentBEB[21]; 
Double_t fractionEB[21];
Double_t numberpresentEEP[21],numberpresentBEEP[21];
Double_t fractionEEP[21];
Double_t numberpresentEEM[21],numberpresentBEEM[21];
Double_t fractionEEM[21];

Double_t aveampEB  = getAverageAmplitude(ebAmpProfile,25.);
Double_t aveampEEP = getAverageAmplitude(eepAmpProfile,15.);
Double_t aveampEEM = getAverageAmplitude(eemAmpProfile,15.);

std::cout << " The EB Average Amplitdue is " << aveampEB << std::endl;
std::cout << " The EE+ Average Amplitdue is " << aveampEEP << std::endl;
std::cout << " The EE- Average Amplitdue is " << aveampEEM << std::endl;

for (Int_t i = 0; i < 21; ++i)
{
   ampthres[i]=((Double_t)i)*0.05;
   
   numberpresentEB[i]=getNumAboveThreshold(ebAmpProfile,(ampthres[i])*aveampEB);
   numberpresentBEB[i]=(Double_t)numXtalsEB-numberpresentEB[i];
   fractionEB[i]=(numberpresentEB[i])/((Double_t) numXtalsEB );
   
   numberpresentEEP[i]=getNumAboveThreshold(eepAmpProfile,ampthres[i]*aveampEEP);
   numberpresentBEEP[i]=(Double_t)numXtalsEEP-numberpresentEEP[i];
   fractionEEP[i]=numberpresentEEP[i]/((Double_t) numXtalsEEP );
   
   numberpresentEEM[i]=getNumAboveThreshold(eemAmpProfile,ampthres[i]*aveampEEM);
   numberpresentBEEM[i]=(Double_t)numXtalsEEM-numberpresentEEM[i];
   fractionEEM[i]=numberpresentEEM[i]/((Double_t) numXtalsEEM );
}


TGraph *numEBGr = new TGraph(21, ampthres, numberpresentEB);
numEBGr->SetName("EBnumGraph");
numEBGr->SetTitle(";Fraction of Amplitude;Number of Active crystals above this threshold");

TGraph *numEEPGr = new TGraph(21, ampthres, numberpresentEEP);
numEEPGr->SetName("EEPnumGraph");
numEEPGr->SetTitle(";Fraction of Amplitude;Number of Active crystals above this threshold");

TGraph *numEEMGr = new TGraph(21, ampthres, numberpresentEEM);
numEEMGr->SetName("EEMnumGraph");
numEEMGr->SetTitle(";Fraction of Amplitude;Number of Active crystals above this threshold");

TGraph *numBEBGr = new TGraph(21, ampthres, numberpresentBEB);
numBEBGr->SetName("EBnumBELOWGraph");
numBEBGr->SetTitle(";Fraction of Amplitude;Number of Active crystals below this threshold");

TGraph *numBEEPGr = new TGraph(21, ampthres, numberpresentBEEP);
numBEEPGr->SetName("EEPnumBELOWGraph");
numBEEPGr->SetTitle(";Fraction of Amplitude;Number of Active crystals below this threshold");

TGraph *numBEEMGr = new TGraph(21, ampthres, numberpresentBEEM);
numBEEMGr->SetName("EEMnumBELOWGraph");
numBEEMGr->SetTitle(";Fraction of Amplitude;Number of Active crystals below this threshold");


TGraph *numEBFr = new TGraph(21, ampthres, fractionEB);
numEBFr->SetName("EBnumFractionGraph");
numEBFr->SetTitle(";Fraction of Amplitude;Fraction of Active crystals above this threshold");

TGraph *numEEPFr = new TGraph(21, ampthres, fractionEEP);
numEEPFr->SetName("EEPnumFractionGraph");
numEEPFr->SetTitle(";Fraction of Amplitude;Fraction of Active crystals above this threshold");

TGraph *numEEMFr = new TGraph(21, ampthres, fractionEEM);
numEEMFr->SetName("EEMnumFractionGraph");
numEEMFr->SetTitle(";Fraction of Amplitude;Fraction of Active crystals above this threshold");

TFile *outfile = new TFile("outfile.root","RECREATE");
outfile->cd();
numEBGr->Write();
numEEPGr->Write();
numEEMGr->Write();
numBEBGr->Write();
numBEEPGr->Write();
numBEEMGr->Write();
numEBFr->Write();
numEEPFr->Write();
numEEMFr->Write();
outfile->Close();

}//END of the MIN FUNCTION HERE!!!!!!!!!!!!!!!


Int_t getNumActiveXtals (TProfile2D *tprof)
{

Int_t numBinseta = tprof->GetNbinsX();
Int_t numBinsphi = tprof->GetNbinsY();
std::cout << " numBinseta " << numBinseta << std::endl;
std::cout << " numBinsphi " << numBinsphi << std::endl;
Int_t biniter = 0;
Int_t numXtals = 0;

for (Int_t iphi = 0; iphi <= numBinsphi+1; ++iphi)
{
   for (Int_t ieta = 0; ieta <= numBinseta+1; ++ieta)
   {
    
	Double_t numEntries = tprof->GetBinEntries(biniter);
	if (numEntries > 0 ) 
	{
	   numXtals++;
	}
	biniter++;
   }
}

return numXtals;
}

Double_t getAverageAmplitude(TProfile2D *tprof, Double_t thres)
{

Int_t numBinseta = tprof->GetNbinsX();
Int_t numBinsphi = tprof->GetNbinsY();
Int_t biniter = 0;
Double_t numXtals = 0;
Double_t amplitude = 0;

for (Int_t iphi = 0; iphi <= numBinsphi+1; ++iphi)
{
   for (Int_t ieta = 0; ieta <= numBinseta+1; ++ieta)
   {
	Double_t amp = tprof->GetBinContent(biniter);
	if (amp > thres ) 
	{
	   numXtals++;
	   amplitude+=amp;
	}
	biniter++;
   }
}

if ( numXtals > 0 )return (amplitude/numXtals);
else return 0;

}

Double_t getNumAboveThreshold(TProfile2D *tprof, Double_t thres)
{

Int_t numBinseta = tprof->GetNbinsX();
Int_t numBinsphi = tprof->GetNbinsY();
Int_t biniter = 0;
Double_t numXtals = 0;

for (Int_t iphi = 0; iphi <= numBinsphi+1; ++iphi)
{
   for (Int_t ieta = 0; ieta <= numBinseta+1; ++ieta)
   {
	Double_t amp = tprof->GetBinContent(biniter);
	if (amp > thres ) 
	{

	   numXtals++;
	}
	biniter++;
   }
}


return numXtals;

}

TProfile2D* getProfileWithThreshold(TProfile2D *tprof, Double_t thres)
{
TProfile2D* tempProf=0;


return tempProf;
}





