#include <iostream>
#include <fstream>
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "DQMServices/Diagnostic/interface/HDQMInspector.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TH2.h"
#include "TH1.h"
#include "TH1F.h"
#include "TFile.h"
#include "TLeaf.h"
#include "TStyle.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TRandom.h" 

using namespace std;

//******************************************************************************


int main (int argc, char* argv[]) 
{


char dbString[120];
sprintf(dbString,"sqlite_file:%s",argv[1]);
printf("%s\n",dbString);

Int_t DetID = atoi(argv[2]);

char histoString[120];
sprintf(histoString,"%d@%s@%s",DetID,argv[3],argv[4]);
printf("%s\n",histoString);
char gifString[120];
sprintf(gifString,"%s@%s.gif",argv[3],argv[4]);
printf("%s\n",gifString);

gROOT->Reset();


HDQMInspector A;
A.setDB(dbString,"HDQM_test","cms_cond_strip","w3807dev","");

A.setDebug(1);
A.setDoStat(1);


//createTrend(std::string ListItems, std::string CanvasName="", int logy=0,std::string Conditions="", unsigned int firstRun=109524, unsigned int lastRun=110520);
A.createTrend(histoString,gifString,0,"");


  cout << "Ending... " <<  endl;
  return 0;

}

