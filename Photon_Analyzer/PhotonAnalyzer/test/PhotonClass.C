#define PhotonClass_cxx
#include "PhotonClass.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include "TH1F.h"
#include <vector>
#include "TStopwatch.h"
#include <cstring>
#include <list>
#include <TStyle.h>




using namespace std;

int main(int argc, const char* argv[])
{
  Long64_t maxEvents = atof(argv[3]);
  if (maxEvents < -1LL)
    {
      std::cout<<"Please enter a valid value for maxEvents (parameter 3)."<<std::endl;
      return 1;
    }
  int reportEvery = atof(argv[4]);
  if (reportEvery < 1)
    {
      std::cout<<"Please enter a valid value for reportEvery (parameter 4)."<<std::endl;
      return 1;
    }
  PhotonClass t(argv[1],argv[2]);
  t.Loop(maxEvents,reportEvery);
  return 0;
}





void PhotonClass::Loop(Long64_t maxEvents, int reportEvery)
{


  if (fChain == 0) return;
   int nTotal;
   nTotal = 0;
 

   Long64_t nentries = fChain->GetEntriesFast();
   std::cout << "Total entries: " << nentries << std::endl;
   Long64_t nentriesToCheck = nentries;

   if (maxEvents != -1LL && nentries > maxEvents)
     nentriesToCheck = maxEvents;
     nTotal = nentriesToCheck;
 
   Long64_t nbytes = 0, nb = 0;
   std::cout<<"Running over "<<nTotal<<" events."<<std::endl;
   TStopwatch sw;
   sw.Start();


   for (Long64_t jentry=0; jentry<nentries;jentry++)
     {

       Long64_t ientry = LoadTree(jentry);
       if (ientry < 0) break;
       nb = fChain->GetEntry(jentry);   nbytes += nb;
       Photon_nPho->Fill(nPho);


      for( int i=0; i<nPho; i++ )
	{
	  
	  if(  phoEt->at(i) > 30 && fabs(phoSCEta->at(i)) < 1.4442 )
	    {   
	      Photon_HoverE->Fill(phoHoverE->at(i));
	      Photon_phoPFChIso->Fill(phoPFChIso->at(i));
	      Photon_phoPFPhoIso->Fill(phoPFPhoIso->at(i));
	       Photon_phoPFNeuIso->Fill(phoPFNeuIso->at(i));
	       Photon_phoEt->Fill(phoEt->at(i));
	       Photon_phoEta->Fill(phoEta->at(i));
	       Photon_phohasPixelSeed->Fill(phohasPixelSeed->at(i));
	       Photon_phoR9->Fill(phoR9->at(i));
	     

	      std::cout<<"Running"<<std::endl;	      
	      // if (Cut(ientry) < 0) continue;
	    }
	  //	tree->Fill();	  
	}
     }

   if((nentriesToCheck-1)%reportEvery != 0)
     std::cout<<"Finished entry "<<(nentriesToCheck-1)<<"/"<<(nentriesToCheck-1)<<std::endl;
   sw.Stop();

}



void PhotonClass::Histograms(const char* file2)
      {
        fileName = new TFile(file2, "RECREATE");
	fileName->cd();
	Photon_HoverE= new TH1F("Photon_HoverE","",100,0,1);
	Photon_phoPFChIso= new TH1F("Photon_phoPFChIso","",50,0,5);
	Photon_phoPFPhoIso= new TH1F("Photon_phoPFPhoIso","",100,0,10);
	Photon_phoPFNeuIso= new	TH1F ("Photon_phoPFNeuIso","",50,0,5);
	Photon_phoEt=new TH1F ("Photon_phoEt","",100,30,1000);
	Photon_phoEta=new TH1F ("Photon_phoEta","",100,-5,5);
	Photon_phohasPixelSeed= new TH1F ("Photon_phohasPixelSeed","",2,0,2);
	Photon_phoR9= new TH1F ("Photon_phoR9","",50,0,1);
	Photon_nPho=new TH1I ("Photon_nPho","",20,0,20);


      }


