#include <fstream>
#include <iomanip>
#include "TH1F.h"
#include "TCanvas.h"
#include "TRandom.h"
#include "TGraphErrors.h"
#include "Utilities.h"
#include "Definitions.h"
void GenerateUncertainty()
{
  //gROOT->ProcessLine(".X setEnv.C");
  int i,j;
  double x,y,eta;
  ofstream UncertaintyFile;
  UncertaintyFile.open((SINGLE_UNCERTAINTY_TAG+".txt").c_str());
  UncertaintyFile.setf(ios::right);
  ///////////////////////////////////////////////////////////// 
  TGraphErrors *gTot[NETA];
  TGraph *gErrTot[NETA];
  for(i=0;i<NETA;i++)
    {
      eta = 0.5*(etaBoundaries[i]+etaBoundaries[i+1]);
      gTot[i] = UncVsPt(eta,10,1000,50);
      gErrTot[i] = getError(gTot[i]);
      cout<<"Processing eta bin ["<<etaBoundaries[i]<<","<<etaBoundaries[i+1]<<"] "<<endl;
      UncertaintyFile << setw(13) << etaBoundaries[i] << setw(13) << etaBoundaries[i+1] << setw(13) << 3*gErrTot[i]->GetN() << setw(13);
      
      for(j=0;j<gErrTot[i]->GetN();j++)
        {
	   gErrTot[i]->GetPoint(j,x,y);
	   UncertaintyFile << x << setw(13) << y << setw(13) << y << setw(13);
	}
      UncertaintyFile << "\n";	
    }	   
  UncertaintyFile.close();
  cout<<endl;
  cout<<"File "<< SINGLE_UNCERTAINTY_TAG+".txt" << " written"<<endl;
}  

//////////////////////////////////////////////////////////////
TGraphErrors* UncVsPt(double eta, double ptMin, double ptMax, int Npoints)
{
  int i,j,k,ind;
  double c,r,e,energy,corPt,corE,finalC,pt,ratio;
  double x[NTRIALS],y[NTRIALS],ex[NTRIALS],ey[NTRIALS];

  vector<string> vL = parseLevels(LEVELS);
  vector<string> vT = parseLevels(CORRECTION_TAGS);
  vector<string> vU = parseLevels(UNCERTAINTY_TAGS);
  vector<int> vOn; 
  CombinedJetCorrector *JetCorrector[10];
  JetCorrectionUncertainty *JetUnc[10];
  for(i=0;i<vL.size();i++)
    {
      JetCorrector[i] = new CombinedJetCorrector(vL[i],vT[i]);
      string tmp = "../../../CondFormats/JetMETObjects/data/"+vU[i]+".txt";
      JetUnc[i] = new JetCorrectionUncertainty(tmp);
      ind = ACTIVE_UNCERTAINTIES.find(vL[i]);
      if (ind>=0)
        vOn.push_back(1);
      else
        vOn.push_back(0);
    }
  TH1F *hCor = new TH1F("Cor","Cor",1000,0,6);
  TRandom *rnd = new TRandom();
  /////////////////////////////////////////////////////////
  double max = TMath::Min(ptMax,0.5*CM_ENERGY/cosh(eta));
  ratio = pow(max/ptMin,1./Npoints);
  pt = ptMin;
  k = 0;
  while (pt<=max)
    {   
      hCor->Reset();
      for(i=0;i<NTRIALS;i++)
        {  
          corPt = pt;
          energy = pt*cosh(eta); 
          corE = energy;  
          for(j=0;j<vL.size();j++)
           {      
             c = JetCorrector[j]->getCorrection(corPt,eta,corE);
             e = JetUnc[j]->uncertaintyPtEta(corPt,eta,"UP");
             if (vOn[j]==0)
               r = c;
             else
               r = rnd->Gaus(c,e*c);
             corPt*=r;
           }
          finalC = corPt/pt;
          hCor->Fill(finalC);
       }
      x[k] = pt;
      y[k] = hCor->GetMean();
      ex[k] = 0;
      ey[k] = hCor->GetRMS();
      pt*=ratio; 
      k++;
    }
  delete hCor;
  TGraphErrors *g = new TGraphErrors(k,x,y,ex,ey);
  return g;
}  
