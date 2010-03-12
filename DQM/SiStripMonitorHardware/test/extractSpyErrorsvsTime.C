#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <map>
#include <vector>

#include "TH1F.h"
#include "TH2F.h"
#include "TLegend.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TString.h"
#include "TDirectory.h"

int main(int argc, char** argv) {//main

  if (argc < 4){
    std::cout << "Usage: " << argv[0] 
	      << " <path to file> <number of runs> <run number list>"  
	      << std::endl;
    return 0;
  }


  unsigned int nRuns;
  std::istringstream(argv[2])>>nRuns;

  std::vector<unsigned int> runs;

  //std::map<unsigned int,std::vector<std::pair<unsigned int,float> > > lMap;
  //std::pair<std::map<unsigned int,std::vector<std::pair<unsigned int,float> > >::iterator,bool> lMapIter;

  //std::map<unsigned int,double> lMapFED; 

  unsigned int nDirNotFound = 0;
  unsigned int nFileNotFound = 0;

  std::ofstream txtoutfile;                                                                                         
  txtoutfile.open("SpyErrors.dat",std::ios::out);                                                                   
  txtoutfile << "| Error type | run # | fedID | rate of error |" << std::endl;

  TString baseName = argv[1];
  baseName += "/";

  unsigned int nErr = 0;

  for (unsigned int r(0); r<nRuns; r++){//loop on runs

    unsigned int lRun;
    std::istringstream(argv[3+r])>>lRun;

    std::cout << "Processing run " << lRun << std::endl;

    runs.push_back(lRun);


    TString fileName  = baseName;
    fileName += lRun;
    fileName += "/DQMStore.root";

    TFile *f = TFile::Open(fileName);
    
    if (!f) {
      std::cout << " -- File " << fileName << " not found." << std::endl;
      return 0;
    }

    TString dirName = "DQMData/SiStrip/ReadoutView/";
    std::cout << " -- Directory " << dirName << std::endl;

    if (!f->cd(dirName)) {
      std::cerr << " -- Folder not found. Check if file valid or source code valid. Going to next run." << std::endl;
      nDirNotFound++;
      //return 0;
      continue;
    }

    //looking for object with name chNames etc...
    TString normDir = dirName;
    normDir += "SpyMonitoringSummary";
    if (!f->cd(normDir)) {
      std::cerr << "Folder not found, please check file " << __FILE__ << ", line " << __LINE__ << std::endl;
      nDirNotFound++;
      return 0;
    };

    TH1F *hNorm = (TH1F*)gDirectory->Get("nNoData");
    double norm = hNorm->GetEntries();

    //lMapFED.insert(std::pair<unsigned int,double>(lRun,hNorm->GetMean()));                                          
                                                                                                                     
    //find out which FEDs are in error, and which error
    const unsigned int nErrs = 11;
    TString lFedName[nErrs] = {"MinZero","MaxSat",
			       "LowTrimDAC","HighTrimDAC",
			       "LowRange","HighRange",
			       "OutOfSync","OtherPbs",
			       "ApvErrorBit","ApvAddressError",
			       "NegativePeds"};                           
                                                                                                                     
    for (unsigned int iH(0); iH<nErrs; iH++){                                                                           
      TH1F *FEDobj = (TH1F*)gDirectory->Get(lFedName[iH]);
      if (!FEDobj) { 
	std::cout << "Error, histogram " << lFedName[iH] << " not found. Continue..." << std::endl;
	continue;//return 0;
      }
      else { 
	if (FEDobj->GetEntries() != 0) {
	  for (int bin(1); bin<FEDobj->GetNbinsX()+1; bin++){
	    if (FEDobj->GetBinContent(bin)>0){
	      unsigned int iFed = bin+49;
	      float lStat = FEDobj->GetBinContent(bin)*1./norm;
	      txtoutfile << lFedName[iH] << " " << lRun << " " << iFed << " " << lStat << std::endl;
	    }
	  }
	}
      }
    }

    
    for (unsigned int ifed(50); ifed<500;ifed++){//loop on FEDs

      TString fedDir = dirName;
      fedDir += "FrontEndDriver";
      fedDir += ifed;

      if (!f->cd(fedDir)) continue;
      else {
	std::cout << " - Errors detected for FED " << ifed << std::endl;
	txtoutfile << " ***************************************** " << std::endl
		   << " **** Channel/APV errors for FED " << ifed << " **** " << std::endl
		   << " ***************************************** " << std::endl;
      }
      
      TDirectory *current = gDirectory;

      //channel histos
      for (unsigned int iH(0); iH<nErrs; iH++){                                                                           
	TString objName = lFedName[iH];
	objName += "ForFED";
	objName += ifed;
	TH1F *obj = (TH1F*)current->Get(objName);
	
	if (!obj) {
	  std::cout << "Error, histogram " << objName << " not found. Exiting..." << std::endl;
	  continue;//return 0;
	}
	else {
	  if (obj->GetEntries() != 0) {
	    for (int bin(1); bin<obj->GetNbinsX()+1; bin++){
	      unsigned int iCh = bin-1;
	      //if (obj->GetNbinsX() < 100) iCh = 96*ifed+bin-1;
	      //else iCh = 192*ifed+bin-1;
	      if (obj->GetBinContent(bin)>0){ 
		float lStat = obj->GetBinContent(bin)/norm*1.;
		//std::vector<std::pair<unsigned int, float> > lVec;
		//lVec.push_back(std::pair<unsigned int, float>(lRun,lStat));
		//lMapIter = lMap.insert(std::pair<unsigned int,std::vector<std::pair<unsigned int,float> > >(iCh,lVec));
		//if (!lMapIter.second) ((lMapIter.first)->second).push_back(std::pair<unsigned int, float>(lRun,lStat));
		
		txtoutfile << objName << " " << lRun << " " << ifed << " " << iCh << " " << lStat << std::endl;
		nErr++;
	      }
	    }
	  }
	}
      }//loop on errors
    }//loop on feds

  }//loop on runs

  //unsigned int nErr =  lMap.size();

  assert (runs.size() == nRuns);

  std::cout << "Found " << nErr << " Channels/APVs with errors in " << nRuns << " runs processed." << std::endl;
  std::cout << "Number of runs where file was not found : " << nFileNotFound << std::endl;
  std::cout << "Number of runs where folder was not found : " << nDirNotFound << std::endl;

//   TFile *outfile = TFile::Open("APVAddressErrors.root","RECREATE");
//   outfile->cd();
  
//   TH1F *h[nErr];
//   std::map<unsigned int,std::vector<std::pair<unsigned int,float> > >::iterator lIter = lMap.begin();

//   unsigned int e = 0;

//   for (;lIter!=lMap.end(); lIter++){

//     unsigned int lCh = lIter->first;
//     std::ostringstream lName;
//     lName << "ErrorFed" << static_cast<unsigned int>(lCh/192.) << "Ch" << static_cast<unsigned int>(lCh%192/2.) << "APV" << lCh%2;

//     h[e] = new TH1F(lName.str().c_str(),"rate of APVAddressError vs run",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);

//     std::vector<std::pair<unsigned int,float> > lVec = lIter->second;
//     unsigned int nBins = lVec.size();

//     for (unsigned int b(0); b<nBins; b++){
//       //std::cout <<"Run " << lVec.at(b).first << ", runs[0] = " << runs[0] << ", runs[" << nRuns-1 << "] = " << runs[nRuns-1] << std::endl; 

//       h[e]->SetBinContent(lVec.at(b).first-runs[0]+1,lVec.at(b).second);
//       //std::cout << "Setting bin " << lVec.at(b).first-runs[0]+1 << " content to " << lVec.at(b).second << std::endl;
//     }

//     //h[e]->Write();

//     e++;

//   }

//   TH1F *hRate = new TH1F("hRate",";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
//   std::map<unsigned int,double>::iterator lIterFED = lMapFED.begin();
//   for (;lIterFED!=lMapFED.end(); lIterFED++){
//     hRate->SetBinContent(lIterFED->first-runs[0]+1,lIterFED->second);
//   }



//   outfile->Write();
//   outfile->Close();

  txtoutfile.close();

  return 1;
  
}//main
