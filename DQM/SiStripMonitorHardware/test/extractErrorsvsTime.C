#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cassert>
#include <map>
#include <vector>
#include <cmath>

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TLegend.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TString.h"
#include "TDirectory.h"

//struct Clusters {
//  unsigned int nEvents;
//  double mean;
//  double meanErr;
//};

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
  const unsigned int nHistsDetailed = 5;

  std::map<unsigned int,std::vector<std::pair<unsigned int,float> > > lMap[nHistsDetailed];
  std::pair<std::map<unsigned int,std::vector<std::pair<unsigned int,float> > >::iterator,bool> lMapIter;

  std::map<unsigned int,double> lMapFED; 
  std::map<unsigned int,std::pair<double,double> > lMapFEDChannels; 
  std::map<unsigned int,std::pair<double,double> > lMapChannels; 
  std::map<unsigned int,std::pair<double,double> > lMapAll; 

  //4 histos, 4 partitions
  //pair nb events, mean+err value of histo
  //std::map<unsigned int,Clusters> lMapClust[4][4];
  //std::map<unsigned int,Clusters>::iterator lIterClust;
  //TString clustDir[4] = {"TIB","TID","TOB","TEC"};

  unsigned int nDirNotFound = 0;
  //unsigned int nClustDirNotFound = 0;
  unsigned int nFileNotFound = 0;

  std::ofstream txtoutfile;                                                                                         
  txtoutfile.open("FEDErrors_16xyyy.dat",std::ios::out);                                                                   
  txtoutfile << "| Error type | run # | fedID | rate of error |" << std::endl;

  for (unsigned int r(0); r<nRuns; r++){//loop on runs

    unsigned int lRun;
    std::istringstream(argv[3+r])>>lRun;

    std::cout << "Processing run " << lRun << std::endl;

    runs.push_back(lRun);

    //TString histName  = "APVAddressErrorBits";
    TString histName[nHistsDetailed]  = {"FEMissing","APVAddressErrorBits","APVErrorBits","UnlockedBits","OOSBits"};


    TString fileName = argv[1];

    //     fileName += "/";
    //     fileName += static_cast<unsigned int>(lRun/1000.);
    //     fileName += "/";
    //     if (lRun - static_cast<unsigned int>(lRun/1000.)*1000 < 10) fileName += "00" ;
    //     else if (lRun - static_cast<unsigned int>(lRun/1000.)*1000 < 100) fileName += "0" ;
    //     fileName += lRun - static_cast<unsigned int>(lRun/1000.)*1000;

    unsigned int lSubFolder = static_cast<unsigned int>(lRun/100.);

    TString baseName = fileName;
    fileName += "/000";
    fileName += lSubFolder;
    fileName += "xx/DQM_V0001_SiStrip_R000";
    fileName += lRun;
    fileName += ".root";

    
    //    fileName += "/DQM_V0005_R000";
    //     fileName += lRun;
    //     fileName += ".root";


    TFile *f = TFile::Open(fileName);
    
 //    if (!f) {
//       std::cout << "Cannot open file " << fileName << " for reading !" << std::endl;
//       fileName = baseName;
//       fileName += "/DQM_V0004_R000";
//       fileName += lRun;
//       fileName += ".root";
      
//       f = TFile::Open(fileName);
//       if (!f) {
// 	std::cout << "Cannot open file " << fileName << " for reading !" << std::endl;
// 	fileName = baseName;
// 	fileName += "/DQM_V0003_R000";
// 	fileName += lRun;
// 	fileName += ".root";
	
// 	f = TFile::Open(fileName);
// 	if (!f) {
// 	  std::cout << "Cannot open file " << fileName << " for reading !" << std::endl;
// 	  fileName = baseName;
// 	  fileName += "/DQM_V0002_R000";
// 	  fileName += lRun;
// 	  fileName += ".root";
	  
// 	  f = TFile::Open(fileName);
// 	  if (!f) {
// 	    std::cout << "Cannot open file " << fileName << " for reading !" << std::endl;
// 	    fileName = baseName;
// 	    fileName += "/DQM_V0001_R000";
// 	    fileName += lRun;
// 	    fileName += ".root";
	    
// 	    f = TFile::Open(fileName);
    if (!f) {
      std::cout << "Cannot open file " << fileName << " for reading ! Exiting ..." << std::endl;
      nFileNotFound++;
      return 0;
    }
// 	  }
// 	}
//       }
//     }

    TString dirName = "DQMData/Run ";
    dirName += lRun;
    dirName += "/SiStrip/Run summary/ReadoutView/";
    std::cout << "Directory " << dirName << std::endl;

    if (!f->cd(dirName)) {
      std::cerr << "Folder not found. Check if file valid or source code valid in " << __FILE__ << ", variable dirName! Going to next run." << std::endl;
      nDirNotFound++;
      //return 0;
      continue;
    }

    //looking for object with name chNames etc...
    TString normDir = dirName;
    normDir += "FedMonitoringSummary/";
    if (!f->cd(normDir)) {
      std::cerr << "Folder " << normDir << " not found, please modify source code " << __FILE__ << ", line " << __LINE__ << std::endl;
      nDirNotFound++;
      return 0;
    };

  

    TH1F *hNorm = (TH1F*)gDirectory->Get("FED/nFEDErrors");
    if (!hNorm) hNorm = (TH1F*)gDirectory->Get("FEDLevel/nFEDErrors");
    if (!hNorm) hNorm = (TH1F*)gDirectory->Get("nFEDErrors");
    if (!hNorm) {
      std::cout << "Error, histogram nFEDErrors not found. Continue..." << std::endl;
      continue;//return 0;
    }
    double norm = hNorm->GetEntries();

    lMapFED.insert(std::pair<unsigned int,double>(lRun,hNorm->GetMean()));                                          
          
    TH1F *hChannels = (TH1F*)gDirectory->Get("Fiber/nBadChannelStatusBits");
    if (!hChannels) hChannels = (TH1F*)gDirectory->Get("FiberLevel/nBadChannelStatusBits");
    if (!hChannels) hChannels = (TH1F*)gDirectory->Get("nBadChannelStatusBits");
    if (!hChannels){
      std::cout << "Error, histogram nBadChannelStatusBits not found. Continue..." << std::endl;
      continue;
    }

    lMapChannels.insert(std::pair<unsigned int,std::pair<double,double> >(lRun,std::pair<double,double>(hChannels->GetMean(),hChannels->GetMeanError())));               

    

    TProfile *hAll = (TProfile*)gDirectory->Get("Trends/nTotalBadChannelsvsTime");
    if (!hAll) hAll = (TProfile*)gDirectory->Get("ErrorsVsTime/nTotalBadChannelsvsTime");
    if (!hAll) hAll = (TProfile*)gDirectory->Get("nTotalBadChannelsvsTime");
    if (!hAll) {
      std::cout << "Error, histogram nTotalBadChannelsvsTime not found. Continue..." << std::endl;
      continue;
    }
    lMapAll.insert(std::pair<unsigned int,std::pair<double,double> >(lRun,std::pair<double,double>(hAll->GetMean(2),hAll->GetMeanError(2))));

    if (hAll->GetMean(2) < 0.000001) std::cout << " -- Run " << lRun << " has no errors." << std::endl; 

    lMapFEDChannels.insert(std::pair<unsigned int,std::pair<double,double> >(lRun,std::pair<double,double>(hAll->GetMean(2)-hChannels->GetMean(),sqrt(hAll->GetMeanError(2)*hAll->GetMeanError(2)+hChannels->GetMeanError()*hChannels->GetMeanError()))));

    //find out which FEDs are in error, and which error
    //TString lFedName[6] = {"AnyFEDErrors","CorruptBuffers","AnyFEProblems","AnyDAQProblems","DataMissing","BadDAQCRCs"};
    TString lFedName[4] = {"FED/VsId/AnyFEDErrors","FED/VsId/CorruptBuffers","FED/VsId/AnyFEProblems","FED/VsId/BadDAQCRCs"};
    TString lFedNameBis[4] = {"FEDLevel/VsFedId/AnyFEDErrors","FEDLevel/VsFedId/CorruptBuffers","FEDLevel/VsFedId/AnyFEProblems","FEDLevel/VsFedId/BadDAQCRCs"};
    TString lFedNameTer[4] = {"AnyFEDErrors","CorruptBuffers","AnyFEProblems","BadDAQCRCs"};
    
    for (unsigned int iH(0); iH<4; iH++){                                                                           
      TH1F *FEDobj = (TH1F*)gDirectory->Get(lFedName[iH]);
      if (!FEDobj) FEDobj = (TH1F*)gDirectory->Get(lFedNameBis[iH]);
      if (!FEDobj) FEDobj = (TH1F*)gDirectory->Get(lFedNameTer[iH]);

      if (!FEDobj) { 
	std::cout << "Error, histogram " << lFedName[iH] << " not found. Continue..." << std::endl;
	continue;//return 0;
      }
      else { 
	if (FEDobj->GetEntries() != 0) {
	  for (int bin(1); bin<FEDobj->GetNbinsX()+1; bin++){
	    if (FEDobj->GetBinContent(bin)>0){
	      unsigned int iFed = bin+49;
	      float lStat = FEDobj->GetBinContent(bin)/norm*1.;
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
      }
      
      TDirectory *current = gDirectory;

      //channel histos
      for (unsigned int iH(0); iH<nHistsDetailed; iH++){                                                                           
 
	TString objName = histName[iH];
	objName += "ForFED";
	objName += ifed;
	TH1F *obj = (TH1F*)current->Get(objName);
	
	if (!obj) {
	  std::cout << "Error, histogram " << objName << " not found. Next ..." << std::endl;
	  continue;//return 0;
	}
	else {
	  std::cout << "Processing histogram " << objName << ", nentries = " << obj->GetEntries() << std::endl;
	  if (obj->GetEntries() != 0) {
	    for (int bin(1); bin<obj->GetNbinsX()+1; bin++){
	      if (obj->GetBinContent(bin)>0){ 
		//unsigned int iCh = static_cast<int>((bin-1)/2.);
		//unsigned int iAPV = (bin-1);//%2;
		unsigned int iCh = bin-1;
		if (iH==0) iCh = iCh*12;
		float lStat = obj->GetBinContent(bin)/norm*1.;
		std::vector<std::pair<unsigned int, float> > lVec;
		lVec.push_back(std::pair<unsigned int, float>(lRun,lStat));
		//lMapIter = lMap.insert(std::pair<unsigned int,std::vector<std::pair<unsigned int,float> > >(192*ifed+iAPV,lVec));
		lMapIter = lMap[iH].insert(std::pair<unsigned int,std::vector<std::pair<unsigned int,float> > >(96*ifed+iCh,lVec));
		if (!lMapIter.second) ((lMapIter.first)->second).push_back(std::pair<unsigned int, float>(lRun,lStat));


	      }
	    }
	  }
	}
      }
      
    }//loop on feds

//     TString clustDirBase = "DQMData/Run ";
//     clustDirBase += lRun;
//     clustDirBase += "/SiStrip/Run summary/MechanicalView/";
//     TString histClust[4] = {"Summary_MeanNumberOfClusters__",
// 			    "Summary_MeanNumberOfDigis__",
// 			    "Summary_TotalNumberOfClusters_OffTrack_in_",
// 			    "Summary_TotalNumberOfClusters_OnTrack_in_"};

//     for (unsigned iP(0); iP<4; iP++){//loop on partitions
//       TString lClustDir = clustDirBase+clustDir[iP];
//       if (!f->cd(lClustDir)) {
//       std::cerr << "Folder not found. Check if file valid or source code valid in " << __FILE__ << ", variable dirName! Going to next run." << std::endl;
//       nClustDirNotFound++;
//       //return 0;
//       continue;
//       }

//       for (unsigned int iH(0); iH<4; iH++){
// 	TString lHistName = histClust[iH];
// 	lHistName += clustDir[iP];

// 	TH1F *hClust = (TH1F*)gDirectory->Get(lHistName);
// 	if (!hClust) {
// 	  std::cout << "Can't find object " << lHistName << " in directory " 
// 		    << lClustDir
// 		    << ", continuing..."
// 		    << std::endl;
// 	  continue;
// 	}
// 	Clusters lClust;
// 	lClust.nEvents = hClust->GetEntries();
// 	lClust.mean = hClust->GetMean();
// 	lClust.meanErr = hClust->GetMeanError();
// 	lMapClust[iP][iH].insert(std::pair<unsigned int,Clusters >(lRun,lClust));
//       }


//     }//loop on partitions




  }//loop on runs

  const unsigned int nErr[nHistsDetailed] = {lMap[0].size(),lMap[1].size(),lMap[2].size(),lMap[3].size(),lMap[4].size()};

  assert (runs.size() == nRuns);

  //std::cout << "Found " << nErr << " APVs with errors in " << nRuns << " runs processed." << std::endl;
  std::cout << "Found " << nErr << " channels with errors in " << nRuns << " runs processed." << std::endl;
  std::cout << "Number of runs where file was not found : " << nFileNotFound << std::endl;
  std::cout << "Number of runs where folder was not found : " << nDirNotFound << std::endl;
  //std::cout << "Number of runs where cluster folder was not found : " << nClustDirNotFound << std::endl;

  //TFile *outfile = TFile::Open("APVAddressErrors.root","RECREATE");
  TFile *outfile = TFile::Open("MyHDQM_run16xyyy.root","RECREATE");
  outfile->cd();
  
  for (unsigned int iH=0; iH<nHistsDetailed; iH++){

    TH1F *h[nErr[iH]];
    std::map<unsigned int,std::vector<std::pair<unsigned int,float> > >::iterator lIter = lMap[iH].begin();

    unsigned int e = 0;

    for (;lIter!=lMap[iH].end(); lIter++,e++){

      unsigned int lCh = lIter->first;
      std::ostringstream lName;
      //lName << "ErrorFed" << static_cast<unsigned int>(lCh/192.) << "Ch" << static_cast<unsigned int>(lCh%192/2.) << "APV" << lCh%2;
      lName << "Hist" << iH << "ErrorFed" << static_cast<unsigned int>(lCh/96.) << "Ch" << static_cast<unsigned int>(lCh%96) ;//<< "APV" << lCh%2;

      //h[e] = new TH1F(lName.str().c_str(),"rate of APVAddressError vs run",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
      h[e] = new TH1F(lName.str().c_str(),"rate of Error vs run; run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);

      std::vector<std::pair<unsigned int,float> > lVec = lIter->second;
      unsigned int nBins = lVec.size();

      for (unsigned int b(0); b<nBins; b++){
	//std::cout <<"Run " << lVec.at(b).first << ", runs[0] = " << runs[0] << ", runs[" << nRuns-1 << "] = " << runs[nRuns-1] << std::endl; 

	h[e]->SetBinContent(lVec.at(b).first-runs[0]+1,lVec.at(b).second);
	//std::cout << "Setting bin " << lVec.at(b).first-runs[0]+1 << " content to " << lVec.at(b).second << std::endl;
      }

      //h[e]->Write();

    }
  }

  TH1F *hRate = new TH1F("hRate",";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
  std::map<unsigned int,double>::iterator lIterFED = lMapFED.begin();
  for (;lIterFED!=lMapFED.end(); lIterFED++){
    hRate->SetBinContent(lIterFED->first-runs[0]+1,lIterFED->second);
  }


  TH1F *hRateFEDChannels = new TH1F("hRateFEDChannels",";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
  std::map<unsigned int,std::pair<double,double> >::iterator lIterFEDCh = lMapFEDChannels.begin();
  for (;lIterFEDCh!=lMapFEDChannels.end(); lIterFEDCh++){
    hRateFEDChannels->SetBinContent(lIterFEDCh->first-runs[0]+1,lIterFEDCh->second.first/36392.*100);
    hRateFEDChannels->SetBinError(lIterFEDCh->first-runs[0]+1,lIterFEDCh->second.second/36392.*100);
  }

  TH1F *hRateChannels = new TH1F("hRateChannels",";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
  std::map<unsigned int,std::pair<double,double> >::iterator lIterCh = lMapChannels.begin();
  for (;lIterCh!=lMapChannels.end(); lIterCh++){
    hRateChannels->SetBinContent(lIterCh->first-runs[0]+1,lIterCh->second.first/36392.*100);
    hRateChannels->SetBinError(lIterCh->first-runs[0]+1,lIterCh->second.second/36392.*100);
  }

  TH1F *hRateAll = new TH1F("hRateAll",";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
  std::map<unsigned int,std::pair<double,double> >::iterator lIterAll = lMapAll.begin();
  for (;lIterAll!=lMapAll.end(); lIterAll++){
    hRateAll->SetBinContent(lIterAll->first-runs[0]+1,lIterAll->second.first/36392.*100);
    hRateAll->SetBinError(lIterAll->first-runs[0]+1,lIterAll->second.second/36392.*100);
  }



//   TH1F *hRateClust[4][4];
//   for (unsigned int iP(0); iP<4; iP++){
//     for (unsigned int iH(0); iH<4; iH++){
//       std::ostringstream lNameClust;
//       lNameClust << "hRateClust_" << clustDir[iP] << "_" << iH ;
//       hRateClust[iP][iH] = new TH1F(lNameClust.str().c_str(),";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
//       lIterClust = lMapClust[iP][iH].begin();
//       for (;lIterClust!=lMapClust[iP][iH].end(); lIterClust++){
// 	hRateClust[iP][iH]->SetBinContent(lIterClust->first-runs[0]+1,lIterClust->second.mean);
// 	hRateClust[iP][iH]->SetBinError(lIterClust->first-runs[0]+1,lIterClust->second.meanErr);
//       }
//     }
//   }


  outfile->Write();
  outfile->Close();
  txtoutfile.close();

  return 1;
  
}//main
