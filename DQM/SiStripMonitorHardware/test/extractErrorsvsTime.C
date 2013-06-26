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

  if (argc < 6){
    std::cout << "Usage: " << argv[0] 
	      << " <path to file (paths available given in getCommand script)> <first run (0=all)> <last run (0=all)> <total number of runs> <run number list>"  
	      << std::endl;
    return 0;
  }
  std::string lPath = argv[1];
  
  bool lAfsFiles = true;

  if (lPath.find("/afs") == lPath.npos) lAfsFiles = false;

  std::string lFolder = lPath.substr(lPath.find("xxxx")-5,9);
  if (lAfsFiles) lFolder += "_afs";
  std::cout << " -- Folder = " << lFolder << std::endl;
  
  unsigned int nRuns;
  std::istringstream(argv[4])>>nRuns;

  unsigned int lRunMin=0;
  unsigned int lRunMax=10000000;
  std::istringstream(argv[2])>>lRunMin;
  std::istringstream(argv[3])>>lRunMax;

  if (lRunMax==0) lRunMax=10000000;

  std::vector<unsigned int> runs;
  const unsigned int nHistsDetailed = 5;

  std::map<unsigned int,std::vector<std::pair<unsigned int,float> > > lMap[nHistsDetailed];
  std::pair<std::map<unsigned int,std::vector<std::pair<unsigned int,float> > >::iterator,bool> lMapIter;
  std::map<unsigned int,std::vector<std::pair<unsigned int,float> > > lMapRun[nHistsDetailed];
  std::pair<std::map<unsigned int,std::vector<std::pair<unsigned int,float> > >::iterator,bool> lMapRunIter;

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
  std::ostringstream txtname;
  txtname << "FEDErrors_" << lFolder << "_" << lRunMin << "_" << lRunMax << ".dat";

  txtoutfile.open(txtname.str().c_str(),std::ios::out);

  txtoutfile << "| Error type | run # | fedID | rate of error |" << std::endl;
    //TString histName  = "APVAddressErrorBits";

  TString histName[nHistsDetailed]  = {"FEMissing","APVAddressErrorBits","APVErrorBits","UnlockedBits","OOSBits"};

  std::vector<double> lNormVal[nHistsDetailed];
  for (unsigned int r(0); r<nRuns; ++r){//loop on runs
    
    unsigned int lRun;
    std::istringstream(argv[5+r])>>lRun;

    std::cout << "Processing run " << lRun << std::endl;

    if (lRun < lRunMin || lRun > lRunMax) continue;

    runs.push_back(lRun);


    TString fileName = lPath;

    //     fileName += "/";
    //     fileName += static_cast<unsigned int>(lRun/1000.);
    //     fileName += "/";
    //     if (lRun - static_cast<unsigned int>(lRun/1000.)*1000 < 10) fileName += "00" ;
    //     else if (lRun - static_cast<unsigned int>(lRun/1000.)*1000 < 100) fileName += "0" ;
    //     fileName += lRun - static_cast<unsigned int>(lRun/1000.)*1000;

    unsigned int lSubFolder = static_cast<unsigned int>(lRun/100.);

    TString baseName = fileName;
    if (lAfsFiles) {
      std::cout << " Reading files from afs..." << std::endl;
      fileName += "/000";
      fileName += lSubFolder;
      fileName += "xx";
    }
    fileName += "/DQM_V0001_SiStrip_R000";
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
      std::cout << "Cannot open file " << fileName << " for reading ! Continue ..." << std::endl;
      nFileNotFound++;
      //return 0;
      continue;
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
      std::cerr << "Folder " << normDir << " not found, continuing... " << __FILE__ << ", line " << __LINE__ << std::endl;
      nDirNotFound++;
      //return 0;
      continue;
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

    const unsigned int nHistos = 5;

    std::string lFedName[nHistos] = {"FED/VsId/AnyFEDErrors","FED/VsId/CorruptBuffers","FE/VsId/AnyFEProblems","FED/VsId/BadDAQCRCs","FED/VsId/DataMissing"};
    std::string lFedNameBis[nHistos] = {"FEDLevel/VsFedId/AnyFEDErrors","FEDLevel/VsFedId/CorruptBuffers","FrontEndLevel/VsFedId/AnyFEProblems","FEDLevel/VsFedId/BadDAQCRCs","FEDLevel/VsFedId/DataMissing"};
    std::string lFedNameTer[nHistos] = {"AnyFEDErrors","CorruptBuffers","AnyFEProblems","BadDAQCRCs","DataMissing"};
    
    for (unsigned int iH(0); iH<nHistos; ++iH){                                                                           
      TH1F *FEDobj = (TH1F*)gDirectory->Get(lFedName[iH].c_str());
      if (!FEDobj) FEDobj = (TH1F*)gDirectory->Get(lFedNameBis[iH].c_str());
      if (!FEDobj) FEDobj = (TH1F*)gDirectory->Get(lFedNameTer[iH].c_str());

      if (!FEDobj) { 
	std::cout << "Error, histogram " << lFedName[iH] << " not found. Continue..." << std::endl;
	continue;//return 0;
      }
      else { 
	if (FEDobj->GetEntries() != 0) {
	  unsigned int nFeds = 0;
	  for (int bin(1); bin<FEDobj->GetNbinsX()+1; ++bin){
	    if (FEDobj->GetBinContent(bin)>0){
	      ++nFeds;
	    }
	  }
	  for (int bin(1); bin<FEDobj->GetNbinsX()+1; ++bin){
	    if (FEDobj->GetBinContent(bin)>0){
	      unsigned int iFed = bin+49;
	      float lStat = FEDobj->GetBinContent(bin)/norm*1.;
	      if (nFeds < 90) {// lesss than any entire partition....
		txtoutfile << lFedName[iH] << " " << lRun << " " << iFed << " " << lStat << std::endl;
	      }
	      else {
		txtoutfile << lFedName[iH] << " " << lRun << " " << nFeds << " FEDs with rate: " << lStat << std::endl;
		break;
	      }
	    }
	  }
	}
      }
    }

    for (unsigned int ifed(50); ifed<500;++ifed){//loop on FEDs

      TString fedDir = dirName;
      fedDir += "FrontEndDriver";
      fedDir += ifed;

      if (!f->cd(fedDir)) continue;
      else {
	//std::cout << " - Errors detected for FED " << ifed << std::endl;
      }
      
      TDirectory *current = gDirectory;

      //channel histos
      for (unsigned int iH(0); iH<nHistsDetailed; ++iH){                                                                           
 
	TString objName = histName[iH];
	objName += "ForFED";
	objName += ifed;
	TH1F *obj = (TH1F*)current->Get(objName);
	
	if (!obj) {
	  //std::cout << "Error, histogram " << objName << " not found. Next ..." << std::endl;
	  continue;//return 0;
	}
	else {
	  std::cout << "Processing histogram " << objName << ", nentries = " << obj->GetEntries() << std::endl;
	  if (obj->GetEntries() != 0) {
	    for (int bin(1); bin<obj->GetNbinsX()+1; ++bin){
	      if (obj->GetBinContent(bin)>0){ 
		//unsigned int iCh = static_cast<int>((bin-1)/2.);
		//unsigned int iAPV = (bin-1);//%2;
		unsigned int iCh = bin-1;
		if (iH==0) iCh = iCh*12;//fe level
		unsigned int iAPV = 0;
		if (iH==1 || iH==2) {
		  iAPV = iCh%2;
		  iCh = static_cast<unsigned int>(iCh/2.);
		}
		float lStat = obj->GetBinContent(bin)/norm*1.;
		std::vector<std::pair<unsigned int, float> > lVec;
		lVec.push_back(std::pair<unsigned int, float>(lRun,lStat));
		std::vector<std::pair<unsigned int, float> > lVecRun;
		unsigned int lChIdx = 2*96*ifed+2*iCh+iAPV;
		lVecRun.push_back(std::pair<unsigned int, float>(lChIdx,lStat));
		//lMapIter = lMap.insert(std::pair<unsigned int,std::vector<std::pair<unsigned int,float> > >(192*ifed+iAPV,lVec));
		lMapIter = lMap[iH].insert(std::pair<unsigned int,std::vector<std::pair<unsigned int,float> > >(lChIdx,lVec));
		if (!lMapIter.second) ((lMapIter.first)->second).push_back(std::pair<unsigned int, float>(lRun,lStat));
		lMapRunIter = lMapRun[iH].insert(std::pair<unsigned int,std::vector<std::pair<unsigned int,float> > >(lRun,lVecRun));
		if (!lMapRunIter.second) ((lMapRunIter.first)->second).push_back(std::pair<unsigned int, float>(lChIdx,lStat));
		else {
		  lNormVal[iH].push_back(norm);
		}
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



    f->Close();
  }//loop on runs

  const unsigned int nErr[nHistsDetailed] = {lMap[0].size(),lMap[1].size(),lMap[2].size(),lMap[3].size(),lMap[4].size()};

  //assert (runs.size() == nRuns);
  nRuns = runs.size();

  //std::cout << "Found " << nErr << " APVs with errors in " << nRuns << " runs processed." << std::endl;
  for (unsigned int iH(0); iH<nHistsDetailed; ++iH){
    txtoutfile << "Found " << nErr[iH] << " channels with errors " << histName[iH] << " in " << lMapRun[iH].size() << " runs with this type of errors." << std::endl;
  }
  std::cout << "Number of runs where file was not found : " << nFileNotFound << std::endl;
  std::cout << "Number of runs where folder was not found : " << nDirNotFound << std::endl;
  //std::cout << "Number of runs where cluster folder was not found : " << nClustDirNotFound << std::endl;

  //TFile *outfile = TFile::Open("APVAddressErrors.root","RECREATE");

  std::ostringstream lrootname;
  lrootname << "MyHDQM_" << lFolder << "_" << runs[0] << "_" << runs[nRuns-1] << ".root";

  TFile *outfile = TFile::Open(lrootname.str().c_str(),"RECREATE");
  outfile->cd();


  txtoutfile << " ****************************************** " 
	     << std::endl
	     << " ** Summary of errors per FED/CH per run ** " 
	     << std::endl
	     << " ***** for rate < 1% and nRuns < 5 ******** "
	     << std::endl
	     << " ****************************************** " 
	     <<std::endl;
  
  for (unsigned int iH=0; iH<nHistsDetailed; ++iH){//loop on histos

    TH1F *h[nErr[iH]];
    std::map<unsigned int,std::vector<std::pair<unsigned int,float> > >::iterator lIter = lMap[iH].begin();
    std::ostringstream lNameRun;
    lNameRun << "Number_" << histName[iH] << "_vsRun";
    TH1F *hRun = new TH1F(lNameRun.str().c_str(),"Number of channel with Error vs run; run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
    lNameRun.str("");
    lNameRun << "FEDS_" << histName[iH] << "_vsRun";
    TH2F *hFEDRun_apv0 = new TH2F((lNameRun.str()+"_apv0").c_str(),"Number of runs with FED channels APV0 with Errors; FED id ; ch id",440,50,490,96,0,96);
    TH2F *hFEDRun_apv1 = new TH2F((lNameRun.str()+"_apv1").c_str(),"Number of runs with FED channels APV1 with Errors; FED id ; ch id",440,50,490,96,0,96);
    lNameRun.str("");
    lNameRun << "FEDS_" << histName[iH] << "_vsRate";
    TH2F *hFEDRate_apv0 = new TH2F((lNameRun.str()+"_apv0").c_str(),"Mean rate of errors of FED channels APV0 with Errors; FED id ; ch id",440,50,490,96,0,96);
    TH2F *hFEDRate_apv1 = new TH2F((lNameRun.str()+"_apv1").c_str(),"Mean rate of errors of FED channels APV1 with Errors; FED id ; ch id",440,50,490,96,0,96);
    std::map<unsigned int,std::vector<std::pair<unsigned int,float> > >::iterator lIterRun = lMapRun[iH].begin();

    assert(lNormVal[iH].size() == lMapRun[iH].size());
    unsigned int lIdx = 0;
    for (;lIterRun!=lMapRun[iH].end(); ++lIterRun,++lIdx){//loop on elements

      std::vector<std::pair<unsigned int,float> > lVec = lIterRun->second;
      hRun->Fill(lIterRun->first,lVec.size());
      for (unsigned int ele(0);ele<lVec.size();++ele){
	unsigned int lCh = lVec[ele].first;
	unsigned int lElements = lMap[iH][lCh].size();
	unsigned int lApv = static_cast<unsigned int>(lCh%2);
	if (lElements<5 && lVec[ele].second < 0.01) {
	  txtoutfile << histName[iH] 
		     << " Run " << lIterRun->first << " fed " 
		     << static_cast<unsigned int>(lCh/192.) << " ch " 
		     << static_cast<unsigned int>((lCh%192)/2.) << " apv "
		     << lApv
		     << " rate " << lVec[ele].second 
		     << " (" << 1/lNormVal[iH][lIdx] << ")"
		     << " nRuns " << lElements
		     << std::endl;
	}
	if (lApv==0) {
	  hFEDRun_apv0->Fill(static_cast<unsigned int>(lCh/192.),static_cast<unsigned int>((lCh%192)/2.));
	  hFEDRate_apv0->Fill(static_cast<unsigned int>(lCh/192.),static_cast<unsigned int>((lCh%192)/2.),lVec[ele].second/lElements);
	}
	else {
	  hFEDRun_apv1->Fill(static_cast<unsigned int>(lCh/192.),static_cast<unsigned int>((lCh%192)/2.));
	  hFEDRate_apv1->Fill(static_cast<unsigned int>(lCh/192.),static_cast<unsigned int>((lCh%192)/2.),lVec[ele].second/lElements);
	}
      }
    }//loop on elements




//     unsigned int e = 0;
//     for (;lIter!=lMap[iH].end(); lIter++,e++){//loop on elements

//       unsigned int lCh = lIter->first;
//       std::ostringstream lName;
//       //lName << "ErrorFed" << static_cast<unsigned int>(lCh/192.) << "Ch" << static_cast<unsigned int>(lCh%192/2.) << "APV" << lCh%2;
//       lName << "Hist" << iH << "ErrorFed" << static_cast<unsigned int>(lCh/96.) << "Ch" << static_cast<unsigned int>(lCh%96) ;//<< "APV" << lCh%2;

//       //h[e] = new TH1F(lName.str().c_str(),"rate of APVAddressError vs run",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
//       h[e] = new TH1F(lName.str().c_str(),"rate of Error vs run; run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);

//       std::vector<std::pair<unsigned int,float> > lVec = lIter->second;
//       unsigned int nBins = lVec.size();
      
//       for (unsigned int b(0); b<nBins; b++){
// 	//std::cout <<"Run " << lVec.at(b).first << ", runs[0] = " << runs[0] << ", runs[" << nRuns-1 << "] = " << runs[nRuns-1] << std::endl; 

// 	h[e]->SetBinContent(lVec.at(b).first-runs[0]+1,lVec.at(b).second);
// 	//std::cout << "Setting bin " << lVec.at(b).first-runs[0]+1 << " content to " << lVec.at(b).second << std::endl;
//       }

//       //h[e]->Write();

//     }//loop on elements
  }//loop on histos

  TH1F *hRate = new TH1F("hRate",";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
  std::map<unsigned int,double>::iterator lIterFED = lMapFED.begin();
  for (;lIterFED!=lMapFED.end(); ++lIterFED){
    hRate->SetBinContent(lIterFED->first-runs[0]+1,lIterFED->second);
  }


  TH1F *hRateFEDChannels = new TH1F("hRateFEDChannels",";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
  std::map<unsigned int,std::pair<double,double> >::iterator lIterFEDCh = lMapFEDChannels.begin();
  for (;lIterFEDCh!=lMapFEDChannels.end(); ++lIterFEDCh){
    hRateFEDChannels->SetBinContent(lIterFEDCh->first-runs[0]+1,lIterFEDCh->second.first/36392.*100);
    hRateFEDChannels->SetBinError(lIterFEDCh->first-runs[0]+1,lIterFEDCh->second.second/36392.*100);
  }

  TH1F *hRateChannels = new TH1F("hRateChannels",";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
  std::map<unsigned int,std::pair<double,double> >::iterator lIterCh = lMapChannels.begin();
  for (;lIterCh!=lMapChannels.end(); ++lIterCh){
    hRateChannels->SetBinContent(lIterCh->first-runs[0]+1,lIterCh->second.first/36392.*100);
    hRateChannels->SetBinError(lIterCh->first-runs[0]+1,lIterCh->second.second/36392.*100);
  }

  TH1F *hRateAll = new TH1F("hRateAll",";run #",runs[nRuns-1]-runs[0]+1,runs[0],runs[nRuns-1]+1);
  std::map<unsigned int,std::pair<double,double> >::iterator lIterAll = lMapAll.begin();
  for (;lIterAll!=lMapAll.end(); ++lIterAll){
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
