#include "CalibCalorimetry/EcalTiming/interface/EcalTimeTreeContent.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "TH2F.h"
#include "TTree.h"
#include "TFile.h"
#include "TCanvas.h"

#include <iostream>


int main(int argc,  char * argv[])
{
  // Binary to view the hits per crystal maps of a TTree
  // Also prints out information on run range of the TTree and number of: hits, entries, and consecutive duplicate hits with amplitude above 26 (47) in EB (EE) and |t| < 5 (7) ns
  // Duplicate hits require the same time, amplitude, time error and in the barrel also the same swiss cross noise. Some duplicates are expected by random chance when enough hits are present in the input tree
  // To view for instance the occupancy EB map produced by this binary, open root and type .x occupancyEB.C
  // Jared Turkewitz
  // Usage: MakeEcalTimingTreePlots <TTree input file>

  char* infile = argv[1];
  if(!infile)
  {
    std::cout << "Missing input file." << std::endl;
    return -1;
  }

  EcalTimeTreeContent treeVars_;

  TFile* myFile = new TFile(infile);
  TTree *myInputTree_ = (TTree*)myFile->Get("EcalTimeAnalysis");

  float lastCryTimesEB[EBDetId::kSizeForDenseIndexing];
  for(int i=0; i<EBDetId::kSizeForDenseIndexing;++i)
    lastCryTimesEB[i] = 0;
  float lastCryAmpsEB[EBDetId::kSizeForDenseIndexing];
  for(int i=0; i<EBDetId::kSizeForDenseIndexing;++i)
    lastCryAmpsEB[i] = 0;
  float lastCrySwissCrossEB[EBDetId::kSizeForDenseIndexing];
  for(int i=0; i<EBDetId::kSizeForDenseIndexing;++i)
    lastCrySwissCrossEB[i] = 0;
  float lastCryTimeErrorEB[EBDetId::kSizeForDenseIndexing];
  for(int i=0; i<EBDetId::kSizeForDenseIndexing;++i)
    lastCryTimeErrorEB[i] = 0;

  float lastCryTimesEE[EEDetId::kSizeForDenseIndexing];
  for(int i=0; i<EEDetId::kSizeForDenseIndexing;++i)
    lastCryTimesEE[i] = 0;
  float lastCryAmpsEE[EEDetId::kSizeForDenseIndexing];
  for(int i=0; i<EEDetId::kSizeForDenseIndexing;++i)
    lastCryAmpsEE[i] = 0;
  float lastCryTimeErrorEE[EEDetId::kSizeForDenseIndexing];
  for(int i=0; i<EEDetId::kSizeForDenseIndexing;++i)
    lastCryTimeErrorEE[i] = 0;

  float hitCounterPerCrystalEB[EBDetId::kSizeForDenseIndexing];
  for(int i=0; i<EBDetId::kSizeForDenseIndexing;++i)
    hitCounterPerCrystalEB[i] = 0;
  float hitCounterPerCrystalEE[EEDetId::kSizeForDenseIndexing];
  for(int i=0; i<EEDetId::kSizeForDenseIndexing;++i)
    hitCounterPerCrystalEE[i] = 0;
  
  int runFirst = 999999;
  int runLast = 0;
  
  TCanvas* canvas = new TCanvas();
  TH2F* occupancyEB = new TH2F("hitsPerCryMapEB","Hits used in each crystal;i#phi;i#eta",360,1.,361.,171,-85,86);
  TH2F* occupancyEBNoDuplicates = new TH2F("occupancyEBNoDuplicates","occupancyEBNoDuplicates",360,1,361,171,-85,86); 
  TH2F* duplicatesEB = new TH2F("duplicatesEB","duplicatesEB",360,1,361,171,-85,86);
  TH2F* occupancyEEM = new TH2F("hitsPerCryMapEEM","Hits per cry EEM;ix;iy",100,1,101,100,1,101);
  TH2F* occupancyEEP = new TH2F("hitsPerCryMapEEP","Hits per cry EEP;ix;iy",100,1,101,100,1,101);

  setBranchAddresses(myInputTree_,treeVars_);
  int hitCounterEBAll = 0;
  int duplicateCounterEBAll = 0;
  int hitCounterEEAll = 0;
  int duplicateCounterEEAll = 0;
  int nEntries = myInputTree_->GetEntries();
  for(int entry = 0; entry < nEntries; ++entry)
  {
    myInputTree_->GetEntry(entry);
    // Loop over the clusters in the event
    for(int bCluster=0; bCluster < treeVars_.nClusters; bCluster++)
    {
      for(int cryInBC=0; cryInBC < treeVars_.nXtalsInCluster[bCluster]; cryInBC++)
      {
        int hashedIndex = treeVars_.xtalInBCHashedIndex[bCluster][cryInBC];
        float cryTime = treeVars_.xtalInBCTime[bCluster][cryInBC];
        float cryTimeError = treeVars_.xtalInBCTimeErr[bCluster][cryInBC];
        float cryAmp = treeVars_.xtalInBCAmplitudeADC[bCluster][cryInBC];
        float crySwissCrossNoise = treeVars_.xtalInBCSwissCross[bCluster][cryInBC];

	int runNumber = treeVars_.runId;
	if (runNumber < runFirst)
	{
	  runFirst = runNumber;
	}
	if (runNumber > runLast)
	{
	  runLast = runNumber;
	}
	
        if(treeVars_.xtalInBCIEta[bCluster][0] != -999999)
        {
	  EBDetId det = EBDetId::unhashIndex(hashedIndex);
          if(det==EBDetId()) // make sure DetId is valid
	  {
            continue;
          }
          int ieta = det.ieta();
          int iphi = det.iphi();

          // RecHit cuts
          bool keepHit = cryAmp >= 26
            && crySwissCrossNoise < 0.95
	    && cryTime >= -5
	    && cryTime <= 5;
          if(!keepHit)
            continue;
	  hitCounterEBAll++;
          hitCounterPerCrystalEB[hashedIndex]++;

          if(cryTime==lastCryTimesEB[hashedIndex] && cryAmp==lastCryAmpsEB[hashedIndex] && crySwissCrossNoise == lastCrySwissCrossEB[hashedIndex] && cryTimeError==lastCryTimeErrorEB[hashedIndex])
          {
            duplicateCounterEBAll++;
          }
          occupancyEB->Fill(iphi,ieta);
          if(cryTime!=lastCryTimesEB[hashedIndex])
            occupancyEBNoDuplicates->Fill(iphi,ieta);

          if(cryTime==lastCryTimesEB[hashedIndex])
	  {
	     duplicatesEB->Fill(iphi,ieta);
	  }
          //Store information on previous hit for duplicate checking
          lastCryTimesEB[hashedIndex] = cryTime;
	  lastCryAmpsEB[hashedIndex] = cryAmp;
	  lastCrySwissCrossEB[hashedIndex] = crySwissCrossNoise;
	  lastCryTimeErrorEB[hashedIndex] = cryTimeError;
        }
	else
       	{

	  EEDetId det = EEDetId::unhashIndex(hashedIndex);
	  if(det==EEDetId()) // make sure DetId is valid
	    continue; 

	  int ix = det.ix();
	  int iy = det.iy();
	  int zside = det.zside();

          bool keepHit = cryAmp >= 47 
	    && cryTime >= -7
	    && cryTime <= 7;
          if(!keepHit)
            continue;
          hitCounterPerCrystalEE[hashedIndex]++;
	  hitCounterEEAll++;
          if(cryTime==lastCryTimesEE[hashedIndex] && cryAmp==lastCryAmpsEE[hashedIndex] && cryTimeError==lastCryTimeErrorEE[hashedIndex])
          {
            duplicateCounterEEAll++;
          }
	  if(zside == -1)
            occupancyEEM->Fill(ix,iy);
	  if(zside == 1)
            occupancyEEP->Fill(ix,iy);

          //Store information on previous hit for duplicate checking
          lastCryTimesEE[hashedIndex] = cryTime;
	  lastCryAmpsEE[hashedIndex] = cryAmp;
	  lastCryTimeErrorEE[hashedIndex] = cryTimeError;
        }
      }
    }
  }
  if(runFirst > runLast)
  {
    std::cout << "No runs in the tree!" << std::endl;
  }
  std::cout << "Run Range: " << runFirst << "-" << runLast << std::endl;
  std::cout << "nEntries: " << nEntries << std::endl;  
  std::cout << "hitCounterEBAll: " << hitCounterEBAll << " duplicateCounterEBAll: " << duplicateCounterEBAll << std::endl;
  std::cout << "hitCounterEEAll: " << hitCounterEEAll << " duplicateCounterEEAll: " << duplicateCounterEEAll << std::endl;
  int nXtalsUnder10HitsEB = 0;
  for(int i=0; i<EBDetId::kSizeForDenseIndexing;++i)
    if(hitCounterPerCrystalEB[i] <= 10)
    {
      nXtalsUnder10HitsEB++; 
    } 
  int nXtalsUnder10HitsEE = 0;
  for(int i=0; i<EEDetId::kSizeForDenseIndexing;++i)
    if(hitCounterPerCrystalEE[i] <= 10)
    {
      nXtalsUnder10HitsEE++; 
    } 

  int nXtalsWithZeroHitsEB = 0;
  for(int i=0; i<EBDetId::kSizeForDenseIndexing;++i)
    if(hitCounterPerCrystalEB[i] == 0)
    {
      nXtalsWithZeroHitsEB++; 
    } 
  int nXtalsWithZeroHitsEE = 0;
  for(int i=0; i<EEDetId::kSizeForDenseIndexing;++i)
    if(hitCounterPerCrystalEE[i] == 0)
    {
      nXtalsWithZeroHitsEE++; 
    } 

  std::cout << "nXtalsWithZeroHitsEB: " << nXtalsWithZeroHitsEB << std::endl;
  std::cout << "nXtalsUnder10HitsEB: " << nXtalsUnder10HitsEB << std::endl;
  std::cout << "nXtalsWithZeroHitsEE: " << nXtalsWithZeroHitsEE << std::endl;
  std::cout << "nXtalsUnder10HitsEE: " << nXtalsUnder10HitsEE << std::endl;
  
  canvas->cd();
  occupancyEB->Draw("colz");
  canvas->Print("occupancyEB.C");

  occupancyEEM->Draw("colz");
  canvas->Print("occupancyEEM.C");

  occupancyEEP->Draw("colz");
  canvas->Print("occupancyEEP.C");

  occupancyEBNoDuplicates->Draw("colz");
  canvas->Print("occupancyEBNoDuplicates.C");
  
  duplicatesEB->Draw("colz");
  canvas->Print("duplicatesEB.C");
}
