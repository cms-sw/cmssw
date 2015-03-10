#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TFile.h"
#include "TProfile2D.h"
#include "TStyle.h"
#include "TDirectory.h"
#include "TError.h"
#include "TTree.h"
#include "TGraphErrors.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"
#include "CondTools/Ecal/interface/EcalTimeCalibConstantsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalTimeCalibErrorsXMLTranslator.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "CreateEcalTimingCalibs.h"

#define ADCtoGeVEB   0.039
#define ADCtoGeVEE   0.063

// Globals
EcalTimeTreeContent treeVars_;
std::vector<std::string> listOfFiles_;
const float sigmaNoiseEE = 2.10;  // ADC ; using total single-sample noise
TF1* timeCorrectionEE_;

// ****************************************************************
std::string intToString(int num)
{
  using namespace std;
  ostringstream myStream;
  myStream << num << flush;
  return(myStream.str()); //returns the string form of the stringstream object
}

// ****************************************************************
//XXX: main
int main(int argc, char* argv[])
{
  //
  // **** May 20 2010 update ****
  // Usage: CreateEcalTimingCalibsEE fileWithTree options...
  //

  using namespace std;
  // Ao dependent timing corrections
  timeCorrectionEE_ = new TF1("timeCorrectionEE_","pol4(0)",0,1.2);
  //coefficients obtained in the interval (0, 1.5) from Low eta region < 2.2, run 144011
  timeCorrectionEE_->SetParameters(-0.461192,0.0876435,-0.234752,0.143774,-0.051990);

  // For selection cuts
  string inBxs, inOrbits, inTrig, inTTrig, inLumi, inRuns;
  float avgTimeMin, avgTimeMax;
  float minAmpEB, minAmpEE;
  float maxE1E9, maxSwissCrossNoise;  // EB only, no spikes seen in EE
  float maxHitTimeEE, minHitTimeEE;
  // init to sensible defaults
  avgTimeMin = -1; // in ns
  avgTimeMax = 1; // in ns
  minAmpEB = 5; // GeV
  minAmpEE = 5; // GeV
  maxHitTimeEE = 15; // ns
  minHitTimeEE = -15; // ns
  maxE1E9 = 0.95; // EB only
  maxSwissCrossNoise = 0.95; // EB only
  inBxs = "-1";
  inOrbits = "-1";
  inTrig = "-1";
  inTTrig = "-1";
  inLumi = "-1";
  inRuns = "-1";

  char* infile = argv[1];
  if (!infile)
  {
    cout << " No input file specified !" << endl;
    return -1;
  }

  //TODO: Replace this with the parseArguments function from the pi0 binary
  std::string stringGenericOption    = "--";
  for (int i=1 ; i<argc ; i++) {
    if (argv[i] == std::string("-bxs") && argc>i+1) inBxs = std::string(argv[i+1]);
    if (argv[i] == std::string("-orbits") && argc>i+1) inOrbits = std::string(argv[i+1]);
    if (argv[i] == std::string("-trig") && argc>i+1) inTrig = std::string(argv[i+1]);
    if (argv[i] == std::string("-ttrig") && argc>i+1) inTTrig = std::string(argv[i+1]);
    if (argv[i] == std::string("-lumi") && argc>i+1) inLumi = std::string(argv[i+1]);
    if (argv[i] == std::string("-runs") && argc>i+1) inRuns = std::string(argv[i+1]);
    if (argv[i] == std::string("-ebampmin") && argc>i+1) minAmpEB = atof(argv[i+1]);
    if (argv[i] == std::string("-eeampmin") && argc>i+1) minAmpEE = atof(argv[i+1]);
    if (argv[i] == std::string("-e1e9max") && argc>i+1) maxE1E9 = atof(argv[i+1]);
    if (argv[i] == std::string("-swisskmax") && argc>i+1) maxSwissCrossNoise = atof(argv[i+1]);
    if (argv[i] == std::string("-avgtimemin") && argc>i+1) avgTimeMin = atof(argv[i+1]);
    if (argv[i] == std::string("-avgtimemax") && argc>i+1) avgTimeMax = atof(argv[i+1]);
    if (argv[i] == std::string("-eehittimemax") && argc>i+1) maxHitTimeEE = atof(argv[i+1]);
    if (argv[i] == std::string("-eehittimemin") && argc>i+1) minHitTimeEE = atof(argv[i+1]);
    // handle here the case of multiple arguments for input files
    if (argv[i] == std::string("--i"))// && argc>i+1)
    {
      for (int u=i+1; u<argc; u++)
      {
        if ( 0==std::string(argv[u]).find( stringGenericOption ) )
        {
          if ( 0==listOfFiles_.size())  {std::cout << "no input files listed" << std::cout;}
          else  {std::cout << "no more files listed, found: " << argv[u] << std::cout;}
          break;
        }
        else
        {
          listOfFiles_.push_back(argv[u]);
          i++;
        }
      }// loop on arguments following --i
      continue;
    }//end 'if input files'

  }
  // Open the input files
  if (listOfFiles_.size()==0){
    std::cout << "\tno input file found" << std::endl;
    return(1);
  }
  else{
    std::cout << "\tfound " << listOfFiles_.size() << " input files: " << std::endl;
    for(std::vector<std::string>::const_iterator  file_itr=listOfFiles_.begin(); file_itr!=listOfFiles_.end(); file_itr++){
      std::cout << "\t" << (*file_itr) << std::endl;
    }
  }
  // Tree construction
  TChain* chain = new TChain ("EcalTimeAnalysis") ;
  std::vector<std::string>::const_iterator file_itr;
  for(file_itr=listOfFiles_.begin(); file_itr!=listOfFiles_.end(); file_itr++){
    chain->Add( (*file_itr).c_str() );
  }


  cout << "Running with options: "
    << "avgTimeMin: " << avgTimeMin << " avgTimeMax: " << avgTimeMax
    << " minAmpEB: " << minAmpEB << " minAmpEE: " << minAmpEE
    << " maxHitTimeEE: " << maxHitTimeEE << " minHitTimeEE: " << minHitTimeEE
    << " inTrig: " << inTrig << " inTTrig: " << inTTrig << " inLumi: " << inLumi 
    << " inBxs: " << inBxs << " inRuns: " << inRuns << " inOrbits: " << inOrbits
    << endl;

  // Ignore warnings
  gErrorIgnoreLevel = 2001;

  setBranchAddresses(chain,treeVars_);

  // Generate all the vectors for skipping selections
  std::vector<std::vector<double> > bxIncludeVector;
  std::vector<std::vector<double> > bxExcludeVector;
  std::vector<std::vector<double> > orbitIncludeVector;
  std::vector<std::vector<double> > orbitExcludeVector;
  std::vector<std::vector<double> > trigIncludeVector;
  std::vector<std::vector<double> > trigExcludeVector;
  std::vector<std::vector<double> > ttrigIncludeVector;
  std::vector<std::vector<double> > ttrigExcludeVector;
  std::vector<std::vector<double> > lumiIncludeVector;
  std::vector<std::vector<double> > lumiExcludeVector;
  std::vector<std::vector<double> > runIncludeVector;
  std::vector<std::vector<double> > runExcludeVector;
  //recall: string inBxs, inOrbits, inTrig, inTTrig, inLumi, inRuns;
  genIncludeExcludeVectors(inBxs,bxIncludeVector,bxExcludeVector);
  genIncludeExcludeVectors(inOrbits,orbitIncludeVector,orbitExcludeVector);
  genIncludeExcludeVectors(inTrig,trigIncludeVector,trigExcludeVector);
  genIncludeExcludeVectors(inTTrig,ttrigIncludeVector,ttrigExcludeVector);
  genIncludeExcludeVectors(inLumi,lumiIncludeVector,lumiExcludeVector);
  genIncludeExcludeVectors(inRuns,runIncludeVector,runExcludeVector);

  // Open output file and book hists
  string fileNameBeg = "timingCalibsEE";
  string rootFilename = fileNameBeg+".root";
  TFile* outfile = new TFile(rootFilename.c_str(),"RECREATE");
  outfile->cd();

  TH1F* calibHistEE = new TH1F("timingCalibsEE","timingCalibs EE [ns]",2000,-100,100);
  TH1F* calibErrorHistEE = new TH1F("timingCalibErrorEE","timingCalibError EE [ns]",500,0,5);
  calibHistEE->Sumw2();
  calibErrorHistEE->Sumw2();

  TH2F* calibsVsErrors = new TH2F("timingCalibsAndErrors","TimingCalibs vs. errors [ns]",500,0,5,100,0,10);
  calibsVsErrors->Sumw2();

  //TH2F* calibMapEE = new TH2F("calibMapEE","time calib map EE",360,1,361,170,-86,86);
  //TH2F* calibMapEEFlip = new TH2F("calibMapEEFlip","time calib map EE",170,-86,86,360,1,361);
  //TH2F* calibMapEEPhase = new TH2F("calibMapEEPhase","time calib map EE (phase of Tmax)",360,1,361,170,-86,86);
  
  //TH2F* calibMapEtaAvgEE = new TH2F("calibMapEtaAvgEE","time calibs raw eta avg map EE",360,1,361,170,-86,86);
  //TH1F* calibHistEtaAvgEE = new TH1F("timingCalibsEtaAvgEE","EtaAvgTimingCalibs EE [ns]",2000,-100,100);

  TH2F* hitsPerCryMapEEM = new TH2F("hitsPerCryMapEEM","Hits per cry EEM;ix;iy",100,1,101,100,1,101);
  TH2F* hitsPerCryMapEEP = new TH2F("hitsPerCryMapEEP","Hits per cry EEP;ix;iy",100,1,101,100,1,101);
  TH1F* hitsPerCryHistEEM = new TH1F("hitsPerCryHistEEM","Hits per cry EEM;hashedIndex",14648,0,14648);
  TH1F* hitsPerCryHistEEP = new TH1F("hitsPerCryHistEEP","Hits per cry EEP;hashedIndex",14648,0,14648);
  //TH1C* eventsEEMHist = new TH1C("numEventsEEM","Number of events, EEM",100,0,100);
  //TH1C* eventsEEPHist = new TH1C("numEventsEEP","Number of events, EEP",100,0,100);
  TProfile* ampProfileEEM = new TProfile("ampProfileEEM","Amp. profile EEM;hashedIndex",14648,0,14648);
  TProfile* ampProfileEEP = new TProfile("ampProfileEEP","Amp. profile EEP;hashedIndex",14648,0,14648);
  TProfile2D* ampProfileMapEEP = new TProfile2D("ampProfileMapEEP","Amp. profile EEP;ix;iy",100,1,101,100,1,101);
  TProfile2D* ampProfileMapEEM = new TProfile2D("ampProfileMapEEM","Amp. profile EEM;ix;iy",100,1,101,100,1,101);

  //TH1F* eventsEEHist = new TH1F("numEventsEE","Number of events, EE",100,0,100);
  //TH1F* calibSigmaHist = new TH1F("timingSpreadEE","Crystal timing spread [ns]",1000,-5,5);

  TH1F* sigmaHistEE = new TH1F("sigmaCalibsEE"," Sigma of calib distributions EE [ns]",100,0,1);

  //TH1F* chiSquaredEachEventHist = new TH1F("chi2eachEvent","Chi2 of each event",500,0,500);
  //TH2F* chiSquaredVsAmpEachEventHist = new TH2F("chi2VsAmpEachEvent","Chi2 vs. amplitude of each event",500,0,500,750,0,750);
  //TH2F* chiSquaredHighMap = new TH2F("chi2HighMap","Channels with event #Chi^{2} > 100",360,1,361,170,-86,86);
  //TH1F* chiSquaredTotalHist = new TH1F("chi2Total","Total chi2 of all events in each crystal",500,0,500);
  //TH1F* chiSquaredSingleOverTotalHist = new TH1F("chi2SingleOverTotal","Chi2 of each event over total chi2",100,0,1);
  //TH1F* ampEachEventHist = new TH1F("energyEachEvent","Energy of all events [GeV]",1000,0,10);
  //TH1F* numPointsErasedHist = new TH1F("numPointsErased","Number of points erased per crystal",25,0,25);
  
  //TProfile2D* myAmpProfile = (TProfile2D*)EBampProfile->Clone();
  //myAmpProfile->Write();
  TH1F* expectedStatPresHistEEM = new TH1F("expectedStatPresEEM","Avg. expected statistical precision EEM [ns], all crys",200,0,2);
  TH2F* expectedStatPresVsObservedMeanErrHistEEM = new TH2F("expectedStatPresVsObsEEM","Expected stat. pres. vs. obs. error on mean each event EEM [ns]",200,0,2,200,0,2);
  TH1F* expectedStatPresEachEventHistEEM = new TH1F("expectedStatPresSingleEventEEM","Expected stat. pres. each event EEM [ns]",200,0,2);
  TH1F* expectedStatPresHistEEP = new TH1F("expectedStatPresEEP","Avg. expected statistical precision EEP [ns], all crys",200,0,2);
  TH2F* expectedStatPresVsObservedMeanErrHistEEP = new TH2F("expectedStatPresVsObsEEP","Expected stat. pres. vs. obs. error on mean each event [ns] EEP",200,0,2,200,0,2);
  TH1F* expectedStatPresEachEventHistEEP = new TH1F("expectedStatPresSingleEventEEP","Expected stat. pres. each event EEP [ns]",200,0,2);
  
  TH2F* errorOnMeanVsNumEvtsHist = new TH2F("errorOnMeanVsNumEvts","Error_on_mean vs. number of events",50,0,50,200,0,2);
  errorOnMeanVsNumEvtsHist->Sumw2();
  
  TH1F* calibHistEEM = new TH1F("timingCalibsEEM","timingCalibs EEM [ns]",500,-25,25);
  TH1F* calibHistEEP = new TH1F("timingCalibsEEP","timingCalibs EEP [ns]",500,-25,25);
  TH1F* calibErrorHistEEM = new TH1F("calibErrorEEM","timingCalibError EEM [ns]",250,0,5);
  TH1F* calibErrorHistEEP = new TH1F("calibErrorEEP","timingCalibError EEP [ns]",250,0,5);
  calibHistEEM->Sumw2();
  calibHistEEP->Sumw2();
  calibErrorHistEEM->Sumw2();
  calibErrorHistEEP->Sumw2();

  TH2F* calibMapEEM = new TH2F("calibMapEEM","time calib map EEM",100,1,101,100,1,101);
  TH2F* calibMapEEP = new TH2F("calibMapEEP","time calib map EEP",100,1,101,100,1,101);
  calibMapEEM->Sumw2();
  calibMapEEP->Sumw2();

  TH2F* sigmaMapEEM = new TH2F("sigmaMapEEM","Sigma of time calib map EEM [ns];ix;iy",100,1.,101.,100,1,101);
  TH2F* sigmaMapEEP = new TH2F("sigmaMapEEP","Sigma of time calib map EEP [ns];ix;iy",100,1.,101.,100,1,101);
  TH2F* calibErrorMapEEM = new TH2F("calibErrorMapEEM","Error of time calib map EEM [ns];ix;iy",100,1.,101.,100,1,101);
  TH2F* calibErrorMapEEP = new TH2F("calibErrorMapEEP","Error of time calib map EEP [ns];ix;iy",100,1.,101.,100,1,101);

  TDirectory* cryDirEEP = gDirectory->mkdir("crystalTimingHistsEEP");
  cryDirEEP->cd();
  TH1C* cryTimingHistsEEP[100][100]; // [0][0] = ix 1, iy 1
  for(int x=0; x < 100; ++x)
  {
    for(int y=0; y < 100; ++y)
    {
      if(!EEDetId::validDetId(x+1,y+1,1))
        continue;
      string histname = "EEP_cryTiming_ix";
      histname+=intToString(x+1);
      histname+="_iy";
      histname+=intToString(y+1);
      cryTimingHistsEEP[x][y] = new TH1C(histname.c_str(),histname.c_str(),660,-33,33);
      cryTimingHistsEEP[x][y]->Sumw2();
    }
  }
  outfile->cd();
  TDirectory* cryDirEEM = gDirectory->mkdir("crystalTimingHistsEEM");
  cryDirEEM->cd();
  TH1C* cryTimingHistsEEM[100][100]; // [0][0] = ix 1, iy 1
  for(int x=0; x < 100; ++x)
  {
    for(int y=0; y < 100; ++y)
    {
      if(!EEDetId::validDetId(x+1,y+1,-1))
        continue;
      string histname = "EEM_cryTiming_ix";
      histname+=intToString(x+1);
      histname+="_iy";
      histname+=intToString(y+1);
      cryTimingHistsEEM[x][y] = new TH1C(histname.c_str(),histname.c_str(),660,-33,33);
      cryTimingHistsEEM[x][y]->Sumw2();
    }
  }
  outfile->cd();
  
  cout << "Making calibs..." << endl;

  CrystalCalibration* eeCryCalibs[14648];
  //XXX: Making calibs with weighted/unweighted mean
  for(int i=0; i < 14648; ++i)
    eeCryCalibs[i] = new CrystalCalibration(); //use weighted mean!
    //eeCryCalibs[i] = new CrystalCalibration(false); //don't use weighted mean!

  cout << "Looping over TTree...";

  // Loop over the TTree
  int numEventsUsed = 0;
  int nEntries = chain->GetEntries();
  cout << "Begin loop over TTree." << endl;

  for(int entry = 0; entry < nEntries; ++entry)
  {
    chain->GetEntry(entry);

    // Loop once to calculate average event time
    float sumTime = 0;
    int numCrysEE = 0;
    for(int bCluster=0; bCluster < treeVars_.nClusters; bCluster++)
    {
      if(treeVars_.xtalInBCIEta[bCluster][0] != -999999) continue; // skip EB clusters
      for(int cryInBC=0; cryInBC < treeVars_.nXtalsInCluster[bCluster]; cryInBC++)
      {
        sumTime += treeVars_.xtalInBCTime[bCluster][cryInBC];
        numCrysEE++;
      }
    }
    //debug
    //cout << "Number of EE crys in event: " << numEEcrys << endl;

    //XXX: Event cuts
    if(sumTime/numCrysEE > avgTimeMax || sumTime/numCrysEE < avgTimeMin)
    {
      //cout << "Average event time: " << sumTime/numCrysEE  << " so event rejected." << endl;
      continue;
    }
    // check BX, orbit, lumi, run, L1 tech/phys triggers
    bool keepEvent = includeEvent(treeVars_.bx,bxIncludeVector,bxExcludeVector)
      && includeEvent(treeVars_.orbit,orbitIncludeVector,orbitExcludeVector)
      && includeEvent(treeVars_.lumiSection,lumiIncludeVector,lumiExcludeVector)
      && includeEvent(treeVars_.runId,runIncludeVector,runExcludeVector)
      && includeEvent(treeVars_.l1ActiveTriggers,
          treeVars_.l1NActiveTriggers,trigIncludeVector,trigExcludeVector)
      && includeEvent(treeVars_.l1ActiveTechTriggers,
          treeVars_.l1NActiveTechTriggers,ttrigIncludeVector,ttrigExcludeVector);
    if(!keepEvent)
      continue;
      
    numEventsUsed++;

    // Loop over the EE crys and fill the map
    for(int bCluster=0; bCluster < treeVars_.nClusters; bCluster++)
    {
      if(treeVars_.xtalInBCIEta[bCluster][0] != -999999) continue; // skip EB clusters
      for(int cryInBC=0; cryInBC < treeVars_.nXtalsInCluster[bCluster]; cryInBC++)
      {
        int hashedIndex = treeVars_.xtalInBCHashedIndex[bCluster][cryInBC];
        float cryTime = treeVars_.xtalInBCTime[bCluster][cryInBC];
        float cryTimeError = treeVars_.xtalInBCTimeErr[bCluster][cryInBC];
        float cryAmp = treeVars_.xtalInBCAmplitudeADC[bCluster][cryInBC];
	float Ao = cryAmp/sigmaNoiseEE;
	float AoLog = log10(Ao/25);

        EEDetId det = EEDetId::unhashIndex(hashedIndex);
        if(det==EEDetId()) // make sure DetId is valid
          continue;

        int ix = det.ix();
        int iy = det.iy();

        //XXX: RecHit cuts
        bool keepHit = cryAmp >= minAmpEE
          && cryTime > minHitTimeEE
          && cryTime < maxHitTimeEE
	  && AoLog > 0
          && AoLog < 1.2;
        if(!keepHit)
          continue;

        //cout << "STUPID DEBUG: " << hashedIndex << " cryTime: " << cryTime << " cryTimeError: " << cryTimeError << " cryAmp: " << cryAmp << endl;

	// Timing correction to take out the energy dependence if log10(ampliOverSigOfThis/25)
        // is between 0 and 1.2 (about 1 and 13 GeV)
	// amplitude dependent timing corrections
        float timing = cryTime - timeCorrectionEE_->Eval(AoLog);
        //FIXME
        cryTimeError = 1;
        eeCryCalibs[hashedIndex]->insertEvent(cryAmp,timing,cryTimeError,false);

        //SIC Use when we don't have time_error available
        //eeCryCalibs[hashedIndex]->insertEvent(cryAmp,cryTime,35/(cryAmp/1.2),false);
        if(det.zside() < 0)
        {
          ampProfileEEM->Fill(hashedIndex,cryAmp);
          ampProfileMapEEM->Fill(ix,iy,cryAmp);
        }
        else
        {
          ampProfileEEP->Fill(hashedIndex,cryAmp);
          ampProfileMapEEP->Fill(ix,iy,cryAmp);
        }
      }
    }
  }

  //create output text file
  ofstream fileStream;
  string fileName = fileNameBeg+".calibs.txt";
  fileStream.open(fileName.c_str());
  if(!fileStream.good() || !fileStream.is_open())
  {
    cout << "Couldn't open text file." << endl;
    return -1;
  }
  //create problem channels text file
  ofstream fileStreamProb;
  string fileName2 = fileNameBeg+".problems.txt";
  fileStreamProb.open(fileName2.c_str());
  if(!fileStreamProb.good() || !fileStreamProb.is_open())
  {
    cout << "Couldn't open text file." << endl;
    return -1;
  }

  // Create calibration container objects
  EcalTimeCalibConstants timeCalibConstants;
  EcalTimeCalibErrors timeCalibErrors;

  cout << "Using " << numEventsUsed << " out of " << nEntries << " in the tree." << endl;
  cout << "Creating calibs..." << endl;
  float cryCalibAvg = 0;
  int numCrysCalibrated = 0;
  vector<int> hashesToCalibrateToAvg;
  //Loop over all the crys
  for(int hashedIndex=0; hashedIndex < 14648; ++hashedIndex)
  {
    EEDetId det = EEDetId::unhashIndex(hashedIndex);
    if(det==EEDetId())
      continue;
    CrystalCalibration cryCalib = *(eeCryCalibs[hashedIndex]);
    int x = det.ix();
    int y = det.iy();

    //chiSquaredTotalHist->Fill(cryCalib.totalChi2);
    //expectedStatPresHistEB->Fill(sqrt(1/expectedPresSumEB));
    //expectedStatPresVsObservedMeanErrHistEB->Fill(sigmaM,sqrt(1/expectedPresSumEB));

    //XXX: Filter events at default 0.5*meanE threshold
    cryCalib.filterOutliers();
    
    //numPointsErasedHist->Fill(numPointsErased);
    
    //Write cryTimingHists
    vector<TimingEvent> times = cryCalib.timingEvents;
    for(vector<TimingEvent>::const_iterator timeItr = times.begin();
        timeItr != times.end(); ++timeItr)
    {
      if(det.zside() < 0)
      {
        float weight = 1/((timeItr->sigmaTime)*(timeItr->sigmaTime));
        cryTimingHistsEEM[x-1][y-1]->Fill(timeItr->time,weight);
      }
      else
      {
        float weight = 1/((timeItr->sigmaTime)*(timeItr->sigmaTime));
        cryTimingHistsEEP[x-1][y-1]->Fill(timeItr->time,weight);
      }
    }
    if(det.zside() < 0)
    {
      cryDirEEM->cd();
      cryTimingHistsEEM[x-1][y-1]->Write();
    }
    else
    {
      cryDirEEP->cd();
      cryTimingHistsEEP[x-1][y-1]->Write();
    }
    outfile->cd();

    if(det.zside() < 0)
    {
      hitsPerCryHistEEM->SetBinContent(hashedIndex+1,cryCalib.timingEvents.size());
      hitsPerCryMapEEM->Fill(x,y,cryCalib.timingEvents.size());
    }
    else
    {
      hitsPerCryHistEEP->SetBinContent(hashedIndex+1,cryCalib.timingEvents.size());
      hitsPerCryMapEEP->Fill(x,y,cryCalib.timingEvents.size());
    }
    
    // Make timing calibs
    double p1 = cryCalib.mean;
    double p1err = cryCalib.meanE;
    //cout << "cry ieta: " << ieta << " cry iphi: " << iphi << " p1: " << p1 << " p1err: " << p1err << endl;
    if(cryCalib.timingEvents.size() < 10)
    {
      fileStreamProb << "Cry (only " << cryCalib.timingEvents.size() << " events) was calibrated to avg: " << det.zside() << ", "
                                                                                        << x <<", " << y << ", hash: "
                                                                        << hashedIndex
                                                                          << "\t Calib: " << p1 << "\t Error: " << p1err << std::endl;
      hashesToCalibrateToAvg.push_back(hashedIndex);
      continue;
    }
    // Make it so we can add calib to reco time
    p1*=-1;
    if(p1err < 0.5 && p1err > 0)
    {
      fileStream << "EE\t" << hashedIndex << "\t" << p1 << "\t\t" << p1err << endl;
      if(det.zside() < 0)
      {
        calibHistEEM->Fill(p1);
        //calibMapEEMFlip->Fill(y-85,x+1,p1);
        calibMapEEM->Fill(x,y,p1);
        //calibMapEEMPhase->Fill(x+1,y-85,p1/25-floor(p1/25));
        //errorOnMeanVsNumEvtsHist->Fill(times.size(),p1err);
      }
      else
      {
        calibHistEEP->Fill(p1);
        //calibMapEEPFlip->Fill(y-85,x+1,p1);
        calibMapEEP->Fill(x,y,p1);
        //calibMapEEPPhase->Fill(x+1,y-85,p1/25-floor(p1/25));
        //errorOnMeanVsNumEvtsHist->Fill(times.size(),p1err);
      }
      cryCalibAvg+=p1;
      ++numCrysCalibrated;
      //Store in timeCalibration container
      EcalTimeCalibConstant tcConstant = p1;
      EcalTimeCalibError tcError = p1err;
      uint32_t rawId = EEDetId::unhashIndex(hashedIndex);
      timeCalibConstants[rawId] = tcConstant;
      timeCalibErrors[rawId] = tcError;
    }
    else
    {
      //std::cout << "Cry: " << y <<", " << x << ", hash: " << itr->first
      //  << "\t Calib: " << p1 << "\t Error: " << p1err << std::endl;
      fileStreamProb << "Cry was calibrated to avg: " << x <<", " << y << ", hash: " << hashedIndex
        << "\t Calib: " << p1 << "\t Error: " << p1err << std::endl;
    }

    sigmaHistEE->Fill(cryCalib.stdDev);
    if(det.zside() < 0)
    {
      //calibsVsErrorsEEM->Fill(p1err, p1 > 0 ? p1 : -1*p1);
      calibErrorHistEEM->Fill(p1err);
      calibErrorMapEEM->Fill(x,y,p1err);
      sigmaMapEEM->Fill(x,y,cryCalib.stdDev);
    }
    else
    {
      //calibsVsErrorsEEP->Fill(p1err, p1 > 0 ? p1 : -1*p1);
      calibErrorHistEEP->Fill(p1err);
      calibErrorMapEEP->Fill(x,y,p1err);
      sigmaMapEEP->Fill(x,y,cryCalib.stdDev);
    }
  }
  
  fileStream.close();
  fileStreamProb.close();
  // Calc average
  if(numCrysCalibrated > 0)
    cryCalibAvg/=numCrysCalibrated;
  cryCalibAvg-= 2.0833; // Global phase shift
  // calibrate uncalibratable crys
  for(vector<int>::const_iterator hashItr = hashesToCalibrateToAvg.begin();
      hashItr != hashesToCalibrateToAvg.end(); ++hashItr)
  {
    //Store in timeCalibration container
    EcalTimeCalibConstant tcConstant = cryCalibAvg;
    EcalTimeCalibError tcError = 999;
    uint32_t rawId = EEDetId::unhashIndex(*hashItr);
    timeCalibConstants[rawId] = tcConstant;
    timeCalibErrors[rawId] = tcError;
  }

  //Write XML files
  cout << "Writing XML files." << endl;
  EcalCondHeader header;
  header.method_="testmethod";
  header.version_="testversion";
  header.datasource_="testdata";
  header.since_=123;
  header.tag_="testtag";
  header.date_="Mar 24 1973";
  string timeCalibFile = "EcalTimeCalibsEE.xml";
  string timeCalibErrFile = "EcalTimeCalibErrorsEE.xml";
  // Hack to prevent seg fault
  EcalTimeCalibConstant tcConstant = 0;
  EcalTimeCalibError tcError = 0;
  uint32_t rawId = EBDetId::unhashIndex(0);
  timeCalibConstants[rawId] = tcConstant;
  timeCalibErrors[rawId] = tcError;
  // End hack
  EcalTimeCalibConstantsXMLTranslator::writeXML(timeCalibFile,header,timeCalibConstants);
  EcalTimeCalibErrorsXMLTranslator::writeXML(timeCalibErrFile,header,timeCalibErrors);

  cout << "Writing histograms." << endl;
  outfile->cd();
  calibHistEEM->SetXTitle("timingCalib [ns]");
  calibHistEEM->Write();
  calibHistEEP->SetXTitle("timingCalib [ns]");
  calibHistEEP->Write();
  calibErrorHistEEP->SetXTitle("uncertainty on mean [ns]");
  calibErrorHistEEP->Write();
  calibErrorHistEEM->SetXTitle("uncertainty on mean [ns]");
  calibErrorHistEEM->Write();
  calibErrorMapEEP->Write();
  calibErrorMapEEM->Write();
  sigmaHistEE->Write();
  sigmaMapEEM->Write();
  sigmaMapEEP->Write();

  //can->Print("calibs1D.png");
  //cout << "Writing calibVsErrors" << endl;
  //calibsVsErrors->SetYTitle("AbsCalibConst");
  //calibsVsErrors->SetXTitle("calibConstError");
  //calibsVsErrors->Write();

  //Move empty bins out of the way
  int nxbins = calibMapEEM->GetNbinsX();
  int nybins = calibMapEEM->GetNbinsY();
  for(int i=0;i<=(nxbins+2)*(nybins+2); ++i)
  {
    double binentsM = calibMapEEM->GetBinContent(i);
    if(binentsM==0)
    {
      calibMapEEM->SetBinContent(i,-1000);
    }
    double binentsP = calibMapEEP->GetBinContent(i);
    if(binentsP==0)
    {
      calibMapEEP->SetBinContent(i,-1000);
    }
  }
  calibMapEEM->SetXTitle("ix");
  calibMapEEM->SetYTitle("iy");
  calibMapEEM->Write();
  calibMapEEP->SetXTitle("ix");
  calibMapEEP->SetYTitle("iy");
  calibMapEEP->Write();

  //calibSigmaHist->SetXTitle("#sigma_{cryTime} [ns]");
  //calibSigmaHist->Write();
  
  // Old hist, commented Jun 15 2009
  //avgAmpVsSigmaTHist->SetXTitle("#sigma_{cryTime} [ns]");
  //avgAmpVsSigmaTHist->SetYTitle("Avg. amp. [adc]");
  //avgAmpVsSigmaTHist->Write();
 
  //errorOnMeanVsNumEvtsHist->SetXTitle("Events");
  //errorOnMeanVsNumEvtsHist->SetYTitle("Error_on_mean [ns]");
  //TProfile* theProf = (TProfile*) errorOnMeanVsNumEvtsHist->ProfileX();
  //TF1* myFit = new TF1("myFit","[0]/sqrt(x)+[1]",0,50);
  //myFit->SetRange(0,50);
  ////theProf->Fit("myFit");
  //theProf->Write();
  //errorOnMeanVsNumEvtsHist->Write();
  //
  //chiSquaredEachEventHist->Write();
  //chiSquaredVsAmpEachEventHist->SetXTitle("amplitude [ADC]");
  //chiSquaredVsAmpEachEventHist->SetYTitle("#Chi^{2}");
  //chiSquaredVsAmpEachEventHist->Write();
  //chiSquaredHighMap->SetXTitle("iphi");
  //chiSquaredHighMap->SetYTitle("ieta");
  //chiSquaredHighMap->Write();
  //chiSquaredTotalHist->Write();
  //chiSquaredSingleOverTotalHist->Write();

  expectedStatPresHistEEM->Write();
  expectedStatPresVsObservedMeanErrHistEEM->Write();
  expectedStatPresEachEventHistEEM->Write();
  expectedStatPresHistEEP->Write();
  expectedStatPresVsObservedMeanErrHistEEP->Write();
  expectedStatPresEachEventHistEEP->Write();
  //ampEachEventHist->Write();
  //numPointsErasedHist->Write();
  hitsPerCryHistEEP->Write();
  hitsPerCryMapEEP->Write();
  hitsPerCryHistEEM->Write();
  hitsPerCryMapEEM->Write();
  ampProfileEEP->Write();
  ampProfileMapEEP->Write();
  ampProfileEEM->Write();
  ampProfileMapEEM->Write();
  
  //cout << "All done!  Close input." << endl;
  //f->Close();
  //cout << "Close output and quit!" << endl;
  outfile->Close();
  cout << "done." << endl;
}


