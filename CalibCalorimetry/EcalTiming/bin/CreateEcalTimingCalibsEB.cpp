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
const float sigmaNoiseEB = 1.06;  // ADC ; using total single-sample noise
TF1* timeCorrectionEB_; 

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
  // Usage: CreateEcalTimingCalibsEB fileWithTree options...
  //

  using namespace std;
  // Ao dependent timing corrections
  // By the definition of these corrections, the timing should be zero for the hits in
  // Module 1 or Low eta EE within the valid A/sigma ranges.
  // Earlier data will have positive time due to the gradual timing shifts in the positive direction.
  timeCorrectionEB_ = new TF1("timeCorrectionEB_","pol4(0)",0,1.2);
  //coefficients obtained in the interval (0, 1.5) from Module 1 of run 144011 EB data; 
  timeCorrectionEB_->SetParameters(0.0399144,-1.32993,2.00013,-1.51769,0.407406);
  //coefficients obtained in the interval (-0.5, 2.0)
  //timeCorrectionEB_->SetParameters(0.0544539,-1.51924,2.57379,-2.11848,0.606632);

  // For selection cuts
  string inBxs, inOrbits, inTrig, inTTrig, inLumi, inRuns;
  float avgTimeMin, avgTimeMax;
  float minAmpEB, minAmpEE;
  float maxSwissCrossNoise;  // EB only, no spikes seen in EE
  float maxHitTimeEB, minHitTimeEB;
  // init to sensible defaults
  avgTimeMin = -1; // in ns
  avgTimeMax = 1; // in ns
  minAmpEB = 5; // GeV
  minAmpEE = 5; // GeV
  maxHitTimeEB = 15; // ns
  minHitTimeEB = -15; // ns
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
    if (argv[i] == std::string("-swisskmax") && argc>i+1) maxSwissCrossNoise = atof(argv[i+1]);
    if (argv[i] == std::string("-avgtimemin") && argc>i+1) avgTimeMin = atof(argv[i+1]);
    if (argv[i] == std::string("-avgtimemax") && argc>i+1) avgTimeMax = atof(argv[i+1]);
    if (argv[i] == std::string("-ebhittimemax") && argc>i+1) maxHitTimeEB = atof(argv[i+1]);
    if (argv[i] == std::string("-ebhittimemin") && argc>i+1) minHitTimeEB = atof(argv[i+1]);
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
    << " maxSwissCrossNoise (EB): " << maxSwissCrossNoise
    << " maxHitTimeEB: " << maxHitTimeEB << " minHitTimeEB: " << minHitTimeEB                                                                   
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
  string fileNameBeg = "timingCalibsEB";
  string rootFilename = fileNameBeg+".root";
  TFile* outfile = new TFile(rootFilename.c_str(),"RECREATE");
  outfile->cd();

  TH1F* calibHistEB = new TH1F("timingCalibsEB","timingCalibs EB [ns]",2000,-100,100);
  TH1F* calibErrorHistEB = new TH1F("calibErrorEB","timingCalibError EB [ns]",500,0,5);
  calibHistEB->Sumw2();
  calibErrorHistEB->Sumw2();

  TH2F* calibsVsErrors = new TH2F("timingCalibsAndErrors","TimingCalibs vs. errors [ns]",500,0,5,100,0,10);
  calibsVsErrors->Sumw2();

  TH1F* expectedStatPresHistEB = new TH1F("expectedStatPresEB","Avg. expected statistical precision EB [ns], all crys",200,0,2);
  TH2F* expectedStatPresVsObservedMeanErrHistEB = new TH2F("expectedStatPresVsObsEB","Expected stat. pres. vs. obs. error on mean each event EB [ns]",200,0,2,200,0,2);
  TH1F* expectedStatPresEachEventHistEB = new TH1F("expectedStatPresSingleEventEB","Expected stat. pres. each event EB [ns]",200,0,2);
  
  TH2F* errorOnMeanVsNumEvtsHist = new TH2F("errorOnMeanVsNumEvts","Error_on_mean vs. number of events",50,0,50,200,0,2);
  errorOnMeanVsNumEvtsHist->Sumw2();

  TH1F* hitsPerCryHistEB = new TH1F("hitsPerCryEB","Hits used in each crystal;hashedIndex",61200,0,61200);
  TH2F* hitsPerCryMapEB = new TH2F("hitsPerCryMapEB","Hits used in each crystal;i#phi;i#eta",360,1.,361.,172,-86,86);

  TProfile2D* ampProfileMapEB = new TProfile2D("ampProfileMapEB","amp profile map [ADC];i#phi;i#eta",360,1.,361.,172,-86,86);
  TProfile* ampProfileEB = new TProfile("ampProfileEB","Average amplitude in cry [ADC];hashedIndex",61200,0,61200);

  TH1F* sigmaHistEB = new TH1F("sigmaCalibsEB"," Sigma of calib distributions EB [ns]",100,0,1);

  //=============Special Bins for TT and Modules borders=============================
  double ttEtaBins[36] = {-85, -80, -75, -70, -65, -60, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86 };
  // double modEtaBins[10]={-85, -65, -45, -25, 0, 1, 26, 46, 66, 86};
  double ttPhiBins[73];
  double modPhiBins[19];
  double timingBins[79];
  double highEBins[11];
  for (int i = 0; i < 79; ++i)
  {
    timingBins[i]=-7.+double(i)*14./78.;
    if (i<73)
    {
      ttPhiBins[i]=1+5*i;
      if ( i < 19) 
      {
        modPhiBins[i]=1+20*i;
        if (i < 11)
        {
          highEBins[i]=10.+double(i)*20.;
        }
      }
    }

  } 
  TH2F* calibMapEB = new TH2F("calibMapEB","time calib map EB [ns];i#phi;i#eta",360,1.,361.,172,-86,86);
  calibMapEB->Sumw2();
  TH2F* sigmaMapEB = new TH2F("sigmaMapEB","Sigma of time calib map EB [ns];i#phi;i#eta",360,1.,361.,172,-86,86);
  TH2F* calibErrorMapEB = new TH2F("calibErrorMapEB","Error of time calib map EB [ns];i#phi;i#eta",360,1.,361.,172,-86,86);
  TProfile2D* calibTTMapEB = new TProfile2D("calibTTMapEB","time calib map EB (TT) [ns];i#phi;i#eta",360/5,ttPhiBins,35, ttEtaBins);

  TDirectory* cryDirEB = gDirectory->mkdir("crystalTimingHistsEB");
  cryDirEB->cd();
  TH1C* cryTimingHistsEB[61200];
  EBDetId det;
  for(int hi=0; hi < 61200; ++hi)
  {
    det = EBDetId::unhashIndex(hi);
    if(det==EBDetId())
      continue;
    string histname = "EB_cryTiming_ieta";
    histname+=intToString(det.ieta());
    histname+="_iphi";
    histname+=intToString(det.iphi());
    cryTimingHistsEB[hi] = new TH1C(histname.c_str(),histname.c_str(),660,-33,33);
    cryTimingHistsEB[hi]->Sumw2();
  }
  outfile->cd();
  
  cout << "Making calibs...";

  CrystalCalibration* ebCryCalibs[61200];
  //XXX: Making calibs with weighted/unweighted mean
  for(int i=0; i < 61200; ++i)
    ebCryCalibs[i] = new CrystalCalibration(); //use weighted mean!
    //ebCryCalibs[i] = new CrystalCalibration(false); //don't use weighted mean!

  // Loop over the TTree
  int numEventsUsed = 0;
  int nEntries = chain->GetEntries();
  cout << "Begin loop over TTree." << endl;

  for(int entry = 0; entry < nEntries; ++entry)
  {
    chain->GetEntry(entry);

    // Loop once to calculate average event time
    float sumTime = 0;
    int numCrysEB = 0;
    for(int bCluster=0; bCluster < treeVars_.nClusters; bCluster++)
    {
      if(treeVars_.xtalInBCIEta[bCluster][0] == -999999) continue; // skip EE clusters
      for(int cryInBC=0; cryInBC < treeVars_.nXtalsInCluster[bCluster]; cryInBC++)
      {
        sumTime += treeVars_.xtalInBCTime[bCluster][cryInBC];
        numCrysEB++;
      }
    }
    //debug
    //cout << "Number of EB crys in event: " << numEBcrys << endl;

    //XXX: Event cuts
    if(sumTime/numCrysEB > avgTimeMax || sumTime/numCrysEB < avgTimeMin)
    {
      //cout << "Average event time: " << sumTime/numCrysEB  << " so event rejected." << endl;
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

    // Loop over the EB crys and fill the map
    for(int bCluster=0; bCluster < treeVars_.nClusters; bCluster++)
    {
      if(treeVars_.xtalInBCIEta[bCluster][0] == -999999) continue; // skip EE clusters
      for(int cryInBC=0; cryInBC < treeVars_.nXtalsInCluster[bCluster]; cryInBC++)
      {
        int hashedIndex = treeVars_.xtalInBCHashedIndex[bCluster][cryInBC];
        float cryTime = treeVars_.xtalInBCTime[bCluster][cryInBC];
        float cryTimeError = treeVars_.xtalInBCTimeErr[bCluster][cryInBC];
        float cryAmp = treeVars_.xtalInBCAmplitudeADC[bCluster][cryInBC];
        //float cryEt = treeVars_.cryETEB_[cryIndex]; // not in the tree
        //SIC FEB 14,16 2011 - removed E/E9
        //                - spike cleaning done higher up
        float crySwissCrossNoise = treeVars_.xtalInBCSwissCross[bCluster][cryInBC];
	float Ao = cryAmp/sigmaNoiseEB;
	float AoLog = log10(Ao/25);

        EBDetId det = EBDetId::unhashIndex(hashedIndex);
        if(det==EBDetId()) // make sure DetId is valid
          continue;

        int ieta = det.ieta();
        int iphi = det.iphi();

        //XXX: RecHit cuts
        bool keepHit = cryAmp >= minAmpEB
          && crySwissCrossNoise < maxSwissCrossNoise
          && cryTime > minHitTimeEB
          && cryTime < maxHitTimeEB
          && AoLog > 0
          && AoLog < 1.2;
        if(!keepHit)
          continue;

        //cout << "STUPID DEBUG: " << hashedIndex << " cryTime: " << cryTime << " cryTimeError: " << cryTimeError << " cryAmp: " << cryAmp << endl;

        // Timing correction to take out the energy dependence if log10(ampliOverSigOfThis/25)
        // is between 0 and 1.2 (about 1 and 13 GeV)
	// amplitude dependent timing corrections
        float timing = cryTime - timeCorrectionEB_->Eval(AoLog);
        //FIXME
        cryTimeError = 1;
        ebCryCalibs[hashedIndex]->insertEvent(cryAmp,timing,cryTimeError,false);
        //SIC Use when we don't have time_error available
        //ebCryCalibs[hashedIndex]->insertEvent(cryAmp,cryTime,35/(cryAmp/1.2),false);
        ampProfileEB->Fill(hashedIndex,cryAmp);
        ampProfileMapEB->Fill(iphi,ieta,cryAmp);
        //if(cryTime > 33 || cryTime < -33)
        //  cout << "Crystal: " << det << " event time is over/underflow: " << cryTime << endl;
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
  for(int hashedIndex=0; hashedIndex < 61200; ++hashedIndex)
  {
    EBDetId det = EBDetId::unhashIndex(hashedIndex);
    if(det==EBDetId())
      continue;
    CrystalCalibration cryCalib = *(ebCryCalibs[hashedIndex]);
    int ieta = det.ieta();
    int iphi = det.iphi();

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
      float weight = 1/((timeItr->sigmaTime)*(timeItr->sigmaTime));
      cryTimingHistsEB[hashedIndex]->Fill(timeItr->time,weight);
    }
    cryDirEB->cd();
    cryTimingHistsEB[hashedIndex]->Write();
    outfile->cd();
    hitsPerCryHistEB->SetBinContent(hashedIndex+1,cryCalib.timingEvents.size());
    hitsPerCryMapEB->Fill(iphi,ieta,cryCalib.timingEvents.size());
    
    // Make timing calibs
    double p1 = cryCalib.mean;
    double p1err = cryCalib.meanE;
    //cout << "cry ieta: " << ieta << " cry iphi: " << iphi << " p1: " << p1 << " p1err: " << p1err << endl;
    if(cryCalib.timingEvents.size() < 10)
    {
      fileStreamProb << "Cry (only " << cryCalib.timingEvents.size() << " events) was calibrated to avg: " << ieta <<", " << iphi << ", hash: "
                                                                        << hashedIndex
                                                                        << "\t Calib: " << p1 << "\t Error: " << p1err << std::endl;
      hashesToCalibrateToAvg.push_back(hashedIndex);
      continue;
    }
    // Make it so we can add calib to reco time
    p1*=-1;
    if(p1err < 0.5 && p1err > 0)
    {
      fileStream << "EB\t" << hashedIndex << "\t" << p1 << "\t\t" << p1err << endl;
      calibHistEB->Fill(p1);
      //calibMapEEMFlip->Fill(y-85,x+1,p1);
      calibMapEB->Fill(iphi,ieta,p1);
      calibTTMapEB->Fill(iphi,ieta,p1);
      //calibMapEEMPhase->Fill(x+1,y-85,p1/25-floor(p1/25));
      //errorOnMeanVsNumEvtsHist->Fill(times.size(),p1err);
      cryCalibAvg+=p1;
      ++numCrysCalibrated;
      
      //Store in timeCalibration container
      EcalTimeCalibConstant tcConstant = p1;
      EcalTimeCalibError tcError = p1err;
      uint32_t rawId = EBDetId::unhashIndex(hashedIndex);
      timeCalibConstants[rawId] = tcConstant;
      timeCalibErrors[rawId] = tcError;
    }
    else
    {
      //std::cout << "Cry: " << ieta <<", " << iphi << ", hash: " << itr->first
      //  << "\t Calib: " << p1 << "\t Error: " << p1err << std::endl;
      fileStreamProb << "Cry was calibrated to avg: " << ieta <<", " << iphi << ", hash: " << hashedIndex
        << "\t Calib: " << p1 << "\t Error: " << p1err << std::endl;
      hashesToCalibrateToAvg.push_back(hashedIndex);
    }
    //calibsVsErrorsEB->Fill(p1err, p1 > 0 ? p1 : -1*p1);
    calibErrorHistEB->Fill(p1err);
    calibErrorMapEB->Fill(iphi,ieta,p1err);
    sigmaHistEB->Fill(cryCalib.stdDev);
    sigmaMapEB->Fill(iphi,ieta,cryCalib.stdDev);
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
    uint32_t rawId = EBDetId::unhashIndex(*hashItr);
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
  string timeCalibFile = "EcalTimeCalibsEB.xml";
  string timeCalibErrFile = "EcalTimeCalibErrorsEB.xml";
  // Hack to prevent seg fault
  EcalTimeCalibConstant tcConstant = 0;
  EcalTimeCalibError tcError = 0;
  uint32_t rawId = EEDetId::unhashIndex(0);
  timeCalibConstants[rawId] = tcConstant;
  timeCalibErrors[rawId] = tcError;
  // End hack
  EcalTimeCalibConstantsXMLTranslator::writeXML(timeCalibFile,header,timeCalibConstants);
  EcalTimeCalibErrorsXMLTranslator::writeXML(timeCalibErrFile,header,timeCalibErrors);

  cout << "Writing histograms." << endl;
  outfile->cd();
  calibHistEB->SetXTitle("timingCalib [ns]");
  calibHistEB->Write();
  sigmaHistEB->Write();
  calibErrorHistEB->SetXTitle("uncertainty on mean [ns]");
  calibErrorHistEB->Write();
  //eventsEBHist->Write();

  //can->Print("calibs1D.png");
  //cout << "Writing calibVsErrors" << endl;
  //calibsVsErrors->SetYTitle("AbsCalibConst");
  //calibsVsErrors->SetXTitle("calibConstError");
  //calibsVsErrors->Write();

  //cout << "Writing calibErrorHists" << endl;
  //calibErrorHistEB->Write();

  //cout << "Writing calib maps" << endl;
  sigmaMapEB->Write();
  calibMapEB->Write();
  calibErrorMapEB->Write();
  calibTTMapEB->Write();
  //calibMapEBFlip->SetXTitle("ieta");
  //calibMapEBFlip->SetYTitle("iphi");
  //calibMapEBFlip->Write();
  //calibMapEBPhase->SetXTitle("iphi");
  //calibMapEBPhase->SetYTitle("ieta");
  //calibMapEBPhase->Write();
  
  //Move empty bins out of the way
  //int nxbins = calibMapEEM->GetNbinsX();
  //int nybins = calibMapEEM->GetNbinsY();
  //for(int i=0;i<=(nxbins+2)*(nybins+2); ++i)
  //{
  //  double binentsM = calibMapEEM->GetBinContent(i);
  //  if(binentsM==0)
  //  {
  //    calibMapEEM->SetBinContent(i,-1000);
  //  }
  //  double binentsP = calibMapEEP->GetBinContent(i);
  //  if(binentsP==0)
  //  {
  //    calibMapEEP->SetBinContent(i,-1000);
  //  }
  //}
  //calibMapEEM->SetXTitle("ix");
  //calibMapEEM->SetYTitle("iy");
  //calibMapEEM->Write();
  //calibMapEEP->SetXTitle("ix");
  //calibMapEEP->SetYTitle("iy");
  //calibMapEEP->Write();

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

  expectedStatPresHistEB->Write();
  expectedStatPresVsObservedMeanErrHistEB->Write();
  expectedStatPresEachEventHistEB->Write();
  //ampEachEventHist->Write();
  //numPointsErasedHist->Write();

  //calibMapEtaAvgEB->SetXTitle("i#phi");
  //calibMapEtaAvgEB->SetYTitle("i#eta");
  //calibMapEtaAvgEB->Write();
  //calibHistEtaAvgEB->Write();
  
  hitsPerCryHistEB->Write();
  hitsPerCryMapEB->Write();
  ampProfileMapEB->Write();
  ampProfileEB->Write();
  
  //cout << "All done!  Close input." << endl;
  //f->Close();
  //cout << "Close output and quit!" << endl;
  outfile->Close();
  cout << "done." << endl;
}


