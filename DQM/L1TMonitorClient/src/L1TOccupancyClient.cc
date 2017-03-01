#include "DQM/L1TMonitorClient/interface/L1TOccupancyClient.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Histograms/interface/MEtoEDMFormat.h"
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <vector>
#include <TMath.h>
#include <limits.h>
#include <TFile.h>
#include <TDirectory.h>
#include <TProfile.h>

using namespace std;
using namespace edm;

//____________________________________________________________________________
// Function: L1TOccupancyClient
// Description: This is the constructor, basic variable initialization
// Inputs: 
// * const edm::ParameterSet& ps = Parameter for this analyzer
//____________________________________________________________________________
L1TOccupancyClient::L1TOccupancyClient(const edm::ParameterSet& ps){

  
  // Get parameters
  parameters_ = ps;
  verbose_    = ps.getParameter<bool>                      ("verbose");
  tests_      = ps.getParameter<std::vector<ParameterSet> >("testParams");

  if(verbose_){cout << "[L1TOccupancyClient:] Called constructor" << endl;}
}

//____________________________________________________________________________
// Function: ~L1TOccupancyClient
// Description: This is the destructor, basic variable deletion
//____________________________________________________________________________
L1TOccupancyClient::~L1TOccupancyClient(){
  if(verbose_){cout << "[L1TOccupancyClient:] Called destructor" << endl;}
}

//____________________________________________________________________________
// Function: beginRun
// Description: This is will be run at the begining of each run
// Inputs: 
// * const Run&        r       = Run information 
// * const EventSetup& context = Event Setup information
//____________________________________________________________________________
void L1TOccupancyClient::book(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter){

  hservice_ = new L1TOccupancyClientHistogramService(parameters_,ibooker,verbose_); 
  
  if(verbose_){
    cout << "[L1TOccupancyClient:] Called beginRun" << endl;

    // In verbose mode we will produce an extra output file with several tests
    file_ = TFile::Open("DQM_L1TOccupancyClient_Snapshots_LS.root","RECREATE");
  }
  
  ibooker.setCurrentFolder("L1T/L1TOccupancy/");
  //dbe_->setCurrentFolder("L1T/L1TOccupancy/Results");
  //dbe_->setCurrentFolder("L1T/L1TOccupancy/BadCellValues");
  //dbe_->setCurrentFolder("L1T/L1TOccupancy/Certification");
  
  // Loop over all tests in defined 
  for (vector<ParameterSet>::iterator it = tests_.begin(); it != tests_.end(); it++) {

    // If the test algorithm is XYSymmetry we create the necessary histograms 
    if((*it).getUntrackedParameter<string>("algoName","XYSymmetry")=="XYSymmetry") {

      // Getting Parameters for the test
      string       testName       = (*it).getParameter<string>         ("testName");
      ParameterSet algoParameters = (*it).getParameter<ParameterSet>   ("algoParams");
      string       histPath       = algoParameters.getParameter<string>("histPath");
      
      if(verbose_){
        cout << "[L1TOccupancyClient:] Monitored histogram path: " << histPath << endl;
      
	// Creating verbose file directory structure
        // test_name/test_name_Results, 
	// test_name/test_name_Histos
        // TDirectory *td  = file_->mkdir(testName.c_str()             ,testName.c_str());
        //FIXME: sub never used gcc361 warning
        //TDirectory *sub = td   ->mkdir((testName+"_Results").c_str(),string("_Results").c_str());

        //sub = td->mkdir((testName+"_Histos").c_str()      ,(testName+"_Histos").c_str());
        //sub = td->mkdir((testName+"_Histos_AllLS").c_str(),(testName+"_Histos_AllLS").c_str());
      }
      
      // Load histograms in service instance
      if(hservice_->loadHisto(igetter, testName,histPath)){

      
      
        // Mask channels specified in python file
        hservice_->setMaskedBins(testName,algoParameters.getParameter<vector<ParameterSet> >("maskedAreas")); 

      // Book MonitorElements
      // * Test results
      ibooker.setCurrentFolder("L1T/L1TOccupancy/Results");
      string          title = testName;
      MonitorElement* m     = ibooker.book2D(title.c_str(),hservice_->getDifferentialHistogram(testName));
      m->setTitle(title.c_str()); 
      m->Reset();
      meResults[title] = m; 

      // * Which cells are masked as bad
      ibooker.setCurrentFolder("L1T/L1TOccupancy/HistogramDiff");
      title = testName;
      m = ibooker.book2D(title.c_str(),hservice_->getDifferentialHistogram(testName));
      m->Reset();
      m->setTitle(title.c_str());
      meDifferential[title] = m;
      
      // * Fraction of bad cells
      ibooker.setCurrentFolder("L1T/L1TOccupancy/Certification");
      title = testName;
      m = ibooker.book1D(title.c_str(),title.c_str(),2500,-.5,2500.-.5);
      m->setTitle(title.c_str());
      meCertification[title] = m;
   
        mValidTests.push_back(&(*it));
	
      }
      
    }
  }
}

//____________________________________________________________________________
// Function: endRun
// Description: This is will be run at the end of each run
// Inputs: 
// * const Run&        r       = Run information 
// * const EventSetup& context = Event Setup information
//____________________________________________________________________________
void L1TOccupancyClient::dqmEndJob(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter){

  book(ibooker, igetter);
  
  if(verbose_){cout << "[L1TOccupancyClient:] Called endRun()" << endl;}

  // Loop over every test in python
  for (std::vector<ParameterSet*>::iterator it = mValidTests.begin(); it != mValidTests.end(); it++) {

    ParameterSet &test     = (**it);
    string       algo_name = test.getUntrackedParameter<string>("algoName","XYSymmetry");
    string       test_name = test.getParameter         <string>("testName");
    
    if(verbose_) {cout << "[L1TOccupancyClient:] Starting calculations for: " << algo_name << " on: " << test_name << endl;}
    
    if(algo_name == "XYSymmetry") {

      ParameterSet ps       = (**it).getParameter<ParameterSet>("algoParams");
      string       histPath = ps.getParameter<string>("histPath");

      vector<pair<int,double> > deadChannels;
      vector<pair<int,double> > statDev;
      bool enoughStats = false;

      // Make final block
      hservice_->updateHistogramEndRun(test_name);

      // Perform the test
      double dead = xySymmetry(ps,test_name,deadChannels,statDev,enoughStats);
      stringstream str;
      str << test_name << "_cumu_LS_EndRun";

      if(verbose_) {
        TH2F* cumulative_save = (TH2F*) hservice_->getDifferentialHistogram(test_name)->Clone(str.str().c_str());

        cumulative_save->SetTitle(str.str().c_str());

        TDirectory* td = file_->GetDirectory(test_name.c_str());

        td->cd(string(test_name+"_Histos_AllLS").c_str());

        cumulative_save->Write();
      }
      // If we have enough statistics, we can write test result 
      if(enoughStats) {

	// Make the result histogram
        printDeadChannels(deadChannels,meResults[test_name]->getTH2F(),statDev,test_name);
	
	if(verbose_) {
          TH2F* cumulative_save = (TH2F*) hservice_->getDifferentialHistogram(test_name)->Clone(str.str().c_str());
          cumulative_save->SetTitle(str.str().c_str());
          TDirectory* td = file_->GetDirectory(("DQM_L1TOccupancyClient_Snapshots_LS.root:/"+test_name).c_str());
          td->cd(string(test_name+"_Histos").c_str());
          cumulative_save->Write();

          // save the result histo
          TH2F* h2f = meResults[test_name]->getTH2F();
          stringstream str2;
          str2 << test_name << "_result_LS_EndRun";
          TH2F* dead_save = (TH2F*) h2f->Clone(str2.str().c_str());
        
          td->cd(string(test_name+"_Results").c_str());
          dead_save->SetTitle(str2.str().c_str());
          dead_save->Write();
        }
        
        // Updating test results
        meDifferential[test_name]->Reset();
        meDifferential[test_name]->getTH2F()->Add(hservice_->getDifferentialHistogram(test_name));
        
        vector<int> lsCertification = hservice_->getLSCertification(test_name);

        // Fill fraction of dead channels
        for(unsigned int i=0;i<lsCertification.size();i++){
          int bin = meCertification[test_name]->getTH1()->FindBin(lsCertification[i]);
          meCertification[test_name]->getTH1()->SetBinContent(bin,1-dead);
        }
        
        // Reset differential histo
        hservice_->resetHisto(test_name);

        if(verbose_) {cout << "Now we have enough statstics for " << test_name << endl;}

      }else{
        if(verbose_){cout << "we don't have enough statstics for " << test_name << endl;}
        
        // Getting LS which this test monitored
        vector<int> lsCertification = hservice_->getLSCertification(test_name);

        // Fill fraction of dead channels
        for(unsigned int i=0;i<lsCertification.size();i++){
          int bin = meCertification[test_name]->getTH1()->FindBin(lsCertification[i]);
          meCertification[test_name]->getTH1()->SetBinContent(bin,-1);
        }
      }
    }else {if(verbose_){cout << "No valid algorithm" << std::endl;}}
  }

  if(verbose_){file_->Close();}
 
  delete hservice_;

}

//____________________________________________________________________________
// Function: beginLuminosityBlock
// Description: This is will be run at the begining of each luminosity block
// Inputs: 
// * const LuminosityBlock& lumiSeg = Luminosity Block information 
// * const EventSetup&      context = Event Setup information
//____________________________________________________________________________

//____________________________________________________________________________
// Function: endLuminosityBlock
// Description: This is will be run at the end of each luminosity block
// Inputs: 
// * const LuminosityBlock& lumiSeg = Luminosity Block information 
// * const EventSetup&      context = Event Setup information
//____________________________________________________________________________
void L1TOccupancyClient::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter, const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c){

  book(ibooker, igetter);
  
  int eventLS = lumiSeg.id().luminosityBlock();

  if(verbose_) {
    cout << "[L1TOccupancyClient:] Called endLuminosityBlock()" << endl;
    cout << "[L1TOccupancyClient:] Lumisection: " << eventLS << endl;
  }
  
  // Loop over every test in python
  for (std::vector<ParameterSet*>::const_iterator it = mValidTests.begin(); it != mValidTests.end(); it++) {
    
    ParameterSet &test     = (**it);
    string       algo_name = test.getUntrackedParameter<string>("algoName","XYSymmetry");
    string       test_name = test.getParameter         <string>("testName");

    if(verbose_) {cout << "[L1TOccupancyClient:] Starting calculations for " << algo_name << " on:" << test_name << endl;}
   
    if(algo_name == "XYSymmetry") {

      ParameterSet ps       = (**it).getParameter<ParameterSet>("algoParams");
      string       histPath = ps.getParameter<string>("histPath");
      
      vector<pair<int,double> > deadChannels;
      vector<pair<int,double> > statDev;
      bool enoughStats = false;

      // Update histo's data with data of this LS
      hservice_->updateHistogramEndLS(igetter, test_name,histPath,eventLS);

      // Perform the test
      double dead = xySymmetry(ps,test_name,deadChannels,statDev,enoughStats);
      stringstream str;
      str << test_name << "_cumu_LS_" << eventLS;

      if(verbose_) {
        TH2F* cumulative_save = (TH2F*) hservice_->getDifferentialHistogram(test_name)->Clone(str.str().c_str());
        cumulative_save->SetTitle(str.str().c_str());
        TDirectory* td = file_->GetDirectory(test_name.c_str());
        td->cd(string(test_name+"_Histos_AllLS").c_str());
        cumulative_save->Write();
      }
      
      // If we have enough statistics, we can write test result 
      if(enoughStats) {

	// Make the result histogram
        printDeadChannels(deadChannels,meResults[test_name]->getTH2F(),statDev,test_name);
	
	if(verbose_) {
          TH2F* cumulative_save = (TH2F*) hservice_->getDifferentialHistogram(test_name)->Clone(str.str().c_str());
          cumulative_save->SetTitle(str.str().c_str());
          TDirectory* td = file_->GetDirectory(("DQM_L1TOccupancyClient_Snapshots_LS.root:/"+test_name).c_str());
          td->cd(string(test_name+"_Histos").c_str());
          cumulative_save->Write();

          // save the result histo
          TH2F* h2f = meResults[test_name]->getTH2F();
          stringstream str2;
          str2 << test_name << "_result_LS_" << eventLS;
          TH2F* dead_save = (TH2F*) h2f->Clone(str2.str().c_str());
        
          td->cd(string(test_name+"_Results").c_str());
          dead_save->SetTitle(str2.str().c_str());
          dead_save->Write();
        }
        
        // Updating test results
        meDifferential[test_name]->Reset();
        meDifferential[test_name]->getTH2F()->Add(hservice_->getDifferentialHistogram(test_name));
        
        vector<int> lsCertification = hservice_->getLSCertification(test_name);

        // Fill fraction of dead channels
        for(unsigned int i=0;i<lsCertification.size();i++){
	  int bin = meCertification[test_name]->getTH1()->FindBin(lsCertification[i]);
	  meCertification[test_name]->getTH1()->SetBinContent(bin,1-dead);  
	}
	        
        // Reset differential histo
        hservice_->resetHisto(test_name);

        if(verbose_) {cout << "Now we have enough statstics for " << test_name << endl;}

      }else{if(verbose_){cout << "we don't have enough statstics for " << test_name << endl;}}
    }else {if(verbose_){cout << "No valid algorithm" << std::endl;}}
  }
}

//____________________________________________________________________________
// Function: analyze
// Description: This is will be run for every event
// Inputs: 
// * const Event&      e       = Event information 
// * const EventSetup& context = Event Setup information
//____________________________________________________________________________
//void L1TOccupancyClient::analyze(const Event& e, const EventSetup& context){}

//____________________________________________________________________________
// Function: xySymmetry
// Description: This method preforms the XY Symmetry test
// Inputs: 
// * ParameterSet                     ps           = Parameters for the test 
// * std::string                      test_name    = Test name of the test to be executed
// * std::vector< pair<int,double> >& deadChannels = Vector of 
// * std::vector< pair<int,double> >& statDev      = 
// * bool&                            enoughStats  = 
// Outputs:
// * double = fraction of bins that failed test, DeadChannels in vector, in: ParameterSet of test parameters
//____________________________________________________________________________
double L1TOccupancyClient::xySymmetry(const ParameterSet  &              ps, 
                                      string                      iTestName, 
                                      vector< pair<int,double> >& deadChannels, 
                                      vector< pair<int,double> >& statDev, 
                                      bool&                       enoughStats){
  
  // Getting differential histogram for this this thes
  TH2F* diffHist = hservice_->getDifferentialHistogram(iTestName);
  
  int    pAxis              = ps.getUntrackedParameter<int>   ("axis",1);
  int    pAverageMode       = ps.getUntrackedParameter<int>   ("averageMode",2); // 1=arith. mean, 2=median
  int    nBinsX             = diffHist->GetNbinsX();           // actual number of bins x
  int    nBinsY             = diffHist->GetNbinsY();           // actual number of bins y
  
  // Axis==1 : Means symmetry axis is vertical
  if(pAxis==1){
    
    int maxBinStrip, centralBinStrip; // x-coordinate of strips
    
    maxBinStrip = nBinsX;
    
    // If takeCenter=true  determine central bin of the pAxis
    // If takeCenter=false determine the bin to use based user input
    if(ps.getUntrackedParameter<bool>("takeCenter",true)){centralBinStrip = nBinsX / 2 + 1;}
    else {
      double pAxisSymmetryValue = ps.getParameter         <double>("axisSymmetryValue");
      getBinCoordinateOnAxisWithValue(diffHist, pAxisSymmetryValue, centralBinStrip, 1);  
    }
    
    // Assuming odd number of strips --> first comparison is middle strip to itself
    int upBinStrip  = centralBinStrip;
    int lowBinStrip = centralBinStrip; 

    // If even number decrease lowBinstrip by one
    if(nBinsX%2==0){lowBinStrip--;}
    
    // Do we have enough statistics? Min(Max(strip_i,strip_j))>threshold
    std::unique_ptr<double[]> maxAvgs(new double[maxBinStrip-upBinStrip+1]);

    int nActualStrips=0; //number of strips that are not fully masked
    for(int i=0, j=upBinStrip, k=lowBinStrip;j<=maxBinStrip;i++,j++,k--) {
      double avg1 = getAvrg(diffHist,iTestName,pAxis,nBinsY,j,pAverageMode);
      double avg2 = getAvrg(diffHist,iTestName,pAxis,nBinsY,k,pAverageMode);
      
      // Protection for when both strips are masked
      if(!hservice_->isStripMasked(iTestName,j,pAxis) && !hservice_->isStripMasked(iTestName,k,pAxis)) {
        maxAvgs[i] = TMath::Max(avg1,avg2);
        nActualStrips++;
      }
    }
    
    vector<double> defaultMu0up;
    defaultMu0up.push_back(13.7655);
    defaultMu0up.push_back(184.742);
    defaultMu0up.push_back(50735.3);
    defaultMu0up.push_back(-97.6793);
        
    TF1 tf("myFunc","[0]*(TMath::Log(x*[1]+[2]))+[3]",10.,11000.);
    vector<double> params = ps.getUntrackedParameter< vector<double> >("params_mu0_up",defaultMu0up);
    for(unsigned int i=0;i<params.size();i++) {tf.SetParameter(i,params[i]);}
    int statsup = (int)tf.Eval(hservice_->getNBinsHistogram(iTestName));

    vector<double> defaultMu0low;
    defaultMu0low.push_back(2.19664);
    defaultMu0low.push_back(1.94546);
    defaultMu0low.push_back(-99.3263);
    defaultMu0low.push_back(19.388);
    
    params = ps.getUntrackedParameter<vector<double> >("params_mu0_low",defaultMu0low);
    for(unsigned int i=0;i<params.size();i++) {tf.SetParameter(i,params[i]);}
    int statslow = (int)tf.Eval(hservice_->getNBinsHistogram(iTestName));
    
    if(verbose_) {
      cout << "nbins: "   << hservice_->getNBinsHistogram(iTestName) << endl;
      cout << "statsup= " << statsup << ", statslow= " << statslow << endl;
    }
    
    enoughStats = TMath::MinElement(nActualStrips,maxAvgs.get())>TMath::Max(statsup,statslow);
    if(verbose_) {
      cout << "stats: " << TMath::MinElement(nActualStrips,maxAvgs.get()) << ", statsAvg: " << diffHist->GetEntries()/hservice_->getNBinsHistogram(iTestName) << ", threshold: " << TMath::Max(statsup,statslow) << endl;
    }
    
    //if enough statistics
    //make the test
    if(enoughStats) {
      for(;upBinStrip<=maxBinStrip;upBinStrip++,lowBinStrip--) {
        double avg = getAvrg(diffHist, iTestName, pAxis, nBinsY, upBinStrip, pAverageMode);
        compareWithStrip(diffHist,iTestName,lowBinStrip,nBinsY,pAxis,avg,ps,deadChannels); //compare with lower side
        
        avg = getAvrg(diffHist, iTestName, pAxis, nBinsY, lowBinStrip, pAverageMode);
        compareWithStrip(diffHist,iTestName,upBinStrip,nBinsY,pAxis,avg,ps,deadChannels); //compare with upper side
      }
    }
  }
  
  // pAxis==2 : Means symetry pAxis is horizontal
  else if(pAxis==2) {
    int maxBinStrip, centralBinStrip; //x-coordinate of strips
    
    maxBinStrip = nBinsY;
    
    // Determine center of diagram: either with set pAxis or middle of diagram
    if(ps.getUntrackedParameter<bool>("takeCenter",true)){centralBinStrip = nBinsY / 2 + 1;}
    else {
      double pAxisSymmetryValue = ps.getParameter<double>("axisSymmetryValue");
      getBinCoordinateOnAxisWithValue(diffHist, pAxisSymmetryValue, centralBinStrip, 2);
      
    }
    
    //assuming odd number of strips --> first comparison is middle strip to itself
    int lowBinStrip = centralBinStrip, upBinStrip = centralBinStrip;
    
    //if even number
    if(nBinsX%2==0) {
      //decrease lowBinstrip by one
      lowBinStrip--; 
    }
    
    //do we have enough statistics? Min(Max(strip_i,strip_j))>threshold
    std::unique_ptr<double[]> maxAvgs(new double[maxBinStrip-upBinStrip+1]);
    int nActualStrips = 0;
    for(int i=0, j=upBinStrip, k=lowBinStrip;j<=maxBinStrip;i++,j++,k--) {
      double avg1 = getAvrg(diffHist, iTestName, pAxis, nBinsX, j, pAverageMode);
      double avg2 = getAvrg(diffHist, iTestName, pAxis, nBinsX, k, pAverageMode);
      if(!hservice_->isStripMasked(iTestName,j,pAxis) && !hservice_->isStripMasked(iTestName,k,pAxis)) {
        maxAvgs[i] = TMath::Max(avg1,avg2);
        nActualStrips++;
      }
    }
    
    vector<double> defaultMu0up;
    defaultMu0up.push_back(13.7655);
    defaultMu0up.push_back(184.742);
    defaultMu0up.push_back(50735.3);
    defaultMu0up.push_back(-97.6793);
        
    vector<double> params = ps.getUntrackedParameter<std::vector<double> >("params_mu0_up",defaultMu0up);
    TF1 tf("myFunc","[0]*(TMath::Log(x*[1]+[2]))+[3]",10.,11000.);
    for(unsigned int i=0;i<params.size();i++) {
      tf.SetParameter(i,params[i]);
    }
    int statsup = (int)tf.Eval(hservice_->getNBinsHistogram(iTestName));
    
    vector<double> defaultMu0low;
    defaultMu0low.push_back(2.19664);
    defaultMu0low.push_back(1.94546);
    defaultMu0low.push_back(-99.3263);
    defaultMu0low.push_back(19.388);

    params = ps.getUntrackedParameter<std::vector<double> >("params_mu0_low",defaultMu0low);
    for(unsigned int i=0;i<params.size();i++) {
      tf.SetParameter(i,params[i]);
    }
    int statslow = (int)tf.Eval(hservice_->getNBinsHistogram(iTestName));
    if(verbose_) {
      cout << "statsup= " << statsup << ", statslow= " << statslow << endl;
    }
    enoughStats = TMath::MinElement(nActualStrips,maxAvgs.get())>TMath::Max(statsup,statslow);
    if(verbose_) {
      cout << "stats: " << TMath::MinElement(nActualStrips,maxAvgs.get()) << ", statsAvg: " << diffHist->GetEntries()/hservice_->getNBinsHistogram(iTestName) << ", threshold: " << TMath::Max(statsup,statslow) << endl;
    }
    
    //if we have enough statistics
    //make the test
    if(enoughStats) {
      for(;upBinStrip<=maxBinStrip;upBinStrip++,lowBinStrip--) {
        double avg = getAvrg(diffHist, iTestName, pAxis, nBinsX, upBinStrip, pAverageMode);
        compareWithStrip(diffHist,iTestName, lowBinStrip,nBinsX,pAxis,avg,ps,deadChannels); //compare with lower side
        
        avg = getAvrg(diffHist, iTestName, pAxis, nBinsX, lowBinStrip, pAverageMode);
        compareWithStrip(diffHist,iTestName, upBinStrip,nBinsX,pAxis,avg,ps,deadChannels); //compare with upper side
      }
    }
  }
  else {if(verbose_){cout << "Invalid axis" << endl;}}
  
  return (deadChannels.size()-hservice_->getNBinsMasked(iTestName))*1.0/hservice_->getNBinsHistogram(iTestName);
}

//____________________________________________________________________________
// Function: getAvrg
// Description: Calculate strip average with method iAvgMode, where strip is 
// prependicular to iAxis at bin iBinStrip of histogram iHist
// Inputs: 
// * TH2F*  iHist     = Histogram to be tested
// * string iTestName = Name of the test
// * int    iAxis     = Axis prependicular to plot symmetry
// * int    iNBins     = Number of bins in the strip
// * int    iBinStrip = Bin corresponding to the strip in iAxis
// * int    iAvgMode  = Type of average mode 1) Average 2) Median
// Outputs:
// * double = Average of input strip
//____________________________________________________________________________
double L1TOccupancyClient::getAvrg(TH2F* iHist, string iTestName, int iAxis, int iNBins, int iBinStrip, int iAvgMode) {

  double avg = 0.0;
  TH1D* proj = NULL;
  TH2F* histo = new TH2F(*iHist);

  std::vector<double> values;
  int marked;

  if(iAxis==1) {

    switch(iAvgMode) {

      // arithmetic average
      case 1: 
        marked = hservice_->maskBins(iTestName,histo,iBinStrip,iAxis);
        proj   = histo->ProjectionX();
        avg    = proj->GetBinContent(iBinStrip)/(iNBins-marked);
        break;

      // median
      case 2:
	marked = hservice_->maskBins(iTestName,histo,iBinStrip,iAxis);
        proj = histo->ProjectionY("_py",iBinStrip,iBinStrip);
        for(int i=0;i<iNBins;i++) {
          values.push_back(proj->GetBinContent(i+1));
        }
        avg = TMath::Median(iNBins,&values[0]);
        break;
      default:
        if(verbose_){cout << "Invalid averaging mode!" << endl;}
        break;
    }
  }
  else if(iAxis==2) {

    switch(iAvgMode) {
      // arithmetic average
      case 1:
	marked = hservice_->maskBins(iTestName,histo,iBinStrip,iAxis);
        proj = histo->ProjectionY();
        avg = proj->GetBinContent(iBinStrip)/(iNBins-marked);
        break;
      // median
      case 2:
	marked = hservice_->maskBins(iTestName,histo,iBinStrip,iAxis);
        proj = histo->ProjectionX("_px",iBinStrip,iBinStrip);
        for(int i=0;i<iNBins;i++) {
          values.push_back(proj->GetBinContent(i+1));
        }
      
        avg = TMath::Median(iNBins,&values[0]);
        break;
      default: 
        if(verbose_) { cout << "invalid averaging mode!" << endl;}
        break;
    }
  }
  else {
    if(verbose_) {cout << "invalid axis" << endl;}
  }
  delete proj;
  delete histo;
  return avg;
}

//____________________________________________________________________________
// Function: printDeadChannels
// Description: 
// Inputs: 
// * vector< pair<int,double> > iDeadChannels     = List of bin that are masked of failed tthe test
// * TH2F*                      oHistDeadChannels = Histogram where test results should be printed
// * vector< pair<int,double> > statDev           = ???
// * string                     iTestName         = Name of the test
//____________________________________________________________________________
void L1TOccupancyClient::printDeadChannels(const vector< pair<int,double> >& iDeadChannels, TH2F* oHistDeadChannels, const vector<std::pair<int,double> >& statDev, string iTestName) {

  // Reset the dead channels histogram
  oHistDeadChannels->Reset();
  if(verbose_) {cout << "suspect or masked channels of " << iTestName << ": ";}

  int x,y,z;
  float chi2 = 0.0;

  // put all bad (value=1) and masked (value=-1) cells in histo
  for (std::vector<pair<int,double> >::const_iterator it = iDeadChannels.begin(); it != iDeadChannels.end(); it++) {

    int bin = (*it).first;
    oHistDeadChannels->GetBinXYZ(bin,x,y,z);

    if(hservice_->isMasked(iTestName,x,y)){
      oHistDeadChannels->SetBinContent(bin,-1); 
      if(verbose_){printf("(%4i,%4i) Masked\n",x,y);}
    }
    else{
      oHistDeadChannels->SetBinContent(bin, 1); 
      if(verbose_){printf("(%4i,%4i) Failed test\n",x,y);}
    }
  }

  // FIXME: Is this needed?
  for (std::vector<pair<int,double> >::const_iterator it = statDev.begin(); it != statDev.end(); it++) {
    double dev = (*it).second;
    chi2 += dev;
  }
  //put total chi2 in float

  if(verbose_) {
   cout << "total number of suspect channels: " << (iDeadChannels.size()-(hservice_->getNBinsMasked(iTestName))) << endl;
  }
}

//____________________________________________________________________________
// Function: compareWithStrip
// Description: Evaluates statistical compatibility of a strip (cell by cell) against a given average
// Inputs: 
// * TH2F*                      iHist      = Histogram to be tested
// * string                     iTestName  = Which test to apply
// * int                        iBinStrip  = Bin Coordinate (in bin units) of the stripo
// * int                        iNBins     = Number of Bins in the strip
// * int                        iAxis      = Which Axis is prependicular to the plot symmetry.
// * double                     iAvg       = Average of the strip
// * ParameterSet               iPS        = Parameters for the test
// * vector<pair<int,double> >& oChannels  = Output of bin that are masked or failed the test
// Outputs:
// * int = Number of dead channels
//____________________________________________________________________________
int L1TOccupancyClient::compareWithStrip(TH2F* iHist, string iTestName, int iBinStrip, int iNBins, int iAxis, double iAvg, const ParameterSet& iPS, vector<pair<int,double> >& oChannels) {

  int dead = 0;
  
  //
  if(iAxis==1) {
	
    // Get and set parameters for working curves
    TF1* fmuup  = new TF1("fmuup" ,"TMath::Log(TMath::PoissonI(x,[0])/TMath::PoissonI(x,[1]))",-10000.,10000.);
    TF1* fmulow = new TF1("fmulow","TMath::Log(TMath::PoissonI(x,[0])/TMath::PoissonI(x,[1]))",-10000.,10000.);
    fmuup ->SetParameter(0,iAvg*iPS.getUntrackedParameter<double>("factorup",2.0));
    fmuup ->SetParameter(1,iAvg);
    fmulow->SetParameter(0,iAvg*iPS.getUntrackedParameter<double>("factorlow",0.1));
    fmulow->SetParameter(1,iAvg);
    
    TF1* fchi = new TF1("fchi","[0]*x**2+[1]*x+[2]",0.,1500.);
    
    // Evaluate sigma up
    vector<double> defaultChi2up;
    defaultChi2up.push_back(5.45058e-05);
    defaultChi2up.push_back(0.268756);
    defaultChi2up.push_back(-11.7515);
    
    vector<double> params = iPS.getUntrackedParameter< vector<double> >("params_chi2_up",defaultChi2up);
    for(unsigned int i=0; i<params.size(); i++){fchi->SetParameter(i,params[i]);}
    double sigma_up = fchi->Eval(iAvg);

    // Evaluate sigma low
    vector<double> defaultChi2low;
    defaultChi2low.push_back(4.11095e-05);
    defaultChi2low.push_back(0.577451);
    defaultChi2low.push_back(-10.378);
    
    params = iPS.getUntrackedParameter< vector<double> >("params_chi2_low",defaultChi2low);
    for(unsigned int i=0; i<params.size(); i++){fchi->SetParameter(i,params[i]);}
    double sigma_low = fchi->Eval(iAvg);
    
    if(verbose_){cout << "binstrip= " << iBinStrip << ", sigmaup= " << sigma_up << ", sigmalow= " << sigma_low << endl;}
    
    for(int i=1;i<=iNBins;i++) {
      if(verbose_) {
        cout << "    " << i << " binContent: up:" << fmuup ->Eval(iHist->GetBinContent(iBinStrip,i)) 
	                    << " low: "           << fmulow->Eval(iHist->GetBinContent(iBinStrip,i)) << endl;
      }
      
      // Evaluate chi2 for cells
      double muup  = fmuup ->Eval(iHist->GetBinContent(iBinStrip,i));
      double mulow = fmulow->Eval(iHist->GetBinContent(iBinStrip,i));
      
      // If channel is masked -> set it to value -1
      if(hservice_->isMasked(iTestName,iBinStrip,i)) {
        oChannels.push_back(pair<int,double>(iHist->GetBin(iBinStrip,i),-1.0));
      }
      //else perform test
      else if(muup  > sigma_up || 
              mulow > sigma_low || 
	      ((fabs(muup)  == std::numeric_limits<double>::infinity()) && (
		fabs(mulow) == std::numeric_limits<double>::infinity()))) {
        dead++;
        oChannels.push_back(pair<int,double>(iHist->GetBin(iBinStrip,i),abs(iHist->GetBinContent(iBinStrip,i)-iAvg)/iAvg));
      }
    }
  }
  // 
  else if(iAxis==2){
	
    //get and set parameters for working curves
    TF1* fmuup  = new TF1("fmuup" ,"TMath::Log(TMath::PoissonI(x,[0])/TMath::PoissonI(x,[1]))",-10000.,10000.);
    TF1* fmulow = new TF1("fmulow","TMath::Log(TMath::PoissonI(x,[0])/TMath::PoissonI(x,[1]))",-10000.,10000.);
    fmuup ->SetParameter(0,iAvg*iPS.getUntrackedParameter<double>("factorup",2.0));
    fmuup ->SetParameter(1,iAvg);
    fmulow->SetParameter(0,iAvg*iPS.getUntrackedParameter<double>("factorlow",0.1));
    fmulow->SetParameter(1,iAvg);
    
    TF1* fchi = new TF1("fchi","[0]*x**2+[1]*x+[2]",0.,1500.);
    
    // Evaluate sigma up
    vector<double> defaultChi2up;
    defaultChi2up.push_back(5.45058e-05);
    defaultChi2up.push_back(0.268756);
    defaultChi2up.push_back(-11.7515);
    
    vector<double> params = iPS.getUntrackedParameter<vector<double> >("params_chi2_up",defaultChi2up);
    for(unsigned int i=0;i<params.size();i++){fchi->SetParameter(i,params[i]);}
    double sigma_up = fchi->Eval(iAvg);

    // Evaluate sigma low
    vector<double> defaultChi2low;
    defaultChi2low.push_back(4.11095e-05);
    defaultChi2low.push_back(0.577451);
    defaultChi2low.push_back(-10.378);

    params = iPS.getUntrackedParameter<vector<double> >("params_chi2_low",defaultChi2low);
    for(unsigned int i=0;i<params.size();i++){fchi->SetParameter(i,params[i]);}
    double sigma_low = fchi->Eval(iAvg);
    
    if(verbose_) {cout << "binstrip= " << iBinStrip << ", sigmaup= " << sigma_up << ", sigmalow= " << sigma_low << endl;}

    for(int i=1;i<=iNBins;i++) {
      if(verbose_) {
        cout << "    " << i << " binContent: up:" << fmuup ->Eval(iHist->GetBinContent(i,iBinStrip)) 
                            << " low: "           << fmulow->Eval(iHist->GetBinContent(i,iBinStrip)) << endl;
      }
      
      //evaluate chi2 for cells
      double muup  = fmuup ->Eval(iHist->GetBinContent(i,iBinStrip));
      double mulow = fmulow->Eval(iHist->GetBinContent(i,iBinStrip));
      
      //if channel is masked -> set it to value -1
      if(hservice_->isMasked(iTestName,i,iBinStrip)) {
        oChannels.push_back(pair<int,double>(iHist->GetBin(iBinStrip,i),-1.0));
      }
      //else perform test
      else if(muup > sigma_up ||  
              mulow > sigma_low || 
            ((fabs(muup) == std::numeric_limits<double>::infinity()) && 
            (fabs(mulow) == std::numeric_limits<double>::infinity()))) {
        dead++;
        oChannels.push_back(pair<int,double>(iHist->GetBin(i,iBinStrip),abs(iHist->GetBinContent(i,iBinStrip)-iAvg)/iAvg));
      }
    }
  }
  else {if(verbose_) {cout << "invalid axis" << endl;}}
  
  return dead;
}

//____________________________________________________________________________
// Function: getBinCoordinateOnAxisWithValue
// Description: Returns the bin global bin number with the iValue in the iAxis 
// Inputs: 
// * TH2F*  iHist          = Histogram to be tested
// * double iValue         = Value to be evaluated in the histogram iHist
// * int&   oBinCoordinate = (output) bin number (X or Y) for iValue 
// * int    iAxis          = Axis to be used
//____________________________________________________________________________
void L1TOccupancyClient::getBinCoordinateOnAxisWithValue(TH2F* iHist, double iValue, int& oBinCoordinate, int iAxis) {

  int nBinsX = iHist->GetNbinsX(); //actual number of bins x
  int nBinsY = iHist->GetNbinsY(); //actual number of bins y
  
  if(iAxis==1){
    int global = iHist->GetXaxis()->FindFixBin(iValue);
    
    // If parameter exceeds axis' value: set to maximum number of bins in x-axis
    if(global > nBinsX*nBinsY) {global = iHist->GetXaxis()->GetLast();}
    
    // Get coordinates of bin
    int y,z;
    iHist->GetBinXYZ(global,oBinCoordinate,y,z);
  }
  else if(iAxis==2){
    int global = iHist->GetYaxis()->FindFixBin(iValue);
    
    // If parameter exceeds axis' value: set to maximum number of bins in x-axis
    if(global > nBinsX*nBinsY) {global = iHist->GetYaxis()->GetLast();}
    
    // Get coordinates of bin
    int x,z;
    iHist->GetBinXYZ(global,x,oBinCoordinate,z);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TOccupancyClient);
