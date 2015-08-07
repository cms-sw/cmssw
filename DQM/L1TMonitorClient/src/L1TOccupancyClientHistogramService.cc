#include "DQM/L1TMonitorClient/interface/L1TOccupancyClientHistogramService.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/QReport.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <stdio.h>
#include <sstream>
#include <math.h>
#include <vector>
#include <TMath.h>

using namespace edm;
using namespace std;

L1TOccupancyClientHistogramService::L1TOccupancyClientHistogramService(){}

//___________________________________________________________________________
// Function: L1TOccupancyClientHistogramService
// Description: Constructor
// Inputs: 
// * ParameterSet iParameters = Input parameter set
// * DQMStore*    iDBE        = Pointer to the DQMStore
// * bool         iVerbose    = Verbose control
//____________________________________________________________________________
L1TOccupancyClientHistogramService::L1TOccupancyClientHistogramService(const ParameterSet& iParameters, DQMStore::IBooker &ibooker, bool iVerbose) {
  //mDBE        = iDBE;
  mVerbose    = iVerbose;
  mParameters = iParameters;
}

//___________________________________________________________________________
// Function: getNBinsHistogram
// Description: Returns the number of tested bin (i.e.: not masked) of the 
//              histogram with name test. 
// Inputs: 
//   * string iHistName = Name of the histogram
// Outputs:
//   * unsigned int = Total number un-masked bins of histogram iHistName 
//____________________________________________________________________________
unsigned int L1TOccupancyClientHistogramService::getNBinsHistogram(string iHistName) {

  TH2F* pHistogram  = getDifferentialHistogram(iHistName);
  int   nBinsX      = pHistogram->GetNbinsX();
  int   nBinsY      = pHistogram->GetNbinsY();
  int   nMasked     = getNBinsMasked(iHistName);
  unsigned int  nBinsActive = (nBinsX*nBinsY)-nMasked;

  return nBinsActive;

}

//____________________________________________________________________________
// Function: setMaskedBins
// Description: Reads user defined masked areas and populates a list of masked 
//              bins accordingly
// Inputs: 
//   * string               iHistName    = Name of the histogram
//   * vector<ParameterSet> iMaskedAreas = Vector areas to be masked
//____________________________________________________________________________
void L1TOccupancyClientHistogramService::setMaskedBins(string iHistName, const vector<ParameterSet>& iMaskedAreas) {

  TH2F* histo = mHistograms[iHistName].first;
  vector<pair<int,int> >* m = new vector<pair<int,int> >();
  
  if(mVerbose){printf("Masked areas for: %s\n",iHistName.c_str());}

  for(unsigned int i=0;i<iMaskedAreas.size();i++) {

    ParameterSet iMA        = iMaskedAreas[i];
    int          iTypeUnits = iMA.getParameter<int>("kind");

    //get boundaries from python
    double xmin = iMA.getParameter<double>("xmin");
    double xmax = iMA.getParameter<double>("xmax");
    double ymin = iMA.getParameter<double>("ymin");
    double ymax = iMA.getParameter<double>("ymax");

    if(mVerbose){
      string sTypeUnits;
      if     (iTypeUnits == 0){sTypeUnits = "Histogram Units";}
      else if(iTypeUnits == 1){sTypeUnits = "Bin Units";}
      else                    {sTypeUnits = "Unknown Units";}
      printf("Area %3i: xmin=%6.2f xmax=%6.2f ymin=%6.2f ymax=%6.2f %s\n",i,xmin,xmax,ymin,ymax,sTypeUnits.c_str());
    }

    int xfirst,yfirst,xlast,ylast;

    //if min < max: change
    if(!(xmin<=xmax)){int z=xmax; xmax=xmin; xmin=z;}
    if(!(ymin<=ymax)){int z=ymax; ymax=ymin; ymin=z;}

    // If masked area are defined in terms of units of the histogram get bin coordinates
    if(iTypeUnits==0) {

      // We get the global bin number for this coordinates
      int globalMaxBin = histo->FindBin(xmax,ymax);
      int globalMinBin = histo->FindBin(xmax,ymax);

      // Dummy value for this variable since is a 2D histogram
      int binZ = 0; 

      // We convert global bins in axis bin numbers
      histo->GetBinXYZ(globalMinBin,xfirst,yfirst,binZ);
      histo->GetBinXYZ(globalMaxBin,xlast ,ylast ,binZ);

      // If the max edge (on X or Y) coincide with the lower edge of the current bin
      // pass one bin bellow (since bins are defined from [a,b[)
      if(histo->GetXaxis()->GetBinLowEdge(globalMaxBin)==xmax){xlast--;}
      if(histo->GetYaxis()->GetBinLowEdge(globalMaxBin)==ymax){ylast--;}

    }
    // Else units are bin coordinates just convert from double to int
    else {
      xfirst = (int) xmin;
      xlast  = (int) xmax;
      yfirst = (int) ymin;
      ylast  = (int) ymax;
    }
    
    // Now we generate coordinate pairs for each bin in the masked area
    // and we store them for future use
    for(int x=xfirst; x<=xlast; x++) {
      for(int y=yfirst; y<=ylast; y++) {
        pair<int,int> p;
        p.first  = x;
        p.second = y;
        m->push_back(p);
      }
    }
  }
  
  delete[] mMaskedBins[iHistName];
  mMaskedBins[iHistName]=m;

}

//____________________________________________________________________________
// Function: getMaskedBins
// Description: Returns the vector of masked channels
// Inputs: 
// * string iHistName = Name of the histogram
// Outputs:
// * vector<pair<int,int> > = Vector of masked bin coordinates (x,y) 
//____________________________________________________________________________
vector<pair<int,int> > L1TOccupancyClientHistogramService::getMaskedBins(string iHistName) {
  return (*mMaskedBins[iHistName]);
}

//____________________________________________________________________________
// Function: getNBinsMasked
// Description: Returns the total number of masked channels
// Inputs: 
// * string iHistName = Name of the histogram
// Outputs:
// * unsigned int = Total number of masked bins
//____________________________________________________________________________
unsigned int L1TOccupancyClientHistogramService::getNBinsMasked(string iHistName) {
  return mMaskedBins[iHistName]->size();
}


//____________________________________________________________________________
// Function: maskBins
// Description: masks channels of a certain strip for calculating average in L1TOccupancyClient::getAvrg()
// Inputs: 
// * string iHistName = Name of the histogram
// * TH2F*  oHist     =
// * int    iStrip     =
// * int    iAxis      =
// Outputs:
// * int = 
//____________________________________________________________________________
int L1TOccupancyClientHistogramService::maskBins(string iHistName, TH2F* oHist, int iStrip, int iAxis) {

  vector<pair<int,int> > m = (*mMaskedBins[iHistName]);
  int count=0;
  
  // iAxis==1 : Means symmetry axis is vertical
  if(iAxis==1) {
    for(unsigned int i=0;i<m.size();i++) {
      pair<int,int> &p = m[i];
      if(p.first==iStrip) {
        oHist->SetBinContent(p.first,p.second,0.0);
        count++;
      }
    }
  }
  // iAxis==2 : Means symmetry axis is horizontal
  else if(iAxis==2) {
    for(unsigned int i=0;i<m.size();i++) {
      pair<int,int> &p = m[i];
      if(p.second==iStrip) {
        oHist->SetBinContent(p.first,p.second,0.0);
        count++;
      }
    }
  }
  else {
    if(mVerbose) {cout << "invalid axis" << endl;}
  }
  
  return count;
}

//____________________________________________________________________________
// Function: isMasked
// Description: Returns if bin (iBinX,iBinY) of histogram iHistName is masked
// Inputs: 
// * string iHistName = Name of the histogram to be tested
// * int    iBinX     = X coordinate of the bin to be tested
// * int    iBinY     = Y coordinate of the bin to be tested
// Outputs:
// * unsigned int = Total number of masked bins
//____________________________________________________________________________
bool L1TOccupancyClientHistogramService::isMasked(string iHistName, int iBinX, int iBinY) {

  vector<pair<int,int> > *thisHistMaskedBins = mMaskedBins[iHistName];

  bool binIsMasked = false;

  for(unsigned int i=0; i<thisHistMaskedBins->size(); i++) {
    if((*thisHistMaskedBins)[i].first ==iBinX && 
       (*thisHistMaskedBins)[i].second==iBinY){
      binIsMasked=true;
      break;
    }
  }

  return binIsMasked;
}

//____________________________________________________________________________
// Function: isStripMasked
// Description: Returns if a whole strip is masked 
// Inputs:
// * string iHistName = Name of the histogram to be tested
// * int    iBinStrip = Which is the strip to be checked (in bin units)
// * int    iAxis     = Which is the axis where the symmetry is defined
// Outputs:
// * bool = Returns is all bins in a strip are masked
//____________________________________________________________________________
bool L1TOccupancyClientHistogramService::isStripMasked(string iHistName, int iBinStrip, int iAxis) {

  bool stripIsMasked = true;
  vector<pair<int,int> > *thisHistMaskedBins = mMaskedBins[iHistName];

  // If the histogram to be tested had strips defined along Y
  if(iAxis==1) {
    int count=0;
    for(unsigned int i=0; i<thisHistMaskedBins->size(); i++) {
      if((*thisHistMaskedBins)[i].first==iBinStrip){count++;}
    }
    stripIsMasked = getDifferentialHistogram(iHistName)->GetYaxis()->GetNbins()==count;
  }
  // If the histogram to be tested had strips defined along X
  else {
    int count=0;
    for(unsigned int i=0; i<thisHistMaskedBins->size(); i++) {
      if((*thisHistMaskedBins)[i].second==iBinStrip){count++;}
    }
    stripIsMasked = getDifferentialHistogram(iHistName)->GetXaxis()->GetNbins()==count;
  }

  return stripIsMasked;
}


//____________________________________________________________________________
// Function: loadHisto
// Description: Load the histogram iHistName  at iHistLocation
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
// * string iHistLocation = Location of the histogram in the directory structure
// Outputs:
// * TH2F* = Returns a pointer the differential histogram
//____________________________________________________________________________
TH2F* L1TOccupancyClientHistogramService::loadHisto(DQMStore::IGetter &igetter, string iHistName, string iHistLocation) {

  pair<TH2F*,TH2F*> histPair; 

  // Histogram to be monitored should be loaded  in the begining of the run
  TH2F* pHist = getRebinnedHistogram(igetter, iHistName,iHistLocation);
  
  if(mHistValid[iHistName]){
  
    histPair.first = pHist; 
  
    TH2F* histDiff = new TH2F(*histPair.first); // Clone the rebinned histogram to be monitored
    histDiff->Reset();                          // We reset histDiff so we are sure it is empty
    histPair.second=histDiff;
  
    mHistograms[iHistName]=histPair;
  
    // Stating the previous closed LS Block Histogram Diff 
    mHistDiffMinus1[iHistName]=new TH2F(*histDiff);
    
  }
    
  return pHist;  //return pointer to the differential histogram
  
}

//____________________________________________________________________________
// Function: getRebinnedHistogram
// Description: Get an histogram from iHistLocation and rebins it
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
// * string iHistLocation = Location of the histogram in the directory structure
// Outputs:
// * TH2F* = Returns a pointer the differential histogram
//____________________________________________________________________________
TH2F* L1TOccupancyClientHistogramService::getRebinnedHistogram(DQMStore::IGetter &igetter, string iHistName, string iHistLocation) {
  
  MonitorElement* me = igetter.get(iHistLocation);

  TH2F* histMonitor;
  
  if(!me){
    histMonitor           = 0;
    mHistValid[iHistName] = false;
  }
  else{
    mHistValid[iHistName] = true;
    histMonitor = new TH2F(*(igetter.get(iHistLocation)->getTH2F()));

    // Default rebin factors
    int rebinFactorX=1;
    int rebinFactorY=1;

    vector<ParameterSet> testParameters = mParameters.getParameter< vector<ParameterSet> >("testParams");
    for(unsigned int i=0 ; i<testParameters.size() ; i++){
      if(testParameters[i].getParameter<string>("testName")==iHistName){
        ParameterSet algoParameters = testParameters[i].getParameter<ParameterSet>("algoParams");
        rebinFactorX = algoParameters.getUntrackedParameter<int>("rebinFactorX",1);
        rebinFactorY = algoParameters.getUntrackedParameter<int>("rebinFactorY",1);
        break;
      }
    }

    // Rebinning
    if(rebinFactorX!=1){histMonitor->RebinY(rebinFactorX);}
    if(rebinFactorY!=1){histMonitor->RebinY(rebinFactorY);}

  }

  return histMonitor;
  
}

//____________________________________________________________________________
// Function: updateHistogramEndLS
// Description: Update de differential histogram
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
// * string iHistLocation = Location of the histogram in the directory structure
//____________________________________________________________________________
void L1TOccupancyClientHistogramService::updateHistogramEndLS(DQMStore::IGetter &igetter, string iHistName,string iHistLocation,int iLS) {
    
  if(mHistValid[iHistName]){
  
    TH2F* histo_curr = getRebinnedHistogram(igetter, iHistLocation,iHistLocation); // Get the rebinned histogram current cumulative iHistLocation

    TH2F* histo_old = new TH2F(*histo_curr);            // Clonecout <<"WP01"<<end; 
    histo_curr->Add(mHistograms[iHistName].first,-1.0); //calculate the difference to previous cumulative histo

    mLSListDiff[iHistName].push_back(iLS);

    delete mHistograms[iHistName].first;            //delete old cumulateive histo 
    mHistograms[iHistName].first=histo_old;         //save old as new
    mHistograms[iHistName].second->Add(histo_curr); //save new as current
    delete histo_curr;
  }  
}

//____________________________________________________________________________
// Function: updateHistogramEndRun
// Description: Update de differential histogram and LS list to certify in the 
//              end of the run by merging the last 2 LS Blocks (open + closed) 
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
//____________________________________________________________________________
void L1TOccupancyClientHistogramService::updateHistogramEndRun(string iHistName){
  
  if(mHistValid[iHistName]){
  
    mHistograms[iHistName].second->Add(mHistDiffMinus1[iHistName]);
    mLSListDiff[iHistName].insert(mLSListDiff      [iHistName].end(), 
                                  mLSListDiffMinus1[iHistName].begin(),
                                  mLSListDiffMinus1[iHistName].end());
  }
}

//____________________________________________________________________________
// Function: resetHisto
// Description: Resets a differential histogram by iHistName
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
//____________________________________________________________________________
void L1TOccupancyClientHistogramService::resetHisto(string iHistName){
  
  if(mHistValid[iHistName]){
  
    // Replacing mHistDiffMinus1
    delete mHistDiffMinus1[iHistName];
    mHistDiffMinus1[iHistName] = new TH2F(*mHistograms[iHistName].second);  

    // Resetting
    mHistograms[iHistName].second->Reset();

    // LS Accounting  
    mLSListDiffMinus1[iHistName] = mLSListDiff[iHistName]; // Replacing
    mLSListDiff      [iHistName].clear();                 // Resetting
    
  }
}

//____________________________________________________________________________
// Function: getLSCertification
// Description: Get the list of LS used for this test differential statistics
//              which in turn are the ones being certified
// Inputs: 
// * string iHistName = Name of the histogram to be tested
// Output:
// vector<int> = List of LS analysed
//____________________________________________________________________________
vector<int> L1TOccupancyClientHistogramService::getLSCertification(string iHistName){return mLSListDiff[iHistName];}

//____________________________________________________________________________
// Function: getDifferentialHistogram
// Description: Gets a differential histogram by iHistName
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
// Outputs:
// * TH2F* = Returns a pointer the differential histogram
//____________________________________________________________________________
TH2F* L1TOccupancyClientHistogramService::getDifferentialHistogram(string iHistName) {  
  if(mHistValid[iHistName]){return mHistograms[iHistName].second;}
  return 0;
}                                            
