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
L1TOccupancyClientHistogramService::L1TOccupancyClientHistogramService(ParameterSet iParameters, DQMStore* iDBE, bool iVerbose) {
  dbe_        = iDBE;
  verbose_    = iVerbose;
  mParameters = iParameters;
}

//___________________________________________________________________________
// Function: getNbinsHisto
// Description: Returns the number of tested bin (i.e.: not masked) of the 
//              histogram with name test. 
// Inputs: 
//   * string iHistName = Name of the histogram
// Outputs:
//   * uint = Total number un-masked bins of histogram iHistName 
//____________________________________________________________________________
uint L1TOccupancyClientHistogramService::getNbinsHisto(string iHistName) {

  TH2F* pHistogram  = getDifferentialHistogram(iHistName);
  int   nBinsX      = pHistogram->GetNbinsX();
  int   nBinsY      = pHistogram->GetNbinsY();
  int   nMasked     = getNMarkedChannels(iHistName);
  uint  nBinsActive = (nBinsX*nBinsY)-nMasked;

  return nBinsActive;

}

//____________________________________________________________________________
// Function: setMarkedChannels
// Description: Reads user defined masked areas and populates a list of masked 
//              bins accordingly
// Inputs: 
//   * string               iHistName    = Name of the histogram
//   * vector<ParameterSet> iMaskedAreas = Vector areas to be masked
//____________________________________________________________________________
void L1TOccupancyClientHistogramService::setMarkedChannels(string iHistName, vector<ParameterSet> iMaskedAreas) {

  TH2F* histo = histos_[iHistName].first;
  vector<pair<int,int> >* m = new vector<pair<int,int> >();
  
  if(verbose_){printf("Masked areas for: %s\n",iHistName.c_str());}

  for(uint i=0;i<iMaskedAreas.size();i++) {

    ParameterSet iMA        = iMaskedAreas[i];
    int          iTypeUnits = iMA.getParameter<int>("kind");

    //get boundaries from python
    double xmin = iMA.getParameter<double>("xmin");
    double xmax = iMA.getParameter<double>("xmax");
    double ymin = iMA.getParameter<double>("ymin");
    double ymax = iMA.getParameter<double>("ymax");

    if(verbose_){
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
    if(iTypeUnits == 1) {

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
// Function: getMarkedChannels
// Description: Returns the vector of masked channels
// Inputs: 
// * string iHistName = Name of the histogram
// Outputs:
// * vector<pair<int,int> > = Vector of masked bin coordinates (x,y) 
//____________________________________________________________________________
vector<pair<int,int> > L1TOccupancyClientHistogramService::getMarkedChannels(string iHistName) {
  return (*mMaskedBins[iHistName]);
}

//____________________________________________________________________________
// Function: getNMarkedChannels
// Description: Returns the total number of masked channels
// Inputs: 
// * string iHistName = Name of the histogram
// Outputs:
// * uint = Total number of masked bins
//____________________________________________________________________________
uint L1TOccupancyClientHistogramService::getNMarkedChannels(string iHistName) {
  return mMaskedBins[iHistName]->size();
}

//____________________________________________________________________________
// TODO: Investigate what does this function do!!!
//masks channels of a certain strip for calculating average in L1TOccupancyClient::getAvrg()
//____________________________________________________________________________
int L1TOccupancyClientHistogramService::markChannels(string iHistName, TH2F* histo, int strip, int axis) {

  vector<pair<int,int> > m = (*mMaskedBins[iHistName]);
  int count=0;
  
  if(axis==1) {
    for(uint i=0;i<m.size();i++) {
      pair<int,int> p = m[i];
      if(p.first==strip) {
        histo->SetBinContent(p.first,p.second,0.0);
        count++;
      }
    }
  }
  else if(axis==2) {
    for(uint i=0;i<m.size();i++) {
      pair<int,int> p = m[i];
      if(p.second==strip) {
        histo->SetBinContent(p.first,p.second,0.0);
        count++;
      }
    }
  }
  else {
    if(verbose_) {cout << "invalid axis" << endl;}
  }
  
  return count;
}

//____________________________________________________________________________
// Function: isMarked
// Description: Returns if bin (iBinX,iBinY) of histogram iHistName is masked
// Inputs: 
// * string iHistName = Name of the histogram to be tested
// * int    iBinX     = X coordinate of the bin to be tested
// * int    iBinY     = Y coordinate of the bin to be tested
// Outputs:
// * uint = Total number of masked bins
//____________________________________________________________________________
bool L1TOccupancyClientHistogramService::isMarked(string iHistName, int iBinX, int iBinY) {

  vector<pair<int,int> > *thisHistMaskedBins = mMaskedBins[iHistName];

  bool binIsMasked = false;

  for(uint i=0; i<thisHistMaskedBins->size(); i++) {
    if((*thisHistMaskedBins)[i].first ==iBinX && 
       (*thisHistMaskedBins)[i].second==iBinY){
      binIsMasked=true;
      break;
    }
  }

  return binIsMasked;
}

//____________________________________________________________________________
// Function: isWholeStripMarked
// Description: Returns if a whole strip is masked 
// Inputs:
// * string iHistName = Name of the histogram to be tested
// * int    iBinStrip = Which is the strip to be checked (in bin units)
// * int    iAxis     = Which is the axis where the symmetry is defined
// Outputs:
// * bool = Returns is all bins in a strip are masked
//____________________________________________________________________________
bool L1TOccupancyClientHistogramService::isWholeStripMarked(string iHistName, int iBinStrip, int iAxis) {

  bool stripIsMasked = true;
  vector<pair<int,int> > *thisHistMaskedBins = mMaskedBins[iHistName];

  // If the histogram to be tested had strips defined along Y
  if(iAxis==1) {
    int count=0;
    for(uint i=0; i<thisHistMaskedBins->size(); i++) {
      if((*thisHistMaskedBins)[i].first==iBinStrip){count++;}
    }
    stripIsMasked = getDifferentialHistogram(iHistName)->GetYaxis()->GetNbins()==count;
  }
  // If the histogram to be tested had strips defined along X
  else {
    int count=0;
    for(uint i=0; i<thisHistMaskedBins->size(); i++) {
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
TH2F* L1TOccupancyClientHistogramService::loadHisto(string iHistName, string iHistLocation) {

  pair<TH2F*,TH2F*> histPair; 
  
  // Histogram to be monitored should be loaded  in the begining of the run
  histPair.first = getRebinnedHisto(iHistName,iHistLocation); 
  
  TH2F* histDiff = new TH2F(*histPair.first); // Clone the rebinned histogram to be monitored
  histDiff->Reset();                          // We reset histDiff so we are sure it is empty
  histPair.second=histDiff;
  
  histos_[iHistName]=histPair;
  
  return histDiff;  //return pointer to the differential histogram
  
}

//____________________________________________________________________________
// Function: getRebinnedHisto
// Description: Get an histogram from iHistLocation and rebins it
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
// * string iHistLocation = Location of the histogram in the directory structure
// Outputs:
// * TH2F* = Returns a pointer the differential histogram
//____________________________________________________________________________
TH2F* L1TOccupancyClientHistogramService::getRebinnedHisto(string iHistName, string iHistLocation) {
  
  // We clone the histogram to be monitored 
  TH2F* histMonitor = new TH2F(*(dbe_->get(iHistLocation)->getTH2F())); 
  
  int rebinFactorX=1;
  int rebinFactorY=1;
  
  vector<ParameterSet> testParameters = mParameters.getParameter< vector<ParameterSet> >("testParams");
  for(uint i=0 ; i<testParameters.size() ; i++){
    if(testParameters[i].getParameter<string>("testName")==iHistName){
      ParameterSet algoParameters = testParameters[i].getParameter<ParameterSet>("algoParams");
      rebinFactorX = algoParameters.getUntrackedParameter<int>("rebinFactorX",1);
      rebinFactorY = algoParameters.getUntrackedParameter<int>("rebinFactorY",1);
      break;
    }
  }
  
  if(rebinFactorX!=1){histMonitor->RebinY(rebinFactorX);}
  if(rebinFactorY!=1){histMonitor->RebinY(rebinFactorY);}
  
  return histMonitor;
  
}

//____________________________________________________________________________
// Function: updateHisto
// Description: Update de differential histogram
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
// * string iHistLocation = Location of the histogram in the directory structure
//____________________________________________________________________________
void L1TOccupancyClientHistogramService::updateHisto(string iHistName,string iHistLocation) {
  
  TH2F* histo_curr = getRebinnedHisto(iHistLocation,iHistLocation); // Get the rebinned histogram current cumulative iHistLocation
  TH2F* histo_old = new TH2F(*histo_curr);            // Clone 
  histo_curr->Add(histos_[iHistName].first,-1.0);  //calculate the difference to previous cumulative histo
  
  delete histos_[iHistName].first;            //delete old cumulateive histo 
  histos_[iHistName].first=histo_old;         //save old as new
  histos_[iHistName].second->Add(histo_curr); //save new as current

}

//____________________________________________________________________________
// Function: resetHisto
// Description: Resets a differential histogram by iHistName
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
//____________________________________________________________________________
void L1TOccupancyClientHistogramService::resetHisto(string iHistName) {
  histos_[iHistName].second->Reset();
}

//____________________________________________________________________________
// Function: getDifferentialHistogram
// Description: Resets a differential histogram by iHistName
// Inputs: 
// * string iHistName     = Name of the histogram to be tested
// Outputs:
// * TH2F* = Returns a pointer the differential histogram
//____________________________________________________________________________
TH2F* L1TOccupancyClientHistogramService::getDifferentialHistogram(string iHistName) {
  return histos_[iHistName].second;
}                                            
