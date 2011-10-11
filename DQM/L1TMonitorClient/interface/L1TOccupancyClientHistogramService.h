#ifndef DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENTHISTOGRAMSERVICE_H
#define DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENTHISTOGRAMSERVICE_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <TH1F.h>
#include <TH1D.h>
#include <TH2F.h>

class L1TOccupancyClientHistogramService {

  public:
  
    L1TOccupancyClientHistogramService(); 
    L1TOccupancyClientHistogramService(edm::ParameterSet iParameters, DQMStore* iDBE, bool iVerbose);
  
    //loads the histo of test into histos_
    TH2F* loadHisto  (std::string test,std::string histo); 

    //updates histo (i.e. calculates differential to previous LS and adds it to cumulatice histo)
    void  updateHisto(std::string test,std::string histo); 

    //resets the cumulative histo (after performing the test in L1TOccupancyClient)
    void  resetHisto(std::string test);  

    //marks channels of histo in specific strip to perform L1TOccupancyClient::getAvrg()
    int  markChannels(std::string test, TH2F* histo, int strip, int axis);  

    bool isWholeStripMarked(std::string test, int binStrip, int axis); //checks if a whole strip is masked
    bool isMarked(std::string test, int x, int y); //checks if cells is masked

    void setMarkedChannels(std::string test, std::vector<edm::ParameterSet> mark); //set masked channels specified in python

    unsigned int getNbinsHisto(std::string test);  //returns actual number of bins in a histo (i.e. nBins-nMaskedBins)
    std::vector<std::pair<int,int> > getMarkedChannels(std::string test);  //returns masked channels of test
    TH2F* getDifferentialHistogram(std::string test); //returns cumulative histogram
    uint  getNMarkedChannels(std::string test); //returns number of masked channels in histo
    TH2F* getRebinnedHisto(std::string iHistName, std::string histo);  //returns a rebinned version of the histo

  private:

    DQMStore*    dbe_;        // storage service
    bool         verbose_;    // verbose mode
    edm::ParameterSet mParameters; // Copy of the parameters
    
    // Maps
    std::map<std::string,std::pair<TH2F*,TH2F*> >        histos_; //test name, previous histo, cumulative histo
    std::map<std::string,std::vector<std::pair<int,int> >* > mMaskedBins; //marked channels

};

#endif
