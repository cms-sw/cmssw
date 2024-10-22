#ifndef DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENTHISTOGRAMSERVICE_H
#define DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENTHISTOGRAMSERVICE_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMStore.h"
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
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  L1TOccupancyClientHistogramService();
  L1TOccupancyClientHistogramService(const edm::ParameterSet& iParameters, DQMStore::IBooker& ibooker, bool iVerbose);

  //loads the histo of test into histos_
  TH2F* loadHisto(DQMStore::IGetter& igetter, std::string test, std::string histo);

  //updates histo (i.e. calculates differential to previous LS and adds it to cumulatice histo)
  void updateHistogramEndLS(DQMStore::IGetter& igetter, std::string test, std::string histo, int iLS);
  void updateHistogramEndRun(std::string iHistName);

  //resets the cumulative histo (after performing the test in L1TOccupancyClient)
  void resetHisto(std::string test);

  // Masks channels of histo in specific strip to perform L1TOccupancyClient::getAvrg()
  int maskBins(std::string test, TH2F* histo, int strip, int axis);

  bool isMasked(std::string test, int x, int y);                 //checks if cells is masked
  bool isStripMasked(std::string test, int binStrip, int axis);  //checks if a whole strip is masked

  void setMaskedBins(std::string test,
                     const std::vector<edm::ParameterSet>& mask);     //set masked channels specified in python
  std::vector<std::pair<int, int> > getMaskedBins(std::string test);  //returns masked channels of test

  unsigned int getNBinsMasked(std::string test);     // Get number of masked bins in test
  unsigned int getNBinsHistogram(std::string test);  // Get actual number of bins in test (i.e. nBins-nMaskedBins)
  TH2F* getDifferentialHistogram(std::string test);  // Get cumulative histogram
  TH2F* getRebinnedHistogram(DQMStore::IGetter& igetter,
                             std::string iHistName,
                             std::string iHistLocation);  // Get rebinned version of the hist

  std::vector<int> getLSCertification(std::string iHistName);  // Get list of tested LS for test iHistName

private:
  //DQMStore*         mDBE;        // storage service
  bool mVerbose;                  // verbose mode
  edm::ParameterSet mParameters;  // Copy of the parameters

  // Maps
  std::map<std::string, bool> mHistValid;                                 // Map of valid histograms (i.e. that exist)
  std::map<std::string, std::pair<TH2F*, TH2F*> > mHistograms;            // The cumulative histograms
  std::map<std::string, std::vector<std::pair<int, int> >*> mMaskedBins;  // Marked Bins
  std::map<std::string, TH2F*> mHistDiffMinus1;                           // Last already closed LS Block Histogram Diff
  std::map<std::string, std::vector<int> > mLSListDiff;                   // LS list of current block
  std::map<std::string, std::vector<int> > mLSListDiffMinus1;             // LS list of block -1
};

#endif
