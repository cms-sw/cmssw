#ifndef DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENTHISTOGRAMSERVICE_H
#define DQM_L1TMONITORCLIENT_L1TOCCUPANCYCLIENTHISTOGRAMSERVICE_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
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
  TH2F* loadHisto(DQMStore::IGetter& igetter, const std::string& test, std::string histo);

  //updates histo (i.e. calculates differential to previous LS and adds it to cumulatice histo)
  void updateHistogramEndLS(DQMStore::IGetter& igetter, const std::string& test, const std::string& histo, int iLS);
  void updateHistogramEndRun(const std::string& iHistName);

  //resets the cumulative histo (after performing the test in L1TOccupancyClient)
  void resetHisto(const std::string& test);

  // Masks channels of histo in specific strip to perform L1TOccupancyClient::getAvrg()
  int maskBins(const std::string& test, TH2F* histo, int strip, int axis);

  bool isMasked(const std::string& test, int x, int y);                 //checks if cells is masked
  bool isStripMasked(const std::string& test, int binStrip, int axis);  //checks if a whole strip is masked

  void setMaskedBins(const std::string& test,
                     const std::vector<edm::ParameterSet>& mask);            //set masked channels specified in python
  std::vector<std::pair<int, int> > getMaskedBins(const std::string& test);  //returns masked channels of test

  unsigned int getNBinsMasked(const std::string& test);     // Get number of masked bins in test
  unsigned int getNBinsHistogram(const std::string& test);  // Get actual number of bins in test (i.e. nBins-nMaskedBins)
  TH2F* getDifferentialHistogram(const std::string& test);  // Get cumulative histogram
  TH2F* getRebinnedHistogram(DQMStore::IGetter& igetter,
                             const std::string& iHistName,
                             const std::string& iHistLocation);  // Get rebinned version of the hist

  std::vector<int> getLSCertification(const std::string& iHistName);  // Get list of tested LS for test iHistName

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
