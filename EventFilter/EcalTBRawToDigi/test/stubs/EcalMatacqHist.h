// -*- Mode: C++; c-basic-offset: 2; indent-tabs-mode: t; tab-width: 8; -*-

/**
 * $Id: EcalMatacqHist.h,v 1.2 2006/09/21 12:16:44 pgras Exp $
 *
 * Test module for matacq data producing some histograms.
 *
 * Parameters:
 * <UL> outputRootFile: untracked string, name of the root file to create for
 * the histograms
 * <LI> nTimePlots: untracked int, number of events whose laser pulse is to
 * be plotted.
 * <LI> firstTimePlotEvent: untracked int, first event for laser pulse time
 * plot starting at 1.
 * </UL>
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <TFile.h>
#include <memory>

class TProfile;
class TH1D;

class EcalMatacqHist: public edm::EDAnalyzer{  
 public:
  EcalMatacqHist(const edm::ParameterSet& ps);  

  virtual ~EcalMatacqHist();
  
 protected:
  void
  analyze( const edm::Event & e, const  edm::EventSetup& c);


private:
  std::string outFileName;
  int nTimePlots;
  int firstTimePlotEvent;
  int iEvent;
  double hTTrigMin;
  double hTTrigMax;
  std::auto_ptr<TFile> outFile;
  std::vector<TProfile> profiles;
  //profile->MATACQ CH ID map
  std::vector<int> profChId;
  TH1D* hTTrig;
};


