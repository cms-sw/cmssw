#ifndef SiStripHistoricPlot_H
#define SiStripHistoricPlot_H
// -*- C++ -*-
// Package:     SiStripHistoricInfoClient
// Class  :     SiStripHistoricPlot
/**\class SiStripHistoricPlot SiStripHistoricPlot.h DQM/SiStripHistoricInfoClient/interface/SiStripHistoricPlot.h
Analyzer that produces the long-term detector  (historic plot) for the silicon strip tracker.
*/
// Original Author:  dkcira
//         Created:  Wed May 9 17:08:21 CEST 2007
// $Id: SiStripHistoricPlot.h,v 1.2 2008/07/23 08:56:57 alebihan Exp $
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "TH1F.h"
#include "TFile.h"
#include "TObjArray.h"
namespace cms{
  class SiStripHistoricPlot : public edm::EDAnalyzer {
  public:
    explicit SiStripHistoricPlot( const edm::ParameterSet& );
    ~SiStripHistoricPlot();
    virtual void beginJob(const edm::EventSetup&);
    virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
    virtual void analyze( const edm::Event&, const edm::EventSetup& );
    virtual void endRun(const edm::Run&, const edm::EventSetup&) ;
    virtual void endJob();
    void fillHistograms(edm::ESHandle<SiStripSummary> pS);
    TString detHistoTitle(uint32_t detid, std::string);
  private:
    bool printdebug_;
    int presentRunNr_;
    TH1F *ClusterSizesAllDets;
    TH1F *ClusterChargeAllDets;
    TH1F *OccupancyAllDets;
    TH1F *PercentNoisyStripsAllDets;
    TObjArray *AllDetHistograms;
    TFile *outputfile;
    std::vector<uint32_t> activeDets;
  };
}
#endif
