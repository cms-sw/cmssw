//  Example of how to calculate ClusterXTalk that is:
//     ( SignalL + SignalR) / SignalM
// for all SiStripClusters and save those values in output ROOT file as well
// as SiStripClusters basic parameters
//
// Author : Samvel Khalatyan (samvel at fnal dot gov)
// Created: 12/06/06
// Licence: GPL

#ifndef CALC_SI_STRIP_CLUSTER_XTALK_INTERFACE
#define CALC_SI_STRIP_CLUSTER_XTALK_INTERFACE

#include <string>

#include "FWCore/Framework/interface/EDAnalyzer.h"

// Save Compile time by forwarding declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"

class TFile;
class TTree;

class CalcSiStripClusterXTalk: public edm::EDAnalyzer {
  public:
    // Constructor
    CalcSiStripClusterXTalk( const edm::ParameterSet &roCONFIG);

    // Destructor
    virtual ~CalcSiStripClusterXTalk();

  private:
    struct Cluster {
      int   nModule;
      int   nPosition;
      int   nWidth;
      float dBaryCenter;

      float dClusterXTalk10;
      float dClusterXTalk5;
      float dClusterXTalk1;

      // See SiStripCluster interface for details
    };

    struct ClusterInfo {
      float dCharge;
    };

    // Prevent objects copying
    CalcSiStripClusterXTalk( const CalcSiStripClusterXTalk &);
    CalcSiStripClusterXTalk &operator =( const CalcSiStripClusterXTalk &);

    // Executed only once right before any analyzes start
    virtual void beginJob( const edm::EventSetup &roEVENT_SETUP);

    // Called for each event during analyzes
    virtual void analyze( const edm::Event	&roEVENT, 
			  const edm::EventSetup &roEVENT_SETUP);

    // Ends upon analyzes finished
    virtual void endJob();

    std::string oOutputFileName_;
    std::string oLblSiStripCluster_;
    std::string oLblSiStripClusterInfo_;
    std::string oLblSiStripDigi_;
    std::string oProdInstName_;

    Cluster     oCluster_;
    ClusterInfo oClusterInfo_;
    TFile       *poOutputFile_;
    TTree       *poClusterTree_;
};

#endif // CALC_SI_STRIP_CLUSTER_XTALK_INTERFACE
