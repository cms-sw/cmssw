//  SiStripClusters Analyzer 
//
// Author : Samvel Khalatyan (samvel at fnal dot gov)
// Created: 01/04/07
// Licence: GPL

#ifndef ANAEX_SISTRIPDETPERF_ANA_SI_STRIP_CLUSTERS_INTERFACE
#define ANAEX_SISTRIPDETPERF_ANA_SI_STRIP_CLUSTERS_INTERFACE

#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"


// Save Compile time by forwarding declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"

class TFile;
class TTree;

class AnaSiStripClusters: public edm::EDAnalyzer {
  public:
    // Constructor
    AnaSiStripClusters( const edm::ParameterSet &roCONFIG);

    // Destructor
    virtual ~AnaSiStripClusters();

  private:
    typedef std::vector<SiStripDigi> DigisVector;

    double getClusterEta( const std::vector<uint16_t> &roSTRIP_AMPLITUDES,
			  const int		      &rnFIRST_STRIP,
			  const DigisVector	      &roDIGIS) const;

    struct Cluster {
      int   nModule;
      int   nLayer;
      int   nSubdet;
      int   nPosition;
      int   nWidth;
      float dBaryCenter;
      float dEta;
      float dEtaTutorial;
      int   nCharge; 

      // See SiStripCluster interface for details
    };

    // Prevent objects copying
    AnaSiStripClusters( const AnaSiStripClusters &);
    AnaSiStripClusters &operator =( const AnaSiStripClusters &);

    // Executed only once right before any analyzes start
    virtual void beginJob( const edm::EventSetup &roEVENT_SETUP);

    // Called for each event during analyzes
    virtual void analyze( const edm::Event	&roEVENT, 
			  const edm::EventSetup &roEVENT_SETUP);

    // Ends upon analyzes finished
    virtual void endJob();

    std::string oOutputFileName_;
    std::string oLblSiStripCluster_;
    std::string oLblSiStripDigi_;
    std::string oProdInstNameDigi_;
    bool        bMTCCMode_;

    Cluster oCluster_;
    TFile   *poOutputFile_;
    TTree   *poClusterTree_;
};

#endif // ANAEX_SISTRIPDETPERF_ANA_SI_STRIP_CLUSTERS_INTERFACE
