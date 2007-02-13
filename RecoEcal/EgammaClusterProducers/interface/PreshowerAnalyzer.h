#ifndef RecoEcal_EgammaClusterProducers_PreshowerAnalyzer_h
#define RecoEcal_EgammaClusterProducers_PreshowerAnalyzer_h
/**\class PreshowerAnalyzer

 Description: Analyzer to fetch collection of objects from event and make simple plots

 Implementation:
     \\\author: Shahram Rahatlou, May 2006
*/
//
// $Id: PreshowerAnalyzer.h,v 1.2 2006/07/21 00:15:59 dbanduri Exp $
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include "TH1.h"
class TFile;

//
// class declaration
//

class PreshowerAnalyzer : public edm::EDAnalyzer {
   
public:
      explicit PreshowerAnalyzer( const edm::ParameterSet& );
      ~PreshowerAnalyzer();

      virtual void analyze( const edm::Event&, const edm::EventSetup& );
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();

 private:

  int nEvt_;         // internal counter of events

  float EminDE_;
  float EmaxDE_;
  int nBinDE_;
  float EminSC_;
  float EmaxSC_;
  int nBinSC_;

  std::string preshClusterCollectionX_;  // secondary name to be given to collection of cluster produced in this module
  std::string preshClusterCollectionY_;  
  std::string preshClusterProducer_;

  double calib_planeX_;
  double calib_planeY_;
  double mip_;
  double gamma_;

  // association parameters:
  std::string islandEndcapSuperClusterCollection1_;
  std::string islandEndcapSuperClusterProducer1_;

  std::string islandEndcapSuperClusterCollection2_;
  std::string islandEndcapSuperClusterProducer2_;

  TH1F* h1_esE_x;
  TH1F* h1_esE_y;
  TH1F* h1_esEta_x;
  TH1F* h1_esEta_y;
  TH1F* h1_esPhi_x;
  TH1F* h1_esPhi_y;
  TH1F* h1_esNhits_x;
  TH1F* h1_esNhits_y;
  TH1F* h1_esDeltaE;
  TH1F* h1_nclu_x;
  TH1F* h1_nclu_y;
  TH1F* h1_islandEESCEnergy1;
  TH1F* h1_islandEESCEnergy2;

  std::string outputFile_; // output file
  TFile*  rootFile_;

};
#endif
