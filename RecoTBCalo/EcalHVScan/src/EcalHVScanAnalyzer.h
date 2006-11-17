#ifndef ECALHVSCANANALYZER_H
#define ECALHVSCANANALYZER_H
/**\class EcalHVScanAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// $Id: EcalHVScanAnalyzer.h,v 1.1 2006/01/02 14:43:57 rahatlou Exp $
//



// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include <string>
#include "TTree.h"
#include "TH1.h"
#include "TGraph.h"
#include "TH2.h"
#include<fstream>
#include<map>
//#include<stl_pair>


struct PNfit {
  double max;
  double t0;
};

class EcalPnDiodeDigi;

//
// class declaration
//

class EcalHVScanAnalyzer : public edm::EDAnalyzer {
   public:
      explicit EcalHVScanAnalyzer( const edm::ParameterSet& );
      ~EcalHVScanAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
      virtual void beginJob(edm::EventSetup const&);
      virtual void endJob();
 private:

      
      PNfit maxPNDiode( const EcalPnDiodeDigi& digi );

      void initPNTTMap();
      std::pair<int,int> pnFromTT( const EcalTrigTowerDetId& tt);
      std::map<int, std::pair<int,int> > pnFromTTMap_; // map of PN diodes as a function of Trig towers

      std::string rootfile_;
      std::string hitCollection_;
      std::string hitProducer_;
      std::string pndiodeProducer_;
      fstream file;
      TH1F h_ampl_;
      TH1F h_jitt_;
      TH1F h1d_pnd_[10];
      TH1F h_pnd_max_[10];
      TH1F h_pnd_t0_[10];
      TH1F h_ampl_xtal_[85][20];
      TH1F h_jitt_xtal_[85][20];
      //TH1F h_alpha_xtal_[85][20];
      //TH1F h_tp_xtal_[85][20];
      TH1F h_norm_ampl_xtal_[85][20];

      TH2F h2d_anfit_tot_;
      TH2F h2d_anfit_bad_;

      // tree with output info
      TTree* tree_;
      int  tnXtal;
      //const int kMaxXtals; Shr 20061117: what is this??
      float tAmpl[85][20],tJitter[85][20],tChi2[85][20], tAmplPN1[85][20],
            tAmplPN2[85][20];
      int   tiEta[85][20],tiPhi[85][20],tiTT[85][20], tiC[85][20];


};



#endif
