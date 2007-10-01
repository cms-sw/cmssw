#ifndef DiJetAnalyzer_h
#define DiJetAnalyzer_h


//c++ include files
#include <memory>
#include <string>
#include <iostream>


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

//root 
#include "TFile.h"
#include "TTree.h"

using namespace std; 

namespace cms
{
class DiJetAnalyzer : public edm::EDAnalyzer {
public: 
      explicit DiJetAnalyzer(const edm::ParameterSet&);
      ~DiJetAnalyzer();
                                                                                                                             
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob(const edm::EventSetup& );
      virtual void endJob();

private: 

      double eta_jet, phi_jet, px_jet, py_jet, pz_jet; 

      // Ecal Hits
      int nEcalHits; 
      double ecalHit_eta[4000], ecalHit_phi[4000], ecalHit_energy[4000]; 

      // HB + HE Hits
      int nHBHEHits; 
      double hbheHit_eta[4000], hbheHit_phi[4000], hbheHit_energy[4000];

      // HO Hits
      int nHOHits; 
      double hoHit_eta[4000], hoHit_phi[4000], hoHit_energy[4000];

      // HF Hits
      int nHFHits;
      double hfHit_eta[4000], hfHit_phi[4000], hfHit_energy[4000];


      const CaloGeometry* geo;

      edm::InputTag jetsLabel_;
      std::vector<edm::InputTag> ecalLabels_;
      edm::InputTag hbheLabel_;
      edm::InputTag hoLabel_;
      edm::InputTag hfLabel_;


      string fOutputFileName;
      TFile* hOutputFile;
      TTree* myTree;

};
}

#endif
