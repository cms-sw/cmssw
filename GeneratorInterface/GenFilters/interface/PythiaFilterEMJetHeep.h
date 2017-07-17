#ifndef PythiaFilterEMJetHeep_h
#define PythiaFilterEMJetHeep_h

/** \class PythiaFilterEMJetHeep
 *
 *  PythiaFilterEMJetHeep filter implements generator-level preselections 
 *  of events with for studying background to high-energetic electrons
 *
 * \author Dmitry Bandurin (KSU), Jeremy Werner (Princeton)
 *
 ************************************************************/

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include "TH1.h"
#include "TTree.h"
class TFile;

namespace edm {
  class HepMCProduct;
}

class PythiaFilterEMJetHeep : public edm::EDFilter {
   public:
      explicit PythiaFilterEMJetHeep(const edm::ParameterSet&);
      ~PythiaFilterEMJetHeep();

      double deltaR(double eta0, double phi0, double eta, double phi);
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void beginJob();
      virtual void endJob();


   private:
      
       edm::EDGetTokenT<edm::HepMCProduct> token_;
       //
       double minEventPt;
       double etaMax;
       double cone_clust;
       double cone_iso;
       unsigned int nPartMin;
       double drMin;
       //
       double ptSeedMin_EB;    
       double fracConePtMin_EB;
       double ptHdMax_EB;
       double fracEmPtMin_EB;
//       double fracHdPtMax_EB;
       double fracTrkPtMax_EB;
       unsigned int ntrkMax_EB;       
       double isoConeMax_EB;   
       //
       double ptSeedMin_EE;    
       double fracConePtMin_EE;
       double ptHdMax_EE;
       double fracEmPtMin_EE;
//       double fracHdPtMax_EE;
       double fracTrkPtMax_EE;
       unsigned int ntrkMax_EE;       
       double isoConeMax_EE;   

// 
       int eventsProcessed; 

       int theNumberOfSelected;
       int maxnumberofeventsinrun;
       std::string outputFile_;

       float pt_photon;
       float setCone_iso;
       float setCone_clust;
       float setEM;
//       float setHAD;
       float setCharged;
       int Ncharged;
       float ptMaxHadron;

       bool minbias;

       bool accepted;       
    
       bool debug;
   
};
#endif
