#ifndef STFilter_h
#define STFilter_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TFile.h"
#include "TH1D.h"
#include "TString.h"

class STFilter : public edm::EDFilter {
   public:
      explicit STFilter(const edm::ParameterSet&);
      ~STFilter();
   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
   private:
      double pTMax_;
      // debug level
      int DEBUGLVL;    
      // counters
      unsigned int input_events;
      unsigned int accepted_events;
      // histograms
      bool m_produceHistos;
      TH1D* hbPt; TH1D* hbPtFiltered;
      TH1D* hbEta; TH1D* hbEtaFiltered;
      // histogram output file
      std::string fOutputFileName ;
      TFile*  hOutputFile ;
      //
      edm::ParameterSet conf_;
      edm::InputTag hepMCProductTag_;
};



#endif
