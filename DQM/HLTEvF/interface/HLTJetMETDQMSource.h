#ifndef HLTJetMETDQMSource_H
#define HLTJetMETDQMSource_H
// -*- C++ -*-
//
// Package:    HLTJetMETDQMSource
// Class:      HLTJetMETDQMSource
// 
/**\class HLTJetMETDQMSource HLTJetMETDQMSource.cc DQM/HLTJetMETDQMSource/src/HLTJetMETDQMSource.cc

 Description: This is a DQM source meant to plot high-level HLT trigger
 quantities as stored in the HLT results object TriggerResults
 
 Jochen Cammin: added more 2D plots: eta vs et and phi vs et

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Peter Wittich
//         Created:  May 2008
// Adapted for JetMET: Jochen Cammin 28 May 2008
// $Id: HLTJetMETDQMSource.h,v 1.0 2008/05/28 13:23:14 cammin Exp $
//
//


// system include files
#include <memory>
#include <unistd.h>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
#include <fstream>
#include <vector>

//
// class decleration
//

class HLTJetMETDQMSource : public edm::EDAnalyzer {
   public:
      explicit HLTJetMETDQMSource(const edm::ParameterSet&);
      ~HLTJetMETDQMSource();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // BeginRun
      void beginRun(const edm::Run& run, const edm::EventSetup& c);

      // EndRun
      void endRun(const edm::Run& run, const edm::EventSetup& c);


      // ----------member data --------------------------- 
      bool isFirst;

      int nev_;
      DQMStore * dbe_;

      MonitorElement* total_;

      bool plotAll_;
      bool resetMe_;
      int currentRun_;
 
      unsigned int nBins_; 
      double ptMin_ ;
      double ptMax_ ;
      
      std::string dirname_;
      bool monitorDaemon_;
      int theHLTOutputType;
      edm::InputTag triggerSummaryLabel_;
      edm::InputTag triggerResultLabel_;

      // helper class to store the data
      class PathInfo {
	PathInfo():
	  pathIndex_(-1), pathName_("unset"), objectType_(-1)
	  {};
      public:
	void setHistos(MonitorElement* const et, MonitorElement* const eta, 
		       MonitorElement* const phi, 
		       MonitorElement* const etavsphi, 
		       MonitorElement* const etavset,
		       MonitorElement* const phivset
		       ) {
	  et_ = et;
	  eta_ = eta;
	  phi_ = phi;
	  etavsphi_ = etavsphi;
	  etavset_  = etavset;
	  phivset_  = phivset;
	}
	MonitorElement * getEtHisto() {
	  return et_;
	}
	MonitorElement * getEtaHisto() {
	  return eta_;
	}
	MonitorElement * getPhiHisto() {
	  return phi_;
	}
	MonitorElement * getEtaVsPhiHisto() {
	  return etavsphi_;
	}
	MonitorElement * getEtaVsEtHisto() {
	  return etavset_;
	}
	MonitorElement * getPhiVsEtHisto() {
	  return phivset_;
	}
	const std::string getName(void ) const {
	  return pathName_;
	}
	~PathInfo() {};
	PathInfo(std::string pathName, size_t type, float ptmin, 
		 float ptmax):
	  pathName_(pathName), objectType_(type),
	  et_(0), eta_(0), phi_(0), etavsphi_(0), etavset_(0), phivset_(0),
	  ptmin_(ptmin), ptmax_(ptmax)
	  {
	  };
	  PathInfo(std::string pathName, size_t type,
		   MonitorElement *et,
		   MonitorElement *eta,
		   MonitorElement *phi,
		   MonitorElement *etavsphi,
		   MonitorElement *etavset,
		   MonitorElement *phivset,
		   float ptmin, float ptmax
		   ):
	    pathName_(pathName), objectType_(type),
	    et_(et), eta_(eta), phi_(phi), etavsphi_(etavsphi),
	    etavset_(etavset), phivset_(phivset), 
	    ptmin_(ptmin), ptmax_(ptmax)
	    {};
	    bool operator==(const std::string v) 
	    {
	      return v==pathName_;
	    }
      private:
	  int pathIndex_;
	  std::string pathName_;
	  int objectType_;

	  // we don't own this data
	  MonitorElement *et_, *eta_, *phi_, *etavsphi_, *etavset_, *phivset_;

	  float ptmin_, ptmax_;

	  const int index() { 
	    return pathIndex_;
	  }
	  const int type() { 
	    return objectType_;
	  }

      public:
	  float getPtMin() const { return ptmin_; }
	  float getPtMax() const { return ptmax_; }
      };

      // simple collection - just 
      class PathInfoCollection: public std::vector<PathInfo> {
      public:
	PathInfoCollection(): std::vector<PathInfo>() 
	  {};
	  std::vector<PathInfo>::iterator find(std::string pathName) {
	    return std::find(begin(), end(), pathName);
	  }
      };
      PathInfoCollection hltPaths_;


};
#endif
