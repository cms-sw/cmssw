#ifndef FOURVECTORHLT_H
#define FOURVECTORHLT_H
// -*- C++ -*-
//
// Package:    FourVectorHLT
// Class:      FourVectorHLT
// 
/**\class FourVectorHLT FourVectorHLT.cc DQM/FourVectorHLT/src/FourVectorHLT.cc

 Description: This is a DQM source meant to plot high-level HLT trigger
 quantities as stored in the HLT results object TriggerResults

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Peter Wittich
//         Created:  May 2008
// $Id: FourVectorHLT.h,v 1.6 2009/10/15 11:31:28 fwyzard Exp $
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

class FourVectorHLT : public edm::EDAnalyzer {
   public:
      explicit FourVectorHLT(const edm::ParameterSet&);
      ~FourVectorHLT();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // BeginRun
      void beginRun(const edm::Run& run, const edm::EventSetup& c);

      // EndRun
      void endRun(const edm::Run& run, const edm::EventSetup& c);


      // ----------member data --------------------------- 
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
		       MonitorElement* const etavsphi ) {
	  et_ = et;
	  eta_ = eta;
	  phi_ = phi;
	  etavsphi_ = etavsphi;
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
	const std::string getName(void ) const {
	  return pathName_;
	}
	~PathInfo() {};
	PathInfo(std::string pathName, size_t type, float ptmin, 
		 float ptmax):
	  pathName_(pathName), objectType_(type),
	  et_(0), eta_(0), phi_(0), etavsphi_(0),
	  ptmin_(ptmin), ptmax_(ptmax)
	  {
	  };
	  PathInfo(std::string pathName, size_t type,
		   MonitorElement *et,
		   MonitorElement *eta,
		   MonitorElement *phi,
		   MonitorElement *etavsphi,
		   float ptmin, float ptmax
		   ):
	    pathName_(pathName), objectType_(type),
	    et_(et), eta_(eta), phi_(phi), etavsphi_(etavsphi),
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
	  MonitorElement *et_, *eta_, *phi_, *etavsphi_;

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
