#ifndef FOURVECTORHLTOFFLINE_H
#define FOURVECTORHLTOFFLINE_H
// -*- C++ -*-
//
// Package:    FourVectorHLTOffline
// Class:      FourVectorHLTOffline
// 
/**\class FourVectorHLTOffline FourVectorHLTOffline.cc DQM/FourVectorHLTOffline/src/FourVectorHLTOffline.cc

 Description: This is a DQM source meant to plot high-level HLT trigger
 quantities as stored in the HLT results object TriggerResults

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeffrey Berryhill
//         Created:  June 2008
// $Id: FourVectorHLTOffline.h,v 1.1 2008/06/12 21:58:48 berryhil Exp $
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

class FourVectorHLTOffline : public edm::EDAnalyzer {
   public:
      explicit FourVectorHLTOffline(const edm::ParameterSet&);
      ~FourVectorHLTOffline();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
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
	void setHistos(MonitorElement* const etOn, 
                       MonitorElement* const etaOn, 
		       MonitorElement* const phiOn, 
		       MonitorElement* const etavsphiOn,  
                       MonitorElement* const etOff, 
                       MonitorElement* const etaOff, 
		       MonitorElement* const phiOff, 
		       MonitorElement* const etavsphiOff,
                       MonitorElement* const etL1, 
                       MonitorElement* const etaL1, 
		       MonitorElement* const phiL1, 
		       MonitorElement* const etavsphiL1) {
	  etOn_ = etOn;
	  etaOn_ = etaOn;
	  phiOn_ = phiOn;
	  etavsphiOn_ = etavsphiOn;
	  etOff_ = etOff;
	  etaOff_ = etaOff;
	  phiOff_ = phiOff;
	  etavsphiOff_ = etavsphiOff;
	  etL1_ = etL1;
	  etaL1_ = etaL1;
	  phiL1_ = phiL1;
	  etavsphiL1_ = etavsphiL1;
	}
	MonitorElement * getEtOnHisto() {
	  return etOn_;
	}
	MonitorElement * getEtaOnHisto() {
	  return etaOn_;
	}
	MonitorElement * getPhiOnHisto() {
	  return phiOn_;
	}
	MonitorElement * getEtaVsPhiOnHisto() {
	  return etavsphiOn_;
	}
	MonitorElement * getEtOffHisto() {
	  return etOff_;
	}
	MonitorElement * getEtaOffHisto() {
	  return etaOff_;
	}
	MonitorElement * getPhiOffHisto() {
	  return phiOff_;
	}
	MonitorElement * getEtaVsPhiOffHisto() {
	  return etavsphiOff_;
	}
	MonitorElement * getEtL1Histo() {
	  return etL1_;
	}
	MonitorElement * getEtaL1Histo() {
	  return etaL1_;
	}
	MonitorElement * getPhiL1Histo() {
	  return phiL1_;
	}
	MonitorElement * getEtaVsPhiL1Histo() {
	  return etavsphiL1_;
	}
	const std::string getName(void ) const {
	  return pathName_;
	}
	~PathInfo() {};
	PathInfo(std::string pathName, size_t type, float ptmin, 
		 float ptmax):
	  pathName_(pathName), objectType_(type),
	  etOn_(0), etaOn_(0), phiOn_(0), etavsphiOn_(0),
	  etOff_(0), etaOff_(0), phiOff_(0), etavsphiOff_(0),
	  etL1_(0), etaL1_(0), phiL1_(0), etavsphiL1_(0),
	  ptmin_(ptmin), ptmax_(ptmax)
	  {
	  };
	  PathInfo(std::string pathName, size_t type,
		   MonitorElement *etOn,
		   MonitorElement *etaOn,
		   MonitorElement *phiOn,
		   MonitorElement *etavsphiOn,
		   MonitorElement *etOff,
		   MonitorElement *etaOff,
		   MonitorElement *phiOff,
		   MonitorElement *etavsphiOff,
		   MonitorElement *etL1,
		   MonitorElement *etaL1,
		   MonitorElement *phiL1,
		   MonitorElement *etavsphiL1,
		   float ptmin, float ptmax
		   ):
	    pathName_(pathName), objectType_(type),
	    etOn_(etOn), etaOn_(etaOn), phiOn_(phiOn), etavsphiOn_(etavsphiOn),
	    etOff_(etOff), etaOff_(etaOff), phiOff_(phiOff), etavsphiOff_(etavsphiOff),
	    etL1_(etL1), etaL1_(etaL1), phiL1_(phiL1), etavsphiL1_(etavsphiL1),
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
	  MonitorElement *etOn_, *etaOn_, *phiOn_, *etavsphiOn_;
	  MonitorElement *etOff_, *etaOff_, *phiOff_, *etavsphiOff_;
	  MonitorElement *etL1_, *etaL1_, *phiL1_, *etavsphiL1_;

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
