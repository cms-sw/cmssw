#ifndef FOURVECTORHLTONLINE_H
#define FOURVECTORHLTONLINE_H
// -*- C++ -*-
//
// Package:    FourVectorHLTOnline
// Class:      FourVectorHLTOnline
// 
/**\class FourVectorHLTOnline FourVectorHLTOnline.cc DQM/FourVectorHLTOnline/src/FourVectorHLTOnline.cc

 Description: This is a DQM source meant to plot high-level HLT trigger
 quantities as stored in the HLT results object TriggerResults

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeffrey Berryhill
//         Created:  June 2008
// $Id: FourVectorHLTOnline.h,v 1.7 2008/10/02 18:43:07 berryhil Exp $
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
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

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

class FourVectorHLTOnline : public edm::EDAnalyzer {
   public:
      explicit FourVectorHLTOnline(const edm::ParameterSet&);
      ~FourVectorHLTOnline();


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
      edm::InputTag triggerResultsLabel_;
      HLTConfigProvider hltConfig_;
      // data across paths
      MonitorElement* scalersSelect;
      // helper class to store the data path

      class PathInfo {
	PathInfo():
	  pathIndex_(-1), pathName_("unset"), filterName_("unset"), objectType_(-1)
	  {};
      public:
	void setHistos(
                       MonitorElement* const NOn, 
                       MonitorElement* const etOn, 
                       MonitorElement* const etaOn, 
		       MonitorElement* const phiOn, 
		       MonitorElement* const etavsphiOn,  
                       MonitorElement* const etL1, 
                       MonitorElement* const etaL1, 
		       MonitorElement* const phiL1, 
		       MonitorElement* const etavsphiL1) {
	  NOn_ = NOn;
	  etOn_ = etOn;
	  etaOn_ = etaOn;
	  phiOn_ = phiOn;
	  etavsphiOn_ = etavsphiOn;
	  etL1_ = etL1;
	  etaL1_ = etaL1;
	  phiL1_ = phiL1;
	  etavsphiL1_ = etavsphiL1;
	}
	MonitorElement * getNOnHisto() {
	  return NOn_;
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
	const std::string getLabel(void ) const {
	  return filterName_;
	}
	void setLabel(std::string labelName){
	  filterName_ = labelName;
          return;
	}
	const std::string getPath(void ) const {
	  return pathName_;
	}
        const edm::InputTag getTag(void) const{
	  edm::InputTag tagName(filterName_,"","HLT");
          return tagName;
	}
	~PathInfo() {};
	PathInfo(std::string pathName, std::string filterName, size_t type, float ptmin, 
		 float ptmax):
	  pathName_(pathName), filterName_(filterName), objectType_(type),
	  NOn_(0),etOn_(0), etaOn_(0), phiOn_(0), etavsphiOn_(0),
	  etL1_(0), etaL1_(0), phiL1_(0), etavsphiL1_(0),
	  ptmin_(ptmin), ptmax_(ptmax)
	  {
	  };
	  PathInfo(std::string pathName, std::string filterName, size_t type,
		   MonitorElement *NOn,
		   MonitorElement *etOn,
		   MonitorElement *etaOn,
		   MonitorElement *phiOn,
		   MonitorElement *etavsphiOn,
		   MonitorElement *etL1,
		   MonitorElement *etaL1,
		   MonitorElement *phiL1,
		   MonitorElement *etavsphiL1,
		   float ptmin, float ptmax
		   ):
	    pathName_(pathName), filterName_(filterName), objectType_(type),
	    NOn_(NOn), etOn_(etOn), etaOn_(etaOn), phiOn_(phiOn), etavsphiOn_(etavsphiOn),
	    etL1_(etL1), etaL1_(etaL1), phiL1_(phiL1), etavsphiL1_(etavsphiL1),
	    ptmin_(ptmin), ptmax_(ptmax)
	    {};
	    bool operator==(const std::string v) 
	    {
	      return v==filterName_;
	    }
      private:
	  int pathIndex_;
	  std::string pathName_;
	  std::string filterName_;
	  int objectType_;

	  // we don't own this data
	  MonitorElement *NOn_, *etOn_, *etaOn_, *phiOn_, *etavsphiOn_;
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
