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
// $Id: FourVectorHLTOffline.h,v 1.8 2008/10/21 22:47:14 berryhil Exp $
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
      std::string processname_;
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
	  pathIndex_(-1), denomPathName_("unset"), pathName_("unset"), l1pathName_("unset"), filterName_("unset"), processName_("unset"), objectType_(-1)
	  {};
      public:
	void setHistos(
                       MonitorElement* const NOn, 
                       MonitorElement* const onEtOn, 
                       MonitorElement* const onEtaOn, 
		       MonitorElement* const onPhiOn, 
		       MonitorElement* const onEtavsonPhiOn,  
                       MonitorElement* const offEtOff, 
                       MonitorElement* const offEtaOff, 
		       MonitorElement* const offPhiOff, 
		       MonitorElement* const offEtavsoffPhiOff,
                       MonitorElement* const l1EtL1, 
                       MonitorElement* const l1EtaL1, 
		       MonitorElement* const l1PhiL1, 
		       MonitorElement* const l1Etavsl1PhiL1,
                       MonitorElement* const l1EtL1On, 
                       MonitorElement* const l1EtaL1On, 
		       MonitorElement* const l1PhiL1On, 
		       MonitorElement* const l1Etavsl1PhiL1On,  
                       MonitorElement* const offEtL1Off, 
                       MonitorElement* const offEtaL1Off, 
		       MonitorElement* const offPhiL1Off, 
		       MonitorElement* const offEtavsoffPhiL1Off,
                       MonitorElement* const offEtOnOff, 
                       MonitorElement* const offEtaOnOff, 
		       MonitorElement* const offPhiOnOff, 
		       MonitorElement* const offEtavsoffPhiOnOff) {
	  NOn_ = NOn;
	  onEtOn_ = onEtOn;
	  onEtaOn_ = onEtaOn;
	  onPhiOn_ = onPhiOn;
	  onEtavsonPhiOn_ = onEtavsonPhiOn;
	  offEtOff_ = offEtOff;
	  offEtaOff_ = offEtaOff;
	  offPhiOff_ = offPhiOff;
	  offEtavsoffPhiOff_ = offEtavsoffPhiOff;
	  l1EtL1_ = l1EtL1;
	  l1EtaL1_ = l1EtaL1;
	  l1PhiL1_ = l1PhiL1;
	  l1Etavsl1PhiL1_ = l1Etavsl1PhiL1;
	  l1EtL1On_ = l1EtL1On;
	  l1EtaL1On_ = l1EtaL1On;
	  l1PhiL1On_ = l1PhiL1On;
	  l1Etavsl1PhiL1On_ = l1Etavsl1PhiL1On;
	  offEtL1Off_ = offEtL1Off;
	  offEtaL1Off_ = offEtaL1Off;
	  offPhiL1Off_ = offPhiL1Off;
	  offEtavsoffPhiL1Off_ = offEtavsoffPhiL1Off;
	  offEtOnOff_ = offEtOnOff;
	  offEtaOnOff_ = offEtaOnOff;
	  offPhiOnOff_ = offPhiOnOff;
	  offEtavsoffPhiOnOff_ = offEtavsoffPhiOnOff;
	}
	MonitorElement * getNOnHisto() {
	  return NOn_;
	}
	MonitorElement * getOnEtOnHisto() {
	  return onEtOn_;
	}
	MonitorElement * getOnEtaOnHisto() {
	  return onEtaOn_;
	}
	MonitorElement * getOnPhiOnHisto() {
	  return onPhiOn_;
	}
	MonitorElement * getOnEtaVsOnPhiOnHisto() {
	  return onEtavsonPhiOn_;
	}
	MonitorElement * getOffEtOffHisto() {
	  return offEtOff_;
	}
	MonitorElement * getOffEtaOffHisto() {
	  return offEtaOff_;
	}
	MonitorElement * getOffPhiOffHisto() {
	  return offPhiOff_;
	}
	MonitorElement * getOffEtaVsOffPhiOffHisto() {
	  return offEtavsoffPhiOff_;
	}
	MonitorElement * getL1EtL1Histo() {
	  return l1EtL1_;
	}
	MonitorElement * getL1EtaL1Histo() {
	  return l1EtaL1_;
	}
	MonitorElement * getL1PhiL1Histo() {
	  return l1PhiL1_;
	}
	MonitorElement * getL1EtaVsL1PhiL1Histo() {
	  return l1Etavsl1PhiL1_;
	}
	MonitorElement * getL1EtL1OnHisto() {
	  return l1EtL1On_;
	}
	MonitorElement * getL1EtaL1OnHisto() {
	  return l1EtaL1On_;
	}
	MonitorElement * getL1PhiL1OnHisto() {
	  return l1PhiL1On_;
	}
	MonitorElement * getL1EtaVsL1PhiL1OnHisto() {
	  return l1Etavsl1PhiL1On_;
	}
	MonitorElement * getOffEtL1OffHisto() {
	  return offEtL1Off_;
	}
	MonitorElement * getOffEtaL1OffHisto() {
	  return offEtaL1Off_;
	}
	MonitorElement * getOffPhiL1OffHisto() {
	  return offPhiL1Off_;
	}
	MonitorElement * getOffEtaVsOffPhiL1OffHisto() {
	  return offEtavsoffPhiL1Off_;
	}
	MonitorElement * getOffEtOnOffHisto() {
	  return offEtOnOff_;
	}
	MonitorElement * getOffEtaOnOffHisto() {
	  return offEtaOnOff_;
	}
	MonitorElement * getOffPhiOnOffHisto() {
	  return offPhiOnOff_;
	}
	MonitorElement * getOffEtaVsOffPhiOnOffHisto() {
	  return offEtavsoffPhiOnOff_;
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
	const std::string getl1Path(void ) const {
	  return l1pathName_;
	}
	const std::string getDenomPath(void ) const {
	  return denomPathName_;
	}
	const std::string getProcess(void ) const {
	  return processName_;
	}
	const int getObjectType(void ) const {
	  return objectType_;
	}

        const edm::InputTag getTag(void) const{
	  edm::InputTag tagName(filterName_,"",processName_);
          return tagName;
	}
	~PathInfo() {};
	PathInfo(std::string denomPathName, std::string pathName, std::string l1pathName, std::string filterName, std::string processName, size_t type, float ptmin, 
		 float ptmax):
	  denomPathName_(denomPathName), pathName_(pathName), l1pathName_(l1pathName), filterName_(filterName), processName_(processName), objectType_(type),
	  NOn_(0),
          onEtOn_(0), onEtaOn_(0), onPhiOn_(0), onEtavsonPhiOn_(0),
	  offEtOff_(0), offEtaOff_(0), offPhiOff_(0), offEtavsoffPhiOff_(0),
	  l1EtL1_(0), l1EtaL1_(0), l1PhiL1_(0), l1Etavsl1PhiL1_(0),
          l1EtL1On_(0), l1EtaL1On_(0), l1PhiL1On_(0), l1Etavsl1PhiL1On_(0),
	  offEtL1Off_(0), offEtaL1Off_(0), offPhiL1Off_(0), offEtavsoffPhiL1Off_(0),
	  offEtOnOff_(0), offEtaOnOff_(0), offPhiOnOff_(0), offEtavsoffPhiOnOff_(0),
	  ptmin_(ptmin), ptmax_(ptmax)
	  {
	  };
	  PathInfo(std::string denomPathName, std::string pathName, std::string l1pathName, std::string filterName, std::string processName, size_t type,
		   MonitorElement *NOn,
		   MonitorElement *onEtOn,
		   MonitorElement *onEtaOn,
		   MonitorElement *onPhiOn,
		   MonitorElement *onEtavsonPhiOn,
		   MonitorElement *offEtOff,
		   MonitorElement *offEtaOff,
		   MonitorElement *offPhiOff,
		   MonitorElement *offEtavsoffPhiOff,
		   MonitorElement *l1EtL1,
		   MonitorElement *l1EtaL1,
		   MonitorElement *l1PhiL1,
		   MonitorElement *l1Etavsl1PhiL1,
		   MonitorElement *l1EtL1On,
		   MonitorElement *l1EtaL1On,
		   MonitorElement *l1PhiL1On,
		   MonitorElement *l1Etavsl1PhiL1On,
		   MonitorElement *offEtL1Off,
		   MonitorElement *offEtaL1Off,
		   MonitorElement *offPhiL1Off,
		   MonitorElement *offEtavsoffPhiL1Off,
		   MonitorElement *offEtOnOff,
		   MonitorElement *offEtaOnOff,
		   MonitorElement *offPhiOnOff,
		   MonitorElement *offEtavsoffPhiOnOff,
		   float ptmin, float ptmax
		   ):
	    denomPathName_(denomPathName), pathName_(pathName), l1pathName_(l1pathName), filterName_(filterName), processName_(processName), objectType_(type),
	    NOn_(NOn), 
            onEtOn_(onEtOn), onEtaOn_(onEtaOn), onPhiOn_(onPhiOn), onEtavsonPhiOn_(onEtavsonPhiOn),
	    offEtOff_(offEtOff), offEtaOff_(offEtaOff), offPhiOff_(offPhiOff), offEtavsoffPhiOff_(offEtavsoffPhiOff),
	    l1EtL1_(l1EtL1), l1EtaL1_(l1EtaL1), l1PhiL1_(l1PhiL1), l1Etavsl1PhiL1_(l1Etavsl1PhiL1),
            l1EtL1On_(l1EtL1On), l1EtaL1On_(l1EtaL1On), l1PhiL1On_(l1PhiL1On), l1Etavsl1PhiL1On_(l1Etavsl1PhiL1On),
	    offEtL1Off_(offEtL1Off), offEtaL1Off_(offEtaL1Off), offPhiL1Off_(offPhiL1Off), offEtavsoffPhiL1Off_(offEtavsoffPhiL1Off),
	    offEtOnOff_(offEtOnOff), offEtaOnOff_(offEtaOnOff), offPhiOnOff_(offPhiOnOff), offEtavsoffPhiOnOff_(offEtavsoffPhiOnOff),
	    ptmin_(ptmin), ptmax_(ptmax)
	    {};
	    bool operator==(const std::string v) 
	    {
	      return v==filterName_;
	    }
      private:
	  int pathIndex_;
	  std::string denomPathName_;
	  std::string pathName_;
	  std::string l1pathName_;
	  std::string filterName_;
	  std::string processName_;
	  int objectType_;

	  // we don't own this data
	  MonitorElement *NOn_, *onEtOn_, *onEtaOn_, *onPhiOn_, *onEtavsonPhiOn_;
	  MonitorElement *offEtOff_, *offEtaOff_, *offPhiOff_, *offEtavsoffPhiOff_;
	  MonitorElement *l1EtL1_, *l1EtaL1_, *l1PhiL1_, *l1Etavsl1PhiL1_;
	  MonitorElement *l1EtL1On_, *l1EtaL1On_, *l1PhiL1On_, *l1Etavsl1PhiL1On_;
	  MonitorElement *offEtL1Off_, *offEtaL1Off_, *offPhiL1Off_, *offEtavsoffPhiL1Off_;
	  MonitorElement *offEtOnOff_, *offEtaOnOff_, *offPhiOnOff_, *offEtavsoffPhiOnOff_;

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
