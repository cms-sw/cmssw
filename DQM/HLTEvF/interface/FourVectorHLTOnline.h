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
// $Id: FourVectorHLTOnline.h,v 1.8 2009/11/20 15:07:18 rekovic Exp $
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

class FourVectorHLTOnline : public edm::EDAnalyzer {
   public:
      explicit FourVectorHLTOnline(const edm::ParameterSet&);
      ~FourVectorHLTOnline();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // BeginRun
      void beginRun(const edm::Run& run, const edm::EventSetup& c);

      // EndRun
      void endRun(const edm::Run& run, const edm::EventSetup& c);

      // EndLuminosityBlock
      void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);   

      void setupHLTMatrix(std::string name, std::vector<std::string> & paths);
      void fillHLTMatrix(TH2F* hist);
      void normalizeHLTMatrix();

      // ----------member data --------------------------- 
      int nev_;
      DQMStore * dbe_;
      edm::Handle<edm::TriggerResults> triggerResults_;

      MonitorElement* total_;
      MonitorElement* ME_HLTPassPass_; 
      MonitorElement* ME_HLTPassFail_; 
      MonitorElement* ME_HLTPassPass_Normalized_; 
      MonitorElement* ME_HLTPassFail_Normalized_; 
      MonitorElement* ME_HLTPass_Normalized_Any_; 
      MonitorElement* ME_HLTPass_Any_; 

      std::vector<MonitorElement*> v_ME_HLTPassPass_;
      std::vector<MonitorElement*> v_ME_HLTPassPass_Normalized_;
      std::vector<MonitorElement*> v_ME_HLTPass_Normalized_Any_;
      std::vector<MonitorElement*> v_ME_HLTPass_Any_;

      bool plotAll_;
      bool resetMe_;
      int currentRun_;
 
      unsigned int nBins_; 
      double ptMin_ ;
      double ptMax_ ;
      unsigned int nBinsOneOver_; 
      double oneOverPtMin_ ;
      double oneOverPtMax_ ;
      
      double electronEtaMax_;
      double electronEtMin_;
      double electronDRMatch_;
      double muonEtaMax_;
      double muonEtMin_;
      double muonDRMatch_;
      double tauEtaMax_;
      double tauEtMin_;
      double tauDRMatch_;
      double jetEtaMax_;
      double jetEtMin_;
      double jetDRMatch_;
      double bjetEtaMax_;
      double bjetEtMin_;
      double bjetDRMatch_;
      double photonEtaMax_;
      double photonEtMin_;
      double photonDRMatch_;
      double trackEtaMax_;
      double trackEtMin_;
      double trackDRMatch_;
      double metMin_;
      double htMin_;
      double sumEtMin_;

      std::vector<std::pair<std::string, std::string> > custompathnamepairs_;
      std::vector<std::string> specialPaths_;


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
                       MonitorElement* const onOneOverEtOn, 
                       MonitorElement* const onEtavsonPhiOn,  
                       MonitorElement* const NL1, 
                       MonitorElement* const l1EtL1, 
                       MonitorElement* const l1OneOverEtL1,
		       MonitorElement* const l1Etavsl1PhiL1,
                       MonitorElement* const NL1On, 
                       MonitorElement* const l1EtL1On, 
		       MonitorElement* const l1Etavsl1PhiL1On,
                       MonitorElement* const NL1OnUM, 
                       MonitorElement* const l1EtL1OnUM, 
		       MonitorElement* const l1Etavsl1PhiL1OnUM,
                       MonitorElement* const filters  
           )

          {
          NOn_ = NOn;
	  onEtOn_ = onEtOn;
	  onOneOverEtOn_ = onOneOverEtOn;
	  onEtavsonPhiOn_ = onEtavsonPhiOn;
          NL1_ = NL1;
	  l1EtL1_ = l1EtL1;
	  l1OneOverEtL1_ = l1OneOverEtL1;
	  l1Etavsl1PhiL1_ = l1Etavsl1PhiL1;
          NL1On_ = NL1On;
	  l1EtL1On_ = l1EtL1On;
	  l1Etavsl1PhiL1On_ = l1Etavsl1PhiL1On;
          NL1OnUM_ = NL1OnUM;
	  l1EtL1OnUM_ = l1EtL1OnUM;
	  l1Etavsl1PhiL1OnUM_ = l1Etavsl1PhiL1OnUM;
    filters_ = filters;
	}
	MonitorElement * getNOnHisto() {
	  return NOn_;
	}
	MonitorElement * getOnEtOnHisto() {
	  return onEtOn_;
	}
	MonitorElement * getOnOneOverEtOnHisto() {
	  return onOneOverEtOn_;
	}
	MonitorElement * getOnEtaVsOnPhiOnHisto() {
	  return onEtavsonPhiOn_;
	}
	MonitorElement * getNL1Histo() {
	  return NL1_;
	}
	MonitorElement * getL1EtL1Histo() {
	  return l1EtL1_;
	}
	MonitorElement * getL1OneOverEtL1Histo() {
	  return l1OneOverEtL1_;
	}
	MonitorElement * getL1EtaVsL1PhiL1Histo() {
	  return l1Etavsl1PhiL1_;
	}
	MonitorElement * getNL1OnHisto() {
	  return NL1On_;
	}
	MonitorElement * getL1EtL1OnHisto() {
	  return l1EtL1On_;
	}
	MonitorElement * getL1EtaVsL1PhiL1OnHisto() {
	  return l1Etavsl1PhiL1On_;
	}
	MonitorElement * getNL1OnUMHisto() {
	  return NL1OnUM_;
	}
	MonitorElement * getL1EtL1OnUMHisto() {
	  return l1EtL1OnUM_;
	}
	MonitorElement * getL1EtaVsL1PhiL1OnUMHisto() {
	  return l1Etavsl1PhiL1OnUM_;
	}
  MonitorElement * getFiltersHisto() {
    return filters_;
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
          NOn_(0), onEtOn_(0), onOneOverEtOn_(0),onEtavsonPhiOn_(0),
	  NL1_(0), l1EtL1_(0), l1OneOverEtL1_(0), l1Etavsl1PhiL1_(0),
          NL1On_(0), l1EtL1On_(0), l1Etavsl1PhiL1On_(0),
          NL1OnUM_(0), l1EtL1OnUM_(0), l1Etavsl1PhiL1OnUM_(0),
          filters_(0),
	  ptmin_(ptmin), ptmax_(ptmax)
	  {
	  };
	  PathInfo(std::string denomPathName, std::string pathName, std::string l1pathName, std::string filterName, std::string processName, size_t type,
		   MonitorElement *NOn,
		   MonitorElement *onEtOn,
		   MonitorElement *onOneOverEtOn,
		   MonitorElement *onEtavsonPhiOn,
		   MonitorElement *NL1,
		   MonitorElement *l1EtL1,
		   MonitorElement *l1OneOverEtL1,
		   MonitorElement *l1Etavsl1PhiL1,
		   MonitorElement *NL1On,
		   MonitorElement *l1EtL1On,
		   MonitorElement *l1Etavsl1PhiL1On,
		   MonitorElement *NL1OnUM,
		   MonitorElement *l1EtL1OnUM,
		   MonitorElement *l1Etavsl1PhiL1OnUM,
		   MonitorElement *filters,
		   float ptmin, float ptmax
		   ):
	    denomPathName_(denomPathName), pathName_(pathName), l1pathName_(l1pathName), filterName_(filterName), processName_(processName), objectType_(type),
            NOn_(NOn), onEtOn_(onEtOn),onOneOverEtOn_(onOneOverEtOn), onEtavsonPhiOn_(onEtavsonPhiOn),
	    NL1_(NL1), l1EtL1_(l1EtL1),l1OneOverEtL1_(l1OneOverEtL1), l1Etavsl1PhiL1_(l1Etavsl1PhiL1),
            NL1On_(NL1On), l1EtL1On_(l1EtL1On), l1Etavsl1PhiL1On_(l1Etavsl1PhiL1On),
            NL1OnUM_(NL1OnUM), l1EtL1OnUM_(l1EtL1OnUM), l1Etavsl1PhiL1OnUM_(l1Etavsl1PhiL1OnUM), filters_(filters),
	    ptmin_(ptmin), ptmax_(ptmax)
	    {};
	    bool operator==(const std::string v) 
	    {
	      return v==filterName_;
	    }

      std::vector< std::pair<std::string,unsigned int> > filtersAndIndices;
      private:
	  int pathIndex_;
	  std::string denomPathName_;
	  std::string pathName_;
	  std::string l1pathName_;
	  std::string filterName_;
	  std::string processName_;
	  int objectType_;

	  // we don't own this data
          MonitorElement *NOn_, *onEtOn_, *onOneOverEtOn_, *onEtavsonPhiOn_;
	  MonitorElement *NL1_, *l1EtL1_, *l1OneOverEtL1_, *l1Etavsl1PhiL1_;
	  MonitorElement *NL1On_, *l1EtL1On_, *l1Etavsl1PhiL1On_;
	  MonitorElement *NL1OnUM_, *l1EtL1OnUM_, *l1Etavsl1PhiL1OnUM_;
	  MonitorElement *filters_;


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
