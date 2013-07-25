#ifndef HLTMONSIMPLEBTAG_H
#define HLTMONSIMPLEBTAG_H

// -*- C++ -*-
//
// Package:    HLTMonSimpleBTag
// Class:      HLTMonSimpleBTag
// 
/**\class HLTMonSimpleBTag HLTMonSimpleBTag.cc DQM/HLTEvF/plugins/HLTMonSimpleBTag.cc DQM/HLTEvF/interface/HLTMonSimpleBTag.h

Description: [one line class summary]

Implementation:
[Notes on implementation]
*/
//
// Original Author:  Freya Blekman (fblekman)
//         Created:  Fri Mar 11 13:20:18 CET 2011
// $Id: HLTMonSimpleBTag.h,v 1.3 2011/03/25 16:10:43 fblekman Exp $
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
#include <string>

//
// class decleration
//

class HLTMonSimpleBTag : public edm::EDAnalyzer {
public:
  explicit HLTMonSimpleBTag(const edm::ParameterSet&);
  ~HLTMonSimpleBTag();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  // BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);

  // EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);

  // ---------- worker functions --------------------

  void calcEff(void);
  
  std::string makeEffName(std::string trig1, std::string trig2){std::string result="effRelative_"+trig1+"_passedanddivby_"+trig2; return result;}
  std::string makeEffNumeratorName(std::string trig1, std::string trig2){std::string result="selected_"+trig1+"_passed_"+trig2; return result;}
  
  void doEffCalc(MonitorElement *eff, MonitorElement *num, MonitorElement *denom);
  // ----------member data --------------------------- 
  int nev_;
  int refresheff_;
  DQMStore * dbe_;

  MonitorElement* total_;

  bool resetMe_;
  int currentRun_;
 
  unsigned int nBins_; 
  double ptMin_ ;
  double ptMax_ ;
  double dRTrigObjMatch_;
      
  std::string dirname_;
  bool monitorDaemon_;
  int theHLTOutputType;
  edm::InputTag triggerSummaryLabel_;
  edm::InputTag triggerResultLabel_;

  // helper class to store the data
  class PathInfo {
    PathInfo():
      pathIndex_(-1), pathName_("unset") 
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
    PathInfo(std::string pathName, float ptmin, 
	     float ptmax):
      pathName_(pathName), 
      et_(0), eta_(0), phi_(0), etavsphi_(0),
      ptmin_(ptmin), ptmax_(ptmax)
    {
    };
    PathInfo(std::string pathName, 
	     MonitorElement *et,
	     MonitorElement *eta,
	     MonitorElement *phi,
	     MonitorElement *etavsphi,
	     double ptmin, double ptmax
	     ):
      pathName_(pathName), 
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

    // we don't own this data
    MonitorElement *et_, *eta_, *phi_, *etavsphi_;

    double ptmin_, ptmax_;

    const int index() { 
      return pathIndex_;
    }

  public:
    double getPtMin() const { return ptmin_; }
    double getPtMax() const { return ptmax_; }
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
  PathInfoCollection hltEfficiencies_;
  
  std::vector<std::pair<std::string,std::string> > triggerMap_; // maps multiple trigger objects in hltPaths_ to each other so they can be used as reference triggers (used by hltEfficiencies_ vector)
};
#endif
