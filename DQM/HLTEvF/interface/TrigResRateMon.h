#ifndef TRIGRESRATEMON_H
#define TRIGRESRATEMON_H
// -*- C++ -*-
//
// Package:    TrigResRateMon
// Class:      TrigResRateMon
// 
/**\class TrigResRateMon TrigResRateMon.cc 

 Module to monitor rates from TriggerResults

 Implementation:
     <Notes on implementation>
*/
// Original Author:
//        Vladimir Rekovic, July 2010
//
//
// $Id: TrigResRateMon.h,v 1.2 2010/09/29 23:07:07 rekovic Exp $
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

// added VR
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

/*
   needs cleaining of include statments (VR)
*/

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"

#include "DataFormats/Math/interface/deltaR.h"
#include  "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DQMServices/Core/interface/MonitorElement.h"



#include <iostream>
#include <fstream>
#include <vector>

namespace edm {
   class TriggerNames;
}

//typedef std::multimap<float,int> fimmap ;
//typedef std::set<fimmap , less<fimmap> > mmset;

class TrigResRateMon : public edm::EDAnalyzer {

   public:
      explicit TrigResRateMon(const edm::ParameterSet&);
      ~TrigResRateMon();

      //void cleanDRMatchSet(mmset& tempSet);

      edm::Handle<trigger::TriggerEvent> fTriggerObj;
      //edm::Handle<reco::BeamSpot> fBeamSpotHandle;

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // BeginRun
      void beginRun(const edm::Run& run, const edm::EventSetup& c);

      // EndRun
      void endRun(const edm::Run& run, const edm::EventSetup& c);
      void setupHltMatrix(const std::string& label, std::vector<std::string> &  paths);
      void setupStreamMatrix(const std::string& label, std::vector<std::string> &  paths);
      void setupHltLsPlots();
      void setupHltBxPlots();
      void countHLTPathHitsEndLumiBlock(const int & lumi);
      void countHLTGroupHitsEndLumiBlock(const int & lumi);
      void countHLTGroupL1HitsEndLumiBlock(const int & lumi);
      void countHLTGroupBXHitsEndLumiBlock(const int & lumi);

  void fillHltMatrix(const edm::TriggerNames & triggerNames, const edm::Event& iEvent, const edm::EventSetup& iSetup);
      void normalizeHLTMatrix();

      int getTriggerTypeParsePathName(const std::string & pathname);
      const std::string getL1ConditionModuleName(const std::string & pathname);
      bool hasL1Passed(const std::string & pathname, const edm::TriggerNames & triggerNames);
      bool hasHLTPassed(const std::string & pathname, const edm::TriggerNames& triggerNames);
      int getThresholdFromName(const std::string & pathname);


      void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);   
      void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);   

      // ----------member data --------------------------- 
      int nev_;
      DQMStore * dbe_;
      bool fLumiFlag;
      bool fIsSetup;

      // JetID helper
      //reco::helper::JetIDHelper *jetID;

  bool jmsDebug;
  bool jmsFakeZBCounts;
  unsigned int zbIndex;
  bool found_zbIndex;

      MonitorElement* ME_HLTAll_LS;
      MonitorElement* ME_HLT_BX;
      MonitorElement* ME_HLT_CUSTOM_BX;
      std::vector<MonitorElement*> v_ME_HLTAll_LS;
      std::vector<MonitorElement*> v_ME_Total_BX;
      std::vector<MonitorElement*> v_ME_Total_BX_Norm;

      std::vector<MonitorElement*> v_ME_HLTPassPass;
      std::vector<MonitorElement*> v_ME_HLTPassPass_Normalized;
      std::vector<MonitorElement*> v_ME_HLTPass_Normalized_Any;

      std::string pathsSummaryFolder_;
      std::string pathsSummaryStreamsFolder_;
      std::string pathsSummaryHLTCorrelationsFolder_;
      std::string pathsSummaryFilterEfficiencyFolder_;
      std::string pathsSummaryFilterCountsFolder_;
      std::string pathsSummaryHLTPathsPerLSFolder_;
      std::string pathsIndividualHLTPathsPerLSFolder_;
      std::string pathsSummaryHLTPathsPerBXFolder_;
      std::string fCustomBXPath;

      std::vector<std::string> fGroupName;


      unsigned int nLS_; 
      double LSsize_ ;
      double thresholdFactor_ ;
      unsigned int referenceBX_; 
      unsigned int Nbx_; 

      bool plotAll_;
      bool doCombineRuns_;
      bool doVBTFMuon_;
      int currentRun_;
      
      unsigned int nBins_; 
      unsigned int nBins2D_; 
      unsigned int nBinsDR_; 
      unsigned int nBinsOneOverEt_; 
      double ptMin_ ;
      double ptMax_ ;
      double dRMax_ ;
      double dRMaxElectronMuon_ ;
      
      double electronEtaMax_;
      double electronEtMin_;
      double electronDRMatch_;
      double electronL1DRMatch_;
      double muonEtaMax_;
      double muonEtMin_;
      double muonDRMatch_;
      double muonL1DRMatch_;
      double tauEtaMax_;
      double tauEtMin_;
      double tauDRMatch_;
      double tauL1DRMatch_;
      double jetEtaMax_;
      double jetEtMin_;
      double jetDRMatch_;
      double jetL1DRMatch_;
      double bjetEtaMax_;
      double bjetEtMin_;
      double bjetDRMatch_;
      double bjetL1DRMatch_;
      double photonEtaMax_;
      double photonEtMin_;
      double photonDRMatch_;
      double photonL1DRMatch_;
      double trackEtaMax_;
      double trackEtMin_;
      double trackDRMatch_;
      double trackL1DRMatch_;
      double metEtaMax_;
      double metMin_;
      double metDRMatch_;
      double metL1DRMatch_;
      double htEtaMax_;
      double htMin_;
      double htDRMatch_;
      double htL1DRMatch_;
      double sumEtMin_;

      // Muon quality cuts
      double dxyCut_;
      double normalizedChi2Cut_;
      int trackerHitsCut_;
      int pixelHitsCut_;
      int muonHitsCut_;
      bool isAlsoTrackerMuon_;
      int nMatchesCut_;

      std::vector<std::pair<std::string, std::string> > custompathnamepairs_;

      std::vector <std::vector <std::string> > triggerFilters_;
      std::vector <std::vector <uint> > triggerFilterIndices_;
      std::vector <std::pair<std::string, float> > fPathTempCountPair;
      std::vector <std::pair<std::string, std::vector<int> > > fPathBxTempCountPair;
      std::vector <std::pair<std::string, float> > fGroupTempCountPair;
      std::vector <std::pair<std::string, float> > fGroupL1TempCountPair;
      std::vector <std::pair<std::string, std::vector<std::string> > > fGroupNamePathsPair;

      std::vector<std::string> specialPaths_;

      std::string dirname_;
      std::string processname_;
      std::string muonRecoCollectionName_;
      bool monitorDaemon_;
      int theHLTOutputType;
      edm::InputTag triggerSummaryLabel_;
      edm::InputTag triggerResultsLabel_;
      edm::InputTag recHitsEBTag_, recHitsEETag_;
      HLTConfigProvider hltConfig_;
      // data across paths
      MonitorElement* scalersSelect;
      // helper class to store the data path

      edm::Handle<edm::TriggerResults> triggerResults_;

      class PathInfo {

       PathInfo():
        pathIndex_(-1), denomPathName_("unset"), pathName_("unset"), l1pathName_("unset"), filterName_("unset"), processName_("unset"), objectType_(-1) {};

       public:

          void setFilterHistos(MonitorElement* const filters) 
          {
              filters_   =  filters;
          }

          void setHistos(

            MonitorElement* const NOn, 
            MonitorElement* const onEtOn, 
            MonitorElement* const onOneOverEtOn, 
            MonitorElement* const onEtavsonPhiOn,  
            MonitorElement* const NOff, 
            MonitorElement* const offEtOff, 
            MonitorElement* const offEtavsoffPhiOff,
            MonitorElement* const NL1, 
            MonitorElement* const l1EtL1, 
            MonitorElement* const l1Etavsl1PhiL1,
            MonitorElement* const NL1On, 
            MonitorElement* const l1EtL1On, 
            MonitorElement* const l1Etavsl1PhiL1On,
            MonitorElement* const NL1Off,   
            MonitorElement* const offEtL1Off, 
            MonitorElement* const offEtavsoffPhiL1Off,
            MonitorElement* const NOnOff, 
            MonitorElement* const offEtOnOff, 
            MonitorElement* const offEtavsoffPhiOnOff,
            MonitorElement* const NL1OnUM, 
            MonitorElement* const l1EtL1OnUM, 
            MonitorElement* const l1Etavsl1PhiL1OnUM,
            MonitorElement* const NL1OffUM,   
            MonitorElement* const offEtL1OffUM, 
            MonitorElement* const offEtavsoffPhiL1OffUM,
            MonitorElement* const NOnOffUM, 
            MonitorElement* const offEtOnOffUM, 
            MonitorElement* const offEtavsoffPhiOnOffUM,
            MonitorElement* const offDRL1Off, 
            MonitorElement* const offDROnOff, 
            MonitorElement* const l1DRL1On)  
         {

              NOn_ = NOn;
              onEtOn_ = onEtOn;
              onOneOverEtOn_ = onOneOverEtOn;
              onEtavsonPhiOn_ = onEtavsonPhiOn;
              NOff_ = NOff;
              offEtOff_ = offEtOff;
              offEtavsoffPhiOff_ = offEtavsoffPhiOff;
              NL1_ = NL1;
              l1EtL1_ = l1EtL1;
              l1Etavsl1PhiL1_ = l1Etavsl1PhiL1;
              NL1On_ = NL1On;
              l1EtL1On_ = l1EtL1On;
              l1Etavsl1PhiL1On_ = l1Etavsl1PhiL1On;
              NL1Off_ = NL1Off;
              offEtL1Off_ = offEtL1Off;
              offEtavsoffPhiL1Off_ = offEtavsoffPhiL1Off;
              NOnOff_ = NOnOff;
              offEtOnOff_ = offEtOnOff;
              offEtavsoffPhiOnOff_ = offEtavsoffPhiOnOff;
              NL1OnUM_ = NL1OnUM;
              l1EtL1OnUM_ = l1EtL1OnUM;
              l1Etavsl1PhiL1OnUM_ = l1Etavsl1PhiL1OnUM;
              NL1OffUM_ = NL1OffUM;
              offEtL1OffUM_ = offEtL1OffUM;
              offEtavsoffPhiL1OffUM_ = offEtavsoffPhiL1OffUM;
              NOnOffUM_ = NOnOffUM;
              offEtOnOffUM_ = offEtOnOffUM;
              offEtavsoffPhiOnOffUM_ = offEtavsoffPhiOnOffUM;
              offDRL1Off_ =  offDRL1Off; 
              offDROnOff_ =  offDROnOff; 
              l1DRL1On_   =  l1DRL1On;

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
         MonitorElement * getNOffHisto() {
           return NOff_;
         }
         MonitorElement * getOffEtOffHisto() {
           return offEtOff_;
         }
         MonitorElement * getOffEtaVsOffPhiOffHisto() {
           return offEtavsoffPhiOff_;
         }
         MonitorElement * getNL1Histo() {
           return NL1_;
         }
         MonitorElement * getL1EtL1Histo() {
           return l1EtL1_;
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
         MonitorElement * getNL1OffHisto() {
           return NL1Off_;
         }
         MonitorElement * getOffEtL1OffHisto() {
           return offEtL1Off_;
         }
         MonitorElement * getOffEtaVsOffPhiL1OffHisto() {
           return offEtavsoffPhiL1Off_;
         }
         MonitorElement * getNOnOffHisto() {
           return NOnOff_;
         }
         MonitorElement * getOffEtOnOffHisto() {
           return offEtOnOff_;
         }
         MonitorElement * getOffEtaVsOffPhiOnOffHisto() {
           return offEtavsoffPhiOnOff_;
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
         MonitorElement * getNL1OffUMHisto() {
           return NL1OffUM_;
         }
         MonitorElement * getOffEtL1OffUMHisto() {
           return offEtL1OffUM_;
         }
         MonitorElement * getOffEtaVsOffPhiL1OffUMHisto() {
           return offEtavsoffPhiL1OffUM_;
         }
         MonitorElement * getNOnOffUMHisto() {
           return NOnOffUM_;
         }
         MonitorElement * getOffEtOnOffUMHisto() {
           return offEtOnOffUM_;
         }
         MonitorElement * getOffEtaVsOffPhiOnOffUMHisto() {
           return offEtavsoffPhiOnOffUM_;
         }
         MonitorElement * getOffDRL1OffHisto() {
           return offDRL1Off_;
         }
         MonitorElement * getOffDROnOffHisto() {
           return offDROnOff_;
         }
         MonitorElement * getL1DROnL1Histo() {
           return l1DRL1On_;
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
         const std::string & getPath(void ) const {
           return pathName_;
         }
         const std::string & getl1Path(void ) const {
           return l1pathName_;
         }
         const int getL1ModuleIndex(void ) const {
           return l1ModuleIndex_;
         }
         const std::string & getDenomPath(void ) const {
           return denomPathName_;
         }
         const std::string & getProcess(void ) const {
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

        PathInfo(std::string denomPathName, std::string pathName, std::string l1pathName, int l1ModuleIndex, std::string filterName, std::string processName, size_t type, float ptmin, float ptmax, float hltThreshold, float l1Threshold):

          denomPathName_(denomPathName), 
          pathName_(pathName), 
          l1pathName_(l1pathName), 
          l1ModuleIndex_(l1ModuleIndex), 
          filterName_(filterName), 
          processName_(processName), 
          objectType_(type),
          NOn_(0), onEtOn_(0), onOneOverEtOn_(0), onEtavsonPhiOn_(0), 
          NOff_(0), offEtOff_(0), offEtavsoffPhiOff_(0),
          NL1_(0), l1EtL1_(0), l1Etavsl1PhiL1_(0),
          NL1On_(0), l1EtL1On_(0), l1Etavsl1PhiL1On_(0),
          NL1Off_(0), offEtL1Off_(0), offEtavsoffPhiL1Off_(0),
          NOnOff_(0), offEtOnOff_(0), offEtavsoffPhiOnOff_(0),
          NL1OnUM_(0), l1EtL1OnUM_(0), l1Etavsl1PhiL1OnUM_(0),
          NL1OffUM_(0), offEtL1OffUM_(0), offEtavsoffPhiL1OffUM_(0),
          NOnOffUM_(0), offEtOnOffUM_(0), offEtavsoffPhiOnOffUM_(0),
          offDRL1Off_(0), offDROnOff_(0), l1DRL1On_(0), filters_(0),
          ptmin_(ptmin), ptmax_(ptmax),
          hltThreshold_(hltThreshold), l1Threshold_(l1Threshold)

        {
        };

        PathInfo(std::string denomPathName, std::string pathName, std::string l1pathName, std::string filterName, std::string processName, size_t type,
          MonitorElement *NOn,
          MonitorElement *onEtOn,
          MonitorElement *onOneOverEtOn,
          MonitorElement *onEtavsonPhiOn,
          MonitorElement *NOff,
          MonitorElement *offEtOff,
          MonitorElement *offEtavsoffPhiOff,
          MonitorElement *NL1,
          MonitorElement *l1EtL1,
          MonitorElement *l1Etavsl1PhiL1,
          MonitorElement *NL1On,
          MonitorElement *l1EtL1On,
          MonitorElement *l1Etavsl1PhiL1On,
          MonitorElement *NL1Off,
          MonitorElement *offEtL1Off,
          MonitorElement *offEtavsoffPhiL1Off,
          MonitorElement *NOnOff,
          MonitorElement *offEtOnOff,
          MonitorElement *offEtavsoffPhiOnOff,
          MonitorElement *NL1OnUM,
          MonitorElement *l1EtL1OnUM,
          MonitorElement *l1Etavsl1PhiL1OnUM,
          MonitorElement *NL1OffUM,
          MonitorElement *offEtL1OffUM,
          MonitorElement *offEtavsoffPhiL1OffUM,
          MonitorElement *NOnOffUM,
          MonitorElement *offEtOnOffUM,
          MonitorElement *offEtavsoffPhiOnOffUM,
          MonitorElement *offDRL1Off, 
          MonitorElement *offDROnOff, 
          MonitorElement *l1DRL1On,
          MonitorElement *filters,
          float ptmin, float ptmax
          ):

            denomPathName_(denomPathName), 
            pathName_(pathName), l1pathName_(l1pathName), 
            filterName_(filterName), processName_(processName), objectType_(type),
            NOn_(NOn), onEtOn_(onEtOn), onOneOverEtOn_(onOneOverEtOn), onEtavsonPhiOn_(onEtavsonPhiOn), 
            NOff_(NOff), offEtOff_(offEtOff), offEtavsoffPhiOff_(offEtavsoffPhiOff),
            NL1_(NL1), l1EtL1_(l1EtL1), l1Etavsl1PhiL1_(l1Etavsl1PhiL1),
            NL1On_(NL1On), l1EtL1On_(l1EtL1On), l1Etavsl1PhiL1On_(l1Etavsl1PhiL1On),
            NL1Off_(NL1Off), offEtL1Off_(offEtL1Off), offEtavsoffPhiL1Off_(offEtavsoffPhiL1Off),
            NOnOff_(NOnOff), offEtOnOff_(offEtOnOff), offEtavsoffPhiOnOff_(offEtavsoffPhiOnOff),
            NL1OnUM_(NL1OnUM), l1EtL1OnUM_(l1EtL1OnUM), l1Etavsl1PhiL1OnUM_(l1Etavsl1PhiL1OnUM),
            NL1OffUM_(NL1OffUM), offEtL1OffUM_(offEtL1OffUM), offEtavsoffPhiL1OffUM_(offEtavsoffPhiL1OffUM),
            NOnOffUM_(NOnOffUM), offEtOnOffUM_(offEtOnOffUM), offEtavsoffPhiOnOffUM_(offEtavsoffPhiOnOffUM),
            offDRL1Off_(offDRL1Off), 
            offDROnOff_(offDROnOff), 
            l1DRL1On_(l1DRL1On),
            filters_(filters),
            ptmin_(ptmin), ptmax_(ptmax)
        {
        };

        bool operator==(const std::string& v) 
        {
          return v==filterName_;
        }

        bool operator!=(const std::string& v) 
        {
          return v!=filterName_;
        }

        float getPtMin() const { return ptmin_; }
        float getPtMax() const { return ptmax_; }
        float getHltThreshold() const { return hltThreshold_; }
        float getL1Threshold() const { return l1Threshold_; }

        std::vector< std::pair<std::string,unsigned int> > filtersAndIndices;


      private:

        int pathIndex_;
        std::string denomPathName_;
        std::string pathName_;
        std::string l1pathName_;
        int l1ModuleIndex_;
        std::string filterName_;
        std::string processName_;
        int objectType_;
        
        // we don't own this data
        MonitorElement *NOn_, *onEtOn_, *onOneOverEtOn_, *onEtavsonPhiOn_;
        MonitorElement *NOff_, *offEtOff_, *offEtavsoffPhiOff_;
        MonitorElement *NL1_, *l1EtL1_, *l1Etavsl1PhiL1_;
        MonitorElement *NL1On_, *l1EtL1On_, *l1Etavsl1PhiL1On_;
        MonitorElement *NL1Off_, *offEtL1Off_, *offEtavsoffPhiL1Off_;
        MonitorElement *NOnOff_, *offEtOnOff_, *offEtavsoffPhiOnOff_;
        MonitorElement *NL1OnUM_, *l1EtL1OnUM_, *l1Etavsl1PhiL1OnUM_;
        MonitorElement *NL1OffUM_, *offEtL1OffUM_, *offEtavsoffPhiL1OffUM_;
        MonitorElement *NOnOffUM_, *offEtOnOffUM_, *offEtavsoffPhiOnOffUM_;
        MonitorElement *offDRL1Off_, *offDROnOff_, *l1DRL1On_;
        MonitorElement *filters_;
        
        float ptmin_, ptmax_;
        float hltThreshold_, l1Threshold_;
        
        const int index() { 
          return pathIndex_;
        }
        const int type() { 
          return objectType_;
        }


     };
     

   public:

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

      PathInfoCollection hltPathsDiagonal_;

};





#endif
