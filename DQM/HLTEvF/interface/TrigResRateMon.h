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
// $Id: TrigResRateMon.h,v 1.11 2011/09/21 16:51:09 lwming Exp $
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
#include <set>

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


  // JMS counts
  // need to fill counts per path with the prescales
  void fillCountsPerPath(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void bookCountsPerPath();
  void printCountsPerPathThisLumi();
  void clearCountsPerPath();
  void fillXsecPerDataset(const int& lumi);

  void filltestHisto(const int& lumi); //Robin
  void bookTestHisto(); //Robin
  // JMS lumi average
  void addLumiToAverage (double lumi);
  void clearLumiAverage ();
  

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
      int nStream_;  //Robin
      int nPass_;  //Robin
      bool passAny;  //Robin 
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

      std::string testPathsFolder_;  //Robin
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


  // JMS Keep track of rates and average lumi
  std::vector<unsigned> rawCountsPerPD; //Robin 
  std::vector<unsigned> rawCountsPerPath; //Robin
  std::vector<unsigned> finalCountsPerPath;  //Robin
  double averageInstLumi;
  double averageInstLumi3LS;
  MonitorElement * meAverageLumiPerLS;

  //Robin---
  int64_t TotalDroppedCounts ;
  MonitorElement * meDiagnostic;
  MonitorElement * meCountsDroppedPerLS;
  MonitorElement * meCountsPassPerLS;
  MonitorElement * meCountsStreamPerLS;
  MonitorElement * meXsecStreamPerLS;
  //  MonitorElement * meXsecPerLS;
  //  MonitorElement * meXsec;
  //  MonitorElement * meXsecPerIL;

  MonitorElement * meXsecPerTestPath;
  std::vector<MonitorElement*> v_ME_XsecPerLS;
  std::vector<MonitorElement*> v_ME_CountsPerLS;
  std::vector<MonitorElement*> v_ME_Xsec;

  // JMS Mask off some paths so that they don't mess with your plots

  std::vector< std::string > maskedPaths_; 
  std::vector< std::string > testPaths_; //Robin

  // JMS calcuate a reference cross section
  // then scale
  std::string referenceTrigInput_;
  std::string referenceTrigName_;
  unsigned referenceTrigIndex_;
  bool foundReferenceTrigger_;
  unsigned referenceTrigCountsPS_;
  //  unsigned testTrigCountsPS_; //Robin
  void findReferenceTriggerIndex();
  //unsigned referenceTrigCounts_;
  
  
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


      // This variable contains the list of PD, then the list of paths per PD
  
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


  // create a class that can store all the strings
  // associated with a primary dataset
  // the 
  class DatasetInfo {

  public:
    std::string datasetName;
    std::vector<std::string> pathNames;
    std::set<std::string> maskedPaths;
    std::set<std::string> testPaths; //Robin
    // this tells you the name of the monitor element
    // that has the counts per path saved
    std::string countsPerPathME_Name;

    // name of monitor element that has xsec per path saved
    std::string xsecPerPathME_Name;

    // name of monitor element that has xsec per path saved
    std::string scaledXsecPerPathME_Name;

    std::string ratePerLSME_Name;
    // this tells you the name of the ME that has
    // raw counts (no prescale accounting)
    // for each path

    std::string rawCountsPerPathME_Name;

    // counts per path

    std::map<std::string, unsigned int> countsPerPath;
    
    //empty default constructor
    DatasetInfo () {};

    // do we need a copy constructor?

    // function to set the paths and
    // create zeroed counts per path
    
    void setPaths (std::vector<std::string> inputPaths){
      pathNames = inputPaths;
      for (std::vector<std::string>::const_iterator iPath = pathNames.begin();
           iPath != pathNames.end();
           iPath++) {
        countsPerPath[*iPath] = 0;
      }
    }//end setPaths

    void clearCountsPerPath () {
      std::map<std::string, unsigned int>::iterator iCounts;
      for (iCounts = countsPerPath.begin();
           iCounts != countsPerPath.end();
           iCounts++){
        iCounts->second = 0;
      }
      
    }// end clearCountsPerPath

    // put this here so that external people
    // don't care how you store counts
    void incrementCountsForPath (std::string targetPath){
      countsPerPath[targetPath]++;
    }
    
    void incrementCountsForPath (std::string targetPath, unsigned preScale){
      countsPerPath[targetPath] += preScale;
    }
    
    void printCountsPerPath () const {
      std::map<std::string, unsigned int>::const_iterator iCounts;
      for (iCounts = countsPerPath.begin();
           iCounts != countsPerPath.end();
           iCounts++){
        std::cout << datasetName
                  << "   " << iCounts->first
                  << "   " << iCounts->second
                  << std::endl;
      }
      
    }// end clearCountsPerPath


    void fillXsecPlot (MonitorElement * myXsecPlot, double currentInstLumi, double secondsPerLS, double referenceXSec) {
      // this will put the reference cross section in the denominator
      fillXsecPlot( myXsecPlot, currentInstLumi*referenceXSec, secondsPerLS);
    }
    
    void fillXsecPlot (MonitorElement * myXsecPlot, double currentInstLumi, double secondsPerLS) {

      
      for (unsigned iPath = 0;
           iPath < pathNames.size();
           iPath++) {
        std::string thisPathName = pathNames[iPath];
        unsigned thisPathCounts = countsPerPath[thisPathName];

        // if this is a masked path, then skip it        
        if (maskedPaths.find(thisPathName) != maskedPaths.end()) {
          //std::cout << "Path " << thisPathName << " is masked, not filling it " << std::endl;
          continue;
        }
	///////
	TString thisName = thisPathName.c_str();
	if ( thisName.Contains("L1") || thisName.Contains("L2") ){
	  continue;
	}

        double xsec = 1.0;
	// what should we fill when averageLumi == 0 ???
        if (currentInstLumi > 0) {
          xsec = thisPathCounts / (currentInstLumi*secondsPerLS);
        } else if ( secondsPerLS > 0) {
	  xsec = (9e-10) ;
	  //          xsec = thisPathCounts / secondsPerLS;
        } else {
	  xsec = (9e-20) ;
	  //          xsec = thisPathCounts;
        }

	myXsecPlot->Fill(iPath, xsec);
        //Robin ???
	//	if (currentInstLumi > 0)  myXsecPlot->Fill(iPath, xsec);
        //std::cout << datasetName << "  " << thisPathName << " filled with xsec  " << xsec << std::endl;
        
      }
    } // end fill Xsec plot

    void fillRawCountsForPath (MonitorElement * myRawCountsPlot, std::string pathName) {
      TH1F* tempRawCounts = myRawCountsPlot->getTH1F();
      int binNumber = tempRawCounts->GetXaxis()->FindBin(pathName.c_str());
      if (binNumber > 0) {
        tempRawCounts->Fill(binNumber);
      } else {
        //std::cout << "Problem finding bin " << pathName << " in plot " << tempRawCounts->GetTitle() << std::endl;
      }
    }// end fill RawCountsForPath

    void setMaskedPaths (std::vector<std::string> inputPaths) {
      for (unsigned i=0; i < inputPaths.size(); i++) {
        std::string maskSubString = inputPaths[i];
        for (unsigned j=0; j < pathNames.size(); j++) {
          // If a path in the DS contains a masked substring
          // Then mask that path
          //
          std::string candidateForRemoval = pathNames[j];
          TString pNameTS (candidateForRemoval);
          if ( pNameTS.Contains(maskSubString)){
            
            maskedPaths.insert(candidateForRemoval);
          } // end if path contains substring
        }// end for each path in ds
      }
    }// end setMaskedPaths

    void printMaskedPaths () {
      std::cout << "========  Printing masked paths for " << datasetName  <<" ======== " << std::endl;
      for ( std::set<std::string>::const_iterator iMask = maskedPaths.begin();
            iMask != maskedPaths.end();
            iMask++) {
        std::cout << (*iMask) << std::endl; 
      }
      std::cout << "======DONE PRINTING=====" << std::endl;
    }
    
  };

  // create a vector of the information
    std::vector<DatasetInfo> primaryDataSetInformation;

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
