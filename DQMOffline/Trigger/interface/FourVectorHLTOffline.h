#ifndef FOURVECTORHLTOFFLINE_H
#define FOURVECTORHLTOFFLINE_H
// -*- C++ -*-
//
// Package:    FourVectorHLTOffline
// Class:      FourVectorHLTOffline
// 
/**\class FourVectorHLTOffline FourVectorHLTOffline.cc 

 Description: This is a DQM source meant to plot high-level HLT trigger
 quantities as stored in the HLT results object TriggerResults

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeffrey Berryhill
//         Created:  June 2008
// Rewritten by: Vladimir Rekovic
//         Date:  May 2009
//
// $Id: FourVectorHLTOffline.h,v 1.67 2011/06/20 10:11:54 bjk Exp $
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
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

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

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TauReco/interface/CaloTauFwd.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "RecoJets/JetProducers/interface/JetIDHelper.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"

/* MC
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
*/
#include "DataFormats/Math/interface/deltaR.h"
#include  "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DQMServices/Core/interface/MonitorElement.h"



#include <iostream>
#include <fstream>
#include <vector>

namespace edm {
   class TriggerNames;
}

typedef std::multimap<float,int> fimmap ;
typedef std::set<fimmap , std::less<fimmap> > mmset;

class FourVectorHLTOffline : public edm::EDAnalyzer {

   public:
      explicit FourVectorHLTOffline(const edm::ParameterSet&);
      ~FourVectorHLTOffline();

      void cleanDRMatchSet(mmset& tempSet);

      edm::Handle<trigger::TriggerEvent> fTriggerObj;
      edm::Handle<edm::TriggerResults> fTriggerResults;
      edm::Handle<reco::BeamSpot> fBeamSpotHandle;

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // BeginRun
      void beginRun(const edm::Run& run, const edm::EventSetup& c);

      // EndRun
      void endRun(const edm::Run& run, const edm::EventSetup& c);
      void fillHltMatrix(const edm::TriggerNames & triggerNames);
      void setupHltMatrix(const std::string& label, std::vector<std::string> &  paths);

      void setupHltLsPlots();
      void setupHltBxPlots();
      void countHLTPathHitsEndLumiBlock(const int & lumi);
      void countHLTGroupHitsEndLumiBlock(const int & lumi);
      void countHLTGroupL1HitsEndLumiBlock(const int & lumi);
      void countHLTGroupBXHitsEndLumiBlock(const int & lumi);
      int getTriggerTypeParsePathName(const std::string & pathname);
      const std::string getL1ConditionModuleName(const std::string & pathname);
      bool hasL1Passed(const std::string & pathname, const edm::TriggerNames & triggerNames);
      bool hasHLTPassed(const std::string & pathname, const edm::TriggerNames& triggerNames);
      int getHltThresholdFromName(const std::string & pathname);

      void selectMuons(const edm::Handle<reco::MuonCollection> & muonHandle);
      bool isVBTFMuon(const reco::Muon& muon);
      void selectElectrons(const edm::Event& iEvent, const edm::EventSetup& iSetup, const edm::Handle<reco::GsfElectronCollection> & eleHandle);
      void selectPhotons(const edm::Handle<reco::PhotonCollection> & phoHandle);
      void selectJets(const edm::Event& iEvent,const edm::Handle<reco::CaloJetCollection> & jetHandle);
      void selectMet(const edm::Handle<reco::CaloMETCollection> & metHandle);
      void selectTaus(const edm::Event& iEvent);
      void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);   
      void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);   
      std::string removeVersions(std::string histVersion);


      // ----------member data --------------------------- 
      int nev_;
      DQMStore * dbe_;
      bool fLumiFlag;
      bool fIsSetup;

      bool useUM;

      // JetID helper
      reco::helper::JetIDHelper *jetID;

      // Tau discriminators
      std::string tauDscrmtrLabel1_;
      std::string tauDscrmtrLabel2_;
      std::string tauDscrmtrLabel3_;

      MonitorElement* ME_HLTAll_LS;
      MonitorElement* ME_HLT_BX;
      MonitorElement* ME_HLT_CUSTOM_BX;
      std::vector<MonitorElement*> v_ME_HLTAll_LS;
      std::vector<MonitorElement*> v_ME_Total_BX;
      std::vector<MonitorElement*> v_ME_Total_BX_Norm;

      std::string pathsSummaryFolder_;
      std::string pathsSummaryHLTCorrelationsFolder_;
      std::string pathsSummaryFilterEfficiencyFolder_;
      std::string pathsSummaryFilterCountsFolder_;
      std::string pathsSummaryHLTPathsPerLSFolder_;
      std::string pathsIndividualHLTPathsPerLSFolder_;
      std::string pathsSummaryHLTPathsPerBXFolder_;
      std::string fCustomBXPath;

      std::vector<std::string> fGroupName;

      reco::MuonCollection * fSelectedMuons;
      edm::Handle<reco::MuonCollection> fSelMuonsHandle;

      reco::GsfElectronCollection * fSelectedElectrons;
      edm::Handle<reco::GsfElectronCollection> fSelElectronsHandle;

      reco::PhotonCollection * fSelectedPhotons;
      edm::Handle<reco::PhotonCollection> fSelPhotonsHandle;

      reco::CaloJetCollection * fSelectedJets;
      edm::Handle<reco::CaloJetCollection> fSelJetsHandle;

      reco::CaloMETCollection * fSelectedMet;
      edm::Handle<reco::CaloMETCollection> fSelMetHandle;

      //reco::CaloTauCollection * fSelectedTaus;
      //edm::Handle<reco::CaloTauCollection> fSelTausHandle;
      reco::PFTauCollection * fSelectedTaus;
      edm::Handle<reco::PFTauCollection> fSelTausHandle;

      unsigned int nLS_; 
      double LSsize_ ;
      double thresholdFactor_ ;
      unsigned int referenceBX_; 
      unsigned int Nbx_; 

      bool plotAll_;
      bool doCombineRuns_;
      int currentRun_;
      
      unsigned int nBins_; 
      unsigned int nBinsDR_; 
      unsigned int nBins2D_; 
      unsigned int nBinsOneOverEt_; 
      double ptMin_ ;
      double ptMax_ ;
      double dRMax_ ;
      
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
      //////////////////////////
      double dxyCut_;
      double normalizedChi2Cut_;
      int trackerHitsCut_;
      int pixelHitsCut_;
      int muonHitsCut_;
      bool isAlsoTrackerMuon_;
      int nMatchesCut_;

      // Electron quality cuts
      //////////////////////////
      float eleMaxOver3x3_;
      // Ecal Barrel
      float dr03TkSumPtEB_;
      float dr04EcalRecHitSumEtEB_;
      float dr04HcalTowerSumEtEB_;
      float hadronicOverEmEB_;
      float deltaPhiSuperClusterTrackAtVtxEB_;
      float deltaEtaSuperClusterTrackAtVtxEB_;
      float sigmaIetaIetaEB_;
      //spikes
      float sigmaIetaIetaSpikesEB_;
      // Ecal Endcap
      float dr03TkSumPtEC_;
      float dr04EcalRecHitSumEtEC_;
      float dr04HcalTowerSumEtEC_;
      float hadronicOverEmEC_;
      float deltaPhiSuperClusterTrackAtVtxEC_;
      float deltaEtaSuperClusterTrackAtVtxEC_;
      float sigmaIetaIetaEC_;
      //spikes
      float sigmaIetaIetaSpikesEC_;

      // Jet quality cuts
      //////////////////////////
      float emEnergyFractionJet_;
      float fHPDJet_;
      int n90Jet_;

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


class BaseMonitor
{
  public:
    virtual void clearSets( void ) = 0;
    virtual void setPath(FourVectorHLTOffline::PathInfoCollection::iterator v) = 0;
    virtual void monitorOffline( void ) = 0;
    virtual void monitorL1( const int l1Index, FourVectorHLTOffline* fv) = 0;
    virtual void monitorOnline(const int hltIndex, const int l1Index, FourVectorHLTOffline* fv) = 0;

    virtual void matchL1Offline(const trigger::TriggerObject& l1FV, FourVectorHLTOffline* fv, const int& NL1, unsigned int& NL1OffUM) = 0;
    virtual void matchOnlineL1(const trigger::TriggerObject& onlineFV, const int& l1Index, FourVectorHLTOffline* fv, const int& NOn) = 0;
    virtual void matchOnlineOffline(const trigger::TriggerObject& onlineFV, FourVectorHLTOffline* fv, const int& NOn) = 0;

    virtual void fillL1Match(FourVectorHLTOffline* fv) = 0;
    virtual void fillOnlineMatch(const int l1Index, FourVectorHLTOffline* fv) = 0;

    virtual bool isTriggerType(int t) = 0;
    virtual ~BaseMonitor(){}

};

template <class T> 
class objMonData:public BaseMonitor {
public:
    objMonData() { EtaMax_= 2.5; EtMin_=3.0; GenJetsFlag_ = false; BJetsFlag_ = false; fL2MuFlag = false; }
    void setLimits(float etaMax, float etMin, float drMatch, float l1drMatch, float dRRange, float thresholdFactor) 
    {
     EtaMax_= etaMax; 
     EtMin_= etMin; 
     DRMatch_= drMatch;
     L1DRMatch_= l1drMatch;
     DRRange_ = dRRange;
     thresholdFactor_ = thresholdFactor;
    }
    void setTriggerType(std::vector<int> trigType) { triggerType_ = trigType; }
    void pushTriggerType(int trigType) { triggerType_.push_back(trigType); }
    void setL1TriggerType(std::vector<int> trigType) { l1triggerType_ = trigType; }
    void pushL1TriggerType(int trigType) { l1triggerType_.push_back(trigType); }
    void setPath(FourVectorHLTOffline::PathInfoCollection::iterator v) { v_ = v; }
    void setReco(edm::Handle<T> offColl) { offColl_ = offColl; }
    void setRecoB(edm::Handle<reco::JetTagCollection> offCollB) { offCollB_ = offCollB; }
    void setRecoMu(edm::Handle<reco::MuonCollection> offCollMu) { offCollMu_ = offCollMu; }
    void setRecoEle(edm::Handle<reco::GsfElectronCollection> offCollEle) { offCollEle_ = offCollEle; }



    // Monitor methods

    void monitorOffline();
    void monitorL1(const int l1Index, FourVectorHLTOffline* fv);
    void monitorOnline(const int hltIndex, const int l1Index, FourVectorHLTOffline* fv);
    void matchL1Offline(const trigger::TriggerObject& l1FV, FourVectorHLTOffline* fv, const int& NL1, unsigned int& NL1OffUM);
    void matchOnlineL1(const trigger::TriggerObject& onlineFV, const int& l1Index, FourVectorHLTOffline* fv, const int& NOn);
    void matchOnlineOffline(const trigger::TriggerObject& onlineFV, FourVectorHLTOffline* fv, const int& NOn);
    void fillOnlineMatch(const int l1Index, FourVectorHLTOffline* fv);
    void fillOnOffMatch(FourVectorHLTOffline* fv);
    void fillOnL1Match(const int l1Index, FourVectorHLTOffline* fv);
    void fillL1Match(FourVectorHLTOffline* fv);
    void fillL1OffMatch(FourVectorHLTOffline* fv);

    void clearSets();

    bool isTriggerType(int t);
    bool isL1TriggerType(int t);

    mmset L1OffDRMatchSet;
    mmset L1MCDRMatchSet;
    mmset OnOffDRMatchSet;
    mmset OnMCDRMatchSet;
    mmset OnL1DRMatchSet;
    mmset OffMCDRMatchSet;


    void setBJetsFlag(bool flag) 
    { 
      BJetsFlag_ = flag; 
    }
    void setL2MuFlag(bool flag) 
    { 
      fL2MuFlag = flag; 
    }
    

private:

    int   pdgId_;
    int   pdgStatus_;

    float EtaMax_;
    float EtMin_;

    float DRMatch_;
    float L1DRMatch_;
    float DRRange_;
    float thresholdFactor_;

    bool GenJetsFlag_;
    bool BJetsFlag_;
    bool fL2MuFlag;

    std::vector<int> triggerType_;
    std::vector<int> l1triggerType_;

    edm::Handle<T> offColl_;
    edm::Handle<reco::JetTagCollection> offCollB_;
    edm::Handle<reco::MuonCollection> offCollMu_;
    edm::Handle<reco::GsfElectronCollection> offCollEle_;

    FourVectorHLTOffline::PathInfoCollection::iterator v_;

};


template <class T> 
bool objMonData<T>::isTriggerType(int t)
{
  bool rc = false;

  for(std::vector<int>::const_iterator it = triggerType_.begin(); it != triggerType_.end(); ++it)
  {

   if(t == *it) { rc = true; break; }

  } // end for

  if (t==0) rc = true;

  return rc;

}


template <class T> 
bool objMonData<T>::isL1TriggerType(int t)
{
  bool rc = false;

  for(std::vector<int>::const_iterator it = l1triggerType_.begin(); it != l1triggerType_.end(); ++it)
  {

   if(fabs(t) == fabs(*it)) { rc = true; break; }

  } // end for

  return rc;

}




template <class T> 
void objMonData<T>::monitorOffline()
{

 if(! isTriggerType(v_->getObjectType()) ) return;

 unsigned int NOff = 0;

 if( offCollB_.isValid()) {
  typedef typename reco::JetTagCollection::const_iterator const_iterator;
  for( const_iterator iter = offCollB_->begin(), iend = offCollB_->end(); iter != iend; ++iter )
  {

    float recoEta = (*iter).first->eta();
    float recoPhi = (*iter).first->phi();
    float recoPt = (*iter).first->pt();


    if (fabs(recoEta) <= EtaMax_ && recoPt >=  EtMin_ )
    {
     
       NOff++;
       v_->getOffEtOffHisto()->Fill(recoPt);
       if(recoPt >= thresholdFactor_*v_->getHltThreshold())
       v_->getOffEtaVsOffPhiOffHisto()->Fill(recoEta, recoPhi);

    }
    /*
    else {

      continue;

    }
    */

  }

 }
 else if(offCollEle_.isValid()) {

  typedef typename reco::GsfElectronCollection::const_iterator const_iterator;
  for( const_iterator iter = offCollEle_->begin(), iend = offCollEle_->end(); iter != iend; ++iter )
  {

   if (fabs(iter->eta()) <= EtaMax_ && iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()) >=  EtMin_ )
   {

     NOff++;
     v_->getOffEtOffHisto()->Fill(iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()));

     if(iter->pt() >= thresholdFactor_*v_->getHltThreshold())
     v_->getOffEtaVsOffPhiOffHisto()->Fill(iter->eta(), iter->phi());

   }
   /*
   else {

     continue;

   }
   */

  }

 } // end else if
 else if(offColl_.isValid()) {

  typedef typename T::const_iterator const_iterator;
  for( const_iterator iter = offColl_->begin(), iend = offColl_->end(); iter != iend; ++iter )
  {

   if (fabs(iter->eta()) <= EtaMax_ && iter->pt() >=  EtMin_ )
   {

     NOff++;
     v_->getOffEtOffHisto()->Fill(iter->pt());

     if(iter->pt() >= thresholdFactor_*v_->getHltThreshold())
     v_->getOffEtaVsOffPhiOffHisto()->Fill(iter->eta(), iter->phi());

   }
   /*
   else {

     continue;

   }
   */

  }

 } // end else if

 if(NOff>0)v_->getNOffHisto()->Fill(NOff);

}


template <class T> 
void objMonData<T>::monitorL1(const int l1Index, FourVectorHLTOffline* fv)
{

  if ( l1Index >= fv->fTriggerObj->sizeFilters() ) return;

  unsigned int NL1=0;
  unsigned int NL1OffUM=0;

  const trigger::TriggerObjectCollection & toc(fv->fTriggerObj->getObjects());
	const trigger::Vids & idtype = fv->fTriggerObj->filterIds(l1Index);
	const trigger::Keys & l1k = fv->fTriggerObj->filterKeys(l1Index);
	bool l1accept = l1k.size() > 0;

  if(!l1accept) return;

  trigger::Vids::const_iterator idtypeiter = idtype.begin(); 

  for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) {

   trigger::TriggerObject l1FV = toc[*l1ki];

   if(isL1TriggerType(*idtypeiter))
   {

     NL1++;

     v_->getL1EtL1Histo()->Fill(l1FV.pt());
     v_->getL1EtaVsL1PhiL1Histo()->Fill(l1FV.eta(), l1FV.phi());

     matchL1Offline(l1FV, fv, NL1, NL1OffUM);

   } // end if isL1TriggerType

   ++idtypeiter;

 } // end for l1ki

 if(NL1 > 0) v_->getNL1Histo()->Fill(NL1);
 if(NL1OffUM > 0) v_->getNL1OffUMHisto()->Fill(NL1OffUM);

}


template <class T> 
void objMonData<T>::matchL1Offline(const trigger::TriggerObject& l1FV, FourVectorHLTOffline* fv, const int& NL1, unsigned int& NL1OffUM)
{

  fimmap L1OffDRMatchMap;

  if (offCollB_.isValid()) {

    int j=0;
    typedef typename reco::JetTagCollection::const_iterator const_iterator;
    for( const_iterator iter = offCollB_->begin(), iend = offCollB_->end(); iter != iend; ++iter )
    {

      float recoEta = (*iter).first->eta();
      float recoPhi = (*iter).first->phi();
      float recoPt = (*iter).first->pt();

      if (fabs(recoEta) <= EtaMax_ && recoPt >=  EtMin_ )
      {

        // fill UM histos (no matching required)
	if (v_->getOffEtL1OffUMHisto() != 0) {
	  if(NL1 == 1) {
	    
	    NL1OffUM++;
	    v_->getOffEtL1OffUMHisto()->Fill(recoPt);
	    
	    if(recoPt >= thresholdFactor_*v_->getHltThreshold())
	      v_->getOffEtaVsOffPhiL1OffUMHisto()->Fill(recoEta,recoPhi);
	    
	  }
	}

         // make maps of matched objects
        float dR = reco::deltaR(recoEta,recoPhi,l1FV.eta(),l1FV.phi());
        if ( dR < DRRange_)
        {

          L1OffDRMatchMap.insert(std::pair<float,int>(dR,j));

        }

      }

      j++;

    }

  }
  else if (offCollMu_.isValid()) {

    int j=0;
    typedef typename reco::MuonCollection::const_iterator const_iterator;
    for( const_iterator iter = offCollMu_->begin(), iend = offCollMu_->end(); iter != iend; ++iter )
    {

      // get Eta, Phi of the MuonDetectorTrack, 
      // looking at the detector most inner Position
      // This should be close to what L1 sees
      float recoEta = iter->outerTrack()->innerPosition().eta();
      float recoPhi = iter->outerTrack()->innerPosition().phi();
      float recoPt = iter->pt();

      if (fabs(recoEta) <= EtaMax_ && recoPt >=  EtMin_ )
      {

        // fill UM histos (no matching required)
	if (v_->getOffEtL1OffUMHisto() != 0) {
	  if(NL1 == 1) {
	    
	    NL1OffUM++;
	    v_->getOffEtL1OffUMHisto()->Fill(recoPt);
	    
	    if(recoPt >= thresholdFactor_*v_->getHltThreshold())
	      v_->getOffEtaVsOffPhiL1OffUMHisto()->Fill(recoEta,recoPhi);
	    
	  }
	}

         // make maps of matched objects
        float dR = reco::deltaR(recoEta,recoPhi,l1FV.eta(),l1FV.phi());
        if ( dR < DRRange_)
        {

          L1OffDRMatchMap.insert(std::pair<float,int>(dR,j));

        }

      }

      j++;

    }

  }
  else if (offCollEle_.isValid()) {

    int j=0;
    typedef typename reco::GsfElectronCollection::const_iterator const_iterator;
    for( const_iterator iter = offCollEle_->begin(), iend = offCollEle_->end(); iter != iend; ++iter )
    {

      if (fabs(iter->eta()) <= EtaMax_ && iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()) >=  EtMin_ )
      {

	if ( v_->getOffEtL1OffUMHisto() != 0) {
	  // fill UM histos (no matching required)
	  if(NL1 == 1) {
	    
	    NL1OffUM++;
	    v_->getOffEtL1OffUMHisto()->Fill(iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()));
	    
	    if(iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()) >= thresholdFactor_*v_->getHltThreshold())
	      v_->getOffEtaVsOffPhiL1OffUMHisto()->Fill(iter->eta(),iter->phi());
	    
	  }
	}

        // make maps of matched objects
        float dR = reco::deltaR(iter->eta(),iter->phi(),l1FV.eta(),l1FV.phi());
        if ( dR < DRRange_) 
        {

         L1OffDRMatchMap.insert(std::pair<float,int>(dR,j));

        }

      }

      j++;

    }

  }
  else if (offColl_.isValid()) {

    int j=0;
    typedef typename T::const_iterator const_iterator;
    for( const_iterator iter = offColl_->begin(), iend = offColl_->end(); iter != iend; ++iter )
    {

      if (fabs(iter->eta()) <= EtaMax_ && iter->pt() >=  EtMin_ )
      {

        // fill UM histos (no matching required)
	if (v_->getOffEtL1OffUMHisto()!= 0 ) {
	  if(NL1 == 1) {
	    
	    NL1OffUM++;
	    v_->getOffEtL1OffUMHisto()->Fill(iter->pt());
	    
	    if(iter->pt() >= thresholdFactor_*v_->getHltThreshold())
	      v_->getOffEtaVsOffPhiL1OffUMHisto()->Fill(iter->eta(),iter->phi());
	    
	  }
	}

        // make maps of matched objects
        float dR = reco::deltaR(iter->eta(),iter->phi(),l1FV.eta(),l1FV.phi());
        if ( dR < DRRange_) 
        {

         L1OffDRMatchMap.insert(std::pair<float,int>(dR,j));

        }

      }

      j++;

    }

  }
  if(! L1OffDRMatchMap.empty())  L1OffDRMatchSet.insert(L1OffDRMatchMap);

}


template <class T> 
void objMonData<T>::monitorOnline(const int hltIndex, const int l1Index, FourVectorHLTOffline* fv)
{

  if(! isTriggerType(v_->getObjectType()) ) return;

  // Get keys of objects passed by the last filter
  const trigger::Keys & k = fv->fTriggerObj->filterKeys(hltIndex);

  const trigger::TriggerObjectCollection & toc(fv->fTriggerObj->getObjects());

  unsigned int NOn=0;

  // Loop over HLT objects
  for (trigger::Keys::const_iterator ki = k.begin(); ki !=k.end(); ++ki ) {

	  trigger::TriggerObject onlineFV = toc[*ki];
	
	  NOn++;    
	
	  v_->getOnEtOnHisto()->Fill(onlineFV.pt());
	  v_->getOnOneOverEtOnHisto()->Fill(1./onlineFV.pt());
	  v_->getOnEtaVsOnPhiOnHisto()->Fill(onlineFV.eta(), onlineFV.phi());
	
	  matchOnlineL1(onlineFV,l1Index, fv, NOn);
	  matchOnlineOffline(onlineFV,fv, NOn);

  } // end loop over HLT objects
  
  if(NOn>0) v_->getNOnHisto()->Fill(NOn);

}

template <class T> 
void objMonData<T>::matchOnlineL1(const trigger::TriggerObject& onlineFV, const int& l1Index, FourVectorHLTOffline* fv, const int& NOn)
{

  if ( l1Index >= fv->fTriggerObj->sizeFilters() ) return;

  unsigned int NOnL1UM=0;

  const trigger::TriggerObjectCollection & toc(fv->fTriggerObj->getObjects());
	const trigger::Vids & idtype = fv->fTriggerObj->filterIds(l1Index);
	const trigger::Keys & l1k = fv->fTriggerObj->filterKeys(l1Index);

  fimmap OnL1DRMatchMap;
  int j=0;
  trigger::Vids::const_iterator idtypeiter = idtype.begin(); 
  for (trigger::Keys::const_iterator l1ki = l1k.begin(); l1ki !=l1k.end(); ++l1ki ) 
  {

      if(isL1TriggerType(*idtypeiter))
      {

        trigger::TriggerObject l1FV = toc[*l1ki];

        // fill UM histos (no matching required)
	if (v_->getL1EtL1OnUMHisto() != 0) {
	  if(NOn == 1) {
	    
	    NOnL1UM++;
	    v_->getL1EtL1OnUMHisto()->Fill(l1FV.pt());
	    v_->getL1EtaVsL1PhiL1OnUMHisto()->Fill(l1FV.eta(),l1FV.phi());
	    
	  }
	}
	  
         float dR = reco::deltaR(l1FV.eta(),l1FV.phi(),onlineFV.eta(),onlineFV.phi());

         if ( dR < DRRange_) 
         {

           OnL1DRMatchMap.insert(std::pair<float,int>(dR,j));

         }

       } // end if isL1TriggerType

       ++idtypeiter;
       j++;

  } // end for

  if(! OnL1DRMatchMap.empty()) OnL1DRMatchSet.insert(OnL1DRMatchMap);

}


template <class T> 
void objMonData<T>::matchOnlineOffline(const trigger::TriggerObject& onlineFV, FourVectorHLTOffline* fv, const int& NOn)
{

  unsigned int NOnOffUM=0;

  fimmap OnOffDRMatchMap;

  if (offCollB_.isValid()) {

     int j=0;
     typedef typename reco::JetTagCollection::const_iterator const_iterator;
     for( const_iterator iter = offCollB_->begin(), iend = offCollB_->end(); iter != iend; ++iter )
     {

       float recoEta = (*iter).first->eta();
       float recoPhi = (*iter).first->phi();
       float recoPt = (*iter).first->pt();

       if (fabs(recoEta) <= EtaMax_ && recoPt >=  EtMin_ )
       {


         // fill UM histos (no matching required)
	 if (v_->getOffEtOnOffUMHisto() != 0) {
	   if(NOn == 1) {
	     
	     NOnOffUM++;
	     v_->getOffEtOnOffUMHisto()->Fill(recoPt);
	     
	     if(recoPt >= thresholdFactor_*v_->getHltThreshold())
	       v_->getOffEtaVsOffPhiOnOffUMHisto()->Fill(recoEta,recoPhi);
	     
	   }
	 }

          // make maps of matched objects
         float dR = reco::deltaR(recoEta,recoPhi,onlineFV.eta(),onlineFV.phi());
         if ( dR < DRRange_)
         {

           OnOffDRMatchMap.insert(std::pair<float,int>(dR,j));

         }

       }

       j++;

     }

  }
  else if (offCollMu_.isValid() && fL2MuFlag) {

    int j=0;
    typedef typename reco::MuonCollection::const_iterator const_iterator;
    for( const_iterator iter = offCollMu_->begin(), iend = offCollMu_->end(); iter != iend; ++iter )
    {

      // get Eta, Phi of the MuonDetectorTrack, 
      // looking at the detector most inner Position
      // This should be close to what L1 sees
      float recoEta = iter->outerTrack()->innerPosition().eta();
      float recoPhi = iter->outerTrack()->innerPosition().phi();
      float recoPt = iter->pt();

      if (fabs(recoEta) <= EtaMax_ && recoPt >=  EtMin_ )
      {
         // fill UM histos (no matching required)
	if (v_->getOffEtOnOffUMHisto() != 0) {
	  if(NOn == 1) {
	    
	    NOnOffUM++;
	    v_->getOffEtOnOffUMHisto()->Fill(iter->pt());
	    
	    if(recoPt >= thresholdFactor_*v_->getHltThreshold())
	      v_->getOffEtaVsOffPhiOnOffUMHisto()->Fill(iter->eta(),iter->phi());
	    
	  }
	}

          // make maps of matched objects
         float dR = reco::deltaR(recoEta,recoPhi,onlineFV.eta(),onlineFV.phi());
         if ( dR < DRRange_)
         {

           OnOffDRMatchMap.insert(std::pair<float,int>(dR,j));

         }

       }

       j++;


     }

  }
  else if (offCollEle_.isValid()) {

     int j=0;

     typedef typename reco::GsfElectronCollection::const_iterator const_iterator;
     for( const_iterator iter = offCollEle_->begin(), iend = offCollEle_->end(); iter != iend; ++iter )
     {

       if (fabs(iter->eta()) <= EtaMax_ && iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()) >=  EtMin_ )
       {

         // fill UM histos (no matching required)
	 if (v_->getOffEtOnOffUMHisto() != 0) {
	   if(NOn == 1) {
	     
	     NOnOffUM++;
	     v_->getOffEtOnOffUMHisto()->Fill(iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()));
	     
	     if(iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()) >= thresholdFactor_*v_->getHltThreshold())
	       v_->getOffEtaVsOffPhiOnOffUMHisto()->Fill(iter->eta(),iter->phi());
	     
	   }
	 }

          // make maps of matched objects
         float dR = reco::deltaR(iter->eta(),iter->phi(),onlineFV.eta(),onlineFV.phi());
         if ( dR < DRRange_)
         {

           OnOffDRMatchMap.insert(std::pair<float,int>(dR,j));

         }

       }

       j++;


     }

  }
  else if (offColl_.isValid()) {

     int j=0;

     typedef typename T::const_iterator const_iterator;
     for( const_iterator iter = offColl_->begin(), iend = offColl_->end(); iter != iend; ++iter )
     {

       if (fabs(iter->eta()) <= EtaMax_ && iter->pt() >=  EtMin_ )
       {

         // fill UM histos (no matching required)
	 if (v_->getOffEtOnOffUMHisto() != 0) {
	   if(NOn == 1) {
	     
	     NOnOffUM++;
	     v_->getOffEtOnOffUMHisto()->Fill(iter->pt());
	     
	     if(iter->pt() >= thresholdFactor_*v_->getHltThreshold())
	       v_->getOffEtaVsOffPhiOnOffUMHisto()->Fill(iter->eta(),iter->phi());
	     
	   }
	 }

          // make maps of matched objects
         float dR = reco::deltaR(iter->eta(),iter->phi(),onlineFV.eta(),onlineFV.phi());
         if ( dR < DRRange_)
         {

           OnOffDRMatchMap.insert(std::pair<float,int>(dR,j));

         }

       }

       j++;


     }

  }

  if(! OnOffDRMatchMap.empty())  OnOffDRMatchSet.insert(OnOffDRMatchMap);
 
}

template <class T> 
void objMonData<T>::fillL1OffMatch(FourVectorHLTOffline* fv)
{

  float NL1Off=0;

  if(L1OffDRMatchSet.size() > 1) {
  
    LogDebug("FourVectorHLTOffline") << " Cleaning L1Off mmset" << std::endl;
    fv->cleanDRMatchSet(L1OffDRMatchSet);

  }
  // clean the set L1-Off
  // now fill histos
  for ( mmset::iterator setIter = L1OffDRMatchSet.begin( ); setIter != L1OffDRMatchSet.end( ); setIter++ ) 
  {

       fimmap tempMap = *setIter;
         
       fimmap::iterator it = tempMap.begin(); 
       int i  = (*it).second ;
       float dR = (*it).first;
       v_->getOffDRL1OffHisto()->Fill(dR);

       if (dR > L1DRMatch_) continue;
       if( offCollB_.isValid()) {

         typedef typename reco::JetTagCollection::const_iterator const_iterator;
         const_iterator iter = offCollB_->begin();
         for (int count = 0; count < i; count++) iter++;


         NL1Off++;
         v_->getOffEtL1OffHisto()->Fill((*iter).first->pt());
         if((*iter).first->pt() >= thresholdFactor_*v_->getHltThreshold())
         v_->getOffEtaVsOffPhiL1OffHisto()->Fill((*iter).first->eta(),(*iter).first->phi());


      }
      else if( offCollMu_.isValid()) {

        typedef typename reco::MuonCollection::const_iterator const_iterator;
        const_iterator iter = offCollMu_->begin();
        for (int count = 0; count < i; count++) iter++;


        NL1Off++;
        v_->getOffEtL1OffHisto()->Fill(iter->pt());
        if(iter->pt() >= thresholdFactor_*v_->getHltThreshold())
        v_->getOffEtaVsOffPhiL1OffHisto()->Fill(iter->outerTrack()->innerPosition().eta(),iter->outerTrack()->innerPosition().phi());

      }
      else if( offCollEle_.isValid()) {

         typedef typename reco::GsfElectronCollection::const_iterator const_iterator;
         const_iterator iter = offCollEle_->begin();
         for (int count = 0; count < i; count++) iter++;


         NL1Off++;
         v_->getOffEtL1OffHisto()->Fill(iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()));
         if(iter->pt() >= thresholdFactor_*v_->getHltThreshold())
         v_->getOffEtaVsOffPhiL1OffHisto()->Fill(iter->eta(),iter->phi());

      }
      else if( offColl_.isValid()) {

         typedef typename T::const_iterator const_iterator;
         const_iterator iter = offColl_->begin();
         for (int count = 0; count < i; count++) iter++;


         NL1Off++;
         v_->getOffEtL1OffHisto()->Fill(iter->pt());
         if(iter->pt() >= thresholdFactor_*v_->getHltThreshold())
         v_->getOffEtaVsOffPhiL1OffHisto()->Fill(iter->eta(),iter->phi());

      }

  }

  if(NL1Off > 0) v_->getNL1OffHisto()->Fill(NL1Off);

}


template <class T> 
void objMonData<T>::fillOnOffMatch(FourVectorHLTOffline* fv)
{

  unsigned int NOnOff=0;

  // clean the set L1-Off
  if(OnOffDRMatchSet.size() > 1){
  
    LogDebug("FourVectorHLTOffline") << " Cleaning OnOff mmset" << std::endl;
    fv->cleanDRMatchSet(OnOffDRMatchSet);

  }
  // now fill histos
  for ( mmset::iterator setIter = OnOffDRMatchSet.begin( ); setIter != OnOffDRMatchSet.end( ); setIter++ ) 
  {


       fimmap tempMap = *setIter;
         
       fimmap::iterator it = tempMap.begin(); 
       int i  = (*it).second ;
       float dR = (*it).first;
       v_->getOffDROnOffHisto()->Fill(dR);
       

       if (dR > DRMatch_) continue;

       if( offCollB_.isValid()) {


         typedef typename reco::JetTagCollection::const_iterator const_iterator;
         const_iterator iter = offCollB_->begin();
         for (int count = 0; count < i; count++) iter++;


         NOnOff++;
         v_->getOffEtOnOffHisto()->Fill((*iter).first->pt());
         if((*iter).first->pt() >= thresholdFactor_*v_->getHltThreshold())
         v_->getOffEtaVsOffPhiOnOffHisto()->Fill((*iter).first->eta(),(*iter).first->phi());

       }
       else if( offCollMu_.isValid() && fL2MuFlag) {

         typedef typename reco::MuonCollection::const_iterator const_iterator;
         const_iterator iter = offCollMu_->begin();
         for (int count = 0; count < i; count++) iter++;


         NOnOff++;
         v_->getOffEtOnOffHisto()->Fill(iter->pt());
         if(iter->pt() >= thresholdFactor_*v_->getHltThreshold())
         v_->getOffEtaVsOffPhiOnOffHisto()->Fill(iter->outerTrack()->innerPosition().eta(),iter->outerTrack()->innerPosition().phi());

      }
       else if( offCollEle_.isValid()) {

         typedef typename reco::GsfElectronCollection::const_iterator const_iterator;
         const_iterator iter = offCollEle_->begin();
         for (int count = 0; count < i; count++) iter++;

         NOnOff++;
         v_->getOffEtOnOffHisto()->Fill(iter->superCluster()->energy()*sin(iter->superCluster()->position().Theta()));
         if(iter->superCluster()->energy()*fabs(sin(iter->superCluster()->position().Theta())) >= thresholdFactor_*v_->getHltThreshold())
         v_->getOffEtaVsOffPhiOnOffHisto()->Fill(iter->eta(),iter->phi());

       }
       else if( offColl_.isValid()) {

         typedef typename T::const_iterator const_iterator;
         const_iterator iter = offColl_->begin();
         for (int count = 0; count < i; count++) iter++;

         NOnOff++;
         v_->getOffEtOnOffHisto()->Fill(iter->pt());
         if(iter->pt() >= thresholdFactor_*v_->getHltThreshold())
         v_->getOffEtaVsOffPhiOnOffHisto()->Fill(iter->eta(),iter->phi());

       }

  }

  v_->getNOnOffHisto()->Fill(NOnOff);

}


template <class T> 
void objMonData<T>::fillOnL1Match(const int l1Index, FourVectorHLTOffline* fv)
{

  const trigger::TriggerObjectCollection & toc(fv->fTriggerObj->getObjects());
	const trigger::Keys & l1k = fv->fTriggerObj->filterKeys(l1Index);

  unsigned int NOnL1=0;

  // clean the set On-L1
  if(OnL1DRMatchSet.size() > 1) {
  
    LogDebug("FourVectorHLTOffline") << " Cleaning L1On mmset" << std::endl;
    fv->cleanDRMatchSet(OnL1DRMatchSet);

  }
  // now fill histos
  for ( mmset::iterator setIter = OnL1DRMatchSet.begin( ); setIter != OnL1DRMatchSet.end( ); setIter++ ) 
  {

    fimmap tempMap = *setIter;
      
    fimmap::iterator it = tempMap.begin(); 
    int i  = (*it).second ;
    float dR = (*it).first;
    v_->getL1DROnL1Histo()->Fill(dR);

    if (dR > L1DRMatch_) continue;

    trigger::Keys::const_iterator l1ki = l1k.begin();
    for (int count = 0; count < i; count++) l1ki++;

    NOnL1++;
    v_->getL1EtL1OnHisto()->Fill(toc[*l1ki].pt());
    v_->getL1EtaVsL1PhiL1OnHisto()->Fill(toc[*l1ki].eta(),toc[*l1ki].phi());

  }

  v_->getNL1OnHisto()->Fill(NOnL1);

}

template <class T> 
void objMonData<T>::fillOnlineMatch(const int l1Index, FourVectorHLTOffline* fv)
{

  if(! isTriggerType(v_->getObjectType()) ) return;
  fillOnOffMatch(fv);

  if ( l1Index >= fv->fTriggerObj->sizeFilters() ) return;
  fillOnL1Match(l1Index, fv);

}

template <class T> 
void objMonData<T>::fillL1Match(FourVectorHLTOffline* fv)
{

  fillL1OffMatch(fv);

}

template <class T> 
void objMonData<T>::clearSets()
{

   L1OffDRMatchSet.clear();
   L1MCDRMatchSet.clear();
   OnOffDRMatchSet.clear();
   OnMCDRMatchSet.clear();
   OnL1DRMatchSet.clear();
   OffMCDRMatchSet.clear();

}



#endif
