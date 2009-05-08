#ifndef DQMOFFLINE_TRIGGER_JETMETHLTOFFLINESOURCE
#define DQMOFFLINE_TRIGGER_JETMETHLTOFFLINESOURCE

// -*- C++ -*-
//
// Package:    JetMETHLTOffline
// Class:      JetMETHLTOffline
// 
/*
 Description: This is a DQM source meant to plot high-level HLT trigger 
 quantities as stored in the HLT results object TriggerResults for the JetMET triggers
*/

//
// Originally create by:  Kenichi Hatakeyama
//                        April 2009
// Owned by:              Shabnam Jabeen
//
//

#include <memory>
#include <unistd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include <iostream>
#include <fstream>
#include <vector>

class DQMStore;
class MonitorElement;

class JetMETHLTOfflineSource : public edm::EDAnalyzer {
 
 private:
  DQMStore* dbe_; //dbe seems to be the standard name for this, I dont know why. We of course dont own it

  //--- Monitoring elements start
  MonitorElement* dqmErrsMonElem_; //monitors DQM errors (ie failing to get trigger info, etc)

  //---

  bool debug_;
  bool verbose_;

  //---
  edm::InputTag triggerResultsLabel_;
  edm::InputTag triggerSummaryLabel_;

  HLTConfigProvider hltConfig_;

  std::string processname_;

  //---
  
  edm::InputTag l1ExtraTaus_;
  edm::InputTag l1ExtraCJets_;
  edm::InputTag l1ExtraFJets_;

  edm::InputTag caloJetsTag_;
  edm::InputTag caloMETTag_;

  //---
 
  std::string  hltTag_;
  std::string dirName_;

  //---

  edm::Handle<edm::TriggerResults> triggerResults_;
  edm::TriggerNames triggerNames_; // TriggerNames class

  edm::Handle<trigger::TriggerEvent> triggerObj_;
  
  edm::Handle<reco::CaloJetCollection> calojetColl_;
  edm::Handle<reco::CaloMETCollection> calometColl_;

  //disabling copying/assignment (copying this class would be bad, mkay)
  JetMETHLTOfflineSource(const JetMETHLTOfflineSource& rhs){}
  JetMETHLTOfflineSource& operator=(const JetMETHLTOfflineSource& rhs){return *this;}

 public:
  explicit JetMETHLTOfflineSource(const edm::ParameterSet& );
  virtual ~JetMETHLTOfflineSource();
  
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  virtual std::string getNumeratorTrigger(const std::string& name);
  virtual std::string getDenominatorTrigger(const std::string& name);
  virtual std::string getTriggerEffLevel(const std::string& name);
  double getTriggerThreshold(const std::string& name);

  virtual void fillMEforEffSingleJet();
  virtual void fillMEforEffDiJetAve();
  virtual void fillMEforEffMET();
  virtual void fillMEforEffMHT();

  virtual void bookMEforEffSingleJet();
  virtual void bookMEforEffDiJetAve();
  virtual void bookMEforEffMET();
  virtual void bookMEforEffMHT();

  virtual void fillMEforMonSingleJet();
  virtual void fillMEforMonDiJetAve();
  virtual void fillMEforMonMET();
  virtual void fillMEforMonMHT();

  virtual void bookMEforMonSingleJet();
  virtual void bookMEforMonDiJetAve();
  virtual void bookMEforMonMET();
  virtual void bookMEforMonMHT();

  virtual bool isBarrel(double eta);
  virtual bool isEndCap(double eta);
  virtual bool isForward(double eta);

  virtual bool validPathHLT(std::string path);

  virtual bool isHLTPathAccepted(std::string pathName);
  virtual bool isTriggerObjectFound(std::string objectName);

 private:
  //
  //--- helper class to store the data path ----------
   class PathInfo { 
     PathInfo(): 
       denomPathName_("unset"),
       denomPathNameL1s_("unset"),
       denomPathNameHLT_("unset"),
       pathName_("unset"),
       pathNameL1s_("unset"),
       pathNameHLT_("unset")
       //processName_("unset"), 
       //objectType_(-1),
       //ptmin_(0.), 
       //ptmax_(0.)
       {}; 
   public: 
      void setHistos(  
 		    MonitorElement* const NumeratorPt,  
 		    MonitorElement* const NumeratorPtBarrel,  
 		    MonitorElement* const NumeratorPtEndCap,  
 		    MonitorElement* const NumeratorPtForward,  
 		    MonitorElement* const NumeratorEta,  
 		    MonitorElement* const NumeratorPhi,  
 		    MonitorElement* const NumeratorEtaPhi,  
 		    MonitorElement* const EmulatedNumeratorPt,  
 		    MonitorElement* const EmulatedNumeratorPtBarrel,  
 		    MonitorElement* const EmulatedNumeratorPtEndCap,  
 		    MonitorElement* const EmulatedNumeratorPtForward,  
 		    MonitorElement* const EmulatedNumeratorEta,  
 		    MonitorElement* const EmulatedNumeratorPhi,  
 		    MonitorElement* const EmulatedNumeratorEtaPhi,  
 		    MonitorElement* const DenominatorPt,  
 		    MonitorElement* const DenominatorPtBarrel,  
 		    MonitorElement* const DenominatorPtEndCap,  
 		    MonitorElement* const DenominatorPtForward,  
 		    MonitorElement* const DenominatorEta,  
 		    MonitorElement* const DenominatorPhi,  
 		    MonitorElement* const DenominatorEtaPhi, 
 		    MonitorElement* const NumeratorPtHLT,  
 		    MonitorElement* const NumeratorPtHLTBarrel,  
 		    MonitorElement* const NumeratorPtHLTEndCap,  
 		    MonitorElement* const NumeratorPtHLTForward,  
 		    MonitorElement* const NumeratorEtaHLT,  
 		    MonitorElement* const NumeratorPhiHLT,  
 		    MonitorElement* const NumeratorEtaPhiHLT,  
 		    MonitorElement* const EmulatedNumeratorPtHLT,  
 		    MonitorElement* const EmulatedNumeratorPtHLTBarrel,  
 		    MonitorElement* const EmulatedNumeratorPtHLTEndCap,  
 		    MonitorElement* const EmulatedNumeratorPtHLTForward,  
 		    MonitorElement* const EmulatedNumeratorEtaHLT,  
 		    MonitorElement* const EmulatedNumeratorPhiHLT,  
 		    MonitorElement* const EmulatedNumeratorEtaPhiHLT,  
 		    MonitorElement* const DenominatorPtHLT,  
 		    MonitorElement* const DenominatorPtHLTBarrel,  
 		    MonitorElement* const DenominatorPtHLTEndCap,  
 		    MonitorElement* const DenominatorPtHLTForward,  
 		    MonitorElement* const DenominatorEtaHLT,  
 		    MonitorElement* const DenominatorPhiHLT,  
 		    MonitorElement* const DenominatorEtaPhiHLT) 
      { 
        NumeratorPt_ 	 =NumeratorPt; 	 
        NumeratorPtBarrel_ 	 =NumeratorPtBarrel; 	 
        NumeratorPtEndCap_ 	 =NumeratorPtEndCap; 	 
        NumeratorPtForward_ 	 =NumeratorPtForward; 	 
        NumeratorEta_ 	 =NumeratorEta; 	 
        NumeratorPhi_ 	 =NumeratorPhi; 	 
        NumeratorEtaPhi_  =NumeratorEtaPhi;  
        EmulatedNumeratorPt_ 	 =EmulatedNumeratorPt; 	 
        EmulatedNumeratorPtBarrel_ 	 =EmulatedNumeratorPtBarrel; 	 
        EmulatedNumeratorPtEndCap_ 	 =EmulatedNumeratorPtEndCap; 	 
        EmulatedNumeratorPtForward_ 	 =EmulatedNumeratorPtForward; 	 
        EmulatedNumeratorEta_ 	 =EmulatedNumeratorEta; 	 
        EmulatedNumeratorPhi_ 	 =EmulatedNumeratorPhi; 	 
        EmulatedNumeratorEtaPhi_  =EmulatedNumeratorEtaPhi;  
        DenominatorPt_    =DenominatorPt; 	 
        DenominatorPtBarrel_    =DenominatorPtBarrel; 	 
        DenominatorPtEndCap_    =DenominatorPtEndCap; 	 
        DenominatorPtForward_    =DenominatorPtForward; 	 
        DenominatorEta_   =DenominatorEta; 	 
        DenominatorPhi_   =DenominatorPhi; 	 
        DenominatorEtaPhi_=DenominatorEtaPhi; 
        NumeratorPtHLT_ 	 =NumeratorPtHLT; 	 
        NumeratorPtHLTBarrel_ 	 =NumeratorPtHLTBarrel; 	 
        NumeratorPtHLTEndCap_ 	 =NumeratorPtHLTEndCap; 	 
        NumeratorPtHLTForward_ 	 =NumeratorPtHLTForward; 	 
        NumeratorEtaHLT_ 	 =NumeratorEtaHLT; 	 
        NumeratorPhiHLT_ 	 =NumeratorPhiHLT; 	 
        NumeratorEtaPhiHLT_  =NumeratorEtaPhiHLT;  
        EmulatedNumeratorPtHLT_ 	 =EmulatedNumeratorPtHLT; 	 
        EmulatedNumeratorPtHLTBarrel_ 	 =EmulatedNumeratorPtHLTBarrel; 	 
        EmulatedNumeratorPtHLTEndCap_ 	 =EmulatedNumeratorPtHLTEndCap; 	 
        EmulatedNumeratorPtHLTForward_ 	 =EmulatedNumeratorPtHLTForward; 	 
        EmulatedNumeratorEtaHLT_ 	 =EmulatedNumeratorEtaHLT; 	 
        EmulatedNumeratorPhiHLT_ 	 =EmulatedNumeratorPhiHLT; 	 
        EmulatedNumeratorEtaPhiHLT_  =EmulatedNumeratorEtaPhiHLT;  
        DenominatorPtHLT_    =DenominatorPtHLT; 	 
        DenominatorPtHLTBarrel_    =DenominatorPtHLTBarrel; 	 
        DenominatorPtHLTEndCap_    =DenominatorPtHLTEndCap; 	 
        DenominatorPtHLTForward_    =DenominatorPtHLTForward; 	 
        DenominatorEtaHLT_   =DenominatorEtaHLT; 	 
        DenominatorPhiHLT_   =DenominatorPhiHLT; 	 
        DenominatorEtaPhiHLT_=DenominatorEtaPhiHLT; 
      }; 
     ~PathInfo() {};
     PathInfo(std::string denomPathName, 
	      std::string pathName,
	      std::string trigEffLevel,
	      double trigThreshold):
	      //std::string processName,
	      //size_t type, 
	      //float ptmin,
	      //float ptmax):
       denomPathName_(denomPathName), 
       pathName_(pathName),
       trigEffLevel_(trigEffLevel),
       trigThreshold_(trigThreshold)
       //processName_(processName), 
       //objectType_(type),
       //ptmin_(ptmin), 
       //ptmax_(ptmax)
	 {
	 };
       float getPtMin() const { return ptmin_; } 
       float getPtMax() const { return ptmax_; }
       int type()       const { return objectType_; } 
       std::string getDenomPathName() { return denomPathName_; }
       std::string getPathName()      { return pathName_; }
       std::string getTrigEffLevel()  { return trigEffLevel_;}
       std::string getPathNameAndLevel() { return trigEffLevel_+"_"+pathName_;}
       double      getTrigThreshold()  { return trigThreshold_;}
       std::string getPathNameL1s()   { return pathNameL1s_;}
       std::string getPathNameHLT()   { return pathNameHLT_;}
       void setPathNameL1s(std::string input) { pathNameL1s_=input;}
       void setPathNameHLT(std::string input) { pathNameHLT_=input;}
       std::string getDenomPathNameL1s()   { return denomPathNameL1s_;}
       std::string getDenomPathNameHLT()   { return denomPathNameHLT_;}
       void setDenomPathNameL1s(std::string input) { denomPathNameL1s_=input;}
       void setDenomPathNameHLT(std::string input) { denomPathNameHLT_=input;}

       MonitorElement * getMENumeratorPt()     { return NumeratorPt_;}
       MonitorElement * getMENumeratorPtBarrel()     { return NumeratorPtBarrel_;}
       MonitorElement * getMENumeratorPtEndCap()     { return NumeratorPtEndCap_;}
       MonitorElement * getMENumeratorPtForward()    { return NumeratorPtForward_;}
       MonitorElement * getMENumeratorEta()    { return NumeratorEta_;}
       MonitorElement * getMENumeratorPhi()    { return NumeratorPhi_;}
       MonitorElement * getMENumeratorEtaPhi() { return NumeratorEtaPhi_;}

       MonitorElement * getMEEmulatedNumeratorPt()     { return EmulatedNumeratorPt_;}
       MonitorElement * getMEEmulatedNumeratorPtBarrel()     { return EmulatedNumeratorPtBarrel_;}
       MonitorElement * getMEEmulatedNumeratorPtEndCap()     { return EmulatedNumeratorPtEndCap_;}
       MonitorElement * getMEEmulatedNumeratorPtForward()    { return EmulatedNumeratorPtForward_;}
       MonitorElement * getMEEmulatedNumeratorEta()    { return EmulatedNumeratorEta_;}
       MonitorElement * getMEEmulatedNumeratorPhi()    { return EmulatedNumeratorPhi_;}
       MonitorElement * getMEEmulatedNumeratorEtaPhi() { return EmulatedNumeratorEtaPhi_;}

       MonitorElement * getMEDenominatorPt()     { return DenominatorPt_;}
       MonitorElement * getMEDenominatorPtBarrel()     { return DenominatorPtBarrel_;}
       MonitorElement * getMEDenominatorPtEndCap()     { return DenominatorPtEndCap_;}
       MonitorElement * getMEDenominatorPtForward()    { return DenominatorPtForward_;}
       MonitorElement * getMEDenominatorEta()    { return DenominatorEta_;}
       MonitorElement * getMEDenominatorPhi()    { return DenominatorPhi_;}
       MonitorElement * getMEDenominatorEtaPhi() { return DenominatorEtaPhi_;}

       MonitorElement * getMENumeratorPtHLT()     { return NumeratorPtHLT_;}
       MonitorElement * getMENumeratorPtHLTBarrel()     { return NumeratorPtHLTBarrel_;}
       MonitorElement * getMENumeratorPtHLTEndCap()     { return NumeratorPtHLTEndCap_;}
       MonitorElement * getMENumeratorPtHLTForward()    { return NumeratorPtHLTForward_;}
       MonitorElement * getMENumeratorEtaHLT()    { return NumeratorEtaHLT_;}
       MonitorElement * getMENumeratorPhiHLT()    { return NumeratorPhiHLT_;}
       MonitorElement * getMENumeratorEtaPhiHLT() { return NumeratorEtaPhiHLT_;}

       MonitorElement * getMEEmulatedNumeratorPtHLT()     { return EmulatedNumeratorPtHLT_;}
       MonitorElement * getMEEmulatedNumeratorPtHLTBarrel()     { return EmulatedNumeratorPtHLTBarrel_;}
       MonitorElement * getMEEmulatedNumeratorPtHLTEndCap()     { return EmulatedNumeratorPtHLTEndCap_;}
       MonitorElement * getMEEmulatedNumeratorPtHLTForward()    { return EmulatedNumeratorPtHLTForward_;}
       MonitorElement * getMEEmulatedNumeratorEtaHLT()    { return EmulatedNumeratorEtaHLT_;}
       MonitorElement * getMEEmulatedNumeratorPhiHLT()    { return EmulatedNumeratorPhiHLT_;}
       MonitorElement * getMEEmulatedNumeratorEtaPhiHLT() { return EmulatedNumeratorEtaPhiHLT_;}

       MonitorElement * getMEDenominatorPtHLT()     { return DenominatorPtHLT_;}
       MonitorElement * getMEDenominatorPtHLTBarrel()     { return DenominatorPtHLTBarrel_;}
       MonitorElement * getMEDenominatorPtHLTEndCap()     { return DenominatorPtHLTEndCap_;}
       MonitorElement * getMEDenominatorPtHLTForward()    { return DenominatorPtHLTForward_;}
       MonitorElement * getMEDenominatorEtaHLT()    { return DenominatorEtaHLT_;}
       MonitorElement * getMEDenominatorPhiHLT()    { return DenominatorPhiHLT_;}
       MonitorElement * getMEDenominatorEtaPhiHLT() { return DenominatorEtaPhiHLT_;}

   bool operator==(const std::string v)
   {
     return v==pathName_;
   }
   private:
       std::string denomPathName_;
       std::string denomPathNameL1s_;
       std::string denomPathNameHLT_;
       std::string pathName_;
       std::string pathNameL1s_;
       std::string pathNameHLT_;
       //std::string processName_;
       std::string trigEffLevel_;
       double trigThreshold_;
       int objectType_;
       float ptmin_, ptmax_;
     
       // we don't own this data 
       MonitorElement *NumeratorPt_; 
       MonitorElement *NumeratorPtBarrel_; 
       MonitorElement *NumeratorPtEndCap_; 
       MonitorElement *NumeratorPtForward_; 
       MonitorElement *NumeratorEta_;
       MonitorElement *NumeratorPhi_;
       MonitorElement *NumeratorEtaPhi_;
       MonitorElement *EmulatedNumeratorPt_; 
       MonitorElement *EmulatedNumeratorPtBarrel_; 
       MonitorElement *EmulatedNumeratorPtEndCap_; 
       MonitorElement *EmulatedNumeratorPtForward_; 
       MonitorElement *EmulatedNumeratorEta_;
       MonitorElement *EmulatedNumeratorPhi_;
       MonitorElement *EmulatedNumeratorEtaPhi_;
       MonitorElement *DenominatorPt_;
       MonitorElement *DenominatorPtBarrel_;
       MonitorElement *DenominatorPtEndCap_;
       MonitorElement *DenominatorPtForward_;
       MonitorElement *DenominatorEta_;
       MonitorElement *DenominatorPhi_;
       MonitorElement *DenominatorEtaPhi_;

       MonitorElement *NumeratorPtHLT_; 
       MonitorElement *NumeratorPtHLTBarrel_; 
       MonitorElement *NumeratorPtHLTEndCap_; 
       MonitorElement *NumeratorPtHLTForward_; 
       MonitorElement *NumeratorEtaHLT_;
       MonitorElement *NumeratorPhiHLT_;
       MonitorElement *NumeratorEtaPhiHLT_;
       MonitorElement *EmulatedNumeratorPtHLT_; 
       MonitorElement *EmulatedNumeratorPtHLTBarrel_; 
       MonitorElement *EmulatedNumeratorPtHLTEndCap_; 
       MonitorElement *EmulatedNumeratorPtHLTForward_; 
       MonitorElement *EmulatedNumeratorEtaHLT_;
       MonitorElement *EmulatedNumeratorPhiHLT_;
       MonitorElement *EmulatedNumeratorEtaPhiHLT_;
       MonitorElement *DenominatorPtHLT_;
       MonitorElement *DenominatorPtHLTBarrel_;
       MonitorElement *DenominatorPtHLTEndCap_;
       MonitorElement *DenominatorPtHLTForward_;
       MonitorElement *DenominatorEtaHLT_;
       MonitorElement *DenominatorPhiHLT_;
       MonitorElement *DenominatorEtaPhiHLT_;
              
   };
   
   // simple collection 
   class PathInfoCollection: public std::vector<PathInfo> {
   public:
     PathInfoCollection(): std::vector<PathInfo>()
       {};
       std::vector<PathInfo>::iterator find(std::string pathName) {
	 return std::find(begin(), end(), pathName);
       }
   };
   //PathInfoCollection hltPaths_;

   // list of trigger paths for getting efficiencies
   PathInfoCollection HLTPathsEffSingleJet_;
   PathInfoCollection HLTPathsEffDiJetAve_;
   PathInfoCollection HLTPathsEffMET_;
   PathInfoCollection HLTPathsEffMHT_;

   //
   //--- helper class to store the data path ----------
   class PathHLTMonInfo { 
     PathHLTMonInfo(): 
       pathName_("unset")
       //processName_("unset"), 
       //objectType_(-1),
       //ptmin_(0.), 
       //ptmax_(0.)
       {}; 
   public: 
      void setHistos(  
 		    MonitorElement* const Pt,  
 		    MonitorElement* const PtBarrel,  
 		    MonitorElement* const PtEndCap,  
 		    MonitorElement* const PtForward,  
 		    MonitorElement* const Eta,  
 		    MonitorElement* const Phi,  
 		    MonitorElement* const EtaPhi,  
 		    MonitorElement* const PtHLT,  
 		    MonitorElement* const PtHLTBarrel,  
 		    MonitorElement* const PtHLTEndCap,  
 		    MonitorElement* const PtHLTForward,  
 		    MonitorElement* const EtaHLT,  
 		    MonitorElement* const PhiHLT,  
 		    MonitorElement* const EtaPhiHLT,
                    MonitorElement* const PtL1s,  
 		    MonitorElement* const PtL1sBarrel,  
 		    MonitorElement* const PtL1sEndCap,  
 		    MonitorElement* const PtL1sForward,  
 		    MonitorElement* const EtaL1s,  
 		    MonitorElement* const PhiL1s,  
 		    MonitorElement* const EtaPhiL1s) 
      { 
        Pt_ 	         =Pt; 	 
        PtBarrel_ 	 =PtBarrel; 	 
        PtEndCap_ 	 =PtEndCap; 	 
        PtForward_ 	 =PtForward; 	 
        Eta_ 	         =Eta; 	 
        Phi_ 	         =Phi; 	 
        EtaPhi_         =EtaPhi;  

        PtHLT_ 	 =PtHLT; 	 
        PtHLTBarrel_ 	 =PtHLTBarrel; 	 
        PtHLTEndCap_ 	 =PtHLTEndCap; 	 
        PtHLTForward_ 	 =PtHLTForward; 	 
        EtaHLT_ 	 =EtaHLT; 	 
        PhiHLT_ 	 =PhiHLT; 	 
        EtaPhiHLT_      =EtaPhiHLT;  

        PtL1s_ 	 =PtL1s; 	 
        PtL1sBarrel_ 	 =PtL1sBarrel; 	 
        PtL1sEndCap_ 	 =PtL1sEndCap; 	 
        PtL1sForward_ 	 =PtL1sForward; 	 
        EtaL1s_ 	 =EtaL1s; 	 
        PhiL1s_ 	 =PhiL1s; 	 
        EtaPhiL1s_      =EtaPhiL1s;  
      }; 
     ~PathHLTMonInfo() {};
     PathHLTMonInfo(std::string pathName):
	      //std::string processName,
	      //size_t type, 
	      //float ptmin,
	      //float ptmax):
       pathName_(pathName)
       //processName_(processName), 
       //objectType_(type),
       //ptmin_(ptmin), 
       //ptmax_(ptmax)
	 {
	 };
       float getPtMin() const { return ptmin_; } 
       float getPtMax() const { return ptmax_; }
       int type()       const { return objectType_; } 
       std::string getPathName()      { return pathName_; }
       std::string getPathNameL1s()   { return pathNameL1s_;}
       std::string getPathNameHLT()   { return pathNameHLT_;}
       void setPathNameL1s(std::string input) { pathNameL1s_=input;}
       void setPathNameHLT(std::string input) { pathNameHLT_=input;}

       MonitorElement * getMEPt()     { return Pt_;}
       MonitorElement * getMEPtBarrel()     { return PtBarrel_;}
       MonitorElement * getMEPtEndCap()     { return PtEndCap_;}
       MonitorElement * getMEPtForward()    { return PtForward_;}
       MonitorElement * getMEEta()    { return Eta_;}
       MonitorElement * getMEPhi()    { return Phi_;}
       MonitorElement * getMEEtaPhi() { return EtaPhi_;}

       MonitorElement * getMEPtHLT()     { return PtHLT_;}
       MonitorElement * getMEPtHLTBarrel()     { return PtHLTBarrel_;}
       MonitorElement * getMEPtHLTEndCap()     { return PtHLTEndCap_;}
       MonitorElement * getMEPtHLTForward()    { return PtHLTForward_;}
       MonitorElement * getMEEtaHLT()    { return EtaHLT_;}
       MonitorElement * getMEPhiHLT()    { return PhiHLT_;}
       MonitorElement * getMEEtaPhiHLT() { return EtaPhiHLT_;}

       MonitorElement * getMEPtL1s()     { return PtL1s_;}
       MonitorElement * getMEPtL1sBarrel()     { return PtL1sBarrel_;}
       MonitorElement * getMEPtL1sEndCap()     { return PtL1sEndCap_;}
       MonitorElement * getMEPtL1sForward()    { return PtL1sForward_;}
       MonitorElement * getMEEtaL1s()    { return EtaL1s_;}
       MonitorElement * getMEPhiL1s()    { return PhiL1s_;}
       MonitorElement * getMEEtaPhiL1s() { return EtaPhiL1s_;}

   bool operator==(const std::string v)
   {
     return v==pathName_;
   }
   private:
       std::string pathName_;
       std::string pathNameL1s_;
       std::string pathNameHLT_;
       //std::string processName_;
       int objectType_;
       float ptmin_, ptmax_;
     
       // we don't own this data 
       MonitorElement *Pt_; 
       MonitorElement *PtBarrel_; 
       MonitorElement *PtEndCap_; 
       MonitorElement *PtForward_; 
       MonitorElement *Eta_;
       MonitorElement *Phi_;
       MonitorElement *EtaPhi_;

       MonitorElement *PtHLT_; 
       MonitorElement *PtHLTBarrel_; 
       MonitorElement *PtHLTEndCap_; 
       MonitorElement *PtHLTForward_; 
       MonitorElement *EtaHLT_;
       MonitorElement *PhiHLT_;
       MonitorElement *EtaPhiHLT_;
       
       MonitorElement *PtL1s_; 
       MonitorElement *PtL1sBarrel_; 
       MonitorElement *PtL1sEndCap_; 
       MonitorElement *PtL1sForward_; 
       MonitorElement *EtaL1s_;
       MonitorElement *PhiL1s_;
       MonitorElement *EtaPhiL1s_;
       
   };
   
   // simple collection 
   class PathHLTMonInfoCollection: public std::vector<PathHLTMonInfo> {
   public:
     PathHLTMonInfoCollection(): std::vector<PathHLTMonInfo>()
       {};
       std::vector<PathHLTMonInfo>::iterator find(std::string pathName) {
	 return std::find(begin(), end(), pathName);
       }
   };
   //PathHLTMonInfoCollection hltMonPaths_;

   PathHLTMonInfoCollection HLTPathsMonSingleJet_;
   PathHLTMonInfoCollection HLTPathsMonDiJetAve_;
   PathHLTMonInfoCollection HLTPathsMonMET_;
   PathHLTMonInfoCollection HLTPathsMonMHT_;

 public:
   PathInfoCollection fillL1andHLTModuleNames(PathInfoCollection hltPaths, 
				       std::string L1ModuleName, std::string HLTModuleName);
   PathHLTMonInfoCollection fillL1andHLTModuleNames(PathHLTMonInfoCollection hltPaths, 
				       std::string L1ModuleName, std::string HLTModuleName);
   virtual bool isTrigAcceptedEmulatedSingleJet(PathInfo v);
   virtual bool isTrigAcceptedEmulatedDiJetAve(PathInfo v);
   virtual bool isTrigAcceptedEmulatedMET(PathInfo v);

};
 
#endif
