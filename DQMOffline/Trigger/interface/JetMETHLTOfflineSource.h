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

 private:
  //
  //--- helper class to store the data path
   class PathInfo { 
     PathInfo(): 
       denomPathName_("unset"),
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
	      std::string pathName):
	      //std::string processName,
	      //size_t type, 
	      //float ptmin,
	      //float ptmax):
       denomPathName_(denomPathName), 
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
       std::string getDenomPathName() { return denomPathName_; }
       std::string getPathName()      { return pathName_; }
       std::string getPathNameL1s()   { return pathNameL1s_;}
       std::string getPathNameHLT()   { return pathNameHLT_;}
       void setPathNameL1s(std::string input) { pathNameL1s_=input;}
       void setPathNameHLT(std::string input) { pathNameHLT_=input;}

       MonitorElement * getMENumeratorPt()     { return NumeratorPt_;}
       MonitorElement * getMENumeratorPtBarrel()     { return NumeratorPtBarrel_;}
       MonitorElement * getMENumeratorPtEndCap()     { return NumeratorPtEndCap_;}
       MonitorElement * getMENumeratorPtForward()    { return NumeratorPtForward_;}
       MonitorElement * getMENumeratorEta()    { return NumeratorEta_;}
       MonitorElement * getMENumeratorPhi()    { return NumeratorPhi_;}
       MonitorElement * getMENumeratorEtaPhi() { return NumeratorEtaPhi_;}

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
       std::string pathName_;
       std::string pathNameL1s_;
       std::string pathNameHLT_;
       //std::string processName_;
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
   //--- helper class to store the data path
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
 		    MonitorElement* const EtaPhiHLT) 
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
   

};
 
#endif
