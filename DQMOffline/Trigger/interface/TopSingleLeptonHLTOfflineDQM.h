#ifndef TOPSINGLELEPTONHLTOFFLINEDQM
#define TOPSINGLELEPTONHLTOFFLINEDQM

#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DQMOffline/Trigger/interface/TopHLTOfflineDQMHelper.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"

/*Originally from DQM/Physics by R. Wolf and J. Andrea*/

/**
  \class   MonitorSingleLepton TopHLTOfflineDQMHelper.h

  \brief   Helper class to define histograms for monitoring of muon/electron/jet/met quantities.

  Helper class to contain histograms for the monitoring of muon/electron/jet/met quantities. 
  This class can be instantiated several times after several event selection steps. It can 
  be used to fill histograms in three different granularity levels according to STANDARD 
  (<10 histograms), VERBOSE(<20 histograms), DEBUG(<30 histgorams). Note that for the sake 
  of simplicity and to force the analyst to keep the number of histograms to be monitored 
  small the MonitorSingleLepton class contains the histograms for all objects at once. It should 
  not contain much more than 10 histograms though in the STANDARD configuration, as these 
  histograms will be monitored at each SelectionStep. Monitoring of histograms after selec-
  tion steps within the same object collection needs to be implemented within the Monitor-
  Ensemble. It will not be covered by the SelectionStep class.
  */

namespace HLTOfflineDQMTopSingleLepton {

  class MonitorSingleLepton {
    public:

    public:
      /// default contructor
      MonitorSingleLepton(const char* label, const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC);
      /// default destructor
      ~MonitorSingleLepton(){};

      /// book histograms in subdirectory _directory_
      void book(DQMStore::IBooker& store_);
      /// fill monitor histograms with electronId and jetCorrections
      void fill(const edm::Event& event, const edm::EventSetup& setup, const HLTConfigProvider& hltConfig, const std::vector<std::string> triggerPaths);

    private:
      /// deduce monitorPath from label, the label is expected
      /// to be of type 'selectionPath:monitorPath'
      std::string monitorPath(const std::string& label) const { return label.substr(label.find(':')+1); };  
      /// deduce selectionPath from label, the label is 
      /// expected to be of type 'selectionPath:monitorPath' 
      std::string selectionPath(const std::string& label) const { return label.substr(0, label.find(':')); };  

      /// set configurable labels for trigger monitoring histograms
      void triggerBinLabels(std::string channel, const std::vector<std::string>& labels);
      /// fill trigger monitoring histograms
      void fill(const edm::Event& event, const edm::TriggerResults& triggerTable, std::string channel, const std::vector<std::string>& labels) const;

      /// check if histogram was booked
      bool booked(const std::string histName) const { return hists_.find(histName.c_str())!=hists_.end(); };
      /// fill histogram if it had been booked before
      void fill(const std::string histName, double value) const { if(booked(histName.c_str())) hists_.find(histName.c_str())->second->Fill(value); };
      /// fill histogram if it had been booked before (2-dim version)
      void fill(const std::string histName, double xValue, double yValue) const { if(booked(histName.c_str())) hists_.find(histName.c_str())->second->Fill(xValue, yValue); };
      /// fill histogram if it had been booked before (2-dim version)
      void fill(const std::string histName, double xValue, double yValue, double zValue) const { if(booked(histName.c_str())) hists_.find(histName.c_str())->second->Fill(xValue, yValue, zValue); };

    private:
      std::string folder_;
      /// instance label 
      std::string label_;
      /// considers a vector of METs
      std::vector< edm::EDGetTokenT< edm::View<reco::MET> > > mets_; 
      /// input sources for monitoring
      edm::EDGetTokenT< edm::View<reco::GsfElectron> > elecs_;
      edm::EDGetTokenT< edm::View<reco::Muon> > muons_;
      edm::EDGetTokenT< edm::View<reco::Jet> > jets_; 
      edm::EDGetTokenT< edm::View<reco::Vertex> > pvs_; 

      /// trigger table
      edm::EDGetTokenT< edm::TriggerResults > triggerTable_;
//     edm::EDGetTokenT< trigger::TriggerEventWithRefs > triggerEventWithRefsTag_;
      edm::EDGetTokenT <trigger::TriggerEventWithRefs> triggerSummaryTokenRAW;
      edm::EDGetTokenT <trigger::TriggerEventWithRefs> triggerSummaryTokenAOD;
      bool hasRawTriggerSummary;
	  	
 
      /// trigger paths for monitoring, expected 
      /// to be of form signalPath:MonitorPath
      std::vector<std::string> triggerPaths_;

      /// electronId label
      edm::EDGetTokenT< edm::ValueMap<float> > electronId_;
      /// electronId pattern we expect the following pattern:
      ///  0: fails
      ///  1: passes electron ID only
      ///  2: passes electron Isolation only
      ///  3: passes electron ID and Isolation only
      ///  4: passes conversion rejection
      ///  5: passes conversion rejection and ID
      ///  6: passes conversion rejection and Isolation
      ///  7: passes the whole selection
      /// As described on https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID
      int eidPattern_;
      /// extra isolation criterion on electron
      StringCutObjectSelector<reco::GsfElectron>* elecIso_;
      /// extra selection on electrons
      StringCutObjectSelector<reco::GsfElectron>* elecSelect_;

      /// extra selection on primary vertices; meant to investigate the pile-up effect
      StringCutObjectSelector<reco::Vertex>* pvSelect_;

      /// extra isolation criterion on muon
      StringCutObjectSelector<reco::Muon>* muonIso_;
      /// extra selection on muons
      StringCutObjectSelector<reco::Muon>* muonSelect_;

      /// jetCorrector
      std::string jetCorrector_;
      /// jetID as an extra selection type 
      edm::EDGetTokenT< reco::JetIDValueMap > jetIDLabel_;
      /// extra jetID selection on calo jets
      StringCutObjectSelector<reco::JetID>* jetIDSelect_;
      /// extra selection on jets (here given as std::string as it depends
      /// on the the jet type, which selections are valid and which not)
      std::string jetSelect_;
      /// include btag information or not
      /// to be determined from the cfg  
      bool includeBTag_;
      /// btag discriminator labels
      edm::EDGetTokenT< reco::JetTagCollection > btagEff_;
      edm::EDGetTokenT< reco::JetTagCollection > btagPur_; 
      edm::EDGetTokenT< reco::JetTagCollection > btagVtx_;
      /// btag working points
      double btagEffWP_, btagPurWP_, btagVtxWP_;
      /// mass window upper and lower edge
      double lowerEdge_, upperEdge_;

      /// number of logged interesting events
      int logged_;
      /// histogram container  
      std::map<std::string,MonitorElement*> hists_;

      /// hlt objects
      std::string          processName_;
      trigger::Vids        electronIds_;
      trigger::VRelectron  electronRefs_;
      trigger::Vids        muonIds_;
      trigger::VRmuon      muonRefs_;
      trigger::Vids        pfjetIds_;
      trigger::VRpfjet     pfjetRefs_;
  };

  inline void 
    MonitorSingleLepton::triggerBinLabels(std::string channel, const std::vector<std::string>& labels)
    {
      for(unsigned int idx=0; idx<labels.size(); ++idx){
        hists_[(channel+"Mon_").c_str()]->setBinLabel( idx+1, "["+monitorPath(labels[idx])+"]", 1);
      }
    }

  inline void 
    MonitorSingleLepton::fill(const edm::Event& event, const edm::TriggerResults& triggerTable, std::string channel, const std::vector<std::string>& labels) const
    {
      for(unsigned int idx=0; idx<labels.size(); ++idx){
        if( acceptHLT(event, triggerTable, monitorPath(labels[idx])) ){
          fill((channel+"Mon_").c_str(), idx+0.5 );
        }
      }
    }

}

#include <utility>

#include "DQMOffline/Trigger/interface/TopHLTOfflineDQMHelper.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

/**
  \class   TopSingleLeptonHLTOfflineDQM TopSingleLeptonHLTOfflineDQM.h 

  \brief   Module to apply a monitored selection of top like events in the semi-leptonic channel

  Plugin to apply a monitored selection of top like events with some minimal flexibility 
  in the number and definition of the selection steps. To achieve this flexibility it 
  employes the SelectionStep class. The MonitorSingleLepton class is used to provide a well 
  defined set of histograms to be monitored after each selection step. The SelectionStep 
  class provides a flexible and intuitive selection via the StringCutParser. SelectionStep 
  and MonitorSingleLepton classes are interleaved. The monitoring starts after a preselection 
  step (which is not monitored in the context of this module) with an instance of the 
  MonitorSingleLepton class. The following objects are supported for selection:

  - jets  : of type reco::Jet (jets), reco::CaloJet (jets/calo) or reco::PFJet (jets/pflow)
  - elecs : of type reco::GsfElectron
  - muons : of type reco::Muon
  - met   : of type reco::MET

  These types have to be present as prefix of the selection step paramter _label_ separated 
  from the rest of the label by a ':' (e.g. in the form "jets:step0"). The class expects 
  selection labels of this type. They will be disentangled by the private helper functions 
  _objectType_ and _seletionStep_ as declared below.
  */

/// define MonitorSingleLepton to be used
//using HLTOfflineDQMTopSingleLepton::MonitorSingleLepton;

class TopSingleLeptonHLTOfflineDQM : public DQMEDAnalyzer  {
  public: 
    /// default constructor
    TopSingleLeptonHLTOfflineDQM(const edm::ParameterSet& cfg);
    /// default destructor
    ~TopSingleLeptonHLTOfflineDQM(){
      if( vertexSelect_ ) delete vertexSelect_;
      if( beamspotSelect_ ) delete beamspotSelect_;
    };

    /// do this during the event loop
    virtual void dqmBeginRun(const edm::Run& r, const edm::EventSetup& c);
    virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
    void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;

  private:
    /// deduce object type from ParameterSet label, the label
    /// is expected to be of type 'objectType:selectionStep'
    std::string objectType(const std::string& label) { return label.substr(0, label.find(':')); };  
    /// deduce selection step from ParameterSet label, the 
    /// label is expected to be of type 'objectType:selectionStep' 
    std::string selectionStep(const std::string& label) { return label.substr(label.find(':')+1); };  

  private:
    /// trigger table
    edm::EDGetTokenT< edm::TriggerResults > triggerTable_;
    /// trigger paths
    std::vector<std::string> triggerPaths_;
    /// primary vertex 
    edm::EDGetTokenT< std::vector<reco::Vertex> > vertex_;
    /// string cut selector
    StringCutObjectSelector<reco::Vertex>* vertexSelect_;

    /// beamspot 
    edm::EDGetTokenT< reco::BeamSpot > beamspot_;
    /// string cut selector
    StringCutObjectSelector<reco::BeamSpot>* beamspotSelect_;

    HLTConfigProvider hltConfig_;



    /// needed to guarantee the selection order as defined by the order of
    /// ParameterSets in the _selection_ vector as defined in the config
    std::vector<std::string> selectionOrder_;
    /// this is the heart component of the plugin; std::string keeps a label 
    /// the selection step for later identification, edm::ParameterSet keeps
    /// the configuration of the selection for the SelectionStep class, 
    /// MonitoringEnsemble keeps an instance of the MonitorSingleLepton class to 
    /// be filled _after_ each selection step
    std::map<std::string, std::pair<edm::ParameterSet, HLTOfflineDQMTopSingleLepton::MonitorSingleLepton*> > selection_; 

    std::map<std::string, SelectionStepHLTBase*> selectmap_;
};

#endif
