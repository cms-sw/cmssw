#ifndef DQMOFFLINE_TRIGGER_TOPELECTRONHLTOFFLINESOURCE
#define DQMOFFLINE_TRIGGER_TOPELECTRONHLTOFFLINESOURCE


// Original Author:  Sarah Boutle
//         Created:  Jan 2010

//#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Common/interface/ValueMap.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class TopElectronHLTOfflineSource : public edm::EDAnalyzer 
{
 public:
  TopElectronHLTOfflineSource(const edm::ParameterSet& conf);
  virtual ~TopElectronHLTOfflineSource();
  
  virtual void beginJob();
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
  //virtual void analyze();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);  


  class EleMEs
    {
    public:
      struct EleMESet
      {
	// kinematics
	MonitorElement* ele_et;
	MonitorElement* ele_eta;
	MonitorElement* ele_phi;
  
	// isolation
	MonitorElement* ele_isolEm;
	MonitorElement* ele_isolHad;
	MonitorElement* ele_minDeltaR;
  
	// event structure
	MonitorElement* global_n30jets;
	MonitorElement* global_sumEt;
  
	// track related
	MonitorElement* ele_gsftrack_etaError;
	MonitorElement* ele_gsftrack_phiError;
	MonitorElement* ele_gsftrack_numberOfValidHits;
	MonitorElement* ele_gsftrack_dzPV;
      };
    public:
      EleMEs()
	{
	}
      
      EleMEs(DQMStore* dbe, const std::vector<std::string>& eleIdNames, bool addExtraId, const std::string& name)
	{
	  setup(dbe, eleIdNames, addExtraId, name);
	}
      
      void setup(DQMStore* dbe, const std::vector<std::string>&, bool addExtraId, const std::string& name);
      void setupMESet(EleMESet& eleSet, DQMStore* dbe, const std::string& name);
      void fill(EleMESet& eleMESet, const reco::GsfElectron& ele, float minDeltaR, float sumEt, int n30jets, float dzPV);
      

      void addMESets(const std::string& name);

      EleMESet& getMESet(size_t namePos, size_t idPos)
      {
        return eleMESets_[namePos+idPos*eleMENames_.size()];
      }
      
      const std::vector<EleMESet>& eleMESets()
      {
        return eleMESets_;
      }
      
      const std::vector<std::string>& eleMENames()
      {
        return eleMENames_;
      }
      
      const std::vector<std::string>& eleIdNames()
      {
        return eleIdNames_;
      }
      
      const std::string& name(size_t i)
      {
        return eleMENames_[i % eleMENames_.size()];
      }
      
      const std::string& idName(size_t i)
      {
        return eleIdNames_[i / eleMENames_.size()];
      }
      
      const std::string fullName(size_t i)
      {
        return name(i)+"_"+idName(i);
      }
     

    private:
      
      std::vector<EleMESet> eleMESets_;
      
      std::vector<std::string> eleMENames_;
      
      std::string name_;
      
      // add vector of references to electron id's
      std::vector<std::string> eleIdNames_;

    };

  virtual void setupHistos(const std::vector<EleMEs>);
  void fill(EleMEs& eleMEs, const edm::Event& iEvent, size_t eleIndex,  const std::vector<const trigger::TriggerObject*>& triggerJets, const std::vector<const trigger::TriggerObject*>& triggerElectrons, const reco::Vertex::Point& vertexPoint);
  
  
 private:
  
  DQMStore* dbe_; //dbe seems to be the standard name for this, I dont know why. We of course dont own it
  
  std::string dirName_;
  
  std::vector<EleMEs> eleMEs_;
  
  std::vector<std::string> electronIdNames_;
  std::string hltTag_;
  
  std::vector<std::string> superTriggerNames_;
  std::vector<std::string> electronTriggerNames_;
  
  edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryLabel_;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsLabel_;
  edm::InputTag triggerJetFilterLabel_;
  edm::InputTag triggerElectronFilterLabel_;
  edm::EDGetTokenT<reco::GsfElectronCollection> electronLabel_;
  edm::EDGetTokenT<reco::VertexCollection> primaryVertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpot_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<float> > > eleIdTokenCollection_;

  edm::Handle<trigger::TriggerEvent> triggerEvent_;
  
  edm::Handle<reco::GsfElectronCollection> eleHandle_;

  // exclude jets with deltaR < 0.1 from deltaR calculation
  bool excludeCloseJets_;
  
  bool requireTriggerMatch_;
  
  double electronMinEt_;
  double electronMaxEta_;
  
  // add extra ID
  bool addExtraId_;
  
  // the extra electron ID cuts
  double extraIdCutsSigmaEta_;
  double extraIdCutsSigmaPhi_;
  double extraIdCutsDzPV_;

  bool hltConfigChanged_;
  bool hltConfigValid_;
  HLTConfigProvider hltConfig_; 
};


#endif
