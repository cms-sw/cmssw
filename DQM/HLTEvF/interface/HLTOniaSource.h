#ifndef HLTONIASOURCE_H
#define HLTONIASOURCE_H

// system include files
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackReco/interface/Track.h"


class PtGreaterRef { 
      public: 
      template <typename T> bool operator () (const T& i, const T& j) { 
        return (i->pt() > j->pt()); 
      } 
    };

class HLTOniaSource : public edm::EDAnalyzer {
   public:
      explicit HLTOniaSource(const edm::ParameterSet&);
      ~HLTOniaSource();


   private:
      virtual void beginJob() ;
      virtual void beginRun(const edm::Run &, const edm::EventSetup &);
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;


      reco::BeamSpot::Point BSPosition_;
 


      //user defined 
      void bookOniaTriggerInvariantMassMEs( std::map<std::string, MonitorElement *> & , std::string , std::string  );
      void bookOniaTriggerMEs( std::map<std::string, MonitorElement *> & , std::string);
      bool  checkHLTConfiguration(const edm::Run &, const edm::EventSetup &,std::string);
      void fillOniaTriggerMEs( edm::Handle<reco::RecoChargedCandidateCollection> &, std::string ,  std::map<std::string, MonitorElement *> & );
      void fillOniaTriggerMEs( edm::Handle<reco::TrackCollection> &, std::string ,  std::map<std::string, MonitorElement *> & );
      void fillOniaTriggerMEs( std::vector<reco::RecoChargedCandidateRef> &, std::string ,  std::map<std::string, MonitorElement *> & );
      void fillInvariantMass(std::vector<reco::RecoChargedCandidateRef> & , reco::RecoChargedCandidateCollection &  , std::string , std::string  );
      void fillInvariantMass(std::vector<reco::RecoChargedCandidateRef> & ,  std::vector<reco::RecoChargedCandidateRef> &, std::string , std::string);
      void fillInvariantMass(std::vector<reco::RecoChargedCandidateRef> & , reco::TrackCollection &  , std::string , std::string  );
      std::string subsystemFolder_;
      std::string hltProcessName_;
      //------------------------
      std::vector< std::string> triggerPath_;
      std::vector<edm::InputTag>  oniaMuonTag_;
      std::vector<edm::InputTag>       pixelTagsAfterFilter_ ;
      std::vector<edm::InputTag>       trackTagsAfterFilter_  ;     
      edm::InputTag   triggerSummaryRAWTag_;
      edm::InputTag  pixelTag_;
      edm::InputTag  beamSpotTag_;
      edm::InputTag  trackTag_;
      DQMStore * dbe_;
      std::map<std::string, MonitorElement *>       pixelAfterFilterME_;
      std::map<std::string, MonitorElement *>       trackAfterFilterME_;
      std::map<std::string, MonitorElement *> pixelME_;
      std::map<std::string, MonitorElement *> muonME_;
      std::map<std::string, MonitorElement *> trackME_;
      std::map<std::string, MonitorElement *> massME_;
      bool hltConfigInit_;
    
};

#endif
