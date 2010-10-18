#ifndef UEJetMultiplicity_H
#define UEJetMultiplicity_H

#include <iostream>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/Common/interface/Handle.h>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
#include <CommonTools/UtilAlgos/interface/TFileService.h>

#include <TROOT.h>
#include <TMath.h>
#include <TH1D.h>
#include <TH2D.h>

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Math/interface/deltaR.h"

// access trigger results
#include <DataFormats/Common/interface/TriggerResults.h>
#include <DataFormats/HLTReco/interface/TriggerEvent.h> 
#include <DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h>

class UEJetMultiplicity : public edm::EDAnalyzer
{
  
public:
  
  //
  explicit UEJetMultiplicity( const edm::ParameterSet& ) ;
  virtual ~UEJetMultiplicity() {}
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob() ;
  virtual void endJob() ;
  
private:
  
  edm::InputTag ChgGenJetsInputTag;
  edm::InputTag TrackJetsInputTag;
  edm::InputTag triggerResultsTag;
  edm::InputTag triggerEventTag;
  edm::InputTag genEventScaleTag;

  reco::GenJetCollection theChgGenJets;
  reco::BasicJetCollection theTrackJets;

  edm::Handle< double              > genEventScaleHandle;
  edm::Handle< reco::GenJetCollection    > ChgGenJetsHandle ;
  edm::Handle< reco::BasicJetCollection  > TrackJetsHandle ;
  edm::Handle< edm::TriggerResults       > triggerResults;
  edm::Handle< trigger::TriggerEvent     > triggerEvent;
  //  edm::Handle<TriggerFilterObjectWithRefs> hltFilter; // not used at the moment: can access objects that fired the trigger
  std::vector<std::string> selectedHLTBits;

  edm::Service<TFileService> fs;

  TH2D** h2d_nJets_vs_minPtJet_chggenjet;
  TH2D** h2d_nJets_vs_minPtJet_trackjet;

  double _eventScaleMin;
  double _eventScaleMax;
  double _PTTHRESHOLD;
  double _ETALIMIT;

  class PtSorter {
  public:
    template <class T> bool operator() ( const T& a, const T& b ) {
      return ( a.pt() > b.pt() );
    }
  };

};

#endif
