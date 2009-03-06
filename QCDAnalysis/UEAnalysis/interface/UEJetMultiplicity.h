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
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include <FWCore/ServiceRegistry/interface/Service.h>
#include <PhysicsTools/UtilAlgos/interface/TFileService.h>

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
#include <FWCore/Framework/interface/TriggerNames.h>
#include <DataFormats/Common/interface/TriggerResults.h>
#include <DataFormats/HLTReco/interface/TriggerEvent.h> 
#include <DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h>

using namespace edm;
using namespace reco;
using namespace trigger;
using std::vector;

class UEJetMultiplicity : public edm::EDAnalyzer
{
  
public:
  
  //
  explicit UEJetMultiplicity( const edm::ParameterSet& ) ;
  virtual ~UEJetMultiplicity() {}
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob( const edm::EventSetup& ) ;
  virtual void endJob() ;
  
private:
  
  InputTag ChgGenJetsInputTag;
  InputTag TrackJetsInputTag;
  InputTag triggerResultsTag;
  InputTag triggerEventTag;
  InputTag genEventScaleTag;

  GenJetCollection theChgGenJets;
  BasicJetCollection theTrackJets;

  Handle< double              > genEventScaleHandle;
  Handle< GenJetCollection    > ChgGenJetsHandle ;
  Handle< BasicJetCollection  > TrackJetsHandle ;
  Handle< TriggerResults      > triggerResults;
  Handle< TriggerEvent        > triggerEvent;
  //  Handle<TriggerFilterObjectWithRefs> hltFilter; // not used at the moment: can access objects that fired the trigger
  TriggerNames triggerNames;
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
