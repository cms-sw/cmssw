#ifndef UEJetValidation_H
#define UEJetValidation_H

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

class UEJetValidation : public edm::EDAnalyzer
{
  
public:
  
  //
  explicit UEJetValidation( const edm::ParameterSet& ) ;
  virtual ~UEJetValidation() {}
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob( const edm::EventSetup& ) ;
  virtual void endJob() ;
  
private:
  
  //virtual void fillHistograms( const Event& e, const EventSetup& es, int iHLTbit );
  virtual void fillHistogramsChgGen( int       iHLTbit,
				     BasicJet &theLeadingTrackJet,
				     GenJet   &theLeadingChgGenJet );

  virtual void fillHistogramsCalo( int       iHLTbit,
				   BasicJet &theLeadingTrackJet,
				   CaloJet  &theLeadingCaloJet );

  InputTag ChgGenJetsInputTag;
  InputTag TrackJetsInputTag;
  InputTag CaloJetsInputTag;
  InputTag triggerResultsTag;
  InputTag triggerEventTag;
  InputTag genEventScaleTag;

  GenJetCollection theChgGenJets;
  BasicJetCollection theTrackJets;
  CaloJetCollection theCaloJets;

  Handle< double              > genEventScaleHandle;
  Handle< GenJetCollection    > ChgGenJetsHandle ;
  Handle< BasicJetCollection  > TrackJetsHandle ;
  Handle< CaloJetCollection   > CaloJetsHandle ;
  Handle< TriggerResults      > triggerResults;
  Handle< TriggerEvent        > triggerEvent;
  //  Handle<TriggerFilterObjectWithRefs> hltFilter; // not used at the moment: can access objects that fired the trigger
  TriggerNames triggerNames;
  std::vector<std::string> selectedHLTBits;

  edm::Service<TFileService> fs;

  TH1D** h_dR_tracksjet_calojet;
  TH1D** h_dR_tracksjet_chggenjet;
  TH1D** h_pTratio_tracksjet_chggenjet;
  TH1D** h_eta_chggenjet;
  TH1D** h_phi_chggenjet;
  TH1D** h_pT_chggenjet;
  TH1D** h_eta_chggenjetMatched;
  TH1D** h_phi_chggenjetMatched;
  TH1D** h_pT_chggenjetMatched;
  TH1D** h_pT_tracksjet;
  TH1D** h_pT_calojet;
  TH1D** h_pT_calojet_hadronic;
  TH1D** h_pT_calojet_electromagnetic;
  TH1D** h_nConstituents_tracksjet;
  TH1D** h_nConstituents_calojet;
  TH1D** h_nConstituents_chggenjet;
  TH1D** h_maxDistance_tracksjet;
  TH1D** h_maxDistance_calojet;
  TH1D** h_maxDistance_chggenjet;
  TH1D** h_jetsizeNchg_tracksjet;
  TH1D** h_jetsizePtsum_tracksjet;
  TH1D** h_jetFragmentation_tracksjet;

  TH2D** h2d_DrTrackJetCaloJet_PtTrackJet;
  TH2D** h2d_DrTrackJetChgGenJet_PtTrackJet;
  TH2D** h2d_pTratio_tracksjet_calojet;
  TH2D** h2d_pTRatio_tracksjet_calojet_hadronic;
  TH2D** h2d_pTRatio_tracksjet_calojet_electromagnetic;
  TH2D** h2d_pTratio_tracksjet_chggenjet;
  TH2D** h2d_nConstituents_tracksjet_calojet;
  TH2D** h2d_nConstituents_tracksjet_chggenjet;
  TH2D** h2d_maxDistance_tracksjet_calojet;
  TH2D** h2d_maxDistance_tracksjet_chggenjet;
  TH2D** h2d_nConstituents_tracksjet; // vs pt(jet)
  TH2D** h2d_nConstituents_calojet;
  TH2D** h2d_nConstituents_chggenjet;
  TH2D** h2d_maxDistance_tracksjet;
  TH2D** h2d_maxDistance_calojet;
  TH2D** h2d_maxDistance_chggenjet;
  TH2D** h2d_jetsizeNchg_tracksjet;
  TH2D** h2d_jetsizePtsum_tracksjet;
  TH2D** h2d_nchg_vs_dR;
  TH2D** h2d_ptsum_vs_dR;

  TH2D* h2d_jetsizeNchg_chggenjet;
  TH2D* h2d_jetsizePtsum_chggenjet;

  double _eventScaleMin;
  double _eventScaleMax;
  double _PTTHRESHOLD;
  double _ETALIMIT;
  double _dR;

  class PtSorter {
  public:
    template <class T> bool operator() ( const T& a, const T& b ) {
      return ( a.pt() > b.pt() );
    }
  };

};

#endif
