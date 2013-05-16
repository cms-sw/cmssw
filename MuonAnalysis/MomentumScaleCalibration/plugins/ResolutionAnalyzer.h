#ifndef RESOLUTIONANALYZER_HH
#define RESOLUTIONANALYZER_HH

// -*- C++ -*-
//
// Package:    ResolutionAnalyzer
// Class:      ResolutionAnalyzer
// 
/**\class ResolutionAnalyzer ResolutionAnalyzer.cc MuonAnalysis/MomentumScaleCalibration/plugins/ResolutionAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Marco De Mattia
//         Created:  Thu Sep 11 12:16:00 CEST 2008
// $Id: ResolutionAnalyzer.h,v 1.15 2010/10/22 17:48:08 wmtan Exp $
//
//

// system include files
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "HepPDT/defs.h"
#include "HepPDT/TableBuilder.hh"
#include "HepPDT/ParticleDataTable.hh"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenEvent.h"
// #include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include <CLHEP/Vector/LorentzVector.h>

#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "Histograms.h"
#include "MuScleFitUtils.h"
//
// class decleration
//

class ResolutionAnalyzer : public edm::EDAnalyzer {
public:
  explicit ResolutionAnalyzer(const edm::ParameterSet&);
  ~ResolutionAnalyzer();

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() {};

  template<typename T>
  std::vector<reco::LeafCandidate> fillMuonCollection (const std::vector<T>& tracks) {
    std::vector<reco::LeafCandidate> muons;
    typename std::vector<T>::const_iterator track;
    for (track = tracks.begin(); track != tracks.end(); ++track){
      reco::Particle::LorentzVector mu(track->px(),track->py(),track->pz(),
				       sqrt(track->p()*track->p() + MuScleFitUtils::mMu2));
      MuScleFitUtils::goodmuon++;
      if (debug_>0) std::cout <<std::setprecision(9)<< "Muon #" << MuScleFitUtils::goodmuon 
			      << ": initial value   Pt = " << mu.Pt() << std::endl;
      reco::LeafCandidate muon(track->charge(),mu);
      // Store muon
      // ----------
      muons.push_back( muon );
    }
    return muons;
  } 

  /// Used to fill the map with the histograms needed
  void fillHistoMap();
  /// Writes the histograms in the map
  void writeHistoMap();
  /// Returns true if the two particles have DeltaR < cut
  bool checkDeltaR(const reco::Particle::LorentzVector & genMu, const reco::Particle::LorentzVector & recMu);

  // ----------member data ---------------------------

  // Collections labels
  // ------------------
  edm::InputTag theMuonLabel_;

  int theMuonType_;
  std::string theRootFileName_;
  std::string theCovariancesRootFileName_;
  bool debug_;
  std::map<std::string, Histograms*> mapHisto_;
  TFile * outputFile_;

  int eventCounter_;
  bool resonance_;
  bool readCovariances_;

  TString treeFileName_;
  int32_t maxEvents_;
  
  double ptMax_;
  
  HCovarianceVSxy * massResolutionVsPtEta_;
  TH2D * recoPtVsgenPt_;
  TH2D * recoPtVsgenPtEta12_;
  TH1D * deltaPtOverPt_;
  TH1D * deltaPtOverPtForEta12_;
};

#endif // RESOLUTIONANALYZER_HH
