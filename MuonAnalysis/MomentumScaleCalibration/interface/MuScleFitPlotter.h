#ifndef MuScleFitPlotter_H
#define MuScleFitPlotter_H

/** \class MuScleFitPlotter
 *  Plotter of the muon info (sim,gen,rec)
 *
 *  $Date: 2010/06/21 15:20:18 $
 *  $Revision: 1.12 $
 *  \author C.Mariotti, S.Bolognesi - INFN Torino / T.Dorigo - INFN Padova
 */

// #include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Histograms.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class LeafCandidate;

class MuScleFitPlotter{

 public:
  // Constructor
  // -----------
  MuScleFitPlotter(std::string);

  // Destructor
  // ----------
  virtual ~MuScleFitPlotter();

  // Operations
  // ----------
  void fillGen1(const reco::GenParticleCollection* genParticles, bool=false);
  void fillGen2(const edm::HepMCProduct* evtMC, bool shepaFlag_);
  void fillSim(edm::Handle<edm::SimTrackContainer> simTracks);
  void fillGenSim(edm::Handle<edm::HepMCProduct> evtMC, edm::Handle<edm::SimTrackContainer> simTracks);
  void fillRec(std::vector<reco::LeafCandidate>& muons);

  // Root tree specific
  void fillRec( const std::vector<std::pair<reco::Particle::LorentzVector, reco::Particle::LorentzVector> > & savedPairs );
  void fillGen( const std::vector<std::pair<reco::Particle::LorentzVector, reco::Particle::LorentzVector> > & genPairs );

  void fillHistoMap();
  void writeHistoMap();

  bool debug;

 protected:

 private:
  // The map of histograms
  // ---------------------
  std::map<std::string, Histograms*> mapHisto;
  TFile * outputFile;

};
#endif


