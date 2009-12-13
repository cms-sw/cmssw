#ifndef MuScleFitPlotter_H
#define MuScleFitPlotter_H

/** \class MuScleFitPlotter
 *  Plotter of the muon info (sim,gen,rec)
 *
 *  $Date: 2009/01/08 17:03:38 $
 *  $Revision: 1.2 $
 *  \author C.Mariotti, S.Bolognesi - INFN Torino / T.Dorigo - INFN Padova
 */

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <DataFormats/FWLite/interface/Handle.h>
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "MuonAnalysis/MomentumScaleCalibration/interface/Histograms.h"
#include <CLHEP/Vector/LorentzVector.h>

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
  MuScleFitPlotter();

  // Destructor
  // ----------
  virtual ~MuScleFitPlotter();

  // Operations
  // ----------
  void fillGen1(edm::Handle<reco::GenParticleCollection> genParticles);
  void fillGen2(edm::Handle<edm::HepMCProduct> evtMC);
  void fillSim(edm::Handle<edm::SimTrackContainer> simTracks);
  void fillGenSim(edm::Handle<edm::HepMCProduct> evtMC, edm::Handle<edm::SimTrackContainer> simTracks);
  void fillRec(std::vector<reco::LeafCandidate>& muons);
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

 
