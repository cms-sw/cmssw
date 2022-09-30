// -*- C++ -*-
//
// Package:    Castor
// Class:      CastorCellProducer
//

/**\class CastorCellProducer CastorCellProducer.cc RecoLocalCalo/Castor/src/CastorCellProducer.cc

 Description: CastorCell Reconstruction Producer. Produce CastorCells from CastorRecHits.
 Implementation:
*/

//
// Original Author:  Hans Van Haevermaet, Benoit Roland
//         Created:  Wed Jul  9 14:00:40 CEST 2008
//
//

// system include
#include <memory>
#include <vector>
#include <iostream>
#include <TMath.h>
#include <TRandom3.h>

// user include
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/Point3D.h"

// Castor Object include
#include "DataFormats/CastorReco/interface/CastorCell.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

//
// class declaration
//

class CastorCellProducer : public edm::global::EDProducer<> {
public:
  explicit CastorCellProducer(const edm::ParameterSet&);

private:
  void beginJob() override;
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void endJob() override;

  // member data
  typedef math::XYZPointD Point;
  typedef ROOT::Math::RhoZPhiPoint CellPoint;
  typedef std::vector<reco::CastorCell> CastorCellCollection;
  edm::EDGetTokenT<CastorRecHitCollection> tok_input_;
};

//
// constants, enums and typedefs
//
namespace {
  constexpr double MYR2D = 180 / M_PI;
}
//
// static data member definitions
//

//
// constructor and destructor
//

CastorCellProducer::CastorCellProducer(const edm::ParameterSet& iConfig) {
  tok_input_ = consumes<CastorRecHitCollection>(
      edm::InputTag(iConfig.getUntrackedParameter<std::string>("inputprocess", "castorreco")));
  // register your products
  produces<CastorCellCollection>();
  // now do what ever other initialization is needed
}

//
// member functions
//

// ------------ method called to produce the data  ------------

void CastorCellProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  using namespace reco;
  using namespace TMath;

  // Produce CastorCells from CastorRecHits

  edm::Handle<CastorRecHitCollection> InputRecHits;
  iEvent.getByToken(tok_input_, InputRecHits);

  auto OutputCells = std::make_unique<CastorCellCollection>();

  // looping over all CastorRecHits

  LogDebug("CastorCellProducer") << "1. entering CastorCellProducer ";

  for (size_t i = 0; i < InputRecHits->size(); ++i) {
    const CastorRecHit& rh = (*InputRecHits)[i];
    int sector = rh.id().sector();
    int module = rh.id().module();
    double energy = rh.energy();
    int zside = rh.id().zside();

    // define CastorCell properties
    double zCell = 0.;
    double phiCell;
    double rhoCell;

    // set z position of the cell
    if (module < 3) {
      // starting in EM section
      if (module == 1)
        zCell = 14415;
      if (module == 2)
        zCell = 14464;
    } else {
      // starting in HAD section
      zCell = 14534 + (module - 3) * 92;
    }

    // set phi position of the cell
    double castorphi[16];
    for (int j = 0; j < 16; j++) {
      castorphi[j] = -2.94524 + j * 0.3927;
    }
    if (sector > 8) {
      phiCell = castorphi[sector - 9];
    } else {
      phiCell = castorphi[sector + 7];
    }

    // add condition to select in eta sides
    if (zside <= 0)
      zCell = -1 * zCell;

    // set rho position of the cell (inner radius 3.7cm, outer radius 14cm)
    rhoCell = 88.5;

    // store cell position
    CellPoint tempcellposition(rhoCell, zCell, phiCell);
    Point cellposition(tempcellposition);

    LogDebug("CastorCellProducer") << "cell number: " << i + 1 << std::endl
                                   << "rho: " << cellposition.rho() << " phi: " << cellposition.phi() * MYR2D
                                   << " eta: " << cellposition.eta() << std::endl
                                   << "x: " << cellposition.x() << " y: " << cellposition.y()
                                   << " z: " << cellposition.z();

    if (energy > 0.) {
      CastorCell newCell(energy, cellposition);
      OutputCells->push_back(newCell);
    }

  }  // end loop over CastorRecHits

  LogDebug("CastorCellProducer") << "total number of cells in the event: " << InputRecHits->size();

  iEvent.put(std::move(OutputCells));
}

// ------------ method called once each job just before starting event loop  ------------
void CastorCellProducer::beginJob() { LogDebug("CastorCellProducer") << "Starting CastorCellProducer"; }

// ------------ method called once each job just after ending the event loop  ------------
void CastorCellProducer::endJob() { LogDebug("CastorCellProducer") << "Ending CastorCellProducer"; }

//define this as a plug-in
DEFINE_FWK_MODULE(CastorCellProducer);
