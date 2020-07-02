///////////////////////////////////////////////////////////////////////////////
// File: FastHFShowerLibrary.cc
// Description: Shower library for Very forward hadron calorimeter
///////////////////////////////////////////////////////////////////////////////

#include "FastSimulation/ShowerDevelopment/interface/FastHFShowerLibrary.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Utilities/interface/RandomEngineAndDistribution.h"
#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalParametersRcd.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"
#include "Geometry/HcalCommonData/interface/HcalSimulationConstants.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "Randomize.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

// Geant4 headers
#include "G4ParticleDefinition.hh"
#include "G4DynamicParticle.hh"
#include "G4DecayPhysics.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleTypes.hh"

// STL headers
#include <memory>

#include <iostream>
#include <mutex>
#include <vector>

//#define DebugLog

static std::once_flag initializeOnce;

FastHFShowerLibrary::FastHFShowerLibrary(edm::ParameterSet const& p) : fast(p) {
  edm::ParameterSet m_HS = p.getParameter<edm::ParameterSet>("HFShowerLibrary");
  applyFidCut = m_HS.getParameter<bool>("ApplyFiducialCut");
}

void const FastHFShowerLibrary::initHFShowerLibrary(const edm::EventSetup& iSetup) {
  edm::LogInfo("FastCalorimetry") << "initHFShowerLibrary::initialization";

  edm::ESHandle<HcalDDDSimConstants> hdc;
  iSetup.get<HcalSimNumberingRecord>().get(hdc);
  hcalConstants = hdc.product();

  edm::ESHandle<HcalSimulationConstants> hdsc;
  iSetup.get<HcalSimNumberingRecord>().get(hdsc);
  const HcalSimulationConstants* hsps = hdsc.product();

  std::string name = "HcalHits";
  numberingFromDDD = std::make_unique<HcalNumberingFromDDD>(hcalConstants);
  hfshower = std::make_unique<HFShowerLibrary>(name, hcalConstants, hsps->hcalsimpar(), fast);

  //only one thread can be allowed to setup the G4 physics table.
  std::call_once(initializeOnce, []() {
    // Geant4 particles
    G4DecayPhysics decays;
    decays.ConstructParticle();
    G4ParticleTable* partTable = G4ParticleTable::GetParticleTable();
    partTable->SetReadiness();
  });
  //G4ParticleTable* partTable = G4ParticleTable::GetParticleTable();
  //hfshower->initRun(partTable, hcalConstants);  // init particle code
}

void FastHFShowerLibrary::SetRandom(const RandomEngineAndDistribution* rnd) {
  // define Geant4 engine per thread
  G4Random::setTheEngine(&(rnd->theEngine()));
  LogDebug("FastHFShowerLibrary::recoHFShowerLibrary")
      << "Begin of event " << G4UniformRand() << "  " << rnd->theEngine().name() << "  " << rnd->theEngine();
}

void FastHFShowerLibrary::recoHFShowerLibrary(const FSimTrack& myTrack) {
#ifdef DebugLog
  edm::LogInfo("FastCalorimetry") << "FastHFShowerLibrary: recoHFShowerLibrary ";
#endif

  if (!myTrack.onVFcal()) {
#ifdef DebugLog
    edm::LogInfo("FastCalorimetry") << "FastHFShowerLibrary: we should not be here ";
#endif
  }

  hitMap.clear();
  double eGen = 1000. * myTrack.vfcalEntrance().e();  // energy in [MeV]
  double delZv = (myTrack.vfcalEntrance().vertex().Z() > 0.0) ? 50.0 : -50.0;
  G4ThreeVector vertex(10. * myTrack.vfcalEntrance().vertex().X(),
                       10. * myTrack.vfcalEntrance().vertex().Y(),
                       10. * myTrack.vfcalEntrance().vertex().Z() + delZv);  // in [mm]

  G4ThreeVector direction(
      myTrack.vfcalEntrance().Vect().X(), myTrack.vfcalEntrance().Vect().Y(), myTrack.vfcalEntrance().Vect().Z());

  bool ok;
  double weight = 1.0;  // rad. damage
  int parCode = myTrack.type();
  double tSlice = 0.1 * vertex.mag() / 29.98;

  std::vector<HFShowerLibrary::Hit> hits =
      hfshower->fillHits(vertex, direction, parCode, eGen, ok, weight, tSlice, false);

  for (unsigned int i = 0; i < hits.size(); ++i) {
    G4ThreeVector pos = hits[i].position;
    int depth = hits[i].depth;
    double time = hits[i].time;
    if (!applyFidCut || (HFFibreFiducial::PMTNumber(pos) > 0)) {
      //    if (!applyFidCut || (applyFidCut && HFFibreFiducial::PMTNumber(pos)>0)) {
      int det = 5;
      int lay = 1;
      uint32_t id = 0;
      HcalNumberingFromDDD::HcalID tmp =
          numberingFromDDD->unitID(det, math::XYZVectorD(pos.x(), pos.y(), pos.z()), depth, lay);
      modifyDepth(tmp);
      id = numberingScheme.getUnitID(tmp);

      CaloHitID current_id(id, time, myTrack.id());
      std::map<CaloHitID, float>::iterator cellitr;
      cellitr = hitMap.find(current_id);
      if (cellitr == hitMap.end()) {
        hitMap.insert(std::pair<CaloHitID, float>(current_id, 1.0));
      } else {
        cellitr->second += 1.0;
      }
    }  // end of isItinFidVolume check
  }    // end loop over hits
}

void FastHFShowerLibrary::modifyDepth(HcalNumberingFromDDD::HcalID& id) {
  if (id.subdet == HcalForward) {
    int ieta = (id.zside == 0) ? -id.etaR : id.etaR;
    if (hcalConstants->maxHFDepth(ieta, id.phis) > 2) {
      if (id.depth <= 2) {
        if (G4UniformRand() > 0.5)
          id.depth += 2;
      }
    }
  }
}
