// Local patched copy of Pythia8Plugins/EvtGen.h adapted to EvtGen 03.x API.
// Diverges from the upstream pythia8 header only in:
//   - EvtGenRandom now overrides setSeed(unsigned long)/lastSeed() pure virtuals
//   - EvtDecayTable::getInstance() returns EvtDecayTable& (not pointer)

#ifndef CMSSW_Pythia8_EvtGen_Patched_H
#define CMSSW_Pythia8_EvtGen_Patched_H

#include "Pythia8/Pythia.h"
#include "EvtGen/EvtGen.hh"
#include "EvtGenBase/EvtRandomEngine.hh"
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtParticleFactory.hh"
#include "EvtGenBase/EvtPDL.hh"
#include "EvtGenBase/EvtDecayTable.hh"
#include "EvtGenBase/EvtParticleDecayList.hh"
#include "EvtGenBase/EvtDecayBase.hh"
#include "EvtGenExternal/EvtExternalGenList.hh"

namespace Pythia8 {

//==========================================================================

// A class to wrap the Pythia random number generator for use by EvtGen.

class EvtGenRandom : public EvtRandomEngine {

public:

  // Constructor.
  EvtGenRandom(Rndm *rndmPtrIn) {rndmPtr = rndmPtrIn;}

  // Return a random number.
  double random() override {if (rndmPtr) return rndmPtr->flat(); else return -1.0;}

  // EvtGen 03 pure virtuals. Record-only: Pythia owns the actual RNG state.
  void setSeed(unsigned long int seed) override {lastSeedVal = seed;}
  unsigned long int lastSeed() const override {return lastSeedVal;}

  // The random number pointer.
  Rndm *rndmPtr;

private:
  unsigned long int lastSeedVal{0};

};

//==========================================================================

// A class to perform decays via the external EvtGen decay program,
// see http://evtgen.warwick.ac.uk/, the program manual provided with
// the EvtGen distribution, and D. J. Lange,
// Nucl. Instrum. Meth. A462, 152 (2001) for details.

// EvtGen performs a series of decays from some initial particle
// decay, rather than just a single decay, and so EvtGen cannot be
// interfaced through the standard external DecayHandler class without
// considerable complication. Consequently, EvtGen is called on the
// complete event record after all steps of Pythia are completed.

// Oftentimes a specific "signal" decay is needed to occur once in an
// event, and all other decays performed normally. This is possible
// via reading in a user decay file (with readDecayFile) and creating
// aliased particles with names ending with signalSuffix. By default,
// this is "_SIGNAL". When decay() is called, all particles in the
// Pythia event record that are of the same types as the signal
// particles are collected. One is selected at random and decayed via
// the channel(s) defined for that aliased signal particle. All other
// particles are decayed normally. The weight for the event is
// calculated and returned.

// It is also possible to specify a status needed to consider a
// particle as a signal candidate. This can be done by modifying the
// signals map, e.g. if the tau- is a signal candidate, then
//     EvtGenDecays.signals[15].status = 201
// will only only select as candidates any tau- with this status. This
// allows the event record to be changed before decays, so only
// certain particles are selected as possible signal candidates
// (e.g. passing kinematic requirements).

// Please note that particles produced from a signal candidate decay
// are not searched for additional signal candidates. This means that
// if B0 and tau- have been designated as signal, then a tau- from a
// W- decay would be a signal candidate, while a tau- from a B0 decay
// would not. This restriction arises from the additional complexity
// of allowing recursive signal decays. The following statuses are
// used: 93 for particles decayed with EvtGen, 94 for particles
// oscillated with EvtGen, 95 for signal particles, and 96 for signal
// particles from an oscillation.

class EvtGenDecays {

public:

  // Constructor.
  EvtGenDecays(Pythia *pythiaPtrIn, string decayFile, string particleDataFile,
    EvtExternalGenList *extPtrIn = 0, EvtAbsRadCorr *fsrPtrIn = 0,
    int mixing = 1, bool xml = false, bool limit = true,
    bool extUse = true, bool fsrUse = true);

  // Destructor.
  ~EvtGenDecays() {
    if (evtgen) delete evtgen;
    if (extOwner && extPtr) delete extPtr;
    if (fsrOwner && fsrPtr) delete fsrPtr;
  }

  // Perform all decays and return the event weight.
  double decay();

  // Stop EvtGen decaying a particle.
  void exclude(int id) {excIds.insert(id);}

  // Update the Pythia particle database from EvtGen.
  void updatePythia();

  // Update the EvtGen particle database from Pythia.
  void updateEvtGen();

  // Read an EvtGen user decay file.
  void readDecayFile(string decayFile, bool xml = false) {
    evtgen->readUDecay(decayFile.c_str(), xml); updateData();}

  // External model pointer and FSR model pointer.
  bool extOwner, fsrOwner;
  EvtExternalGenList *extPtr;
  EvtAbsRadCorr      *fsrPtr;
  std::list<EvtDecayBase*> models;

  // Map of signal particle info.
  struct Signal {int status; EvtId egId; vector<double> bfs; vector<int> map;
    EvtParticleDecayList modes;};
  map<int, Signal> signals;

  // The suffix indicating an EvtGen particle or alias is signal.
  string signalSuffix;

protected:

  // Update the particles to decay with EvtGen, and the selected signals.
  void updateData(bool final = false);

  // Update the Pythia event record with an EvtGen decay tree.
  void updateEvent(Particle *pyPro, EvtParticle *egPro,
    vector<int> *pySigs = 0, vector<EvtParticle*> *egSigs = 0,
    vector<double> *bfs = 0, double *wgt = 0);

  // Check if a particle can decay.
  bool checkVertex(Particle *pyPro);

  // Check if a particle is signal.
  bool checkSignal(Particle *pyPro);

  // Check if an EvtGen particle has oscillated.
  bool checkOsc(EvtParticle *egPro);

  // Number of times to try a decay sampling (constant).
  static const int NTRYDECAY = 1000;

  // The pointer to the associated Pythia object.
  Pythia *pythiaPtr;

  // Random number wrapper for EvtGen.
  EvtGenRandom rndm;

  // The EvtGen object.
  EvtGen *evtgen;

  // Set of particle IDs to include and exclude decays with EvtGen.
  set<int> incIds, excIds;

  // Flag whether the final particle update has been performed.
  bool updated;

  // The selected signal iterator.
  map<int, Signal>::iterator signal;

  // Parameters used to check if a particle should decay (as set via Pythia).
  double tau0Max, tauMax, rMax, xyMax, zMax;
  bool limitTau0, limitTau, limitRadius, limitCylinder, limitDecay;

};

//--------------------------------------------------------------------------

// The constructor.

// The EvtGenDecays object is associated with a single Pythia
// instance. This is to ensure a consistent random number generator
// across the two, as well as any updates to particle data, etc. Note
// that if multiple EvtGenDecays objects exist, that they will modify
// one anothers particle databases due to the design of EvtGen.

// This constructor also sets all particles to be decayed by EvtGen as
// stable within Pythia. The parameters within Pythia used to check if
// a particle should be decayed, as described in the "Particle Decays"
// section of the Pythia manual, are set. Note that if the variable
// "limit" is set to "false", then no check will be made before
// decaying a particle with EvtGen.

// The constructor is designed to have the exact same form as the
// EvtGen constructor except for these five differences.
// (1) The first variable is the pointer to the Pythia object.
// (2) The third last argument is a flag to limit decays based on the
//     Pythia criteria (based on the particle decay vertex).
// (3) The second last argument is a flag if external models should be
//     passed to EvtGen (default is true).
// (4) The last argument is a flag if an FSR model should be passed
//     to EvtGen (default is true).
// (5) No random engine pointer is passed, as this is obtained from
//     Pythia.

//   pythiaPtrIn:      the pointer to the associated Pythia object.
//   decayFile:        the name of the decay file to pass to EvtGen.
//   particleDataFile: the name of the particle data file to pass to EvtGen.
//   extPtrIn:         the optional EvtExternalGenList pointer, this must be
//                     be provided if fsrPtrIn is provided to avoid double
//                     initializations.
//   fsrPtrIn:         the EvtAbsRadCorr pointer to pass to EvtGen.
//   mixing:           the mixing type to pass to EvtGen.
//   xml:              flag to use XML files to pass to EvtGen.
//   limit:            flag to limit particle decays based on Pythia criteria.
//   extUse:           flag to use external models with EvtGen.
//   fsrUse:           flag to use radiative correction engine with EvtGen.

EvtGenDecays::EvtGenDecays(Pythia *pythiaPtrIn, string decayFile,
  string particleDataFile, EvtExternalGenList *extPtrIn,
  EvtAbsRadCorr *fsrPtrIn, int mixing, bool xml, bool limit,
  bool extUse, bool fsrUse) : extPtr(extPtrIn), fsrPtr(fsrPtrIn),
  signalSuffix("_SIGNAL"), pythiaPtr(pythiaPtrIn), rndm(&pythiaPtr->rndm),
  updated(false) {

  // Initialize EvtGen.
  if (!extPtr && fsrPtr) {
    cout << "Error in EvtGenDecays::"
         << "EvtGenDecays: extPtr is null but fsrPtr is provided\n";
    return;
  }
  if (extPtr) extOwner = false;
  else {extOwner = true; extPtr = new EvtExternalGenList();}
  if (fsrPtr) fsrOwner = false;
  else {fsrOwner = true; fsrPtr = extPtr->getPhotosModel();}
  models = extPtr->getListOfModels();
  evtgen = new EvtGen(decayFile.c_str(), particleDataFile.c_str(),
    &rndm, fsrUse ? fsrPtr : 0, extUse ? &models : 0, mixing, xml);

  // Get the Pythia decay limits.
  if (!pythiaPtr) return;
  limitTau0     = pythiaPtr->settings.flag("ParticleDecays:limitTau0");
  tau0Max       = pythiaPtr->settings.parm("ParticleDecays:tau0Max");
  limitTau      = pythiaPtr->settings.flag("ParticleDecays:limitTau");
  tauMax        = pythiaPtr->settings.parm("ParticleDecays:tauMax");
  limitRadius   = pythiaPtr->settings.flag("ParticleDecays:limitRadius");
  rMax          = pythiaPtr->settings.parm("ParticleDecays:rMax");
  limitCylinder = pythiaPtr->settings.flag("ParticleDecays:limitCylinder");
  xyMax         = pythiaPtr->settings.parm("ParticleDecays:xyMax");
  zMax          = pythiaPtr->settings.parm("ParticleDecays:zMax");
  limitDecay    = limit && (limitTau0 || limitTau ||
                            limitRadius || limitCylinder);

}

//--------------------------------------------------------------------------

// Perform all decays and return the event weight.

// All particles in the event record that can be decayed by EvtGen are
// decayed. If a particle is a signal particle, then this is stored in
// a vector of signal particles. A signal particle is only stored if
// its status is the same as the status provided in the signals map. A
// negative status in the signal map indicates that all statuses
// should be accepted. After all signal particles are identified, one
// is randomly chosen and decayed as signal. The remainder are decayed
// normally.

// Forcing a signal decay changes the weight of an event from unity,
// and so the relative event weight is returned, given the forced
// signal decay. A weight of 0 indicates no signal in the event, while
// a weight of -1 indicates something is wrong, e.g. either the Pythia
// or EvtGen pointers are not available or the number of tries has
// been exceeded. For the event weight to be valid, one should not
// change the absolute branching fractions in the signal and inclusive
// definitions, but rather just remove the unwanted decay channels
// from the signal decay definition.

double EvtGenDecays::decay() {

  // Reset the signal and signal counters.
  if (!pythiaPtr || !evtgen) return -1;
  if (!updated) updateData(true);

  // Loop over all particles in the Pythia event.
  Event &event = pythiaPtr->event;
  vector<int> pySigs; vector<EvtParticle*> egSigs, egPrts;
  vector<double> bfs; double wgt(1.);
  for (int iPro = 0; iPro < event.size(); ++iPro) {

    // Check particle is final and can be decayed by EvtGen.
    Particle *pyPro = &event[iPro];
    if (!pyPro->isFinal()) continue;
    if (incIds.find(pyPro->id()) == incIds.end()) continue;
    if (pyPro->status() == 93 || pyPro->status() == 94) continue;

    // Decay the progenitor with EvtGen.
    EvtParticle *egPro = EvtParticleFactory::particleFactory
      (EvtPDL::evtIdFromStdHep(pyPro->id()),
       EvtVector4R(pyPro->e(), pyPro->px(), pyPro->py(), pyPro->pz()));
    egPrts.push_back(egPro);
    egPro->setDiagonalSpinDensity();
    evtgen->generateDecay(egPro);
    pyPro->tau(egPro->getLifetime());
    if (!checkVertex(pyPro)) continue;

    // Add oscillations to event record.
    if (checkOsc(egPro))
      updateEvent(pyPro, egPro, &pySigs, &egSigs, &bfs, &wgt);

    // Undo decay if signal (duplicate to stop oscillations).
    else if (checkSignal(pyPro)) {
      pySigs.push_back(pyPro->index());
      egSigs.push_back(egPro);
      bfs.push_back(signal->second.bfs[0]);
      wgt *= 1 - bfs.back();
      egPro->deleteDaughters();
      EvtParticle *egDau = EvtParticleFactory::particleFactory
        (EvtPDL::evtIdFromStdHep(pyPro->id()),
         EvtVector4R(pyPro->e(), pyPro->px(), pyPro->py(), pyPro->pz()));
      egDau->addDaug(egPro);
      egDau->setDiagonalSpinDensity();

    // If not signal, add to event record.
    } else updateEvent(pyPro, egPro, &pySigs, &egSigs, &bfs, &wgt);
  }
  if (pySigs.size() == 0) {
    for (int iPrt = 0; iPrt < (int)egPrts.size(); ++iPrt)
      egPrts[iPrt]->deleteTree();
    return 0;
  }

  // Determine the decays of the signal particles (signal or background).
  vector<int> modes; int force(-1), n(0);
  for (int iTry = 1; iTry <= NTRYDECAY; ++iTry) {
    modes.clear(); force = pythiaPtr->rndm.pick(bfs); n = 0;
    for (int iSig = 0; iSig < (int)pySigs.size(); ++iSig) {
      if (iSig == force) modes.push_back(0);
      else modes.push_back(pythiaPtr->rndm.flat() > bfs[iSig]);
      if (modes.back() == 0) ++n;
    }
    if (pythiaPtr->rndm.flat() <= 1.0 / n) break;
    if (iTry == NTRYDECAY) {
      wgt = 2.;
      cout << "Warning in EvtGenDecays::decay: maximum "
           << "number of signal decay attempts exceeded";
    }
  }

  // Decay the signal particles and mark forced decay.
  for (int iSig = 0; iSig < (int)pySigs.size(); ++iSig) {
    EvtParticle *egSig = egSigs[iSig];
    Particle    *pySig = &event[pySigs[iSig]];
    signal = signals.find(pySig->id());
    if (egSig->getNDaug() > 0) egSig = egSig->getDaug(0);
    if (modes[iSig] == 0) egSig->setId(signal->second.egId);
    else {
      signal->second.modes.getDecayModel(egSig);
      egSig->setChannel(signal->second.map[egSig->getChannel()]);
    }
    if (iSig == force)
      pySig->status(pySig->status() == 92 || pySig->status() == 94 ? 96 : 95);
    evtgen->generateDecay(egSig);
    updateEvent(pySig, egSigs[iSig]);
  }

  // Delete all EvtGen particles and return weight.
  for (int iPrt = 0; iPrt < (int)egPrts.size(); ++iPrt)
    egPrts[iPrt]->deleteTree();
  return 1. - wgt;

}

//--------------------------------------------------------------------------

// Update the Pythia particle database from EvtGen.

// Note that only the particle spin type, charge type, nominal mass,
// width, minimum mass, maximum mass, and nominal lifetime are
// set. The name string is not set.

void EvtGenDecays::updatePythia() {
  if (!pythiaPtr || !evtgen) return;
  for (int entry = 0; entry < (int)EvtPDL::entries(); ++entry) {
    EvtId egId = EvtPDL::getEntry(entry);
    int   pyId = EvtPDL::getStdHep(egId);
    pythiaPtr->particleData.spinType  (pyId, EvtPDL::getSpinType(egId));
    pythiaPtr->particleData.chargeType(pyId, EvtPDL::chg3(egId));
    pythiaPtr->particleData.m0        (pyId, EvtPDL::getMass(egId));
    pythiaPtr->particleData.mWidth    (pyId, EvtPDL::getWidth(egId));
    pythiaPtr->particleData.mMin      (pyId, EvtPDL::getMinMass(egId));
    pythiaPtr->particleData.mMax      (pyId, EvtPDL::getMaxMass(egId));
    pythiaPtr->particleData.tau0      (pyId, EvtPDL::getctau(egId));
  }
}

//--------------------------------------------------------------------------

// Update the EvtGen particle database from Pythia.

// The particle update database between Pythia and EvtGen is not
// symmetric. Particularly, it is not possible to update the spin
// type, charge, or nominal lifetime in the EvtGen particle database.

void EvtGenDecays::updateEvtGen() {
  if (!pythiaPtr || !evtgen) return;
  int pyId = pythiaPtr->particleData.nextId(1);
  while (pyId != 0) {
    EvtId egId = EvtPDL::evtIdFromStdHep(pyId);
    EvtPDL::reSetMass   (egId, pythiaPtr->particleData.m0(pyId));
    EvtPDL::reSetWidth  (egId, pythiaPtr->particleData.mWidth(pyId));
    EvtPDL::reSetMassMin(egId, pythiaPtr->particleData.mMin(pyId));
    EvtPDL::reSetMassMax(egId, pythiaPtr->particleData.mMax(pyId));
    pyId = pythiaPtr->particleData.nextId(pyId);
  }
}

//--------------------------------------------------------------------------

// Update the particles to decay with EvtGen, and the selected signals.

// If final is false, then only signals are initialized in the signal
// map. Any particle or alias that ends with signalSuffix is taken as
// a signal particle. If final is true all particle entries in EvtGen
// are checked to see if they should be set stable in Pythia. If an
// EvtGen particle has no decay modes, then Pythia is still allowed to
// decay the particle. Additionally, the signal decay channels are
// turned off for the non-aliased signal particle.

void EvtGenDecays::updateData(bool final) {

  // Loop over the EvtGen entries.
  if (!pythiaPtr) return;
  EvtDecayTable &egTable = EvtDecayTable::getInstance();
  for (int iEntry = 0; iEntry < (int)EvtPDL::entries(); ++iEntry) {
    EvtId egId = EvtPDL::getEntry(iEntry);
    int   pyId = EvtPDL::getStdHep(egId);
    if (egTable.getNModes(egId) == 0) continue;
    if (excIds.find(pyId) != excIds.end()) continue;

    // Stop Pythia from decaying the particle and include in decay set.
    if (final)  {
      incIds.insert(pyId);
      pythiaPtr->particleData.mayDecay(pyId, false);
    }

    // Check for signal.
    string egName = EvtPDL::name(egId);
    if (egName.size() <= signalSuffix.size() || egName.substr
        (egName.size() - signalSuffix.size()) != signalSuffix) continue;
    signal = signals.find(pyId);
    if (signal == signals.end()) {
      signals[pyId].status = -1;
      signal = signals.find(pyId);
    }
    signal->second.egId  = egId;
    signal->second.bfs   = vector<double>(2, 0);
    if (!final) continue;

    // Get the signal and background decay modes.
    vector<EvtParticleDecayList> egList = egTable.getDecayTable();
    int sigIdx = egId.getAlias();
    int bkgIdx = EvtPDL::evtIdFromStdHep(pyId).getAlias();
    if (sigIdx > (int)egList.size() || bkgIdx > (int)egList.size()) continue;
    EvtParticleDecayList sigModes = egList[sigIdx];
    EvtParticleDecayList bkgModes = egList[bkgIdx];
    EvtParticleDecayList allModes = egList[bkgIdx];
    double sum(0);

    // Sum signal branching fractions.
    for (int iMode = 0; iMode < sigModes.getNMode(); ++iMode) {
      EvtDecayBase *mode = sigModes.getDecayModel(iMode);
      if (!mode) continue;
      signal->second.bfs[0] += mode->getBranchingFraction();
      sum += mode->getBranchingFraction();
    }

    // Sum remaining background branching fractions.
    for (int iMode = 0; iMode < allModes.getNMode(); ++iMode) {
      EvtDecayBase *mode = allModes.getDecayModel(iMode);
      if (!mode) continue;
      bool match(false);
      for (int jMode = 0; jMode < sigModes.getNMode(); ++jMode)
        if (mode->matchingDecay(*(sigModes.getDecayModel(jMode)))) {
          match = true; break;}
      if (match) bkgModes.removeMode(mode);
      else {
        signal->second.map.push_back(iMode);
        signal->second.bfs[1] += mode->getBranchingFraction();
        sum += mode->getBranchingFraction();
      }
    }
    signal->second.modes = bkgModes;
    for (int iBf = 0; iBf < (int)signal->second.bfs.size(); ++iBf)
      signal->second.bfs[iBf] /= sum;
  }
  if (final) updated = true;

}

//--------------------------------------------------------------------------

// Update the Pythia event record with an EvtGen decay tree.

// The production vertex of each particle (which can also be obtained
// in EvtGen via EvtParticle::get4Pos()) is set by the decay vertex of
// its mother, which in turn is calculated from the mother's
// lifetime. The status code 93 is used to indicate an external decay,
// while the status code 94 is used to indicate an oscillated external
// decay. If the progenitor has a single daughter with the same ID,
// this daughter is used as the progenitor. This is used to prevent
// double oscillations.

// If the arguments after egPro are no NULL and a particle in the
// decay tree is a signal particle, the decay for this particle is
// removed and the particle is stored as a signal candidate in the
// pySigs and egSigs vectors, to be decayed later. However, if any of
// these arguments is NULL then the entire tree is written.

void EvtGenDecays::updateEvent(Particle *pyPro, EvtParticle *egPro,
  vector<int> *pySigs, vector<EvtParticle*> *egSigs,
  vector<double> *bfs, double *wgt) {

  // Set up the mother vector.
  if (!pythiaPtr) return;
  EvtParticle* egMom = egPro;
  if (egPro->getNDaug() == 1 && egPro->getPDGId() ==
      egPro->getDaug(0)->getPDGId()) egMom = egPro->getDaug(0);
  Event &event = pythiaPtr->event;
  vector< pair<EvtParticle*, int> >
    moms(1, pair<EvtParticle*, int>(egMom, pyPro->index()));

  // Loop over the mothers.
  while (moms.size() != 0) {

    // Check if particle can decay.
    egMom = moms.back().first;
    int       iMom  = moms.back().second;
    Particle* pyMom = &event[iMom];
    moms.pop_back();
    if (!checkVertex(pyMom)) continue;
    bool osc(checkOsc(egMom));

    // Set the children of the mother.
    pyMom->daughters(event.size(), event.size() + egMom->getNDaug() - 1);
    pyMom->statusNeg();
    Vec4 vProd = pyMom->vDec();
    for (int iDtr = 0 ; iDtr < (int)egMom->getNDaug(); ++iDtr) {
      EvtParticle *egDtr = egMom->getDaug(iDtr);
      int          id    = egDtr->getPDGId();
      EvtVector4R  p     = egDtr->getP4Lab();
      int idx = event.append(id, 93, iMom, 0, 0, 0, 0, 0, p.get(1),
                             p.get(2), p.get(3), p.get(0), egDtr->mass());
      Particle *pyDtr = &event.back();
      pyDtr->vProd(vProd);
      pyDtr->tau(egDtr->getLifetime());
      if (pySigs && egSigs && bfs && wgt && checkSignal(pyDtr)) {
        pySigs->push_back(pyDtr->index());
        egSigs->push_back(egDtr);
        bfs->push_back(signal->second.bfs[0]);
        (*wgt) *= 1 - bfs->back();
        egDtr->deleteDaughters();
      }
      if (osc) pyDtr->status(94);
      if (egDtr->getNDaug() > 0)
        moms.push_back(pair<EvtParticle*, int>(egDtr, idx));
    }
  }

}

//--------------------------------------------------------------------------

// Check if a particle can decay.

// Modified slightly from ParticleDecays::checkVertex.

bool EvtGenDecays::checkVertex(Particle *pyPro) {
  if (!limitDecay) return true;
  if (limitTau0 && pyPro->tau0() > tau0Max) return false;
  if (limitTau && pyPro->tau() > tauMax) return false;
  if (limitRadius && pow2(pyPro->xDec()) + pow2(pyPro->yDec())
    + pow2(pyPro->zDec()) > pow2(rMax)) return false;
  if (limitCylinder && (pow2(pyPro->xDec()) + pow2(pyPro->yDec())
    > pow2(xyMax) || abs(pyPro->zDec()) > zMax) ) return false;
  return true;
}

//--------------------------------------------------------------------------

// Check if a particle is signal.

bool EvtGenDecays::checkSignal(Particle *pyPro) {
  signal = signals.find(pyPro->id());
  if (signal != signals.end() && (signal->second.status < 0 ||
    signal->second.status == pyPro->status())) return true;
  else return false;
}

//--------------------------------------------------------------------------

// Check if an EvtGen particle has oscillated.

// The criteria defined here for oscillation is a single daughter but
// with a different ID from the mother.

bool EvtGenDecays::checkOsc(EvtParticle *egPro) {
  if (egPro && egPro->getNDaug() == 1 &&
      egPro->getPDGId() != egPro->getDaug(0)->getPDGId()) return true;
  else return false;
}

//==========================================================================

} // end namespace Pythia8

#endif // CMSSW_Pythia8_EvtGen_Patched_H
