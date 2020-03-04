#include <iostream>
#include "GeneratorInterface/PhotosInterface/interface/PhotosppInterface.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosFactory.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "HepMC/GenEvent.h"
#include "HepMC/IO_HEPEVT.h"
#include "HepMC/HEPEVT_Wrapper.h"

using namespace gen;
using namespace edm;
using namespace std;

#include "Photos/Photos.h"
#include "Photos/PhotosHepMCEvent.h"

CLHEP::HepRandomEngine* PhotosppInterface::fRandomEngine = nullptr;

PhotosppInterface::PhotosppInterface(const edm::ParameterSet& pset)
    : fOnlyPDG(-1), fAvoidTauLeptonicDecays(false), fIsInitialized(false), fPSet(nullptr) {
  // add ability to keep brem from hadronizer and only modify specific channels 10/27/2014
  bool UseHadronizerQEDBrem = false;
  fPSet = new ParameterSet(pset);
  std::vector<std::string> par = fPSet->getParameter<std::vector<std::string> >("parameterSets");
  for (unsigned int ip = 0; ip < par.size(); ++ip) {
    std::string curSet = par[ip];
    // Physics settings
    if (curSet == "UseHadronizerQEDBrem")
      UseHadronizerQEDBrem = true;
  }
  if (!UseHadronizerQEDBrem)
    fSpecialSettings.push_back("QED-brem-off:all");
}

void PhotosppInterface::setRandomEngine(CLHEP::HepRandomEngine* decayRandomEngine) {
  fRandomEngine = decayRandomEngine;
}

void PhotosppInterface::configureOnlyFor(int ipdg) {
  fOnlyPDG = ipdg;
  fSpecialSettings.clear();
  return;
}

void PhotosppInterface::init() {
  if (fIsInitialized)
    return;  // do init only once
  Photospp::Photos::initialize();
  Photospp::Photos::createHistoryEntries(true, 746);  // P-H-O
  std::vector<std::string> par = fPSet->getParameter<std::vector<std::string> >("parameterSets");
  for (unsigned int ip = 0; ip < par.size(); ++ip) {
    std::string curSet = par[ip];

    // Physics settings
    if (curSet == "maxWtInterference")
      Photospp::Photos::maxWtInterference(fPSet->getParameter<double>(curSet));
    if (curSet == "setInfraredCutOff")
      Photospp::Photos::setInfraredCutOff(fPSet->getParameter<double>(curSet));
    if (curSet == "setAlphaQED")
      Photospp::Photos::setAlphaQED(fPSet->getParameter<double>(curSet));
    if (curSet == "setInterference")
      Photospp::Photos::setInterference(fPSet->getParameter<bool>(curSet));
    if (curSet == "setDoubleBrem")
      Photospp::Photos::setDoubleBrem(fPSet->getParameter<bool>(curSet));
    if (curSet == "setQuatroBrem")
      Photospp::Photos::setQuatroBrem(fPSet->getParameter<bool>(curSet));
    if (curSet == "setExponentiation")
      Photospp::Photos::setExponentiation(fPSet->getParameter<bool>(curSet));
    if (curSet == "setCorrectionWtForW")
      Photospp::Photos::setCorrectionWtForW(fPSet->getParameter<bool>(curSet));
    if (curSet == "setMeCorrectionWtForScalar")
      Photospp::Photos::setMeCorrectionWtForScalar(fPSet->getParameter<bool>(curSet));
    if (curSet == "setMeCorrectionWtForW")
      Photospp::Photos::setMeCorrectionWtForW(fPSet->getParameter<bool>(curSet));
    if (curSet == "setMeCorrectionWtForZ")
      Photospp::Photos::setMeCorrectionWtForZ(fPSet->getParameter<bool>(curSet));
    if (curSet == "initializeKinematicCorrections")
      Photospp::Photos::initializeKinematicCorrections(fPSet->getParameter<int>(curSet));
    if (curSet == "forceMassFrom4Vector")
      Photospp::Photos::forceMassFrom4Vector(fPSet->getParameter<bool>(curSet));
    if (curSet == "forceMassFromEventRecord")
      Photospp::Photos::forceMassFromEventRecord(fPSet->getParameter<int>(curSet));
    if (curSet == "ignoreParticlesOfStatus")
      Photospp::Photos::ignoreParticlesOfStatus(fPSet->getParameter<int>(curSet));
    if (curSet == "deIgnoreParticlesOfStatus")
      Photospp::Photos::deIgnoreParticlesOfStatus(fPSet->getParameter<int>(curSet));
    if (curSet == "setMomentumConservationThreshold")
      Photospp::Photos::setMomentumConservationThreshold(fPSet->getParameter<double>(curSet));
    if (curSet == "suppressAll")
      if (fPSet->getParameter<bool>(curSet) == true)
        Photospp::Photos::suppressAll();
    if (curSet == "setPairEmission")
      Photospp::Photos::setPairEmission(fPSet->getParameter<bool>(curSet));
    if (curSet == "setPhotonEmission")
      Photospp::Photos::setPhotonEmission(fPSet->getParameter<bool>(curSet));
    if (curSet == "setStopAtCriticalError")
      Photospp::Photos::setStopAtCriticalError(fPSet->getParameter<bool>(curSet));
    if (curSet == "createHistoryEntries")
      Photospp::Photos::createHistoryEntries(fPSet->getParameter<bool>(curSet), 746);

    // Now setup more complicated radiation/mass supression and forcing.
    if (curSet == "suppressBremForBranch") {
      edm::ParameterSet cfg = fPSet->getParameter<edm::ParameterSet>(curSet);
      std::vector<std::string> v = cfg.getParameter<std::vector<std::string> >("parameterSets");
      for (unsigned int i = 0; i < v.size(); i++) {
        std::string vs = v[i];
        std::vector<int> vpar = cfg.getParameter<std::vector<int> >(vs);
        if (vpar.size() == 1)
          Photospp::Photos::suppressBremForBranch(0, vpar[0]);
        if (vpar.size() == 2)
          Photospp::Photos::suppressBremForBranch(0 /*vpar[0]*/, vpar[1]);
        if (vpar.size() == 3)
          Photospp::Photos::suppressBremForBranch(vpar[0], vpar[1], vpar[2]);
        if (vpar.size() == 4)
          Photospp::Photos::suppressBremForBranch(vpar[0], vpar[1], vpar[2], vpar[3]);
        if (vpar.size() == 5)
          Photospp::Photos::suppressBremForBranch(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4]);
        if (vpar.size() == 6)
          Photospp::Photos::suppressBremForBranch(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5]);
        if (vpar.size() == 7)
          Photospp::Photos::suppressBremForBranch(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6]);
        if (vpar.size() == 8)
          Photospp::Photos::suppressBremForBranch(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7]);
        if (vpar.size() == 9)
          Photospp::Photos::suppressBremForBranch(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7], vpar[8]);
        if (vpar.size() == 10)
          Photospp::Photos::suppressBremForBranch(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7], vpar[8], vpar[9]);
      }
    }
    if (curSet == "suppressBremForDecay") {
      edm::ParameterSet cfg = fPSet->getParameter<edm::ParameterSet>(curSet);
      std::vector<std::string> v = cfg.getParameter<std::vector<std::string> >("parameterSets");
      for (unsigned int i = 0; i < v.size(); i++) {
        std::string vs = v[i];
        std::vector<int> vpar = cfg.getParameter<std::vector<int> >(vs);
        if (vpar.size() == 1)
          Photospp::Photos::suppressBremForDecay(0, vpar[0]);
        if (vpar.size() == 2)
          Photospp::Photos::suppressBremForDecay(0 /*vpar[0]*/, vpar[1]);
        if (vpar.size() == 3)
          Photospp::Photos::suppressBremForDecay(vpar[0], vpar[1], vpar[2]);
        if (vpar.size() == 4)
          Photospp::Photos::suppressBremForDecay(vpar[0], vpar[1], vpar[2], vpar[3]);
        if (vpar.size() == 5)
          Photospp::Photos::suppressBremForDecay(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4]);
        if (vpar.size() == 6)
          Photospp::Photos::suppressBremForDecay(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5]);
        if (vpar.size() == 7)
          Photospp::Photos::suppressBremForDecay(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6]);
        if (vpar.size() == 8)
          Photospp::Photos::suppressBremForDecay(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7]);
        if (vpar.size() == 9)
          Photospp::Photos::suppressBremForDecay(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7], vpar[8]);
        if (vpar.size() == 10)
          Photospp::Photos::suppressBremForDecay(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7], vpar[8], vpar[9]);
      }
    }

    if (curSet == "forceBremForBranch") {
      edm::ParameterSet cfg = fPSet->getParameter<edm::ParameterSet>(curSet);
      std::vector<std::string> v = cfg.getParameter<std::vector<std::string> >("parameterSets");
      for (unsigned int i = 0; i < v.size(); i++) {
        std::string vs = v[i];
        std::vector<int> vpar = cfg.getParameter<std::vector<int> >(vs);
        if (vpar.size() == 1)
          Photospp::Photos::forceBremForBranch(0, vpar[0]);
        if (vpar.size() == 2)
          Photospp::Photos::forceBremForBranch(0 /*vpar[0]*/, vpar[1]);
        if (vpar.size() == 3)
          Photospp::Photos::forceBremForBranch(vpar[0], vpar[1], vpar[2]);
        if (vpar.size() == 4)
          Photospp::Photos::forceBremForBranch(vpar[0], vpar[1], vpar[2], vpar[3]);
        if (vpar.size() == 5)
          Photospp::Photos::forceBremForBranch(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4]);
        if (vpar.size() == 6)
          Photospp::Photos::forceBremForBranch(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5]);
        if (vpar.size() == 7)
          Photospp::Photos::forceBremForBranch(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6]);
        if (vpar.size() == 8)
          Photospp::Photos::forceBremForBranch(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7]);
        if (vpar.size() == 9)
          Photospp::Photos::forceBremForBranch(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7], vpar[8]);
        if (vpar.size() == 10)
          Photospp::Photos::forceBremForBranch(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7], vpar[8], vpar[9]);
      }
    }
    if (curSet == "forceBremForDecay") {
      edm::ParameterSet cfg = fPSet->getParameter<edm::ParameterSet>(curSet);
      std::vector<std::string> v = cfg.getParameter<std::vector<std::string> >("parameterSets");
      for (unsigned int i = 0; i < v.size(); i++) {
        std::string vs = v[i];
        std::vector<int> vpar = cfg.getParameter<std::vector<int> >(vs);
        if (vpar.size() == 1)
          Photospp::Photos::forceBremForDecay(0, vpar[0]);
        if (vpar.size() == 2)
          Photospp::Photos::forceBremForDecay(0 /*vpar[0]*/, vpar[1]);
        if (vpar.size() == 3)
          Photospp::Photos::forceBremForDecay(vpar[0], vpar[1], vpar[2]);
        if (vpar.size() == 4)
          Photospp::Photos::forceBremForDecay(vpar[0], vpar[1], vpar[2], vpar[3]);
        if (vpar.size() == 5)
          Photospp::Photos::forceBremForDecay(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4]);
        if (vpar.size() == 6)
          Photospp::Photos::forceBremForDecay(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5]);
        if (vpar.size() == 7)
          Photospp::Photos::forceBremForDecay(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6]);
        if (vpar.size() == 8)
          Photospp::Photos::forceBremForDecay(vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7]);
        if (vpar.size() == 9)
          Photospp::Photos::forceBremForDecay(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7], vpar[8]);
        if (vpar.size() == 10)
          Photospp::Photos::forceBremForDecay(
              vpar[0], vpar[1], vpar[2], vpar[3], vpar[4], vpar[5], vpar[6], vpar[7], vpar[8], vpar[9]);
      }
    }

    if (curSet == "forceMass") {
      edm::ParameterSet cfg = fPSet->getParameter<edm::ParameterSet>(curSet);
      std::vector<std::string> v = cfg.getParameter<std::vector<std::string> >("parameterSets");
      for (unsigned int i = 0; i < v.size(); i++) {
        std::string vs = v[i];
        std::vector<double> vpar = cfg.getParameter<std::vector<double> >(vs);
        if (vpar.size() == 2)
          Photospp::Photos::forceMass((int)vpar[0], vpar[1]);
      }
    }
  }
  // implement options set via c++
  if (fOnlyPDG >= 0) {
    Photospp::Photos::suppressAll();
    Photospp::Photos::forceBremForBranch(0, fOnlyPDG);
    Photospp::Photos::forceBremForBranch(0, -1 * fOnlyPDG);
  }
  if (fAvoidTauLeptonicDecays) {
    Photospp::Photos::suppressBremForDecay(3, 15, 16, 11, -12);
    Photospp::Photos::suppressBremForDecay(3, -15, -16, -11, 12);
    Photospp::Photos::suppressBremForDecay(3, 15, 16, 13, -14);
    Photospp::Photos::suppressBremForDecay(3, -15, -16, -13, -14);
  }
  Photospp::Photos::iniInfo();
  fIsInitialized = true;
  return;
}

HepMC::GenEvent* PhotosppInterface::apply(HepMC::GenEvent* evt) {
  Photospp::Photos::setRandomGenerator(PhotosppInterface::flat);
  if (!fIsInitialized)
    return evt;
  int NPartBefore = evt->particles_size();
  Photospp::PhotosHepMCEvent PhotosEvt(evt);
  PhotosEvt.process();
  //Fix the vertices and barcodes based on Julia Yarba's solution from TauolaInterface
  for (HepMC::GenEvent::vertex_const_iterator vtx = evt->vertices_begin(); vtx != evt->vertices_end(); vtx++) {
    std::vector<int> BCodes;
    BCodes.clear();
    if (*vtx) {
      for (HepMC::GenVertex::particle_iterator pitr = (*vtx)->particles_begin(HepMC::children);
           pitr != (*vtx)->particles_end(HepMC::children);
           ++pitr) {
        if ((*pitr)->barcode() > 10000) {
          BCodes.push_back((*pitr)->barcode());
        }
      }
    }
    if (!BCodes.empty()) {
      for (size_t ibc = 0; ibc < BCodes.size(); ibc++) {
        HepMC::GenParticle* p1 = evt->barcode_to_particle(BCodes[ibc]);
        int nbc = p1->barcode() - 10000 + NPartBefore;
        p1->suggest_barcode(nbc);
      }
    }
  }
  return evt;
}

double PhotosppInterface::flat() {
  if (!fRandomEngine) {
    throw cms::Exception("LogicError")
        << "PhotosppInterface::flat: Attempt to generate random number when engine pointer is null\n"
        << "This might mean that the code was modified to generate a random number outside the\n"
        << "event and beginLuminosityBlock methods, which is not allowed.\n";
  }
  return fRandomEngine->flat();
}

void PhotosppInterface::statistics() { Photospp::Photos::iniInfo(); }

DEFINE_EDM_PLUGIN(PhotosFactory, gen::PhotosppInterface, "Photospp356");
