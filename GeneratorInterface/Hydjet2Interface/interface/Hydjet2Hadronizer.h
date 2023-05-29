#ifndef Hydjet2Hadronizer_h
#define Hydjet2Hadronizer_h

/**
 *    \class HydjetHadronizer
 *    \brief Interface to the HYDJET++ (Hydjet2) generator (since core v. 2.4.2), produces HepMC events
 *    \version 1.1
 *    \author Andrey Belyaev
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"

#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <Hydjet2.h>
#include <InitialParams.h>

namespace CLHEP {
  class HepRandomEngine;
  class RandFlat;
  class RandPoisson;
  class RandGauss;
}  // namespace CLHEP

namespace HepMC {
  class GenEvent;
  class GenParticle;
  class GenVertex;
  class FourVector;
}  // namespace HepMC

namespace gen {
  class Pythia6Service;
  class Hydjet2Hadronizer : public BaseHadronizer {
  public:
    Hydjet2Hadronizer(const edm::ParameterSet &, edm::ConsumesCollector &&);
    ~Hydjet2Hadronizer() override;

    bool readSettings(int);
    bool declareSpecialSettings(const std::vector<std::string> &) { return true; }
    bool initializeForInternalPartons();
    bool initializeForExternalPartons();  //0
    bool generatePartonsAndHadronize();
    bool declareStableParticles(const std::vector<int> &);

    bool hadronize();  //0
    bool decay();      //0
    bool residualDecay();
    void finalizeEvent();
    void statistics();
    const char *classname() const;

  private:
    void doSetRandomEngine(CLHEP::HepRandomEngine *v) override;
    void rotateEvtPlane();
    bool get_particles(HepMC::GenEvent *evt);
    HepMC::GenParticle *build_hyjet2(int index, int barcode);
    HepMC::GenVertex *build_hyjet2_vertex(int i, int id);
    void add_heavy_ion_rec(HepMC::GenEvent *evt);

    std::vector<std::string> const &doSharedResources() const override { return theSharedResources; }
    static const std::vector<std::string> theSharedResources;

    inline double nuclear_radius() const;

    int convertStatusForComponents(int, int, int);
    int convertStatus(int);

    InitialParamsHydjet_t fParams;
    Hydjet2 *hj2;

    bool ev = false;
    bool separateHydjetComponents_;
    bool rotate_;  // Switch to rotate event plane
    HepMC::GenEvent *evt;
    int nsub_;     // number of sub-events
    int nhard_;    // multiplicity of PYTHIA(+PYQUEN)-induced particles in event
    int nsoft_;    // multiplicity of HYDRO-induced particles in event
    double phi0_;  // Event plane angle
    double sinphi0_;
    double cosphi0_;

    unsigned int pythiaPylistVerbosity_;  // pythia verbosity; def=1
    unsigned int maxEventsToPrint_;       // Events to print if verbosity

    edm::ParameterSet pset;
    double Sigin, Sigjet;

    HepMC::FourVector *fVertex_;  // Event signal vertex

    std::vector<double> signalVtx_;  // Pset double vector to set event signal vertex

    Pythia6Service *pythia6Service_;
    edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct>> src_;
  };

  double Hydjet2Hadronizer::nuclear_radius() const {
    // Return the nuclear radius derived from the
    // beam/target atomic mass number.

    return 1.15 * pow((double)fParams.fAw, 1. / 3.);
  }
}  // namespace gen
#endif
