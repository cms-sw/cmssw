#ifndef HydjetHadronizer_h
#define HydjetHadronizer_h

/**
   \class HydjetHadronizer
   \brief Interface to the HYDJET generator (since core v. 1.9.1), produces HepMC events
   \version 2.0
   \authors Camelia Mironov, Yetkin Yilmaz, Andrey Belyaev
*/

#define PYCOMP pycomp_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "GeneratorInterface/Core/interface/BaseHadronizer.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"

#include <map>
#include <string>
#include <vector>
#include <cmath>

namespace CLHEP {
  class HepRandomEngine;
}

namespace HepMC {
  class GenEvent;
  class GenParticle;
  class GenVertex;
  class FourVector;
}  // namespace HepMC

namespace gen {
  class Pythia6Service;

  class HydjetHadronizer : public BaseHadronizer {
  public:
    HydjetHadronizer(const edm::ParameterSet&, edm::ConsumesCollector&&);
    ~HydjetHadronizer() override;

    bool generatePartonsAndHadronize();
    bool hadronize();
    bool decay();
    bool residualDecay();
    bool readSettings(int);
    bool initializeForExternalPartons();
    bool initializeForInternalPartons();
    bool declareStableParticles(const std::vector<int>&);
    bool declareSpecialSettings(const std::vector<std::string>&) { return true; }

    void finalizeEvent();
    void statistics();
    const char* classname() const;

  private:
    void doSetRandomEngine(CLHEP::HepRandomEngine* v) override;
    std::vector<std::string> const& doSharedResources() const override { return theSharedResources; }

    static const std::vector<std::string> theSharedResources;

    void add_heavy_ion_rec(HepMC::GenEvent* evt);
    HepMC::GenParticle* build_hyjet(int index, int barcode);
    HepMC::GenVertex* build_hyjet_vertex(int i, int id);
    bool get_particles(HepMC::GenEvent* evt);
    bool call_hyinit(double energy, double a, int ifb, double bmin, double bmax, double bfix, int nh);
    bool hydjet_init(const edm::ParameterSet& pset);
    inline double nuclear_radius() const;
    void rotateEvtPlane();
    int convertStatus(int st);

    HepMC::GenEvent* evt;
    edm::ParameterSet pset_;
    double abeamtarget_;                  ///< beam/target atomic mass number
    int angularspecselector_;             ///< angular emitted gluon spectrum selection
    double bfixed_;                       ///< fixed impact param (fm); valid only if cflag_=0
    double bmax_;                         ///< max impact param;
                                          ///< units of nucl radius
    double bmin_;                         ///< min impact param;
                                          ///< units of nucl radius
    int cflag_;                           ///< centrality flag
                                          ///< =  0 fixed impact param,
                                          ///< <> 0 between bmin and bmax
    bool embedding_;                      ///< Switch for embedding mode
    double comenergy;                     ///< collision energy
    bool doradiativeenloss_;              ///< DEFAULT = true
    bool docollisionalenloss_;            ///< DEFAULT = true
    double fracsoftmult_;                 ///< fraction of soft hydro induced hadronic multiplicity
                                          ///< proportional to no of nucleon participants
                                          ///< (1-fracsoftmult_)--- fraction of soft
                                          ///< multiplicity proportional to the numebr
                                          ///< of nucleon-nucleon binary collisions
                                          ///< DEFAULT=1., allowed range [0.01,1]
    double hadfreeztemp_;                 ///< hadron freez-out temperature
                                          ///< DEFAULT=0.14MeV, allowed ranges [0.08,0.2]MeV
    std::string hymode_;                  ///< Hydjet running mode
    unsigned int maxEventsToPrint_;       ///< Events to print if verbosity
    double maxlongy_;                     ///< max longitudinal collective rapidity:
                                          ///< controls width of eta-spectra
                                          ///< DEFAULT=4, allowed range [0.01,7.0]
    double maxtrany_;                     ///< max transverse collective rapidity:
                                          ///< controls slope of low-pt spectra
                                          ///< DEFAULT=1.5, allowed range [0.01,3.0]
    int nsub_;                            ///< number of sub-events
    int nhard_;                           ///< multiplicity of PYTHIA(+PYQUEN)-induced particles in event
    int nmultiplicity_;                   ///< mean soft multiplicity in central PbPb
                                          ///< automatically calculated for other centralitie and beams
    int nsoft_;                           ///< multiplicity of HYDRO-induced particles in event
    unsigned int nquarkflavor_;           ///< number of active quark flavors in qgp
                                          ///< DEFAULT=0; allowed values: 0,1,2,3.
    unsigned int pythiaPylistVerbosity_;  ///< pythia verbosity; def=1
    double qgpt0_;                        ///< initial temperature of QGP
                                          ///< DEFAULT = 1GeV; allowed range [0.2,2.0]GeV;
    double qgptau0_;                      ///< proper time of QGP formation
                                          ///< DEFAULT = 0.1 fm/c; allowed range [0.01,10.0]fm/

    double phi0_;  ///< Event plane angle
    double sinphi0_;
    double cosphi0_;
    bool rotate_;  ///< Switch to rotate event plane

    unsigned int shadowingswitch_;   ///< shadowing switcher
                                     ///< 1-ON, 0-OFF
    double signn_;                   ///< inelastic nucleon nucleon cross section [mb]
                                     ///< DEFAULT= 58 mb
    HepMC::FourVector* fVertex_;     ///< Event signal vertex
    std::vector<double> signalVtx_;  ///< Pset double vector to set event signal vertex

    Pythia6Service* pythia6Service_;
    edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct> > src_;
  };

  /**
    Return the nuclear radius derived from the
    beam/target atomic mass number.
 */
  double HydjetHadronizer::nuclear_radius() const { return 1.15 * pow((double)abeamtarget_, 1. / 3.); }
}  // namespace gen
#endif
