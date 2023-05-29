#ifndef Pyquen_Hadronizer_h
#define Pyquen_Hadronizer_h

/**
   \class PyquenHadronizer
   \brief Interface to the PYQUEN generator (since core v. 1.5.4), produces HepMC events
   \version 2.0
   \authors Camelia Mironov, Andrey Belyaev
*/

#include "GeneratorInterface/Core/interface/BaseHadronizer.h"
#include "GeneratorInterface/HiGenCommon/interface/BaseHiGenEvtSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupMixingContent.h"

#include <map>
#include <string>
#include <vector>

#include "HepMC/GenEvent.h"

namespace CLHEP {
  class HepRandomEngine;
}

namespace gen {
  class Pythia6Service;

  class PyquenHadronizer : public BaseHadronizer {
  public:
    PyquenHadronizer(const edm::ParameterSet&, edm::ConsumesCollector&&);
    ~PyquenHadronizer() override;

    bool generatePartonsAndHadronize();
    bool hadronize();
    bool decay();
    bool residualDecay();
    bool readSettings(int);
    bool initializeForExternalPartons();
    bool initializeForInternalPartons();
    bool declareStableParticles(const std::vector<int>&);
    bool declareSpecialSettings(const std::vector<std::string>&) { return true; }
    bool select(HepMC::GenEvent* evtTry) const override { return selector_->filter(evtTry); }
    void finalizeEvent();
    void statistics();
    const char* classname() const;

  private:
    void doSetRandomEngine(CLHEP::HepRandomEngine* v) override;
    std::vector<std::string> const& doSharedResources() const override { return theSharedResources; }

    static const std::vector<std::string> theSharedResources;

    void add_heavy_ion_rec(HepMC::GenEvent* evt);

    bool pyqpythia_init(const edm::ParameterSet& pset);
    bool pyquen_init(const edm::ParameterSet& pset);
    const char* nucleon();
    void rotateEvtPlane(HepMC::GenEvent* evt, double angle);

    edm::ParameterSet pset_;
    double abeamtarget_;                ///< beam/target atomic mass number
    unsigned int angularspecselector_;  ///< angular emitted gluon  spectrum selection
                                        ///< DEFAULT= 0 -- small angular emitted gluon spectrum
                                        ///<        = 1 -- broad angular emitted gluon spectrum
                                        ///<        = 2 -- collinear angular emitted gluon spectrum
    double bmin_;                       ///< min impact param (fm); valid only if cflag_!=0
    double bmax_;                       ///< max impact param (fm); valid only if cflag_!=0
    double bfixed_;                     ///< fixed impact param (fm); valid only if cflag_=0
    int cflag_;                         ///< centrality flag =0 fixed impact param, <>0 minbias
    double comenergy;                   ///< collision energy
    bool doquench_;                     ///< if true perform quenching (default = true)
    bool doradiativeenloss_;            ///< DEFAULT = true
    bool docollisionalenloss_;          ///< DEFAULT = true
    bool doIsospin_;                    ///< Run n&p with proper ratios; if false, only p+p collisions
    int protonSide_;
    bool embedding_;
    double evtPlane_;
    double pfrac_;  ///< Proton fraction in the nucleus

    unsigned int nquarkflavor_;      ///< number of active quark flavors in qgp
                                     ///< DEFAULT=0; allowed values: 0,1,2,3.
    double qgpt0_;                   ///< initial temperature of QGP
                                     ///< DEFAULT = 1GeV; allowed range [0.2,2.0]GeV;
    double qgptau0_;                 ///< proper time of QGP formation
                                     ///< DEFAULT = 0.1 fm/c; allowed range [0.01,10.0]fm/c;
    unsigned int maxEventsToPrint_;  ///< Events to print if verbosity

    HepMC::FourVector* fVertex_;     ///< Event signal vertex
    std::vector<double> signalVtx_;  ///< Pset double vector to set event signal vertex

    bool pythiaHepMCVerbosity_;           ///< HepMC verbosity flag
    unsigned int pythiaPylistVerbosity_;  ///< Pythia PYLIST Verbosity flag

    //    CLHEP::HepRandomEngine* fRandomEngine;
    edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct> > src_;
    Pythia6Service* pythia6Service_;
    std::string filterType_;
    BaseHiGenEvtSelector* selector_;
  };
}  // namespace gen

#endif
