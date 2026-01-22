#ifndef HTo2XTo4LGunProducer_H
#define HTo2XTo4LGunProducer_H

/** \class HTo2XTo4LGunProducer
 *
 * Generates a 4 lepton final state in HepMC format.
 * The gun starts by producing a Higgs of random mass. 
 * The Higgs is then decayed into two long-lived particles which are then propagated to their decay vertex, and decay to 2 leptons each.
 * The resulting leptons are then plugged into the event as two separate verteces - one vertex for each muon pair.
 * Neither the Higgs nor the two long-lived particles are injected into the event.
 *
 * Contact Osvaldo Miguel Colin
 ***************************************/

#include "Math/LorentzVector.h"

#include "IOMC/ParticleGuns/interface/BaseFlatGunProducer.h"

// Forward Declare
namespace HepMC {
  class FourVector;
}

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {
  class HepMCProduct;
}

namespace edm {
  typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > XYZTLorentzVectorD;

  class HTo2XTo4LGunProducer : public BaseFlatGunProducer {
  public:
    explicit HTo2XTo4LGunProducer(const ParameterSet&);
    ~HTo2XTo4LGunProducer() override;

    void produce(Event&, const EventSetup&) override;

  private:
    double min_m_h_;
    double max_m_h_;
    double min_pt_h_;
    double max_pt_h_;
    double min_ctau_llp_;
    double max_ctau_llp_;
    std::string llp_mass_spectrum_;
    double min_m_llp_;
    double max_m_llp_;
    double min_invpt_h_;
    double max_invpt_h_;

    void shoot_llp(CLHEP::HepRandomEngine*,
                   const double&,
                   const double&,
                   XYZTLorentzVectorD&,
                   XYZTLorentzVectorD&,
                   XYZTLorentzVectorD&) const;

    void decay_particle(CLHEP::HepRandomEngine*,
                        const double&,
                        const double&,
                        const double&,
                        const XYZTLorentzVectorD&,
                        XYZTLorentzVectorD&,
                        XYZTLorentzVectorD&,
                        XYZTLorentzVectorD&) const;
  };

}  // namespace edm

#endif
