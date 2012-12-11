// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/FinalState.hh"

namespace Rivet {


  class CMS_2012_I1193338 : public Analysis {
  public:

    CMS_2012_I1193338()
      : Analysis("CMS_2012_I1193338")
    {    }

  public:

    void init() {

      addProjection(ChargedFinalState(-2.4, 2.4, 0.2*GeV), "CFS");
      addProjection(FinalState(),"FS");

      _hist_sigma = bookHistogram1D(1, 1, 1);
	
    }

    void analyze(const Event& event) {

    const double weight = event.weight();

    const ChargedFinalState& cfs = applyProjection<ChargedFinalState>(event, "CFS");
    if (cfs.size() > 1) {_hist_sigma->fill(1.5, weight);}
    if (cfs.size() > 2) {_hist_sigma->fill(2.5, weight);}
    if (cfs.size() > 3) {_hist_sigma->fill(3.5, weight);}

    const FinalState& fs = applyProjection<FinalState>(event, "FS");
    if (fs.size() < 2) vetoEvent; // need at least two particles to calculate gaps

    // Calculate gap sizes and midpoints
    const ParticleVector particlesByEta = fs.particlesByEta(); // sorted from minus to plus
    const size_t num_particles = particlesByEta.size();

    vector<double> gaps;
    vector<double> midpoints;
    for (size_t ip = 1; ip < num_particles; ++ip) {
      const Particle& p1 = particlesByEta[ip-1];
      const Particle& p2 = particlesByEta[ip];
      const double gap = p2.momentum().eta() - p1.momentum().eta();
      const double mid = (p2.momentum().eta() + p1.momentum().eta()) / 2.;
      assert(gap >= 0);
      gaps.push_back(gap);
      midpoints.push_back(mid);
    }

    // Find the midpoint of the largest gap to separate X and Y systems
    int imid = std::distance(gaps.begin(), max_element(gaps.begin(), gaps.end()));
    double gapcenter = midpoints[imid];

    FourMomentum MxFourVector(0.,0.,0.,0.);
    FourMomentum MyFourVector(0.,0.,0.,0.);

    foreach(const Particle& p, fs.particlesByEta()) {
        if (p.momentum().eta() > gapcenter) {
                MxFourVector += p.momentum();
        } else {
                MyFourVector += p.momentum();
        }
    }

    double Mx2 = MxFourVector.mass2();
    double My2 = MyFourVector.mass2();

    const double M2 = (Mx2 > My2 ? Mx2 : My2);
    const double xi = M2/(sqrtS()/GeV * sqrtS()/GeV);

    if (xi < 5*10e-6) vetoEvent;

    _hist_sigma->fill(0.5, weight);

    }

    void finalize() {

      scale(_hist_sigma, crossSection()/millibarn/sumOfWeights());

    }

  private:

    AIDA::IHistogram1D *_hist_sigma;

  };


  DECLARE_RIVET_PLUGIN(CMS_2012_I1193338);

}
