// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/IdentifiedFinalState.hh"
#include "Rivet/Particle.hh"

namespace Rivet {


  class CMS_2011_S8941262 : public Analysis {
  public:


    /// Constructor
    CMS_2011_S8941262()
      : Analysis("CMS_2011_S8941262")
    {
      setNeedsCrossSection(true);
    }


  public:

    /// Book histograms and initialise projections before the run
    void init() {


      _h_total  = bookHistogram1D(1, 1, 1);
      _h_mupt  = bookHistogram1D(2, 1, 1);
      _h_mueta = bookHistogram1D(3, 1, 1);
      nbtot=0;   nbmutot=0;

      IdentifiedFinalState ifs(-2.1, 2.1, 6.0*GeV);
      ifs.acceptIdPair( MUON);
      addProjection(ifs, "IFS");
 
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      const double weight = event.weight();


      // a b-quark must have been produced
      int nb=0;
      foreach (const GenParticle* p, particles(event.genEvent())) {
	if (fabs(p->pdg_id()) == BQUARK) { nb+=1; }
      }
      if(nb==0){
	vetoEvent;
      }

      nbtot+=weight;

      // event must contain a muon
      ParticleVector muons = applyProjection<IdentifiedFinalState>(event, "IFS").particlesByPt();
      if ( (muons.size() < 1) ){ vetoEvent; }
      nbmutot+=weight;

      Particle muon=muons[0];
      FourMomentum pmu = muon.momentum();

      _h_total->fill(      7000/GeV, weight);
      _h_mupt->fill(   pmu.pT()/GeV, weight);
      _h_mueta->fill( pmu.eta()/GeV, weight);
	

    }


    /// Normalise histograms etc., after the run
    void finalize() {

      scale(_h_total, crossSection()/microbarn/sumOfWeights());
      scale(_h_mupt,  crossSection()/nanobarn/sumOfWeights());
      scale(_h_mueta, crossSection()/nanobarn/sumOfWeights());

    }


  private:


    AIDA::IHistogram1D *_h_total;
    AIDA::IHistogram1D *_h_mupt;
    AIDA::IHistogram1D *_h_mueta;
    int nbtot,nbmutot;


  };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2011_S8941262> plugin_CMS_2011_S8941262;


}
