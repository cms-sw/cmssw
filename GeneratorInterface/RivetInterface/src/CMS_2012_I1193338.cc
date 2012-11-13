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
      if (cfs.size() > 1) {_hist_sigma->fill( 1.5, weight);}
      if (cfs.size() > 2) {_hist_sigma->fill( 2.5, weight);}
      if (cfs.size() > 3) {_hist_sigma->fill( 3.5, weight);}

      const FinalState& fs = applyProjection<FinalState>(event, "FS");

      int num_p = 0;
      double gapcenter(0.0), LRG(0.0), etapre(0.0);

      foreach(const Particle& p, fs.particlesByEta()) { // sorted from minus to plus
         num_p += 1;
		if (num_p == 1) { // First particle
			etapre = p.momentum().eta();
		} else if (num_p > 1) {
			double gap = fabs(p.momentum().eta()-etapre);
			if (gap > LRG) {
				LRG = gap; // largest gap
				gapcenter = (p.momentum().eta()+etapre)/2.; // find the center of the gap to separate the X and Y systems. 
			}
			etapre = p.momentum().eta();
		}
      }

      FourMomentum MxFourVector(0.,0.,0.,0.);
      FourMomentum MyFourVector(0.,0.,0.,0.);

      foreach(const Particle& p, fs.particlesByEta()) {
	  if (p.momentum().eta() > gapcenter) {
	  	MxFourVector += p.momentum();
	  } else {
	  	MyFourVector += p.momentum();
	  } 
      }

      double Mx2 = FourMomentum(MxFourVector).mass2();
      double My2 = FourMomentum(MyFourVector).mass2();

      const double M2 = (Mx2 > My2 ? Mx2 : My2);
      const double xi = M2/(7000*7000);   // sqrt(s) = 7000 GeV

      if (xi < 5*10e-6) vetoEvent;

      _hist_sigma->fill( 0.5, weight);

    }

    void finalize() {

      scale(_hist_sigma, crossSection()/millibarn/sumOfWeights());

    }

  private:

    AIDA::IHistogram1D *_hist_sigma;

  };


 //AK DECLARE_RIVET_PLUGIN(CMS_2012_I1193338);
  AnalysisBuilder<CMS_2012_I1193338> plugin_CMS_2012_I1193338;

}
