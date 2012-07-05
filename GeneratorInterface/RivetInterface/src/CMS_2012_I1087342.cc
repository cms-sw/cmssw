// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/FastJets.hh"


namespace Rivet {

  // This analysis is a derived from the class Analysis:
  class CMS_2012_I1087342 : public Analysis {

 
  private:
    AIDA::IHistogram1D* _hist_jetpt_fwdincl;
    AIDA::IHistogram1D* _hist_jetpt_forward;
    AIDA::IHistogram1D* _hist_jetpt_central;

  public:
  
    // Constructor
    CMS_2012_I1087342() : Analysis("CMS_2012_I1087342") {
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(true);
    }

    void init() {      
      const FinalState fs;
      addProjection(FastJets(fs, FastJets::ANTIKT, 0.5),"Jets");

      _hist_jetpt_fwdincl = bookHistogram1D(1,1,1);
      _hist_jetpt_forward = bookHistogram1D(2,1,1);
      _hist_jetpt_central = bookHistogram1D(3,1,1);      
    }

    void analyze(const Event &event) {
      const double weight = event.weight();
      
      const FastJets &fj = applyProjection<FastJets>(event,"Jets");
      const Jets jets = fj.jets(35*GeV, 150*GeV, -4.7, 4.7, ETA);

      double cjet_pt=0.0;
      double fjet_pt=0.0;
      
      foreach(const Jet &j, jets) {
        if(j.momentum().eta() > 3.2 || j.momentum().eta() < -3.2) {
          _hist_jetpt_fwdincl -> fill(j.momentum().pT(), weight);
        }
	double pT = j.momentum().pT()*GeV;
	if (fabs(j.momentum().eta()) < 2.8) {
	  if(cjet_pt < pT) cjet_pt = pT;
	}	
	if (fabs(j.momentum().eta()) < 4.7  && fabs(j.momentum().eta()) > 3.2) {
	  if(fjet_pt < pT) fjet_pt = pT;
	}	
      }

      if (cjet_pt > 35 && fjet_pt > 35) {
	_hist_jetpt_forward->fill(fjet_pt, weight);
	_hist_jetpt_central->fill(cjet_pt, weight);
      }

    }

    void finalize() {
      scale(_hist_jetpt_fwdincl, crossSection() / picobarn / sumOfWeights() / 3.0);
      scale(_hist_jetpt_forward, crossSection() / picobarn / sumOfWeights() / 3.0); 
      scale(_hist_jetpt_central, crossSection() / picobarn / sumOfWeights() / 5.6);
    }
    
  };

  // This global object acts as a hook for the plugin system.
  AnalysisBuilder<CMS_2012_I1087342> plugin_CMS_2012_I1087342;

}

