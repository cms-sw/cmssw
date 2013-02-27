// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/IdentifiedFinalState.hh"
#include "Rivet/Tools/ParticleIdUtils.hh"

namespace Rivet {


  class CMS_2011_I954992 : public Analysis {
  public:

    CMS_2011_I954992()
      : Analysis("CMS_2011_I954992")
    {    }


  public:

    void init() {

      ChargedFinalState cfs(-2.4, 2.4, 0.0*GeV);
      addProjection(cfs,"CFS");

      /// Get muons which pass the initial kinematic cuts
      IdentifiedFinalState muon_fs(-2.1, 2.1, 4.0*GeV);
      muon_fs.acceptIdPair(MUON);
      addProjection(muon_fs, "MUON_FS");

      _h_sigma = bookHistogram1D(1,1,1);

    }

    void analyze(const Event& event) {
      const double weight = event.weight();

      const ChargedFinalState& cfs = applyProjection<ChargedFinalState>(event, "CFS");
      if (cfs.size() != 2) vetoEvent; // no other charged particles in 2.4

      const ParticleVector& muonFS = applyProjection<IdentifiedFinalState>(event, "MUON_FS").particles();
      if(muonFS.size() != 2) vetoEvent;

      if(PID::charge(muonFS.at(0)) != PID::charge(muonFS.at(1))) {

         const double dimuon_mass = (muonFS.at(0).momentum() + muonFS.at(1).momentum()).mass();
	 const double v_angle = muonFS.at(0).momentum().angle(muonFS.at(1).momentum());
         const double dPhi = deltaPhi(muonFS.at(0).momentum().phi(), muonFS.at(1).momentum().phi());
         const double deltaPt = fabs(muonFS.at(0).momentum().pT() - muonFS.at(1).momentum().pT());

         if (dimuon_mass >= 11.5*GeV) { 
	    if (v_angle < 0.95*PI) { 
	       if ( (1-fabs(dPhi/PI)) < 0.1) {
	          if (deltaPt < 1.) {
	             _h_sigma->fill(sqrtS()/GeV, weight);
		  }
	       }
	    }
         }
      }

    }

    /// Normalise histograms etc., after the run
    void finalize() {

      scale(_h_sigma, crossSection()/picobarn/sumOfWeights());

    }

  private:

    AIDA::IHistogram1D * _h_sigma;

  };


  // The hook for the plugin system
  DECLARE_RIVET_PLUGIN(CMS_2011_I954992);

}
