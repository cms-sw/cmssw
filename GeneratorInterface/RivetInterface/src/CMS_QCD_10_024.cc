// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/Beam.hh"
#include "Rivet/Particle.hh"
/// @todo Include more projections as required, e.g. ChargedFinalState, FastJets, ZFinder...

namespace Rivet {


  class CMS_QCD_10_024 : public Analysis {
  public:

    /// @name Constructors etc.
    //@{

    /// Constructor
    CMS_QCD_10_024()
      : Analysis("CMS_QCD_10_024")
    {
      /// @todo Set whether your finalize method needs the generator cross section
      setBeams(PROTON, PROTON);
      setNeedsCrossSection(false);
    }

  private:

    // mandatory methods
    void init();
    void analyze(const Event&);
    void finalize();

    AIDA::IHistogram1D *_hist_dNch_deta_7TeV_pt1_eta08;
    AIDA::IHistogram1D *_hist_dNch_deta_7TeV_pt1_eta24;
    AIDA::IHistogram1D *_hist_dNch_deta_7TeV_pt05_eta08;
    AIDA::IHistogram1D *_hist_dNch_deta_7TeV_pt05_eta24;

    AIDA::IHistogram1D *_hist_dNch_deta_09TeV_pt05_eta08;
    AIDA::IHistogram1D *_hist_dNch_deta_09TeV_pt1_eta08;
    AIDA::IHistogram1D *_hist_dNch_deta_09TeV_pt05_eta24;
    AIDA::IHistogram1D *_hist_dNch_deta_09TeV_pt1_eta24;
    int counter1_, counter2_, counter3_, counter4_, evcounter1_, evcounter2_, evcounter3_, evcounter4_;
    //@}                
  };

  void CMS_QCD_10_024::init() {

      ChargedFinalState cfs(-2.4, 2.4, 0.0*GeV);

      addProjection(cfs, "CFS");

      evcounter1_=0;
      counter1_=0;
      evcounter2_=0;
      evcounter3_=0;
      evcounter4_=0;
      counter2_=0;
      counter3_=0;
      counter4_=0;

      if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)){
      _hist_dNch_deta_7TeV_pt05_eta08 = bookHistogram1D("d01-x01-y01", 24, -2.4, +2.4);
      _hist_dNch_deta_7TeV_pt1_eta08 = bookHistogram1D("d02-x01-y01",24, -2.4, +2.4);
      _hist_dNch_deta_7TeV_pt05_eta24 = bookHistogram1D("d03-x01-y01",24, -2.4, +2.4);
      _hist_dNch_deta_7TeV_pt1_eta24 = bookHistogram1D("d04-x01-y01",24, -2.4, +2.4);
      }

      if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)){
	_hist_dNch_deta_09TeV_pt05_eta08 = bookHistogram1D("d05-x01-y01", 24, -2.4, +2.4);
	_hist_dNch_deta_09TeV_pt1_eta08 = bookHistogram1D("d06-x01-y01",24, -2.4, +2.4);
	_hist_dNch_deta_09TeV_pt05_eta24 = bookHistogram1D("d07-x01-y01",24, -2.4, +2.4);
	_hist_dNch_deta_09TeV_pt1_eta24 = bookHistogram1D("d08-x01-y01",24, -2.4, +2.4);
      }


    }

  void CMS_QCD_10_024::analyze(const Event& event) {
    counter1_=0;
    counter2_=0;
    counter3_=0;
    counter4_=0;
    const double weight = event.weight();

    const ChargedFinalState& charged = applyProjection<ChargedFinalState>(event, "CFS");

    // event selection : require at least one charged particle with certain eta and pt    
      foreach (const Particle& p, charged.particles()) {
        if( fabs(p.momentum().pseudorapidity()) < 0.8 && p.momentum().pT()/GeV > 0.5 ) counter1_++;
        if( fabs(p.momentum().pseudorapidity()) < 0.8 && p.momentum().pT()/GeV > 1.0 ) counter2_++;
        if( fabs(p.momentum().pseudorapidity()) < 2.4 && p.momentum().pT()/GeV > 0.5 ) counter3_++;
        if( fabs(p.momentum().pseudorapidity()) < 2.4 && p.momentum().pT()/GeV > 1.0 ) counter4_++;
     }
	// Nevents passing the selection (used for normalization)
	if( counter1_>=1 ) evcounter1_+=weight;
	if( counter2_>=1 ) evcounter2_+=weight;
    	if( counter3_>=1 ) evcounter3_+=weight;
   	if( counter4_>=1 ) evcounter4_+=weight;

	// plot distributions
	  if ( counter1_ > 0 ) {
	    foreach (const Particle& p, charged.particles()) {
	      if ( p.momentum().pT()/GeV > 0.5 ) {
	        if ( fuzzyEquals(sqrtS()/GeV, 7000, 1E-3) ) _hist_dNch_deta_7TeV_pt05_eta08->fill(p.momentum().pseudorapidity(), weight);
	        if ( fuzzyEquals(sqrtS()/GeV, 900, 1E-3) ) _hist_dNch_deta_09TeV_pt05_eta08->fill(p.momentum().pseudorapidity(), weight);
	      }
	    }
	  }
	  if ( counter2_ > 0 ) {
	    foreach (const Particle& p, charged.particles()) {
	      if ( p.momentum().pT()/GeV > 1.0 ) {
	        if ( fuzzyEquals(sqrtS()/GeV, 7000, 1E-3) ) _hist_dNch_deta_7TeV_pt1_eta08->fill(p.momentum().pseudorapidity(), weight);
	        if ( fuzzyEquals(sqrtS()/GeV, 900, 1E-3) ) _hist_dNch_deta_09TeV_pt1_eta08->fill(p.momentum().pseudorapidity(), weight);
	      }
	    }
	  }
	  if ( counter3_ > 0 ) {
	    foreach (const Particle& p, charged.particles()) {
	      if ( p.momentum().pT()/GeV > 0.5 ) {
	        if ( fuzzyEquals(sqrtS()/GeV, 7000, 1E-3) ) _hist_dNch_deta_7TeV_pt05_eta24->fill(p.momentum().pseudorapidity(), weight);
	        if ( fuzzyEquals(sqrtS()/GeV, 900, 1E-3) ) _hist_dNch_deta_09TeV_pt05_eta24->fill(p.momentum().pseudorapidity(), weight);
	      }
	    }
	  }
	  if ( counter4_ > 0 ) {
	    foreach (const Particle& p, charged.particles()) {
	      if ( p.momentum().pT()/GeV > 1.0 ) {
	        if ( fuzzyEquals(sqrtS()/GeV, 7000, 1E-3) ) _hist_dNch_deta_7TeV_pt1_eta24->fill(p.momentum().pseudorapidity(), weight);
	        if ( fuzzyEquals(sqrtS()/GeV, 900, 1E-3) ) _hist_dNch_deta_09TeV_pt1_eta24->fill(p.momentum().pseudorapidity(), weight);
	      }
	    }   
	  }
	}  

    /// Normalise histograms etc., after the run
  void CMS_QCD_10_024::finalize() {

    const double norm1_ =evcounter1_;
    const double norm2_ =evcounter2_;
    const double norm3_ =evcounter3_;
    const double norm4_ =evcounter4_;

    if(fuzzyEquals(sqrtS()/GeV, 7000, 1E-3)){
      scale(_hist_dNch_deta_7TeV_pt05_eta08, 1.0/norm1_);
      scale(_hist_dNch_deta_7TeV_pt1_eta08, 1.0/norm2_);
      scale(_hist_dNch_deta_7TeV_pt05_eta24, 1.0/norm3_);
      scale(_hist_dNch_deta_7TeV_pt1_eta24, 1.0/norm4_);
    }

    if(fuzzyEquals(sqrtS()/GeV, 900, 1E-3)){
      scale(_hist_dNch_deta_09TeV_pt05_eta08, 1.0/norm1_);
      scale(_hist_dNch_deta_09TeV_pt1_eta08, 1.0/norm2_);
      scale(_hist_dNch_deta_09TeV_pt05_eta24, 1.0/norm3_);
      scale(_hist_dNch_deta_09TeV_pt1_eta24, 1.0/norm4_);
    }
   }

  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_QCD_10_024> plugin_CMS_QCD_10_024;

}
