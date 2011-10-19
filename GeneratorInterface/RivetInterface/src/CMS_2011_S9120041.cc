// -*- C++ -*-

#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/Beam.hh"

using namespace std;

namespace Rivet {

  // UE charged particles vs. leading jet
  class CMS_2011_S9120041 : public Analysis {
  public:

    /// Constructor
    CMS_2011_S9120041() : Analysis("CMS_2011_S9120041") {}


    void init() {
      const ChargedFinalState cfs(-2.0, 2.0, 500*MeV);
      addProjection(cfs, "CFS");

      const ChargedFinalState cfsforjet(-2.5, 2.5, 500*MeV); // tracks accepted are with pT > 0.5 GeV
      addProjection(cfsforjet, "CFSforjet");


      const FastJets jetpro(cfsforjet, FastJets::SISCONE, 0.5);
      addProjection(jetpro, "Jets");
      addProjection(Beam(), "Beam");

      _hist_profile_Nch_pT_7TeV = bookProfile1D(1, 1, 1); //Profile plot for No. of Charged particles vs. pT max for 7 TeV.
      _hist_profile_SumpT_pT_7TeV = bookProfile1D(2, 1, 1);   //Profile plot Trans. Momentum sum vs. pT max for 7 TeV.
      _hist_profile_Nch_09TeV = bookProfile1D(3, 1, 1);
      _hist_profile_Sum_09TeV = bookProfile1D(4, 1, 1);
      _hist_dist_Nch_7TeV_pT3 = bookHistogram1D(5, 1, 1);
      _hist_dist_Sum_7TeV_pT3 = bookHistogram1D(6, 1, 1);
      _hist_dist_pT_7TeV_pT3 = bookHistogram1D(7, 1, 1);
      _hist_dist_Nch_7TeV_pT20 = bookHistogram1D(8, 1, 1);
      _hist_dist_Sum_7TeV_pT20 = bookHistogram1D(9,1,1);
      _hist_dist_pT_7TeV_pT20 = bookHistogram1D(10, 1, 1);
      _hist_dist_Nch_09TeV_pT3 = bookHistogram1D(11, 1, 1); // No. of trans. charged particles Distribution for sqrt(s) = 0.9TeV, pT max > 3GeV.
      _hist_dist_Sum_09TeV_pT3 = bookHistogram1D(12 , 1, 1); // No. of trans. momentum sum Distribution for sqrt(s) = 0.9TeV, pT max > 3GeV.
      _hist_dist_pT_09TeV_pT3 = bookHistogram1D(13, 1, 1); // Trans. momentum Distribution for sqrt(s) = 0.9TeV, pT max > 3GeV.

      _j = 0.0;
      _jj = 0.0;
      _jjj = 0.0;

      _nch_tot_7TeV_pT3 = 0.0;
      _nch_tot_7TeV_pT20 = 0.0;
      _nch_tot_09TeV_pT3 = 0.0;
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      const double weight = event.weight();

      Jets jets = applyProjection<FastJets>(event, "Jets").jetsByPt(1.0*GeV);
      if (jets.size() < 1) vetoEvent;

      FourMomentum p_lead = jets[0].momentum();
      const double philead = p_lead.phi();
      const double pTlead  = p_lead.perp();

      ParticleVector particles =
          applyProjection<ChargedFinalState>(event, "CFS").particlesByPt();

      double nTransverse(0.0), ptSumTransverse(0.0), pT;
      foreach (const Particle& p, particles) {
        double dphi = fabs(deltaPhi(philead, p.momentum().phi()));
        double eta = fabs(p.momentum().eta());
        if (((PI/3.0 < dphi) && (dphi < 2.0*PI/3.0))&&(eta < 2.0)) {
          nTransverse += 1.0;

          ptSumTransverse += p.momentum().perp();
          pT = p.momentum().perp();

          if ( (fuzzyEquals(sqrtS(), 7.0*TeV)) && (pTlead > 20.0*GeV) )
            _hist_dist_pT_7TeV_pT20 -> fill(pT/GeV, weight);

          if ( (fuzzyEquals(sqrtS(), 7.0*TeV)) && (pTlead > 3.0*GeV) )
            _hist_dist_pT_7TeV_pT3 -> fill(pT/GeV, weight);

          if ( (fuzzyEquals(sqrtS(), 900.0*GeV)) && (pTlead > 3.0*GeV) )
            _hist_dist_pT_09TeV_pT3 -> fill(pT/GeV, weight);
        }
      }


      if (fuzzyEquals(sqrtS(), 7.0*TeV))
      {
        _hist_profile_Nch_pT_7TeV -> fill
            (pTlead/GeV, nTransverse / ((8.0 * PI/3.0)), weight);

        _hist_profile_SumpT_pT_7TeV -> fill
            (pTlead/GeV , ptSumTransverse / ((GeV * (8.0 * PI/3.0))) , weight);
        if(pTlead > 3.0*GeV)
        {
          _hist_dist_Nch_7TeV_pT3 -> fill(nTransverse, weight);
          _nch_tot_7TeV_pT3 += nTransverse*weight;
          _j += weight ;
          _hist_dist_Sum_7TeV_pT3 -> fill(ptSumTransverse / GeV, weight);
        }
        if(pTlead > 20.0*GeV)
        {

          _hist_dist_Nch_7TeV_pT20 -> fill(nTransverse, weight);
          _nch_tot_7TeV_pT20 += nTransverse*weight;
          _jj += weight;

          _hist_dist_Sum_7TeV_pT20 -> fill(ptSumTransverse / GeV, weight);
        }
      }
      else if (fuzzyEquals(sqrtS() , 900.0*GeV))
      {
        _hist_profile_Nch_09TeV -> fill(pTlead / GeV, nTransverse / ((8.0 * PI/3.0)), weight);
        _hist_profile_Sum_09TeV -> fill(pTlead / GeV, ptSumTransverse / ((GeV * (8.0 * PI/3.0))), weight);


        if(pTlead > 3.0*GeV)
        {
          _hist_dist_Nch_09TeV_pT3 -> fill(nTransverse, weight);
          _nch_tot_09TeV_pT3 += nTransverse*weight;
          _jjj += weight;

          _hist_dist_Sum_09TeV_pT3 -> fill(ptSumTransverse/GeV, weight);
        }
      }
    }



    /// Normalise histograms etc., after the run
    void finalize() {
      normalize(_hist_dist_Nch_7TeV_pT3);
      normalize(_hist_dist_Sum_7TeV_pT3);
      normalize(_hist_dist_Nch_7TeV_pT20);
      normalize(_hist_dist_Sum_7TeV_pT20);
      normalize(_hist_dist_Nch_09TeV_pT3);
      normalize(_hist_dist_Sum_09TeV_pT3);

      double _nch_avg_7TeV_pT3 = (_nch_tot_7TeV_pT3 / _j);
      double _nch_avg_7TeV_pT20 = (_nch_tot_7TeV_pT20 / _jj);
      double _nch_avg_09TeV_pT3 = (_nch_tot_09TeV_pT3 / _jjj);

      if (_j!=0.0) normalize(_hist_dist_pT_7TeV_pT20, _nch_avg_7TeV_pT20);
      if (_jj!=0.0) normalize(_hist_dist_pT_7TeV_pT3, _nch_avg_7TeV_pT3);
      if (_jjj!=0.0) normalize(_hist_dist_pT_09TeV_pT3, _nch_avg_09TeV_pT3);
    }



  private:

    double  _j;
    double _jj;
    double _jjj;

    double _nch_tot_7TeV_pT3;
    double _nch_tot_7TeV_pT20;
    double _nch_tot_09TeV_pT3;

    AIDA::IProfile1D * _hist_profile_Nch_pT_7TeV;
    AIDA::IProfile1D * _hist_profile_SumpT_pT_7TeV;
    AIDA::IProfile1D * _hist_profile_Nch_09TeV;
    AIDA::IProfile1D * _hist_profile_Sum_09TeV;
    AIDA::IHistogram1D * _hist_dist_Nch_7TeV_pT3 ;
    AIDA::IHistogram1D * _hist_dist_Sum_7TeV_pT3;
    AIDA::IHistogram1D * _hist_dist_pT_7TeV_pT3;
    AIDA::IHistogram1D * _hist_dist_Nch_7TeV_pT20;
    AIDA::IHistogram1D * _hist_dist_Sum_7TeV_pT20;
    AIDA::IHistogram1D * _hist_dist_pT_7TeV_pT20;
    AIDA::IHistogram1D * _hist_dist_Nch_09TeV_pT3;
    AIDA::IHistogram1D * _hist_dist_Sum_09TeV_pT3;
    AIDA::IHistogram1D * _hist_dist_pT_09TeV_pT3;

  };


  // This global object acts as a hook for the plugin system
//AK  DECLARE_RIVET_PLUGIN(CMS_2011_S9120041);
  AnalysisBuilder<CMS_2011_S9120041> plugin_CMS_2011_S9120041;

}




