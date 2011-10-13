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

  class CMS_2011_S9120041 : public Analysis {
    public:


      /// Constructor
      CMS_2011_S9120041()
        : Analysis("CMS_2011_S9120041")
      {
        setNeedsCrossSection(false);
      }

    public:

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

	_j = 0;
	_jj = 0; 
	_jjj = 0;

	_nch_tot_7TeV_pT3 = 0;
	_nch_tot_7TeV_pT20 = 0;
	_nch_tot_09TeV_pT3 = 0;
      }


      /// Perform the per-event analysis
      void analyze(const Event& event) {
        const double weight = event.weight();


        Jets jets = applyProjection<FastJets>(event, "Jets").jetsByPt(1.0*GeV); // we require our jets to have pT > 1 GeV.
        if (jets.size() < 1) vetoEvent;
	
        FourMomentum _p_lead = jets[0].momentum();
        const double _philead = _p_lead.phi();
        const double _pTlead  = _p_lead.perp();

        ParticleVector particles =
          applyProjection<ChargedFinalState>(event, "CFS").particlesByPt();

        double _nTransverse(0.0), _ptSumTransverse(0.0), _pT, _nch_average = 0 ;
        foreach (const Particle& p, particles) {

          double _dphi = fabs(deltaPhi(_philead, p.momentum().phi()));
          double _eta = fabs(p.momentum().eta());
          if (((PI/3.0 < _dphi) && (_dphi < 2*PI/3.0))&&(_eta < 2)) {
            _nTransverse += 1.0;

            _ptSumTransverse += p.momentum().perp();
            _pT = p.momentum().perp();

            if ( (fuzzyEquals(sqrtS(), 7*TeV)) && (_pTlead > 20*GeV) )
              _hist_dist_pT_7TeV_pT20 -> fill(_pT/GeV, weight);

            if ( (fuzzyEquals(sqrtS(), 7*TeV)) && (_pTlead > 3*GeV) )
              _hist_dist_pT_7TeV_pT3 -> fill(_pT/GeV, weight);

            if ( (fuzzyEquals(sqrtS(), 900*GeV)) && (_pTlead > 3*GeV) )
              _hist_dist_pT_09TeV_pT3 -> fill(_pT/GeV, weight);


          }
        }


        if (fuzzyEquals(sqrtS(), 7*TeV)) 
        {
          _hist_profile_Nch_pT_7TeV -> fill
            (_pTlead/GeV, _nTransverse / ((8 * PI/3)), weight);

          _hist_profile_SumpT_pT_7TeV -> fill
            (_pTlead/GeV , _ptSumTransverse / ((GeV * (8 * PI/3))) , weight);
          if(_pTlead > 3*GeV)
          {
            _hist_dist_Nch_7TeV_pT3 -> fill(_nTransverse, weight);
            _nch_average = _nch_average + _nTransverse;
            _nch_tot_7TeV_pT3 = _nch_tot_7TeV_pT3 + _nTransverse;
            _j += weight ;
            _hist_dist_Sum_7TeV_pT3 -> fill(_ptSumTransverse / GeV, weight);
          }
          if(_pTlead > 20*GeV)
          {

            _hist_dist_Nch_7TeV_pT20 -> fill(_nTransverse, weight);
            _nch_tot_7TeV_pT20 = _nch_tot_7TeV_pT20 + _nTransverse;
            _jj += weight;

            _hist_dist_Sum_7TeV_pT20 -> fill(_ptSumTransverse / GeV, weight);
          } 
        }
        else if (fuzzyEquals(sqrtS() , 900*GeV))
        {
          _hist_profile_Nch_09TeV -> fill(_pTlead / GeV, _nTransverse / ((8 * PI/3)), weight);
          _hist_profile_Sum_09TeV -> fill(_pTlead / GeV, _ptSumTransverse / ((GeV * (8 * PI/3))), weight);


          if(_pTlead > 3*GeV)
          {
            _hist_dist_Nch_09TeV_pT3 -> fill(_nTransverse, weight);
            _nch_tot_09TeV_pT3 = _nch_tot_09TeV_pT3 + _nTransverse;
            _jjj += weight;  //counts how many events 

            _hist_dist_Sum_09TeV_pT3 -> fill(_ptSumTransverse/GeV, weight);
          }
        }
      }



      /// Normalise histograms etc., after the run
      void finalize() {
        scale(_hist_dist_Nch_7TeV_pT3 , 1/integral(_hist_dist_Nch_7TeV_pT3 ));
        scale(_hist_dist_Sum_7TeV_pT3 , 1/integral(_hist_dist_Sum_7TeV_pT3 ));
        scale(_hist_dist_Nch_7TeV_pT20 , 1/integral(_hist_dist_Nch_7TeV_pT20 ));
        scale(_hist_dist_Sum_7TeV_pT20 , 1/integral(_hist_dist_Sum_7TeV_pT20 ));
        scale(_hist_dist_Nch_09TeV_pT3 , 1/integral(_hist_dist_Nch_09TeV_pT3 ));
        scale(_hist_dist_Sum_09TeV_pT3 , 1/integral(_hist_dist_Sum_09TeV_pT3 ));

        double  _nch_avg_7TeV_pT3 = 0;
        double  _nch_avg_7TeV_pT20 = 0;
        double _nch_avg_09TeV_pT3 = 0;
        _nch_avg_7TeV_pT3 = (_nch_tot_7TeV_pT3 / _j) ;
        _nch_avg_7TeV_pT20 = (_nch_tot_7TeV_pT20 / _jj) ;
        _nch_avg_09TeV_pT3 = (_nch_tot_09TeV_pT3 / _jjj) ;

        scale(_hist_dist_pT_7TeV_pT20, _nch_avg_7TeV_pT20 / integral(_hist_dist_pT_7TeV_pT20) ); //normalizes pT Distribution so that area = avg Nch
        scale(_hist_dist_pT_7TeV_pT3, _nch_avg_7TeV_pT3 / integral(_hist_dist_pT_7TeV_pT3) ); // normalizes Nch Distribution so that area = avg Nch
        scale(_hist_dist_pT_09TeV_pT3, _nch_avg_09TeV_pT3 / integral(_hist_dist_pT_09TeV_pT3) ); // normalizes SumpT distribtuion so that area = avg Nch

      }



    private:

      int  _j;
      int _jj; 
      int _jjj;
      
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
      //@}

  };



  // This global object acts as a hook for the plugin system
   AnalysisBuilder<CMS_2011_S9120041> plugin_CMS_2011_S9120041;


}




