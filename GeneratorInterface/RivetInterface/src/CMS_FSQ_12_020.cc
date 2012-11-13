// -*- C++ -*-

#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/Beam.hh"
/// @todo Include more projections as required, e.g. ChargedFinalState, FastJets, ZFinder...

using namespace std;

namespace Rivet {


  class CMS_FSQ_12_020 : public Analysis {
  public:

    /// @name Constructors etc.
    //@{

    /// Constructor
    CMS_FSQ_12_020()
      : Analysis("CMS_FSQ_12_020") { }
  //  {
    //  /// @todo Set whether your finalize method needs the generator cross section
    // setNeedsCrossSection(true);
    //}

    //@}


//  public:

    /// @name Analysis methods
    //@{

    /// Book histograms and initialise projections before the run
    void init() {
// MSG_DEBUG("test...test");

      const ChargedFinalState cfs(-0.8, 0.8, 500*MeV);
      addProjection(cfs, "CFS500");
      addProjection(Beam(), "Beam");
       const ChargedFinalState cfslead(-0.8, 0.8, 500*MeV);
       addProjection(cfslead, "CFSlead");

      _hist_profile_Nch_pT_7TeV = bookProfile1D(1, 1, 1); //Profile plot for No. of Charged particles vs. pT max for 7 TeV.
      _hist_profile_SumpT_pT_7TeV = bookProfile1D(2, 1, 1);   //Profile plot Trans. Momentum sum vs. pT max for 7 TeV.
      _hist_profile_Nch_pT_09TeV = bookProfile1D(3, 1, 1);
      _hist_profile_SumpT_pT_09TeV = bookProfile1D(4, 1, 1);

    int  _j = 0;
    int  _jj = 0;
    int  _jjj = 0;


      /// @todo Initialise and register projections here

      /// @todo Book histograms here, e.g.:
      // _h_XXXX = bookProfile1D(1, 1, 1);
      // _h_YYYY = bookHistogram1D(2, 1, 1);

    }

 inline int region_index(double dphi) {
       assert(inRange(dphi, 0.0, PI, CLOSED, CLOSED));
       if (dphi < PI/3.0) return 0;
       if (dphi < 2*PI/3.0) return 1;
       return 2;
     }
    /// Perform the per-event analysis
    void analyze(const Event& event) {
      const double weight = event.weight();
       // Require at least one track in the event with pT >= 1 GeV
       const ChargedFinalState& cfslead = applyProjection<ChargedFinalState>(event, "CFSlead");
//MSG_INFO("Total charged mult. = " << cfslead.size());

       if (cfslead.size() < 1) {
//MSG_INFO("VETO! An empty event ");
         vetoEvent;
       }
      /// @todo Do the event by event analysis here
 // These are the charged particles (tracks) with pT > 500 MeV
      const ChargedFinalState& charged500 = applyProjection<ChargedFinalState>(event, "CFS500");
 // Identify leading track and its phi and pT (this is the same for both the 100 MeV and 500 MeV track cuts)
      ParticleVector particles500 = charged500.particlesByPt();
      Particle p_lead = particles500[0];
      const double philead = p_lead.momentum().phi();
      const double etalead = p_lead.momentum().eta();
      const double pTlead  = p_lead.momentum().perp();
      //MSG_INFO("Leading track: pT = " << pTlead << ", eta = " << etalead << ", phi = " << philead);

   vector<double> num500(3, 0), ptSum500(3, 0.0);

foreach (const Particle& p, particles500) {
         const double pT = p.momentum().pT();
//MSG_INFO(" pT = " << pT << ", phi = " << p.momentum().phi());

         const double dPhi = deltaPhi(philead, p.momentum().phi());
//MSG_INFO(" dphi = " << dPhi*180/PI);

         const int ir = region_index(dPhi);
//MSG_INFO(" ir = " << ir);
// to select the transverse region
      if(ir == 1){
	 num500[ir] += 1;
         ptSum500[ir] += pT;
 }
    //     // Fill temp histos to bin Nch and pT in dPhi
    //     if (p.genParticle() != p_lead.genParticle()) { // We don't want to fill all those zeros from the leading track...
    //       hist_num_dphi_500.fill(dPhi, 1);
    //       hist_pt_dphi_500.fill(dPhi, pT);
    //     }
       }
 const double dEtadPhi = (2*0.8 * 2*PI/3.0);
//MSG_INFO("Trans pT1, pTsum, Nch, <pT>" << pTlead/GeV << ", " <<  ptSum500[1]/GeV << ", " << num500[1] << ", " << ptSum500[1]/GeV/num500[1]);
// foreach (const Particle& p, particles500) {
//        const double pT = p.momentum().pT();
//        const double dPhi = deltaPhi(philead, p.momentum().phi());
//        const int ir = region_index(dPhi);
//        switch (ir) {
//        case 0:
////          _hist_dn_dpt_toward_500->fill(num500[0], pT, weight);
//          break;
//        case 1:
//  //        _hist_dn_dpt_transverse_500->fill(num500[1], pT, weight);
//          break;
//        case 2:
//  //        _hist_dn_dpt_away_500->fill(num500[2], pT, weight);
//          break;
//        default:
//          assert(false && "How did we get here?");
//        }
//      }
 if (fuzzyEquals(sqrtS(), 7.0*TeV))
{
//MSG_INFO("Filling Trans pT1, pTsum, Nch, " << pTlead/GeV << ", " <<  ptSum500[1]/GeV << ", " << num500[1] );
_hist_profile_Nch_pT_7TeV->fill(pTlead/GeV, num500[1] / (dEtadPhi), weight );
_hist_profile_SumpT_pT_7TeV->fill(pTlead/GeV, ptSum500[1] / (GeV * (dEtadPhi)), weight );
}
else if (fuzzyEquals(sqrtS(), 0.9*TeV))
{
_hist_profile_Nch_pT_09TeV->fill(pTlead/GeV, num500[1] / (dEtadPhi), weight );
_hist_profile_SumpT_pT_09TeV->fill(pTlead/GeV, ptSum500[1] / (GeV * (dEtadPhi)), weight );

}
    }


    /// Normalise histograms etc., after the run
    void finalize() {

      /// @todo Normalise, scale and otherwise manipulate histograms here

      // scale(_h_YYYY, crossSection()/sumOfWeights()); # norm to cross section
      // normalize(_h_YYYY); # normalize to unity

    }

    //@}


//  private:

    // Data members like post-cuts event weight counters go here
 

  private:

    /// @name Histograms
    
    double _j;
    double _jj;
    double _jjj;

    AIDA::IProfile1D * _hist_profile_Nch_pT_7TeV;
    AIDA::IProfile1D * _hist_profile_SumpT_pT_7TeV;
    AIDA::IProfile1D * _hist_profile_Nch_pT_09TeV;
    AIDA::IProfile1D * _hist_profile_SumpT_pT_09TeV;

    //AIDA::IProfile1D *_h_XXXX;
    //AIDA::IHistogram1D *_h_YYYY;
   


  };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_FSQ_12_020> plugin_CMS_FSQ_12_020;


}
