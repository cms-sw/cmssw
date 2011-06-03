// -*- C++ -*

// Author: Mohammed Zakaria (mzakaria@ufl.edu)

#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Tools/Logging.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/FastJets.hh"
 #include "Rivet/Projections/Beam.hh"

using namespace std;
int  j = 0 ;
int jj = 0 , jjj = 0;

double Nch_tot_7TeV_pT3 = 0;
double Nch_tot_7TeV_pT20 = 0;
double Nch_tot_09TeV_pT3 = 0;

namespace Rivet {


  class CMS_QCD_10_010 : public Analysis {
  public:

    /// @name Constructors etc.
    //@{

    /// Constructor
    CMS_QCD_10_010()
      : Analysis("CMS_QCD_10_010")
    {
      setNeedsCrossSection(true);
    }

    //@}


  public:

    /// @name Analysis methods
    //@{

    /// Book histograms and initialise projections before the run
    void init() {

      /// Initialise and register projections
      const ChargedFinalState cfs(-2.0, 2.0, 500*MeV);
      addProjection(cfs, "CFS");

//AK
      const ChargedFinalState cfsforjet(-2.5, 2.5, 500*MeV);
      addProjection(cfsforjet, "CFSforjet");


//AK       const FastJets jetpro(FinalState(), FastJets::SISCONE, 0.5);
      const FastJets jetpro(cfsforjet, FastJets::SISCONE, 0.5);
      addProjection(jetpro, "Jets");
      addProjection(Beam(), "Beam");

      /// Book histograms rivet can read it directly from aida file (of Data). 
    _profile_Nch_pT_7TeV = bookProfile1D(1, 1, 1);
    _Profile_SumpT_pT_7TeV = bookProfile1D(2, 1, 1);   
    _hist_Nch_09TeV = bookProfile1D(3, 1, 1);
    _hist_Sum_09TeV = bookProfile1D(4, 1, 1);
    _Dist_Nch_7TeV_pT3 = bookHistogram1D(5, 1, 1); //7TeV pT>3GeV
    _Dist_Sum_7TeV_pT3 = bookHistogram1D(6, 1, 1);
    _Dist_pT_7TeV_pT3 = bookHistogram1D(7, 1, 1);
    _Dist_Nch_7TeV_pT20 = bookHistogram1D(8, 1, 1);
    _Dist_Sum_7TeV_pT20 = bookHistogram1D(9,1,1); 
    _Dist_pT_7TeV_pT20 = bookHistogram1D(10, 1, 1);
    _Dist_Nch_09TeV_pT3 = bookHistogram1D(11, 1, 1);
    _Dist_Sum_09TeV_pT3 = bookHistogram1D(12 , 1, 1); 
    _Dist_pT_09TeV_pT3 = bookHistogram1D(13, 1, 1);
/// _hist_dsigma_dpTjet1 = bookHistogram1D("dsigma_dpTjet1",(50, 0, 50));
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
      const double weight = event.weight();

      // Require at least one jet in the event with pT >= 20 GeV
      Jets jets = applyProjection<FastJets>(event, "Jets").jetsByPt(0.5*GeV);
      if (jets.size() < 1) vetoEvent;

      FourMomentum p_lead = jets[0].momentum();
      const double philead = p_lead.phi();
      const double etalead = p_lead.eta();
      const double pTlead  = p_lead.perp();
      //AK  MSG_INFO("Leading track: pT = " << pTlead << ", eta = " << etalead << ", phi = " << philead);
      // These are the charged particles (tracks) with pT > 500 MeV defined above
      ParticleVector particles =
          applyProjection<ChargedFinalState>(event, "CFS").particlesByPt();


      double nTransverse(0.0), ptSumTransverse(0.0), pT, Nch_average ;
	int  i = 0;
      foreach (const Particle& p, particles) {
 //AK  MSG_INFO( "pTlead =  " << pTlead);

        double dphi=fabs(deltaPhi(philead, p.momentum().phi()));
 //AK  MSG_INFO( "dphi =  " << dphi*180/PI);
       double eta = fabs(p.momentum().eta());
 //AK  MSG_INFO( "eta = " << eta); 
        if (((PI/3.0 < dphi) && (dphi < 2*PI/3.0))&&(eta < 2)) {
//AK  MSG_INFO( "<passes> dphi =  " << dphi*180/PI );       
   nTransverse += 1.0;
//AK  MSG_INFO( "nTransverse =  " << nTransverse );       

          ptSumTransverse += p.momentum().perp();
// //AK  MSG_INFO( "pT =  " << p.momentum().perp());
pT = p.momentum().perp();
if((fuzzyEquals(sqrtS(), 7*TeV))&&(pTlead > 3*GeV))
_Dist_pT_7TeV_pT3->fill(pT/GeV);
if((fuzzyEquals(sqrtS(), 7*TeV))&&(pTlead > 20*GeV))
_Dist_pT_7TeV_pT20->fill(pT/GeV);
if((fuzzyEquals(sqrtS(), 900*GeV))&&(pTlead > 3*GeV))
_Dist_pT_09TeV_pT3->fill(pT/GeV);
        }
      }


//      if (nTransverse > 0)
 if (fuzzyEquals(sqrtS(), 7*TeV)) 
{
        _profile_Nch_pT_7TeV->fill
//            (pTlead/GeV, ptSumTransverse/GeV/nTransverse, weight);
            (pTlead/GeV, nTransverse/((8*PI/3)));
 //AK  MSG_INFO( "filling Nch Profile 7TeV, with ptlead = " << pTlead );
//AK  MSG_INFO("Nch = " << nTransverse );

	_Profile_SumpT_pT_7TeV->fill
		(pTlead/GeV, ptSumTransverse/((GeV*(8*PI/3))), weight);
if(pTlead > 3*GeV)
{
_Dist_Nch_7TeV_pT3->fill(nTransverse);
Nch_average = Nch_average + nTransverse;
Nch_tot_7TeV_pT3 = Nch_tot_7TeV_pT3 + nTransverse;
i = i + 1;
j = j + 1;
 //     //AK  MSG_INFO("Nch_average  = " << Nch_average << " i = "  << i);
/////	//AK  MSG_INFO("Nch_7TeV_pT3 = " << nTransverse << " Nch_tot_7TeV_pT3  = " << Nch_tot_7TeV_pT3 << " j = "  << j);
_Dist_Sum_7TeV_pT3->fill(ptSumTransverse/GeV);
} //       //AK  MSG_INFO("This is a Distribution");

if(pTlead > 20*GeV)
{

_Dist_Nch_7TeV_pT20->fill(nTransverse);
Nch_tot_7TeV_pT20 = Nch_tot_7TeV_pT20 + nTransverse;
jj = jj + 1;
////AK  MSG_INFO("Nch_7TeV_pT20 = " << nTransverse << " Nch_tot_7TeV_pT20  = " << Nch_tot_7TeV_pT20 << " jj = "  << jj);

_Dist_Sum_7TeV_pT20->fill(ptSumTransverse/GeV);
} 
      }
else if (fuzzyEquals(sqrtS(), 900*GeV))
{
	_hist_Nch_09TeV->fill(pTlead/GeV, nTransverse/((8*PI/3)), weight);
        _hist_Sum_09TeV->fill(pTlead/GeV, ptSumTransverse/((GeV*(8*PI/3))), weight);

 //     _hist_dsigma_dpTjet1->fill(pTlead/GeV, weight);

if(pTlead > 3*GeV)
{
_Dist_Nch_09TeV_pT3->fill(nTransverse);
Nch_tot_09TeV_pT3 = Nch_tot_09TeV_pT3 + nTransverse;
jjj = jjj + 1;  //counts how many events 
////AK  MSG_INFO("Nch_09TeV_pT3 = " << nTransverse << " Nch_tot_09TeV_pT3  = " << Nch_tot_09TeV_pT3 << " jjj = "  << jjj);

_Dist_Sum_09TeV_pT3->fill(ptSumTransverse/GeV);
}
}
    }



    /// Normalise histograms etc., after the run
    void finalize() {
//int nxBins = _Dist_Nch_09TeV_pT3.size();
//for(int x = 0 ; x < 200 ; x++)
//{
//   _Dist_Nch_09TeV_pT3.numEvents();


//if(_Dist_Nch_09TeV_pT3 != 0)
//{///_Dist_Nch_09TeV_pT3.empty();
//int i =  _Dist_Nch_09TeV_pT3->binHeight(x);
//      //AK  MSG_INFO("Bin_Value  = " << _Dist_Nch_09TeV_pT3->binHeight(x) << " "  << x);
//}
//}
//      scale(_hist_ptavg_transverse, 3/8*PI);
 /////AK  MSG_INFO( "Area beneath the histogram Nch =  " << integral(_Dist_Nch_7TeV_pT3));
// //AK  MSG_INFO( "Average of the histogram Nch =  " << _Dist_Nch_7TeV_pT3.getMean());


scale(_Dist_Nch_7TeV_pT3, 1/integral(_Dist_Nch_7TeV_pT3 ));
scale(_Dist_Sum_7TeV_pT3, 1/integral(_Dist_Sum_7TeV_pT3 ));
scale(_Dist_Nch_7TeV_pT20, 1/integral(_Dist_Nch_7TeV_pT20 ));
scale(_Dist_Sum_7TeV_pT20, 1/integral(_Dist_Sum_7TeV_pT20 ));
scale(_Dist_Nch_09TeV_pT3, 1/integral(_Dist_Nch_09TeV_pT3 ));
scale(_Dist_Sum_09TeV_pT3, 1/integral(_Dist_Sum_09TeV_pT3 ));
 
double  Nch_avg_7TeV_pT3 = 0;
double  Nch_avg_7TeV_pT20 = 0;
double Nch_avg_09TeV_pT3 = 0;
Nch_avg_7TeV_pT3 = (Nch_tot_7TeV_pT3 / j) ;
//	//AK  MSG_INFO("Nch_avg_7TeV_pT3 = " << Nch_avg_7TeV_pT3);
Nch_avg_7TeV_pT20 = (Nch_tot_7TeV_pT20 / jj) ;
//	//AK  MSG_INFO("Nch_avg_7TeV_pT20 = " << Nch_avg_7TeV_pT20);
Nch_avg_09TeV_pT3 = (Nch_tot_09TeV_pT3 / jjj) ;
//	//AK  MSG_INFO("Nch_avg_09TeV_pT3 = " << Nch_avg_09TeV_pT3);

scale(_Dist_pT_7TeV_pT20, Nch_avg_7TeV_pT20/integral(_Dist_pT_7TeV_pT20) ); //normalizes pT Distribution so that area = avg Nch
scale(_Dist_pT_7TeV_pT3, Nch_avg_7TeV_pT3/integral(_Dist_pT_7TeV_pT3) );
scale(_Dist_pT_09TeV_pT3, Nch_avg_09TeV_pT3/integral(_Dist_pT_09TeV_pT3) );


   }

    //@}


  private:

    /// @name Histograms
    //@{

AIDA::IProfile1D * _profile_Nch_pT_7TeV;
AIDA::IProfile1D * _Profile_SumpT_pT_7TeV;
AIDA::IProfile1D * _hist_Nch_09TeV;
AIDA::IProfile1D * _hist_Sum_09TeV;
AIDA::IHistogram1D * _Dist_Nch_7TeV_pT3 ;
AIDA::IHistogram1D * _Dist_Sum_7TeV_pT3; 
AIDA::IHistogram1D * _Dist_pT_7TeV_pT3;
AIDA::IHistogram1D * _Dist_Nch_7TeV_pT20;
AIDA::IHistogram1D * _Dist_Sum_7TeV_pT20;
AIDA::IHistogram1D * _Dist_pT_7TeV_pT20;
AIDA::IHistogram1D * _Dist_Nch_09TeV_pT3;
AIDA::IHistogram1D * _Dist_Sum_09TeV_pT3;
AIDA::IHistogram1D * _Dist_pT_09TeV_pT3;
//    AIDA::IHistogram1D * _hist_dsigma_dpTjet1;
    //@}

  public:
    string experiment()         const { return "CMS"; }
    string year()               const { return "2011"; }
    string spiresId()           const { return "None"; }
    string collider()           const { return "None"; }
    string summary()            const { return "None"; }
    string description()        const { return "None"; }
    string runInfo()            const { return "None"; }
    string status()             const { return "UNVALIDATED"; }
    vector<string> authors()    const { return vector<string>(); }
    vector<string> references() const { return vector<string>(); }
    vector<std::string> todos() const { return vector<string>(); }
    

  };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_QCD_10_010> plugin_CMS_QCD_10_010;


}

