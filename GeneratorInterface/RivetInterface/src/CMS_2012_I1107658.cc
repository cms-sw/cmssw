// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ZFinder.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/ParticleName.hh"

namespace Rivet {


  class CMS_2012_I1107658 : public Analysis {
  public:

    /// Constructor
    CMS_2012_I1107658()
      : Analysis("CMS_2012_I1107658")
    {
      /// @todo Set whether your finalize method needs the generator cross section
      setNeedsCrossSection(false);
    }

    void init() {

       ZFinder zfinder(-MAXRAPIDITY, MAXRAPIDITY, 0.0*GeV, MUON,4.0*GeV, 140.0*GeV, 0.2, false, false);
       addProjection(zfinder, "ZFinder");

       ChargedFinalState cfs(-2.0, 2.0, 500*MeV); // For charged particles
       addProjection(cfs, "CFS");
       
      _hist_profile_Nchg_towards_pTmumu = bookProfile1D(1, 1, 1);
      _hist_profile_Nchg_transverse_pTmumu = bookProfile1D(2, 1, 1);
      _hist_profile_Nchg_away_pTmumu = bookProfile1D(3, 1, 1);
      _hist_profile_pTsum_towards_pTmumu = bookProfile1D(4, 1, 1);
      _hist_profile_pTsum_transverse_pTmumu = bookProfile1D(5, 1, 1);
      _hist_profile_pTsum_away_pTmumu = bookProfile1D(6, 1, 1);
      _hist_profile_avgpT_towards_pTmumu = bookDataPointSet(7,1,1); 
      _hist_profile_avgpT_transverse_pTmumu = bookDataPointSet(8,1,1);
      _hist_profile_avgpT_away_pTmumu = bookDataPointSet(9,1,1);
      _hist_profile_Nchg_towards_plus_transverse_Mmumu = bookProfile1D(10, 1, 1);
      _hist_profile_pTsum_towards_plus_transverse_Mmumu = bookProfile1D(11, 1, 1);
      _hist_profile_avgpT_towards_plus_transverse_Mmumu = bookDataPointSet(12,1,1);
      _hist_Nchg_towards_zmass_81_101  = bookHistogram1D(13, 1, 1);
      _hist_Nchg_transverse_zmass_81_101  = bookHistogram1D(14, 1, 1);
      _hist_Nchg_away_zmass_81_101  = bookHistogram1D(15, 1, 1); 
      _hist_pT_towards_zmass_81_101  = bookHistogram1D(16, 1, 1);
      _hist_pT_transverse_zmass_81_101  = bookHistogram1D(17, 1, 1);
      _hist_pT_away_zmass_81_101  = bookHistogram1D(18, 1, 1);
      _hist_Nchg_transverse_zpt_5  = bookHistogram1D(19, 1, 1);
      _hist_pT_transverse_zpt_5  = bookHistogram1D(20, 1, 1);
    
    }


    /// Perform the per-event analysis
    void analyze(const Event& event) {
       const double weight = event.weight();
       const ZFinder& zfinder = applyProjection<ZFinder>(event, "ZFinder");
       const FinalState& fs = zfinder.constituentsFinalState();
   
       if (zfinder.particles().size() != 1) vetoEvent;

       Particle lepton0 = fs.particles().at(0);
       Particle lepton1 = fs.particles().at(1);

       if (lepton0.pdgId() != -lepton1.pdgId()) vetoEvent; 

       double pt0 = lepton0.momentum().pT()/GeV;
       double pt1 = lepton1.momentum().pT()/GeV;
       double eta0 = lepton0.momentum().eta();
       double eta1 = lepton1.momentum().eta();  

       if(pt0 > 20. && pt1 > 20 && fabs(eta1) < 2.4 && fabs(eta0) < 2.4){

       	double _Zpt = zfinder.particles()[0].momentum().pT()/GeV;
       	double _Zphi = zfinder.particles()[0].momentum().phi(); 
       	double _Zmass = zfinder.particles()[0].momentum().mass()/GeV;

       	ParticleVector particles =
          applyProjection<ChargedFinalState>(event, "CFS").particlesByPt();

       	double _nTowards(0.0), _ptSumTowards(0.0), _nTransverse(0.0), 
	       _ptSumTransverse(0.0),  _nAway(0.0), _ptSumAway(0.0);

       	foreach (const Particle& p, particles) {
       	  if(fabs(p.pdgId()) !=13){
       	    double _dphi = fabs(deltaPhi(_Zphi, p.momentum().phi()));
            double _eta = fabs(p.momentum().eta());
       
            ///Towards region
            if ( PI/3.0 > _dphi && _eta < 2. ) {
            	_nTowards += 1.0;
             	_ptSumTowards += p.momentum().perp();
             	double _pT = p.momentum().perp();
             	if(_Zmass < 101. && _Zmass > 81.)_hist_pT_towards_zmass_81_101->fill(_pT,weight);
            }
       
       	    ///Transverse region
            if ( PI/3.0 < _dphi && _dphi < 2.*PI/3.0 && _eta < 2. ) {
             	_nTransverse += 1.0;
             	_ptSumTransverse += p.momentum().perp();
             	double _pT = p.momentum().perp();
             	if(_Zmass < 101. && _Zmass > 81.)_hist_pT_transverse_zmass_81_101->fill(_pT,weight);
             	if(_Zpt < 5.)_hist_pT_transverse_zpt_5->fill(_pT,weight);
            }

       	    ///Away region
            if ( 2.*PI/3.0 < _dphi && _eta < 2. ) {
             	_nAway += 1.0;
             	_ptSumAway += p.momentum().perp(); 
             	double _pT = p.momentum().perp();
             	if(_Zmass < 101. && _Zmass > 81.)_hist_pT_away_zmass_81_101->fill(_pT,weight);
            }
 
          }// not muons
        }//Loop over particles


        if(_Zmass < 101. && _Zmass > 81.){         
	  _hist_profile_Nchg_towards_pTmumu->fill(_Zpt/GeV, _nTowards/(((8. * PI)/3.)), weight);
          _hist_profile_Nchg_transverse_pTmumu->fill(_Zpt/GeV, _nTransverse/(((8. * PI)/3.)), weight);
          _hist_profile_Nchg_away_pTmumu->fill(_Zpt/GeV, _nAway/(((8. * PI)/3.)), weight); 
          _hist_profile_pTsum_towards_pTmumu->fill(_Zpt/GeV, _ptSumTowards/(((8. * PI)/3.)), weight);
          _hist_profile_pTsum_transverse_pTmumu->fill(_Zpt/GeV, _ptSumTransverse/(((8. * PI)/3.)), weight);
          _hist_profile_pTsum_away_pTmumu->fill(_Zpt/GeV, _ptSumAway/(((8. * PI)/3.)), weight);
          _hist_Nchg_towards_zmass_81_101->fill(_nTowards,weight);
          _hist_Nchg_transverse_zmass_81_101->fill(_nTransverse,weight);
          _hist_Nchg_away_zmass_81_101->fill(_nAway,weight);
        }//for events around Z resonance, activity as function of pT_mumu  

        if(_Zpt < 5.){
          _hist_profile_Nchg_towards_plus_transverse_Mmumu->fill(_Zmass/GeV, (_nTowards + _nTransverse)/(16.*PI/3.),weight);
          _hist_profile_pTsum_towards_plus_transverse_Mmumu->fill(_Zmass/GeV, (_ptSumTowards + _ptSumTransverse)/(16.*PI/3.),weight);     
          _hist_Nchg_transverse_zpt_5->fill(_nTransverse,weight);         
        }//for events with small recoilm activity as function of M_mumu  

      }//cut on leptons

    }


    /// Normalise histograms etc., after the run
    void finalize() {

      vector<double> yval_avgpt_towards, yerr_avgpt_towards, yval_avgpt_transverse, 
      		     yerr_avgpt_transverse, yval_avgpt_away, yerr_avgpt_away, 
		     yval_avgpt_tt, yerr_avgpt_tt;
    
      for (size_t i = 0;  i < 20; ++i) {        
	double yval_tmp = _hist_profile_pTsum_towards_pTmumu->binHeight(i) / 
			  _hist_profile_Nchg_towards_pTmumu->binHeight(i);
        yval_avgpt_towards.push_back(yval_tmp);
        
	double yerr_tmp = sqrt( _hist_profile_pTsum_towards_pTmumu->binError(i) * _hist_profile_pTsum_towards_pTmumu->binError(i)/
				(_hist_profile_pTsum_towards_pTmumu->binHeight(i) * _hist_profile_pTsum_towards_pTmumu->binHeight(i)) +
				_hist_profile_Nchg_towards_pTmumu->binError(i) * _hist_profile_Nchg_towards_pTmumu->binError(i)/
				(_hist_profile_Nchg_towards_pTmumu->binHeight(i) * _hist_profile_Nchg_towards_pTmumu->binHeight(i))) *
				 yval_tmp;
        yerr_avgpt_towards.push_back(yerr_tmp);       
       
        yval_tmp = _hist_profile_pTsum_transverse_pTmumu->binHeight(i) / _hist_profile_Nchg_transverse_pTmumu->binHeight(i);
        yval_avgpt_transverse.push_back(yval_tmp);
	
        yerr_tmp = sqrt(_hist_profile_pTsum_transverse_pTmumu->binError(i) * _hist_profile_pTsum_transverse_pTmumu->binError(i)/
			(_hist_profile_pTsum_transverse_pTmumu->binHeight(i) * _hist_profile_pTsum_transverse_pTmumu->binHeight(i)) +
			_hist_profile_Nchg_transverse_pTmumu->binError(i) * _hist_profile_Nchg_transverse_pTmumu->binError(i)/
			(_hist_profile_Nchg_transverse_pTmumu->binHeight(i) * _hist_profile_Nchg_transverse_pTmumu->binHeight(i))) * 
			yval_tmp;
        yerr_avgpt_transverse.push_back(yerr_tmp);
             
        yval_tmp = _hist_profile_pTsum_away_pTmumu->binHeight(i) / _hist_profile_Nchg_away_pTmumu->binHeight(i);
        yval_avgpt_away.push_back(yval_tmp);
        yerr_tmp = sqrt(_hist_profile_pTsum_away_pTmumu->binError(i) * _hist_profile_pTsum_away_pTmumu->binError(i)/
			(_hist_profile_pTsum_away_pTmumu->binHeight(i) * _hist_profile_pTsum_away_pTmumu->binHeight(i)) +
			_hist_profile_Nchg_away_pTmumu->binError(i) * _hist_profile_Nchg_away_pTmumu->binError(i)/
			(_hist_profile_Nchg_away_pTmumu->binHeight(i) * _hist_profile_Nchg_away_pTmumu->binHeight(i))) * 
			yval_tmp;
        yerr_avgpt_away.push_back(yerr_tmp); 
      }

      for (size_t i = 0;  i < 10; ++i) {
	
	double yval_tmp = _hist_profile_pTsum_towards_plus_transverse_Mmumu->binHeight(i) / 
	                  _hist_profile_Nchg_towards_plus_transverse_Mmumu->binHeight(i);
	yval_avgpt_tt.push_back(yval_tmp);
                  
	double yerr_tmp = sqrt(_hist_profile_pTsum_towards_plus_transverse_Mmumu->binError(i) * _hist_profile_pTsum_towards_plus_transverse_Mmumu->binError(i)/
			       (_hist_profile_pTsum_towards_plus_transverse_Mmumu->binHeight(i) * _hist_profile_pTsum_towards_plus_transverse_Mmumu->binHeight(i)) + 
			       _hist_profile_Nchg_towards_plus_transverse_Mmumu->binError(i) * _hist_profile_Nchg_towards_plus_transverse_Mmumu->binError(i)/
			       (_hist_profile_Nchg_towards_plus_transverse_Mmumu->binHeight(i) * _hist_profile_Nchg_towards_plus_transverse_Mmumu->binHeight(i))) * 
			       yval_tmp;	
	yerr_avgpt_tt.push_back(yerr_tmp);       
      }
   
      _hist_profile_avgpT_towards_pTmumu->setCoordinate(1, yval_avgpt_towards, yerr_avgpt_towards); 
      _hist_profile_avgpT_transverse_pTmumu->setCoordinate(1, yval_avgpt_transverse, yerr_avgpt_transverse);
      _hist_profile_avgpT_away_pTmumu->setCoordinate(1, yval_avgpt_away, yerr_avgpt_away);
      _hist_profile_avgpT_towards_plus_transverse_Mmumu->setCoordinate(1, yval_avgpt_tt, yerr_avgpt_tt);

      scale(_hist_pT_towards_zmass_81_101,1.0/integral(_hist_Nchg_towards_zmass_81_101));
      scale(_hist_pT_transverse_zmass_81_101,1.0/integral(_hist_Nchg_transverse_zmass_81_101));
      scale(_hist_pT_away_zmass_81_101,1.0/integral(_hist_Nchg_away_zmass_81_101));      
      scale(_hist_pT_transverse_zpt_5,1.0/integral(_hist_Nchg_transverse_zpt_5));
      scale(_hist_Nchg_towards_zmass_81_101,1.0/integral(_hist_Nchg_towards_zmass_81_101));
      scale(_hist_Nchg_transverse_zmass_81_101,1.0/integral(_hist_Nchg_transverse_zmass_81_101));
      scale(_hist_Nchg_away_zmass_81_101,1.0/integral(_hist_Nchg_away_zmass_81_101));
      scale(_hist_Nchg_transverse_zpt_5,1.0/integral(_hist_Nchg_transverse_zpt_5));

    }


  private:

    AIDA::IProfile1D *_hist_profile_Nchg_towards_pTmumu;
    AIDA::IProfile1D *_hist_profile_Nchg_transverse_pTmumu;
    AIDA::IProfile1D *_hist_profile_Nchg_away_pTmumu;
    AIDA::IProfile1D *_hist_profile_pTsum_towards_pTmumu;
    AIDA::IProfile1D *_hist_profile_pTsum_transverse_pTmumu;
    AIDA::IProfile1D *_hist_profile_pTsum_away_pTmumu;
    AIDA::IDataPointSet* _hist_profile_avgpT_towards_pTmumu;
    AIDA::IDataPointSet* _hist_profile_avgpT_transverse_pTmumu;
    AIDA::IDataPointSet* _hist_profile_avgpT_away_pTmumu;
    AIDA::IProfile1D *_hist_profile_Nchg_towards_plus_transverse_Mmumu;
    AIDA::IProfile1D *_hist_profile_pTsum_towards_plus_transverse_Mmumu;   
    AIDA::IDataPointSet*_hist_profile_avgpT_towards_plus_transverse_Mmumu;
    AIDA::IHistogram1D *_hist_Nchg_towards_zmass_81_101;
    AIDA::IHistogram1D *_hist_Nchg_transverse_zmass_81_101; 
    AIDA::IHistogram1D *_hist_Nchg_away_zmass_81_101;

    AIDA::IHistogram1D *_hist_pT_towards_zmass_81_101;
    AIDA::IHistogram1D *_hist_pT_transverse_zmass_81_101;
    AIDA::IHistogram1D *_hist_pT_away_zmass_81_101;

    AIDA::IHistogram1D *_hist_Nchg_transverse_zpt_5;
    AIDA::IHistogram1D *_hist_pT_transverse_zpt_5;
    //@}


  };



  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2012_I1107658> plugin_CMS_2012_I1107658;


}
