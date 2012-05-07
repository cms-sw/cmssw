// -*- C++ -*-
#include "Rivet/Analysis.hh"
#include "Rivet/RivetAIDA.hh"
#include "Rivet/Projections/ChargedFinalState.hh"
#include "Rivet/Projections/HadronicFinalState.hh"
#include "Rivet/Tools/ParticleIdUtils.hh"
#include "Rivet/Projections/Beam.hh"
using namespace std;

namespace Rivet {


  class CMS_2011_S8884919 : public Analysis {
  public:

    CMS_2011_S8884919() : Analysis("CMS_2011_S8884919") {
       setBeams(PROTON, PROTON);
       setNeedsCrossSection(false);
    }
    
    //Container for the moments
   /* struct IMoments {
      int n;
      double mean;
      vector<double> cq;
      
      //default constructor
      IMoments():n(6),mean(0),cq(vector<double>(n,0)){}
    };
    
    IMoments _moments_eta05;
    IMoments _moments_eta24; */
    
    
//AK =====================================================DECLARATIONS
  private:


    vector<AIDA::IHistogram1D*> _h_dNch_dn;
    
    //vector<double>      _v_dNch_dn_binning1_eta05;
    //vector<double>      _v_dNch_dn_binning1_eta24;
    
    AIDA::IHistogram1D* _h_dNch_dn_pt500_eta24;

    //AIDA::IHistogram1D* _h_cq_eta05;
    //AIDA::IHistogram1D* _h_cq_eta24;

    AIDA::IProfile1D*   _h_dmpt_dNch_eta24;

    AIDA::IHistogram1D* _h_KNO_eta05;
    AIDA::IHistogram1D* _h_KNO_eta24;
        
    double _Nevt_after_cuts;
    vector<double> _etabins;
    vector<int> _nch_in_Evt;
    vector<int> _nch_in_Evt_pt500;
    double _sumpt;
    int nch_max;
    
    //mandatory functions
    void init();
    void analyze(const Event&);
    void finalize();
    
    
    /*void moments_add(IMoments& , double , double = 1);
    void makeMoments(AIDA::IHistogram1D* , IMoments&);
    void makeMoments(vector<double>& , IMoments&);
    
    void makeKNO(AIDA::IHistogram1D* nch , AIDA::IHistogram1D* kno){
      
    }*/


  };
  


//----------------------------------------------------------------------------------------------


  /*void CMS_2011_S8884919::moments_add(IMoments& moments , double value , double weight){
    moments.mean += value*weight;

    for(int m = 0 ; m < moments.n ; ++m)
      (moments.cq[m]) += pow(value,m) * weight;
  }


  void CMS_2011_S8884919::makeMoments(AIDA::IHistogram1D* h , IMoments& moments){
    int sumHeight = 0;
  
    for(int i=0 ; i < h->axis().bins() ; ++i){
      moments_add(moments , (h->axis().binLowerEdge(i) + h->axis().binUpperEdge(i)) / 2. , h->binHeight(i));
      sumHeight += h->binHeight(i);
    }
      
    
    //finishing moments
    if(sumHeight != 0){
      moments.mean /= sumHeight;
      for(int m = 0 ; m < moments.n ; ++m)
        if( moments.mean != 0) moments.cq[m] = (moments.cq[m] / sumHeight) / pow(moments.mean , m) ;
    }
  }
  
  void CMS_2011_S8884919::makeMoments(vector<double>& v , IMoments& moments){
    int sumHeight = 0;
  
    for(unsigned i=0 ; i < v.size() ; ++i){
      moments_add(moments , i , v[i]);
      sumHeight += v[i];
    }
      
    
    //finishing moments
    if(sumHeight != 0){
      moments.mean /= sumHeight;
      for(int m = 0 ; m < moments.n ; ++m)
        if( moments.mean != 0) moments.cq[m] = (moments.cq[m] / sumHeight) / pow(moments.mean , m) ;
    }
  }
  */




//AK =====================================================INIT
  void CMS_2011_S8884919::init() {
    ChargedFinalState cfs(-2.4, 2.4, 0.0*GeV);
    addProjection(cfs, "CFS");
    addProjection(Beam(), "Beam");

    nch_max = 400;
    _Nevt_after_cuts = 0;

    //eta bins      
    _etabins.push_back(0.5) ; _etabins.push_back(1.0) ; _etabins.push_back(1.5) ; _etabins.push_back(2.0) ; _etabins.push_back(2.4) ;
    _h_dNch_dn.reserve( _etabins.size() );
    ostringstream t("");
    
    if(fuzzyEquals(sqrtS(), 900, 1E-3)){
      
      for (unsigned ietabin=0; ietabin < _etabins.size(); ietabin++){
        t.str("") ; t << "$|\\eta| <$ " << _etabins[ietabin] << " , $\\sqrt(s)$ = 0.9 TeV" ;
	_h_dNch_dn.push_back( bookHistogram1D( 2 + ietabin, 1, 1 , t.str() , "n" , "$P_{n}$") );
      }
        
      _h_dNch_dn_pt500_eta24 = bookHistogram1D( 20 , 1, 1 , "$p_{T} >$ 500 GeV/c , $|\\eta| <$ 2.4 , $\\sqrt(s)$ = 0.9 TeV" , "n" , "$P_{n}$");

      //_h_cq_eta05 = bookHistogram1D( 17 , 1, 1 , "$|\\eta| <$ 0.5 , $\\sqrt(s)$ = 0.9 TeV" , "q" , "$C_{q}$");
      //_h_cq_eta24 = bookHistogram1D( 17 , 1, 2 , "$|\\eta| <$ 2.4 , $\\sqrt(s)$ = 0.9 TeV" , "q" , "$C_{q}$");

      _h_dmpt_dNch_eta24 = bookProfile1D( 23 , 1, 1 , "$|\\eta| <$ 2.4 , $\\sqrt(s)$ = 0.9 TeV" , "n" , "$< p_{T}> [ GeV/c ]$");

    }
    
    if(fuzzyEquals(sqrtS(), 2360, 1E-3)){
      for (unsigned ietabin=0; ietabin < _etabins.size(); ietabin++){
        t.str("") ; t << "$|\\eta| <$ " << _etabins[ietabin] << " , $\\sqrt(s)$ = 2.36 TeV" ;
	_h_dNch_dn.push_back( bookHistogram1D( 7 + ietabin, 1, 1 , t.str() , "n" , "$P_{n}$") );
      }
        
      _h_dNch_dn_pt500_eta24 = bookHistogram1D( 21 , 1, 1 , "$p_{T} >$ 500 GeV/c , $|\\eta| <$ 2.4 , $\\sqrt(s)$ = 2.36 TeV" , "n" , "$P_{n}$");

      //_h_cq_eta05 = bookHistogram1D( 18 , 1, 1 , "$|\\eta| <$ 0.5 , $\\sqrt(s)$ = 2.36 TeV" , "q" , "$C_{q}$");
      //_h_cq_eta24 = bookHistogram1D( 18 , 1, 2 , "$|\\eta| <$ 2.4 , $\\sqrt(s)$ = 2.36 TeV" , "q" , "$C_{q}$");

      _h_dmpt_dNch_eta24 = bookProfile1D( 24 , 1, 1 , "$|\\eta| <$ 2.4 , $\\sqrt(s)$ = 2.36 TeV" , "n" , "$< p_{T}> [ GeV/c ]$");
    }
    
    if(fuzzyEquals(sqrtS(), 7000, 1E-3)){
      for (unsigned ietabin=0; ietabin < _etabins.size(); ietabin++){
        t.str("") ; t << "$|\\eta| <$ " << _etabins[ietabin] << " , $\\sqrt(s)$ = 7 TeV" ;
	_h_dNch_dn.push_back( bookHistogram1D( 12 + ietabin, 1, 1 , t.str() , "n" , "$P_{n}$") );
      }
        
      _h_dNch_dn_pt500_eta24 = bookHistogram1D( 22 , 1, 1 , "$p_{T} >$ 500 GeV/c , $|\\eta| <$ 2.4 , $\\sqrt(s)$ = 7 TeV" , "n" , "$P_{n}$");

      //_h_cq_eta05 = bookHistogram1D( 19 , 1, 1 , "$|\\eta| <$ 0.5 , $\\sqrt(s)$ = 7 TeV" , "q" , "$C_{q}$");
      //_h_cq_eta24 = bookHistogram1D( 19 , 1, 2 , "$|\\eta| <$ 2.4 , $\\sqrt(s)$ = 7 TeV" , "q" , "$C_{q}$");

      _h_dmpt_dNch_eta24 = bookProfile1D( 25 , 1, 1 , "$|\\eta| <$2.4 , $\\sqrt(s)$ = 7 TeV" , "n" , "$< p_{T}> [ GeV/c ]$");

    }
    
    //_v_dNch_dn_binning1_eta05.assign(nch_max+1,0);
    //_v_dNch_dn_binning1_eta24.assign(nch_max+1,0);    
    
  }

//AK =====================================================ANALYZE
  void CMS_2011_S8884919::analyze(const Event& event) {
    const double weight = event.weight();

    //charge particles
    const ChargedFinalState& charged = applyProjection<ChargedFinalState>(event, "CFS");
    
    //This cut is not needed
    /*if (charged.particles().size()<1) {
      vetoEvent;
    }*/ 
    
    
    _Nevt_after_cuts += weight;
    
    //resetting the multiplicity for the event to 0;
    _nch_in_Evt.assign(_etabins.size() , 0);
    _nch_in_Evt_pt500.assign(_etabins.size() , 0);
    _sumpt = 0;
    
    //std::cout << charged.size() << std::endl;

    //Loop over particles in event
    foreach (const Particle& p, charged.particles()) {
      
      //selecting only charged hadrons
      if(! PID::isHadron(p.pdgId())) continue;

      double pT = p.momentum().pT();	      
      double eta = p.momentum().eta();

      _sumpt+=pT;

      //cout << "eta : " << eta << "   pT : " << pT << endl;

      for (int ietabin=_etabins.size()-1; ietabin >= 0 ; --ietabin){   
        //cout << "  etabin : " << _etabins[ietabin] << "  res : " << (fabs(eta) <= _etabins[ietabin]) << endl;
        if (fabs(eta) <= _etabins[ietabin]){
	  ++(_nch_in_Evt[ietabin]);
          
          if(pT>0.5)
	    ++(_nch_in_Evt_pt500[ietabin]);  
        }
        //break loop to go faster
        else
          break;
        
      }

    }
    
    //filling mutliplicity dependent histogramms
    for (unsigned ietabin=0; ietabin < _etabins.size(); ietabin++){
      _h_dNch_dn[ietabin]->fill(_nch_in_Evt[ietabin] , weight);
    }
    
    //Do only if eta bins are the needed ones
    if(_etabins[4] == 2.4 && _etabins[0] == 0.5){
      if(_nch_in_Evt[4] != 0) _h_dmpt_dNch_eta24->fill(_nch_in_Evt[4] , _sumpt / _nch_in_Evt[4] , weight);
      
      _h_dNch_dn_pt500_eta24->fill(_nch_in_Evt_pt500[4] , weight);
      
      /*if( _nch_in_Evt[4] < nch_max ){
        (_v_dNch_dn_binning1_eta05[_nch_in_Evt[0]])+=weight;
        (_v_dNch_dn_binning1_eta24[_nch_in_Evt[0]])+=weight;
      }*/
      
    }
    else
      getLog() << Log::WARNING << "You changed the number of eta bins, but forgot to propagate it everywhere !! " << endl;   
        
  }
  
//AK =====================================================FINALIZE
  void CMS_2011_S8884919::finalize() {
    getLog() << Log::INFO << "Number of events after event selection: " << _Nevt_after_cuts << endl;	    
    
    /*makeMoments(_v_dNch_dn_binning1_eta05 , _moments_eta05);
    makeMoments(_v_dNch_dn_binning1_eta24 , _moments_eta24);
    
    for(int m = 0 ; m < _moments_eta05.n ; ++m){
      _h_cq_eta05->fill(m , _moments_eta05.cq[m]);
      _h_cq_eta24->fill(m , _moments_eta24.cq[m]);
    }*/
    
    for (unsigned ietabin=0; ietabin < _etabins.size(); ietabin++){
      normalize(_h_dNch_dn[ietabin]);
    }
    normalize(_h_dNch_dn_pt500_eta24);
  }


  // This global object acts as a hook for the plugin system
  AnalysisBuilder<CMS_2011_S8884919> plugin_CMS_2011_S8884919;

}

