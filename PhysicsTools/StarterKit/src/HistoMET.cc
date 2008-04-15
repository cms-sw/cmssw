#include "PhysicsTools/StarterKit/interface/HistoMET.h"
#include "PhysicsTools/StarterKit/interface/PhysVarHisto.h"


using pat::HistoMET;
using pat::MET;

// Constructor:


HistoMET::HistoMET( std::string dir, std::string group,std::string pre,
		    double pt1, double pt2, double m1, double m2 ) 
  : HistoGroup<MET>( dir, group, pre, pt1, pt2, m1, m2)
{



  // book relevant MET histograms 
  addHisto( h_sumEt_              =
	    new PhysVarHisto( pre + "SumEt",              "MET sumEt",               20, 0, 1000, currDir_, "", "vD")  );
  addHisto( h_mEtSig_             =
	    new PhysVarHisto( pre + "MEtSig",             "MET mEtSig",              20, 0, 20, currDir_, "", "vD")  );
  addHisto( h_eLongitudinal_      =
	    new PhysVarHisto( pre + "ELongitudinal",      "MET eLongitudinal",       20, 0, 20, currDir_, "", "vD")  );
  addHisto( h_maxEtInEmTowers_    =
	    new PhysVarHisto( pre + "MaxEtInEmTowers",    "MET maxEtInEmTowers",     20, 0, 20, currDir_, "", "vD")  );
  addHisto( h_maxEtInHadTowers_   =
	    new PhysVarHisto( pre + "MaxEtInHadTowers",   "MET maxEtInHadTowers",    20, 0, 20, currDir_, "", "vD")  );
  addHisto( h_etFractionHadronic_ =
	    new PhysVarHisto( pre + "EtFractionHadronic", "MET etFractionHadronic",  20, 0, 1,  currDir_, "", "vD")  );
  addHisto( h_emEtFraction_       =
	    new PhysVarHisto( pre + "EmEtFraction",       "MET emEtFraction",        20, 0, 1,  currDir_, "", "vD")  );
  addHisto( h_hadEtInHB_          =
	    new PhysVarHisto( pre + "HadEtInHB",          "MET hadEtInHB",           20, pt1, pt2, currDir_, "", "vD")  );
  addHisto( h_hadEtInHO_          =
	    new PhysVarHisto( pre + "HadEtInHO",          "MET hadEtInHO",           20, pt1, pt2, currDir_, "", "vD")  );
  addHisto( h_hadEtInHE_          =
	    new PhysVarHisto( pre + "HadEtInHE",          "MET hadEtInHE",           20, pt1, pt2, currDir_, "", "vD")  );
  addHisto( h_hadEtInHF_          =
	    new PhysVarHisto( pre + "HadEtInHF",          "MET hadEtInHF",           20, pt1, pt2, currDir_, "", "vD")  );
  addHisto( h_emEtInEB_           =
	    new PhysVarHisto( pre + "EmEtInEB",           "MET emEtInEB",            20, pt1, pt2, currDir_, "", "vD")  );
  addHisto( h_emEtInEE_           =
	    new PhysVarHisto( pre + "EmEtInEE",           "MET emEtInEE",            20, pt1, pt2, currDir_, "", "vD")  );
  addHisto( h_emEtInHF_           =
	    new PhysVarHisto( pre + "EmEtInHF",           "MET emEtInHF",            20, pt1, pt2, currDir_, "", "vD")  );



}

HistoMET::~HistoMET()
{
  // Root deletes histograms, not us
}


void HistoMET::fill( const MET * met, uint iPart)
{

  // First fill common 4-vector histograms
  HistoGroup<MET>::fill( met, iPart );

  // fill relevant MET histograms
  h_sumEt_              ->fill( met->sumEt()               , iPart );
  h_mEtSig_             ->fill( met->mEtSig()              , iPart );
  h_eLongitudinal_      ->fill( met->e_longitudinal()      , iPart );
  h_maxEtInEmTowers_    ->fill( met->maxEtInEmTowers()     , iPart );     
  h_maxEtInHadTowers_   ->fill( met->maxEtInHadTowers()    , iPart );    
  h_etFractionHadronic_ ->fill( met->etFractionHadronic () , iPart ); 
  h_emEtFraction_       ->fill( met->emEtFraction()        , iPart );        
  h_hadEtInHB_          ->fill( met->hadEtInHB()           , iPart );           
  h_hadEtInHO_          ->fill( met->hadEtInHO()           , iPart );           
  h_hadEtInHE_          ->fill( met->hadEtInHE()           , iPart );           
  h_hadEtInHF_          ->fill( met->hadEtInHF()           , iPart );           
  h_emEtInEB_           ->fill( met->emEtInEB()            , iPart );            
  h_emEtInEE_           ->fill( met->emEtInEE()            , iPart );            
  h_emEtInHF_           ->fill( met->emEtInHF()            , iPart );            

}


void HistoMET::fillCollection( const std::vector<MET> & coll ) 
{
 
  h_size_->fill( coll.size() );     //! Save the size of the collection.

  std::vector<MET>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoMET::clearVec()
{
  HistoGroup<MET>::clearVec();
}
