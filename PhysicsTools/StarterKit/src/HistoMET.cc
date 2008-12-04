#include "PhysicsTools/StarterKit/interface/HistoMET.h"
#include "PhysicsTools/StarterKit/interface/PhysVarHisto.h"

#include <iostream>

using pat::HistoMET;
using pat::MET;

using namespace std;

// Constructor:


HistoMET::HistoMET( std::string dir, std::string group,std::string pre,
		    double pt1, double pt2, double m1, double m2,
		    TFileDirectory * parentDir) 
  : HistoGroup<MET>( dir, group, pre, pt1, pt2, m1, m2, parentDir)
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


  addHisto( jetME_           =
            new PhysVarHisto( pre + "caloMETReco", "caloMETReco", 3, 1, 4, currDir_, "", "vD")  );
/*
  addHisto( hNevents_           =
            new PhysVarHisto( pre + "METTaskNevents","METTask_Nevents",1,0,1, currDir_, "", "vD")  );
*/
  addHisto( hCaloMEx_           =
            new PhysVarHisto( pre + "METTaskCaloMEx","METTask_CaloMEx",2001,-500,501, currDir_, "", "vD")  );
  addHisto( hCaloMEy_           =
            new PhysVarHisto( pre + "METTaskCaloMEy","METTask_CaloMEy",2001,-500,501, currDir_, "", "vD")  );
  addHisto( hCaloEz_           =
            new PhysVarHisto( pre + "METTaskCaloEz","METTask_CaloEz",2001,-500,501, currDir_, "", "vD")  );
  addHisto( hCaloMET_           =
            new PhysVarHisto( pre + "METTaskCaloMET","METTask_CaloMET",2001,0,2001, currDir_, "", "vD")  );
  addHisto( hCaloMETPhi_           =
            new PhysVarHisto( pre + "METTaskCaloMETPhi","METTask_CaloMETPhi",80,-4,4, currDir_, "", "vD")  );
/*
  addHisto( hCaloHadEtInEB_           =
            new PhysVarHisto( pre + "METTaskCaloHadEtInEB","METTask_CaloHadEtInEB",4001,0,4001, currDir_, "", "vD")  );
  addHisto( hCaloHadEtInEE_           =
            new PhysVarHisto( pre + "METTaskCaloHadEtInEE","METTask_CaloHadEtInEE",4001,0,4001, currDir_, "", "vD")  );
*/

}

HistoMET::~HistoMET()
{
  // Root deletes histograms, not us
}


void HistoMET::fill( const MET * met, uint iPart, double weight)
{

  // First fill common 4-vector histograms
  HistoGroup<MET>::fill( met, iPart, weight );

  // fill relevant MET histograms
  h_sumEt_              ->fill( met->sumEt()               , iPart , weight );
  h_mEtSig_             ->fill( met->mEtSig()              , iPart , weight );
  h_eLongitudinal_      ->fill( met->e_longitudinal()      , iPart , weight );
  h_maxEtInEmTowers_    ->fill( met->maxEtInEmTowers()     , iPart , weight );     
  h_maxEtInHadTowers_   ->fill( met->maxEtInHadTowers()    , iPart , weight );    
  h_etFractionHadronic_ ->fill( met->etFractionHadronic () , iPart , weight ); 
  h_emEtFraction_       ->fill( met->emEtFraction()        , iPart , weight );        
  h_hadEtInHB_          ->fill( met->hadEtInHB()           , iPart , weight );           
  h_hadEtInHO_          ->fill( met->hadEtInHO()           , iPart , weight );           
  h_hadEtInHE_          ->fill( met->hadEtInHE()           , iPart , weight );           
  h_hadEtInHF_          ->fill( met->hadEtInHF()           , iPart , weight );           
  h_emEtInEB_           ->fill( met->emEtInEB()            , iPart , weight );            
  h_emEtInEE_           ->fill( met->emEtInEE()            , iPart , weight );            
  h_emEtInHF_           ->fill( met->emEtInHF()            , iPart , weight );            

  jetME_          ->fill(  1     , iPart , weight );
//  hNevents_          ->fill( met->      , iPart , weight );
  hCaloMEx_          ->fill( met->px()      , iPart , weight );
  hCaloMEy_          ->fill( met->py()      , iPart , weight );
  hCaloEz_          ->fill(  met->pz()     , iPart , weight );
  hCaloMET_          ->fill( met->pt()      , iPart , weight );
  hCaloMETPhi_          ->fill(met->phi()       , iPart , weight );
//  hCaloHadEtInEB_          ->fill( met->      , iPart , weight );
//  hCaloHadEtInEE_          ->fill( met->      , iPart , weight );


}


void HistoMET::fill( const reco::ShallowClonePtrCandidate * pshallow, uint iPart, double weight)
{

  // Get the underlying object that the shallow clone represents
  const pat::MET * met = dynamic_cast<const pat::MET*>(pshallow);

  if ( met == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a met" << endl;
    return;
  }

  

  // First fill common 4-vector histograms from shallow clone

  HistoGroup<MET>::fill( pshallow, iPart, weight);

  // fill relevant MET histograms
  h_sumEt_              ->fill( met->sumEt()               , iPart , weight );
  h_mEtSig_             ->fill( met->mEtSig()              , iPart , weight );
  h_eLongitudinal_      ->fill( met->e_longitudinal()      , iPart , weight );
  h_maxEtInEmTowers_    ->fill( met->maxEtInEmTowers()     , iPart , weight );     
  h_maxEtInHadTowers_   ->fill( met->maxEtInHadTowers()    , iPart , weight );    
  h_etFractionHadronic_ ->fill( met->etFractionHadronic () , iPart , weight ); 
  h_emEtFraction_       ->fill( met->emEtFraction()        , iPart , weight );        
  h_hadEtInHB_          ->fill( met->hadEtInHB()           , iPart , weight );           
  h_hadEtInHO_          ->fill( met->hadEtInHO()           , iPart , weight );           
  h_hadEtInHE_          ->fill( met->hadEtInHE()           , iPart , weight );           
  h_hadEtInHF_          ->fill( met->hadEtInHF()           , iPart , weight );           
  h_emEtInEB_           ->fill( met->emEtInEB()            , iPart , weight );            
  h_emEtInEE_           ->fill( met->emEtInEE()            , iPart , weight );            
  h_emEtInHF_           ->fill( met->emEtInHF()            , iPart , weight );            


  jetME_          ->fill(  1     , iPart , weight );
//  hNevents_          ->fill( met->      , iPart , weight );
  hCaloMEx_          ->fill( met->px()      , iPart , weight );
  hCaloMEy_          ->fill( met->py()      , iPart , weight );
  hCaloEz_          ->fill(  met->pz()     , iPart , weight );
  hCaloMET_          ->fill( met->pt()      , iPart , weight );
  hCaloMETPhi_          ->fill(met->phi()       , iPart , weight );
//  hCaloHadEtInEB_          ->fill( met->      , iPart , weight );
//  hCaloHadEtInEE_          ->fill( met->      , iPart , weight );


}


void HistoMET::fillCollection( const std::vector<MET> & coll, double weight ) 
{
 
  h_size_->fill( coll.size(), 1, weight );     //! Save the size of the collection.

  std::vector<MET>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i, weight);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoMET::clearVec()
{
  HistoGroup<MET>::clearVec();
  h_sumEt_->clearVec();
  h_mEtSig_->clearVec();
  h_eLongitudinal_->clearVec();

  h_maxEtInEmTowers_->clearVec();
  h_maxEtInHadTowers_->clearVec();
  h_etFractionHadronic_->clearVec();
  h_emEtFraction_->clearVec();
  h_hadEtInHB_->clearVec();
  h_hadEtInHO_->clearVec();
  h_hadEtInHE_->clearVec();
  h_hadEtInHF_->clearVec();
  h_emEtInEB_->clearVec();
  h_emEtInEE_->clearVec();
  h_emEtInHF_->clearVec();
  jetME_->clearVec();
  hCaloMEx_->clearVec();
//  hNevents_->clearVec();
  hCaloMEy_->clearVec();
  hCaloEz_->clearVec();
  hCaloMET_->clearVec();
  hCaloMETPhi_->clearVec();
//  hCaloHadEtInEB_->clearVec();
//  hCaloHadEtInEE_->clearVec();

}
