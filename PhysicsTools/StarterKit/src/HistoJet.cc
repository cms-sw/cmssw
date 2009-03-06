#include "PhysicsTools/StarterKit/interface/HistoJet.h"


#include <iostream>

using pat::HistoJet;
using namespace std;

// Constructor:

HistoJet::HistoJet(  std::string dir, std::string group,std::string pre,
		     double pt1, double pt2, double m1, double m2,
		     TFileDirectory * parentDir ) 
  : HistoGroup<Jet>( dir, group, pre, pt1, pt2, m1, m2, parentDir)
{
  // book relevant jet histograms
  addHisto( h_jetFlavour_   =
	    new PhysVarHisto( pre + "Flavour", "Jet Flavour", 21, 0, 21, currDir_, "", "vD" )
	    );
  addHisto( h_BDiscriminant_=
	    new PhysVarHisto( pre + "BDiscriminant", "Jet B Discriminant", 100, -10, 90, currDir_, "", "vD")
	    );
  addHisto( h_jetCharge_    =
	    new PhysVarHisto( pre + "Charge", "Jet Charge", 100, -5, 5, currDir_, "", "vD")
	    );
  addHisto( h_nTrk_         =
	    new PhysVarHisto( pre + "NTrk", "Jet N_{TRK}", 51, -0.5, 50.5, currDir_, "", "vD" )
	    );


  addHisto( jetME_         =
            new PhysVarHisto( pre + "jetReco", "jetReco", 3, 1, 4, currDir_, "", "vD" )
            );
/*
  addHisto( mEta_         =
            new PhysVarHisto( pre + "Eta", "Eta",100, 0, 100, currDir_, "", "vD" )
            );


  addHisto( mPhi_         =
            new PhysVarHisto( pre + "Phi", "Phi", 100, 0, 100,currDir_, "", "vD" )
            );
*/

  addHisto( mE_         =
            new PhysVarHisto( pre + "E",   "E", 100, 0, 100,currDir_, "", "vD" )
            );
  addHisto( mP_         =
            new PhysVarHisto( pre + "P",  "P", 100, 0, 100,currDir_, "", "vD" )
            );
/*
  addHisto( mPt_         =
            new PhysVarHisto( pre + "Pt",  "Pt", 100, 0, 100,currDir_, "", "vD" )
            );
*/
  addHisto( mPt_1_         =
            new PhysVarHisto( pre + "Pt1", "Pt1", 100, 0, 100,currDir_, "", "vD" )
            );
  addHisto( mPt_2_         =
            new PhysVarHisto( pre + "Pt2", "Pt2", 100, 0, 300,currDir_, "", "vD" )
            );
  addHisto( mPt_3_         =
            new PhysVarHisto( pre + "Pt3", "Pt3", 100, 0, 5000,currDir_, "", "vD" )
            );
/*
  addHisto( mMass_         =
            new PhysVarHisto( pre + "Mass", "Mass", 100, 0, 25,currDir_, "", "vD" )
            );
*/
  addHisto( mConstituents_         =
            new PhysVarHisto( pre + "Constituents", "# of Constituents", 100, 0, 30,currDir_, "", "vD" )
            );
//
  addHisto( mEtaFirst_         =
            new PhysVarHisto( pre + "EtaFirst", "EtaFirst", 100, -5, 5,currDir_, "", "vD" )
            );

  addHisto( mPhiFirst_         =
            new PhysVarHisto( pre + "PhiFirst", "PhiFirst", 70, -3.5, 3.5, currDir_, "", "vD" )
            );

  addHisto( mEFirst_         =
            new PhysVarHisto( pre + "EFirst", "EFirst", 100, 0, 1000, currDir_, "", "vD" )
            );

  addHisto( mPtFirst_         =
            new PhysVarHisto( pre + "PtFirst", "PtFirst", 100, 0, 500, currDir_, "", "vD" )
            );
//

  addHisto( mMaxEInEmTowers_         =
            new PhysVarHisto( pre + "MaxEInEmTowers", "MaxEInEmTowers", 100, 0, 100, currDir_, "", "vD" )
            );

  addHisto( mMaxEInHadTowers_         =
            new PhysVarHisto( pre + "MaxEInHadTowers", "MaxEInHadTowers", 100, 0, 100, currDir_, "", "vD" )
            );

  addHisto( mHadEnergyInHO_         =
            new PhysVarHisto( pre + "HadEnergyInHO", "HadEnergyInHO", 100, 0, 10, currDir_, "", "vD" )
            );

  addHisto( mHadEnergyInHB_         =
            new PhysVarHisto( pre + "HadEnergyInHB", "HadEnergyInHB", 100, 0, 50, currDir_, "", "vD" )
            );
  addHisto( mHadEnergyInHF_         =
            new PhysVarHisto( pre + "HadEnergyInHF", "HadEnergyInHF", 100, 0, 50, currDir_, "", "vD" )
            );
  addHisto( mHadEnergyInHE_         =
            new PhysVarHisto( pre + "HadEnergyInHE", "HadEnergyInHE", 100, 0, 100, currDir_, "", "vD" )
            );
  addHisto( mEmEnergyInEB_         =
            new PhysVarHisto( pre + "EmEnergyInEB", "EmEnergyInEB", 100, 0, 10, currDir_, "", "vD" )
            );
  addHisto( mEmEnergyInEE_         =
            new PhysVarHisto( pre + "EmEnergyInEE", "EmEnergyInEE", 100, 0, 50, currDir_, "", "vD" )
            );
  addHisto( mEmEnergyInHF_         =
            new PhysVarHisto( pre + "EmEnergyInHF", "EmEnergyInHF", 120, -20, 100, currDir_, "", "vD" )
            );
  addHisto( mEnergyFractionHadronic_         =
            new PhysVarHisto( pre + "EnergyFractionHadronic", "EnergyFractionHadronic", 120, -0.1, 1.1, currDir_, "", "vD" )
            );
  addHisto( mEnergyFractionEm_         =
            new PhysVarHisto( pre + "EnergyFractionEm", "EnergyFractionEm", 120, -0.1, 1.1, currDir_, "", "vD" )
            );
  addHisto( mN90_         =
            new PhysVarHisto( pre + "N90", "N90", 50, 0, 50, currDir_, "", "vD" )
            );
/*
  // PFlowJet specific
  addHisto( mChargedHadronEnergy_         =
            new PhysVarHisto( pre + "mChargedHadronEnergy", "mChargedHadronEnergy", 100, 0, 100, currDir_, "", "vD" )
            );
  addHisto( mNeutralHadronEnergy_         =
            new PhysVarHisto( pre + "mNeutralHadronEnergy", "mNeutralHadronEnergy", 100, 0, 100, currDir_, "", "vD" )
            );
  addHisto( mChargedEmEnergy_         =
            new PhysVarHisto( pre + "mChargedEmEnergy ", "mChargedEmEnergy ", 100, 0, 100, currDir_, "", "vD" )
            );
  addHisto(  mChargedMuEnergy_         =
            new PhysVarHisto( pre + "mChargedMuEnergy", "mChargedMuEnergy", 100, 0, 100, currDir_, "", "vD" )
            );
  addHisto( mNeutralEmEnergy_         =
            new PhysVarHisto( pre + "mNeutralEmEnergy", "mNeutralEmEnergy", 100, 0, 100, currDir_, "", "vD" )
            );
  addHisto( mChargedMultiplicity_         =
            new PhysVarHisto( pre + "mChargedMultiplicity ", "mChargedMultiplicity ", 100, 0, 100, currDir_, "", "vD" )
            );
  addHisto( mNeutralMultiplicity_         =
            new PhysVarHisto( pre + " mNeutralMultiplicity", "mNeutralMultiplicity", 100, 0, 100, currDir_, "", "vD" )
            );
  addHisto( mMuonMultiplicity_         =
            new PhysVarHisto( pre + "mMuonMultiplicity", "mMuonMultiplicity", 100, 0, 100, currDir_, "", "vD" )
            );
  addHisto( mNeutralFraction_         =
            new PhysVarHisto( pre + "NeutralFraction","Neutral Fraction",100,0,1, currDir_, "", "vD" )
            );
*/

}

HistoJet::~HistoJet()
{
}


void HistoJet::fill( const Jet * jet, uint iJet, double weight )
{

  // First fill common 4-vector histograms
  HistoGroup<Jet>::fill( jet, iJet, weight );

  // fill relevant jet histograms
  h_jetFlavour_     ->fill( jet->partonFlavour(), iJet, weight );
  h_BDiscriminant_  ->fill( jet->bDiscriminator("trackCountingHighPurJetTags"), iJet, weight );
  h_jetCharge_      ->fill( jet->jetCharge(), iJet, weight );
  h_nTrk_           ->fill( jet->associatedTracks().size(), iJet, weight );

  jetME_->fill(1,   iJet,   weight);
 
//  if (mEta_) mEta_->fill(jet->eta(),   iJet,   weight);
//  if (mPhi_) mPhi_->fill(jet->phi(),   iJet,   weight);
  if (mE_) mE_->fill(jet->energy(),   iJet,   weight);
  if (mP_) mP_->fill(jet->p(),   iJet,   weight);
//  if (mPt_) mPt_->fill(jet->pt(),   iJet,   weight);
  if (mPt_1_) mPt_1_->fill(jet->pt(),   iJet,   weight);
  if (mPt_2_) mPt_2_->fill(jet->pt(),   iJet,   weight);
  if (mPt_3_) mPt_3_->fill(jet->pt(),   iJet,   weight);
//  if (mMass_) mMass_->fill(jet->mass(),   iJet,   weight);
  if (mConstituents_) mConstituents_->fill(jet->nConstituents(),   iJet,   weight);
  //  if (jet== jet.begin ()) { // first jet
  //    if (mEtaFirst_) mEtaFirst_->fill(jet->eta(),   iJet,   weight);
  //    if (mPhiFirst_) mPhiFirst_->fill(jet->phi(),   iJet,   weight);
  //    if (mEFirst_) mEFirst_->fill(jet->energy(),   iJet,   weight);
  //    if (mPtFirst_) mPtFirst_->fill(jet->pt(),   iJet,   weight);
  //  }
  if (mMaxEInEmTowers_) mMaxEInEmTowers_->fill(jet->maxEInEmTowers(),   iJet,   weight);
  if (mMaxEInHadTowers_) mMaxEInHadTowers_->fill(jet->maxEInHadTowers(),   iJet,   weight);
  if (mHadEnergyInHO_) mHadEnergyInHO_->fill(jet->hadEnergyInHO(),   iJet,   weight);
  if (mHadEnergyInHB_) mHadEnergyInHB_->fill(jet->hadEnergyInHB(),   iJet,   weight);
  if (mHadEnergyInHF_) mHadEnergyInHF_->fill(jet->hadEnergyInHF(),   iJet,   weight);
  if (mHadEnergyInHE_) mHadEnergyInHE_->fill(jet->hadEnergyInHE(),   iJet,   weight);
  if (mEmEnergyInEB_) mEmEnergyInEB_->fill(jet->emEnergyInEB(),   iJet,   weight);
  if (mEmEnergyInEE_) mEmEnergyInEE_->fill(jet->emEnergyInEE(),   iJet,   weight);
  if (mEmEnergyInHF_) mEmEnergyInHF_->fill(jet->emEnergyInHF(),   iJet,   weight);
  if (mEnergyFractionHadronic_) mEnergyFractionHadronic_->fill(jet->energyFractionHadronic(),   iJet,   weight);
  if (mEnergyFractionEm_) mEnergyFractionEm_->fill(jet->emEnergyFraction(),   iJet,   weight);
  if (mN90_) mN90_->fill(jet->n90(),   iJet,   weight);  
/*
  // PFlowJet specific
  if (mChargedHadronEnergy_)  mChargedHadronEnergy_->fill (jet->chargedHadronEnergy(),  iJet,  weight);
  if (mNeutralHadronEnergy_)  mNeutralHadronEnergy_->fill (jet->neutralHadronEnergy(),  iJet,  weight);
  if (mChargedEmEnergy_) mChargedEmEnergy_->fill(jet->chargedEmEnergy(),  iJet,  weight);
  if (mChargedMuEnergy_) mChargedMuEnergy_->fill (jet->chargedMuEnergy (),  iJet,  weight);
  if (mNeutralEmEnergy_) mNeutralEmEnergy_->fill(jet->neutralEmEnergy(),  iJet,  weight);
  if (mChargedMultiplicity_ ) mChargedMultiplicity_->fill(jet->chargedMultiplicity(),  iJet,  weight);
  if (mNeutralMultiplicity_ ) mNeutralMultiplicity_->fill(jet->neutralMultiplicity(),  iJet,  weight);
  if (mMuonMultiplicity_ )mMuonMultiplicity_->fill (jet-> muonMultiplicity(),  iJet,  weight);

  if (mNeutralFraction_) mNeutralFraction_->fill (jet->neutralMultiplicity()/jet->nConstituents(),  iJet,  weight);
*/



}

void HistoJet::fill( const reco::ShallowClonePtrCandidate * pshallow, uint iJet, double weight )
{

  // Get the underlying object that the shallow clone represents
  const pat::Jet * jet = dynamic_cast<const pat::Jet*>(pshallow);

  if ( jet == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a jet" << endl;
    return;
  }

  // First fill common 4-vector histograms from shallow clone
  HistoGroup<Jet>::fill( pshallow, iJet, weight);

  // fill relevant jet histograms
  h_jetFlavour_     ->fill( jet->partonFlavour(), iJet, weight );
  h_BDiscriminant_  ->fill( jet->bDiscriminator("trackCountingHighPurJetTags"), iJet, weight );
  h_jetCharge_      ->fill( jet->jetCharge(), iJet, weight );
  h_nTrk_           ->fill( jet->associatedTracks().size(), iJet, weight );


  jetME_->fill(1,   iJet,   weight);

//  if (mEta_) mEta_->fill(jet->eta(),   iJet,   weight);
//  if (mPhi_) mPhi_->fill(jet->phi(),   iJet,   weight);
  if (mE_) mE_->fill(jet->energy(),   iJet,   weight);
  if (mP_) mP_->fill(jet->p(),   iJet,   weight);
//  if (mPt_) mPt_->fill(jet->pt(),   iJet,   weight);
  if (mPt_1_) mPt_1_->fill(jet->pt(),   iJet,   weight);
  if (mPt_2_) mPt_2_->fill(jet->pt(),   iJet,   weight);
  if (mPt_3_) mPt_3_->fill(jet->pt(),   iJet,   weight);
//  if (mMass_) mMass_->fill(jet->mass(),   iJet,   weight);
  if (mConstituents_) mConstituents_->fill(jet->nConstituents(),   iJet,   weight);
  //  if (jet== jet.begin ()) { // first jet
  //    if (mEtaFirst_) mEtaFirst_->fill(jet->eta(),   iJet,   weight);
  //    if (mPhiFirst_) mPhiFirst_->fill(jet->phi(),   iJet,   weight);
  //    if (mEFirst_) mEFirst_->fill(jet->energy(),   iJet,   weight);
  //    if (mPtFirst_) mPtFirst_->fill(jet->pt(),   iJet,   weight);
  //  }
  if (mMaxEInEmTowers_) mMaxEInEmTowers_->fill(jet->maxEInEmTowers(),   iJet,   weight);
  if (mMaxEInHadTowers_) mMaxEInHadTowers_->fill(jet->maxEInHadTowers(),   iJet,   weight);
  if (mHadEnergyInHO_) mHadEnergyInHO_->fill(jet->hadEnergyInHO(),   iJet,   weight);
  if (mHadEnergyInHB_) mHadEnergyInHB_->fill(jet->hadEnergyInHB(),   iJet,   weight);
  if (mHadEnergyInHF_) mHadEnergyInHF_->fill(jet->hadEnergyInHF(),   iJet,   weight);
  if (mHadEnergyInHE_) mHadEnergyInHE_->fill(jet->hadEnergyInHE(),   iJet,   weight);
  if (mEmEnergyInEB_) mEmEnergyInEB_->fill(jet->emEnergyInEB(),   iJet,   weight);
  if (mEmEnergyInEE_) mEmEnergyInEE_->fill(jet->emEnergyInEE(),   iJet,   weight);
  if (mEmEnergyInHF_) mEmEnergyInHF_->fill(jet->emEnergyInHF(),   iJet,   weight);
  if (mEnergyFractionHadronic_) mEnergyFractionHadronic_->fill(jet->energyFractionHadronic(),   iJet,   weight);
  if (mEnergyFractionEm_) mEnergyFractionEm_->fill(jet->emEnergyFraction(),   iJet,   weight);
  if (mN90_) mN90_->fill(jet->n90(),   iJet,   weight);

/*
  // PFlowJet specific
  if (mChargedHadronEnergy_)  mChargedHadronEnergy_->fill (jet->chargedHadronEnergy(),  iJet,  weight);
  if (mNeutralHadronEnergy_)  mNeutralHadronEnergy_->fill (jet->neutralHadronEnergy(),  iJet,  weight);
  if (mChargedEmEnergy_) mChargedEmEnergy_->fill(jet->chargedEmEnergy(),  iJet,  weight);
  if (mChargedMuEnergy_) mChargedMuEnergy_->fill (jet->chargedMuEnergy (),  iJet,  weight);
  if (mNeutralEmEnergy_) mNeutralEmEnergy_->fill(jet->neutralEmEnergy(),  iJet,  weight);
  if (mChargedMultiplicity_ ) mChargedMultiplicity_->fill(jet->chargedMultiplicity(),  iJet,  weight);
  if (mNeutralMultiplicity_ ) mNeutralMultiplicity_->fill(jet->neutralMultiplicity(),  iJet,  weight);
  if (mMuonMultiplicity_ )mMuonMultiplicity_->fill (jet-> muonMultiplicity(),  iJet,  weight);

  if (mNeutralFraction_) mNeutralFraction_->fill (jet->neutralMultiplicity()/jet->nConstituents(),  iJet,  weight);
*/


}


void HistoJet::fillCollection( const std::vector<Jet> & coll, double weight ) 
{
 
  h_size_->fill( coll.size(), 1, weight );     //! Save the size of the collection.

  std::vector<Jet>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i, weight);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoJet::clearVec()
{
  HistoGroup<Jet>::clearVec();

  h_jetFlavour_     ->clearVec( );
  h_BDiscriminant_  ->clearVec( );
  h_jetCharge_      ->clearVec( );
  h_nTrk_           ->clearVec( );

  jetME_->clearVec( );

  // Generic Jet Parameters
//  mEta_->clearVec( );
//  mPhi_->clearVec( );
  mE_->clearVec( );
  mP_->clearVec( );
//  mPt_->clearVec( );
  mPt_1_->clearVec( );
  mPt_2_->clearVec( );
  mPt_3_->clearVec( );
//  mMass_->clearVec( );
  mConstituents_->clearVec( );

// Leading Jet Parameters
  mEtaFirst_->clearVec( );
  mPhiFirst_->clearVec( );
  mEFirst_->clearVec( );
  mPtFirst_->clearVec( );

// CaloJet specific
  mMaxEInEmTowers_->clearVec( );
  mMaxEInHadTowers_->clearVec( );
  mHadEnergyInHO_->clearVec( );
  mHadEnergyInHB_->clearVec( );
  mHadEnergyInHF_->clearVec( );
  mHadEnergyInHE_->clearVec( );
  mEmEnergyInEB_->clearVec( );
  mEmEnergyInEE_->clearVec( );
  mEmEnergyInHF_->clearVec( );
  mEnergyFractionHadronic_->clearVec( );
  mEnergyFractionEm_->clearVec( );
  mN90_->clearVec( );
/*
  // PFlowJet specific
  mChargedHadronEnergy_->clearVec( );
  mNeutralHadronEnergy_->clearVec( );
  mChargedEmEnergy_->clearVec( );
  mChargedMuEnergy_->clearVec( );
  mNeutralEmEnergy_->clearVec( );
  mChargedMultiplicity_->clearVec( );
  mNeutralMultiplicity_->clearVec( );
  mMuonMultiplicity_->clearVec( );

  mNeutralFraction_->clearVec( );
*/

}
