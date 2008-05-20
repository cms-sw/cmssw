#include "PhysicsTools/StarterKit/interface/HistoMuon.h"
//#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include <iostream>
#include <sstream>

using pat::HistoMuon;
using namespace std;

// Constructor:


HistoMuon::HistoMuon(std::string dir, std::string group,std::string pre,
		   double pt1, double pt2, double m1, double m2)
  : HistoGroup<Muon>( dir, group, pre, pt1, pt2, m1, m2)
{
  addHisto( h_trackIso_ =
	    new PhysVarHisto( pre + "TrackIso", "Muon Track Isolation", 20, 0, 10, currDir_, "", "vD" )
	   );

  addHisto( h_caloIso_  =
	    new PhysVarHisto( pre + "CaloIso",  "Muon Calo Isolation",  20, 0, 10, currDir_, "", "vD" )
	    );

  addHisto( h_leptonID_ =
            new PhysVarHisto( pre + "LeptonID", "Muon Lepton ID",       20, 0, 1, currDir_, "", "vD" )
            );

  addHisto( h_calCompat_ =
            new PhysVarHisto( pre + "CaloCompat", "Muon Calorimetry Compatability", 100, 0, 1, currDir_, "", "vD" )
            );

  addHisto( h_nChambers_ =
            new PhysVarHisto( pre + "NChamber", "Muon # of Chambers", 51, -0.5, 50.5, currDir_, "", "vD" )
            );

  addHisto( h_caloE_ =
            new PhysVarHisto( pre + "CaloE", "Muon Calorimeter Energy", 50, 0, 50, currDir_, "", "vD" )
            );
  addHisto( h_type_ =
            new PhysVarHisto( pre + "Type", "Muon Type", 65, -0.5, 64.5, currDir_, "", "vD" )
            );
}


// fill a plain ol' muon
void HistoMuon::fill( const Muon *muon, uint iMu, double weight )
{

  // First fill common 4-vector histograms

  HistoGroup<Muon>::fill( muon, iMu, weight);

  // fill relevant muon histograms
  h_trackIso_->fill( muon->trackIso(), iMu , weight);
  h_caloIso_ ->fill( muon->caloIso() , iMu , weight);
  h_leptonID_->fill( muon->leptonID(), iMu , weight);

  const reco::Muon* recoMuon = muon->originalObject();
  h_nChambers_->fill( recoMuon->numberOfChambers(), iMu , weight);

// For CMSSW 1_6_x

  h_calCompat_->fill( recoMuon->caloCompatibility(), iMu, weight );
  h_type_->fill( recoMuon->type(), iMu, weight );
  reco::MuonEnergy muEnergy = recoMuon->calEnergy();


// For CMSSW 2_0_x

//   h_calCompat_->fill( recoMuon->caloCompatibility(), iMu , weight);
//   h_type_->fill( recoMuon->type(), iMu , weight);
//   reco::MuonEnergy muEnergy = recoMuon->calEnergy();

  h_caloE_->fill( muEnergy.em+muEnergy.had+muEnergy.ho, iMu , weight);

}


// fill a muon that is a shallow clone, and take kinematics from 
// shallow clone but detector plots from the muon itself
void HistoMuon::fill( const reco::ShallowCloneCandidate *pshallow, uint iMu, double weight )
{

  // Get the underlying object that the shallow clone represents
  const pat::Muon * muon = dynamic_cast<const pat::Muon*>(pshallow);

  if ( muon == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a muon" << endl;
    return;
  }

  

  // First fill common 4-vector histograms from shallow clone

  HistoGroup<Muon>::fill( pshallow, iMu, weight);

  // fill relevant muon histograms from muon
  h_trackIso_->fill( muon->trackIso(), iMu , weight);
  h_caloIso_ ->fill( muon->caloIso() , iMu , weight);
  h_leptonID_->fill( muon->leptonID(), iMu , weight);

  const reco::Muon* recoMuon = muon->originalObject();
  h_nChambers_->fill( recoMuon->numberOfChambers(), iMu , weight);

// For CMSSW 1_6_x

  h_calCompat_->fill( recoMuon->caloCompatibility(), iMu, weight );
  h_type_->fill( recoMuon->type(), iMu, weight );
  reco::MuonEnergy muEnergy = recoMuon->calEnergy();

// For CMSSW 2_0_x

//   h_calCompat_->fill( recoMuon->caloCompatibility(), iMu , weight);
//   h_type_->fill( recoMuon->type(), iMu , weight);
//   reco::MuonEnergy muEnergy = recoMuon->calEnergy();

  h_caloE_->fill( muEnergy.em+muEnergy.had+muEnergy.ho, iMu , weight);

}

void HistoMuon::fillCollection( const std::vector<Muon> & coll, double weight )
{

  h_size_->fill( coll.size(), 1, weight );     //! Save the size of the collection.

  std::vector<Muon>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i, weight);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  }
}


void HistoMuon::clearVec()
{
  HistoGroup<Muon>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();
  h_leptonID_->clearVec();
}
