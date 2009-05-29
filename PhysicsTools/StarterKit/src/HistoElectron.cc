#include "PhysicsTools/StarterKit/interface/HistoElectron.h"

using namespace std;

// Constructor:

using pat::HistoElectron;

HistoElectron::HistoElectron( std::string dir,std::string group,std::string pre,
			      double pt1, double pt2, double m1, double m2,
			      TFileDirectory * parentDir ) :
  HistoGroup<Electron>( dir, group, pre, pt1, pt2, m1, m2, parentDir)
{
  // book relevant electron histograms

  addHisto( h_trackIso_      =
	    new PhysVarHisto( pre + "TrackIso",       "Electron Track Isolation"    , 100, 0, 10, currDir_, "", "vD")
	    );
  addHisto( h_caloIso_       =
	    new PhysVarHisto( pre + "CaloIso",        "Electron Calo Isolation"     , 100, -20, 20, currDir_, "", "vD")
	    );
  addHisto( h_leptonID_      =
	    new PhysVarHisto( pre + "LeptonID",       "Electron Lepton ID"          , 100, 0, 1, currDir_, "", "vD")
	    );

  std::string type_="SC";
  
  std::string htitle, hlabel;
  hlabel="h"+type_+"eta"; htitle=type_+" #eta";
  addHisto( h_ele_matchingObjectEta_      =
            new PhysVarHisto( pre + hlabel.c_str() , htitle.c_str()                , 100, 0, 1, currDir_, "", "vD")
            );

  hlabel="h"+type_+"Pt"; htitle=type_+" pt";
  addHisto( h_ele_matchingObjectPt_      =
            new PhysVarHisto( pre + hlabel.c_str() , htitle.c_str()  		   , 100, 0, 1, currDir_, "", "vD")
            );

  hlabel="h"+type_+"phi"; htitle=type_+" phi";
  addHisto( h_ele_matchingObjectPhi_      =
            new PhysVarHisto( pre + hlabel.c_str() , htitle.c_str()                , 100, -0, 1, currDir_, "", "vD")
            );

  hlabel="h"+type_+"z"; htitle=type_+" z";
  addHisto( h_ele_matchingObjectZ_      =
            new PhysVarHisto( pre + hlabel.c_str() , htitle.c_str()                , 100, 0, 1, currDir_, "", "vD")
            );


  addHisto( h_ele_matchingObjectEta_matched_      =
            new PhysVarHisto( pre + "matchingObjectEtamatched",       "matching SC #eta", 100, 0, 1, currDir_, "", "vD")
            );

  addHisto( h_ele_matchingObjectPt_matched_      =
            new PhysVarHisto( pre + "matchingObjectPtmatched",       "matching SC p_{T}"          , 100, 0, 1, currDir_, "", "vD")
            );

  addHisto( h_ele_matchingObjectPhi_matched_      =
            new PhysVarHisto( pre + "matchingObjectPhimatched",       "matching SC phi"          , 100, 0, 1, currDir_, "", "vD")
            );

  addHisto( h_ele_matchingObjectZ_matched_      =
            new PhysVarHisto( pre + "matchingObjectZmatched",       "matching SC z"          , 100, 0, 1, currDir_, "", "vD")
            );

//electron basic quantities

  addHisto( h_ele_vertexP_      =
            new PhysVarHisto( pre + "vertexP",       "ele p at vertex"          , 100, 0., 120., currDir_, "", "vD")
            );

  addHisto( h_ele_vertexPt_      =
            new PhysVarHisto( pre + "vertexPt",       "ele p_{T} at vertex"          , 100, 0., 40., currDir_, "", "vD")
            );

  addHisto( h_ele_vertexEta_      =
            new PhysVarHisto( pre + "vertexEta",       "ele #eta at vertex"          , 100, -3.5, 3.5, currDir_, "", "vD")
            );

  addHisto( h_ele_vertexPhi_      =
            new PhysVarHisto( pre + "vertexPhi",       "ele #phi at vertex"          , 100, -3.14, 3.14, currDir_, "", "vD")
            );

  addHisto( h_ele_vertexX_      =
            new PhysVarHisto( pre + "vertexX",       "ele x at vertex"          , 100,- 0.1,0.1, currDir_, "", "vD")
            );
  addHisto( h_ele_vertexY_      =
            new PhysVarHisto( pre + "vertexY",       "ele y at vertex"          , 100, -0.1, 0.1, currDir_, "", "vD")
            );
  addHisto( h_ele_vertexZ_      =
            new PhysVarHisto( pre + "vertexZ",       "ele z at vertex"          , 100, -25., 25., currDir_, "", "vD")
            );
  addHisto( h_ele_charge_      =
            new PhysVarHisto( pre + "charge",       "ele charge"          , 5, -2., 2., currDir_, "", "vD")
            );


//electron matching and ID

  addHisto( h_ele_EoP_      =
            new PhysVarHisto( pre + "EoP",       "ele E/P_{vertex}"          , 100, 0, 10., currDir_, "", "vD")
            );

  addHisto( h_ele_EoPout_      =
            new PhysVarHisto( pre + "EoPout",       "ele E/P_{out}"          , 100, 0, 15., currDir_, "", "vD")
            );
  addHisto( h_ele_dEtaSc_propVtx_      =
            new PhysVarHisto( pre + "dEtaScpropVtx",       "ele #eta_{sc} - #eta_{tr} - prop from vertex"          , 100, -0.2, 0.2, currDir_, "", "vD")
            );
  addHisto( h_ele_dPhiSc_propVtx_      =
            new PhysVarHisto( pre + "dPhiScpropVtx",       "ele #phi_{sc} - #phi_{tr} - prop from vertex"          , 100, -0.2, 0.2, currDir_, "", "vD")
            );
  addHisto( h_ele_dPhiCl_propOut_      =
            new PhysVarHisto( pre + "dPhiClpropOut",       "ele #phi_{cl} - #phi_{tr} - prop from outermost"          , 100, -0.3, 0.3, currDir_, "", "vD")
            );
  addHisto( h_ele_HoE_      =
            new PhysVarHisto( pre + "HoE",       "ele H/E"          , 55, -0.05, 0.5, currDir_, "", "vD")
            );
  addHisto( h_ele_PinMnPout_mode_      =
            new PhysVarHisto( pre + "PinMnPoutmode",       "ele track inner p - outer p, mode"          , 100, -20., 100., currDir_, "", "vD")
            );
  addHisto( h_ele_classes_      =
            new PhysVarHisto( pre + "classes",       "ele electron classes"          , 150, 0., 150., currDir_, "", "vD")
            );
  addHisto( h_ele_eta_golden_      =
            new PhysVarHisto( pre + "etagolden",       "ele electron eta golden"          , 100, 0.0, 3.5, currDir_, "", "vD")
            );
  addHisto( h_ele_eta_shower_      =
            new PhysVarHisto( pre + "etashower",       "ele electron eta showering"          , 100, 0.0, 3.5, currDir_, "", "vD")
            );
  addHisto( h_ele_eta_goldenFrac_      =
            new PhysVarHisto( pre + "etagoldenFrac",       "ele electron eta golden"          , 100, 0., 3.5, currDir_, "", "vD")
            );
  addHisto( h_ele_eta_showerFrac_      =
            new PhysVarHisto( pre + "etashowerFrac",       "ele electron eta showering"          , 100, 0., 3.5, currDir_, "", "vD")
            );

  addHisto( h_ele_eta_      =
            new PhysVarHisto( pre + "eta",       "ele electron eta"          , 100, 0., 3.5, currDir_, "", "vD")
            );

//electron track
  addHisto( h_ele_foundHits_      =
            new PhysVarHisto( pre + "foundHits",       "ele track # found hits"          , 100, 0, 50, currDir_, "", "vD")
            );
  addHisto( h_ele_chi2_      =
            new PhysVarHisto( pre + "chi2",       "ele track #chi^{2}"          , 100, 0., 15., currDir_, "", "vD")
            );

// efficiencies
  addHisto( h_ele_etaEff_      =
            new PhysVarHisto( pre + "etaEff",       "matching SC #eta"          , 100, -3.5, 3.5, currDir_, "", "vD")
            );
  addHisto( h_ele_ptEff_      =
            new PhysVarHisto( pre + "ptEff",       "matching SC p_{T}"          , 100, 5., 20., currDir_, "", "vD")
            );
  addHisto( h_ele_phiEff_      =
            new PhysVarHisto( pre + "phiEff",       "matching SC phi"          , 100, 0., 3.14, currDir_, "", "vD")
            );
  addHisto( h_ele_zEff_      =
            new PhysVarHisto( pre + "zEff",       "matching SC z"          , 100, -25., 25., currDir_, "", "vD")
            );


}

HistoElectron::~HistoElectron()
{
}


void HistoElectron::fill( const Electron * electron, uint iE, double weight )
{

  // First fill common 4-vector histograms
  HistoGroup<Electron>::fill( electron, iE , weight);

  // fill relevant electron histograms
  h_trackIso_       ->fill( electron->trackIso(), iE, weight );
  h_caloIso_        ->fill( electron->caloIso(), iE, weight );
//  h_leptonID_       ->fill( electron->leptonID(), iE, weight );

/*
	  h_ele_matchingObjectEta_       ->fill( moIter->eta(), iE, weight );
	  h_ele_matchingObjectPt_       ->fill( moIter->energy()/cosh(moIter->eta()), iE, weight );
	  h_ele_matchingObjectPhi_       ->fill( moIter->phi(), iE, weight );
	  h_ele_matchingObjectZ_       ->fill( moIter->z(), iE, weight );

      // find best matched electron
//       reco::GsfElectron bestGsfElectron;

	  h_ele_matchingObjectEta_matched_       ->fill( moIter->eta(), iE, weight );
	  h_ele_matchingObjectPt_matched_       ->fill( moIter->energy()/cosh(moIter->eta()), iE, weight );
	  h_ele_matchingObjectPhi_matched_       ->fill( moIter->phi(), iE, weight );
	  h_ele_matchingObjectZ_matched_       ->fill( moIter->z(), iE, weight );
*/
	//electron basic quantities

         // electron related distributions
	  h_ele_vertexP_       ->fill( electron->p(), iE, weight );
	  h_ele_vertexPt_       ->fill( electron->pt(), iE, weight );
	  h_ele_vertexEta_       ->fill( electron->eta(), iE, weight );


	  h_ele_vertexPhi_       ->fill( electron->phi(), iE, weight );
	  h_ele_vertexX_       ->fill( electron->vertex().x(), iE, weight );
	  h_ele_vertexY_       ->fill( electron->vertex().y(), iE, weight );
	  h_ele_vertexZ_       ->fill( electron->vertex().z(), iE, weight );
	  h_ele_charge_       ->fill(  electron->charge(), iE, weight );

	//electron matching and ID
         // match distributions 
	  h_ele_EoP_       ->fill(  electron->eSuperClusterOverP(), iE, weight );
	  h_ele_EoPout_       ->fill( electron->eSeedClusterOverPout(), iE, weight );
	  h_ele_dEtaSc_propVtx_       ->fill( electron->deltaEtaSuperClusterTrackAtVtx(), iE, weight );
	  h_ele_dPhiSc_propVtx_       ->fill( electron->deltaPhiSuperClusterTrackAtVtx(), iE, weight );
	  h_ele_dPhiCl_propOut_       ->fill( electron->deltaPhiSeedClusterTrackAtCalo(), iE, weight );
	  h_ele_HoE_       ->fill( electron->hadronicOverEm(), iE, weight );

        // from electron interface, hence using mode
	  h_ele_PinMnPout_mode_       ->fill( electron->trackMomentumAtVtx().R() - electron->trackMomentumOut().R(), iE, weight );

	//classes
         int eleClass = electron->classification();
	  h_ele_classes_       ->fill( eleClass, iE, weight );

         eleClass = eleClass%100; // get rid of barrel/endcap distinction

	if (eleClass == 0) {
	  h_ele_eta_golden_       ->fill( fabs(electron->eta()), iE, weight );
	}
         if (eleClass == 30 || eleClass == 31 || eleClass == 32  || eleClass == 33 || eleClass == 34 ) {
	  h_ele_eta_shower_       ->fill( fabs(electron->eta()), iE, weight );
	}

	  h_ele_eta_       ->fill( fabs(electron->eta()), iE, weight );

	// electron track
         // track related distributions

	  h_ele_foundHits_       ->fill( electron->gsfTrack()->numberOfValidHits(), iE, weight );
	  h_ele_chi2_       ->fill( electron->gsfTrack()->normalizedChi2(), iE, weight );

//    }//loop over matching object

}


void HistoElectron::fill( const reco::ShallowClonePtrCandidate * pshallow, uint iE, double weight )
{

  // Get the underlying object that the shallow clone represents
  const pat::Electron * electron = dynamic_cast<const pat::Electron*>(pshallow);

  if ( electron == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a electron" << endl;
    return;
  }


  // First fill common 4-vector histograms
  HistoGroup<Electron>::fill( pshallow, iE, weight );

  // fill relevant electron histograms
  h_trackIso_       ->fill( electron->trackIso(), iE, weight );
  h_caloIso_        ->fill( electron->caloIso(), iE, weight );
//  h_leptonID_       ->fill( electron->leptonID(), iE, weight );

/*
	  h_ele_matchingObjectEta_       ->fill( moIter->eta(), iE, weight );
	  h_ele_matchingObjectPt_       ->fill( moIter->energy()/cosh(moIter->eta()), iE, weight );
	  h_ele_matchingObjectPhi_       ->fill( moIter->phi(), iE, weight );
	  h_ele_matchingObjectZ_       ->fill( moIter->z(), iE, weight );

      // find best matched electron
//       reco::GsfElectron bestGsfElectron;

	  h_ele_matchingObjectEta_matched_       ->fill( moIter->eta(), iE, weight );
	  h_ele_matchingObjectPt_matched_       ->fill( moIter->energy()/cosh(moIter->eta()), iE, weight );
	  h_ele_matchingObjectPhi_matched_       ->fill( moIter->phi(), iE, weight );
	  h_ele_matchingObjectZ_matched_       ->fill( moIter->z(), iE, weight );
*/
	//electron basic quantities

         // electron related distributions
	  h_ele_vertexP_       ->fill( electron->p(), iE, weight );
	  h_ele_vertexPt_       ->fill( electron->pt(), iE, weight );
	  h_ele_vertexEta_       ->fill( electron->eta(), iE, weight );


	  h_ele_vertexPhi_       ->fill( electron->phi(), iE, weight );
	  h_ele_vertexX_       ->fill( electron->vertex().x(), iE, weight );
	  h_ele_vertexY_       ->fill( electron->vertex().y(), iE, weight );
	  h_ele_vertexZ_       ->fill( electron->vertex().z(), iE, weight );
	  h_ele_charge_       ->fill(  electron->charge(), iE, weight );

	//electron matching and ID
         // match distributions 
	  h_ele_EoP_       ->fill(  electron->eSuperClusterOverP(), iE, weight );
	  h_ele_EoPout_       ->fill( electron->eSeedClusterOverPout(), iE, weight );
	  h_ele_dEtaSc_propVtx_       ->fill( electron->deltaEtaSuperClusterTrackAtVtx(), iE, weight );
	  h_ele_dPhiSc_propVtx_       ->fill( electron->deltaPhiSuperClusterTrackAtVtx(), iE, weight );
	  h_ele_dPhiCl_propOut_       ->fill( electron->deltaPhiSeedClusterTrackAtCalo(), iE, weight );
	  h_ele_HoE_       ->fill( electron->hadronicOverEm(), iE, weight );

        // from electron interface, hence using mode
	  h_ele_PinMnPout_mode_       ->fill( electron->trackMomentumAtVtx().R() - electron->trackMomentumOut().R(), iE, weight );

	//classes
         int eleClass = electron->classification();
	  h_ele_classes_       ->fill( eleClass, iE, weight );

         eleClass = eleClass%100; // get rid of barrel/endcap distinction

	if (eleClass == 0) {
	  h_ele_eta_golden_       ->fill( fabs(electron->eta()), iE, weight );
	}
         if (eleClass == 30 || eleClass == 31 || eleClass == 32  || eleClass == 33 || eleClass == 34 ) {
	  h_ele_eta_shower_       ->fill( fabs(electron->eta()), iE, weight );
	}

	  h_ele_eta_       ->fill( fabs(electron->eta()), iE, weight );

	// electron track
         // track related distributions

	  h_ele_foundHits_       ->fill( electron->gsfTrack()->numberOfValidHits(), iE, weight );
	  h_ele_chi2_       ->fill( electron->gsfTrack()->normalizedChi2(), iE, weight );

//     } //loop over matching object   

}


void HistoElectron::fillCollection( const std::vector<Electron> & coll,double weight ) 
{
 
  h_size_->fill( coll.size(), 1, weight );     //! Save the size of the collection.

  std::vector<Electron>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i, weight);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoElectron::clearVec()
{
  HistoGroup<Electron>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();
//  h_leptonID_->clearVec();


  h_ele_matchingObjectEta_->clearVec();
  h_ele_matchingObjectPt_->clearVec();
  h_ele_matchingObjectPhi_->clearVec();
  h_ele_matchingObjectZ_->clearVec();

  h_ele_matchingObjectEta_matched_->clearVec();
  h_ele_matchingObjectPt_matched_->clearVec();
  h_ele_matchingObjectPhi_matched_->clearVec();
  h_ele_matchingObjectZ_matched_->clearVec();

// electron basic quantities

  h_ele_vertexP_->clearVec();
  h_ele_vertexPt_->clearVec();
  h_ele_vertexEta_->clearVec();
  h_ele_vertexPhi_->clearVec();
  h_ele_vertexX_->clearVec();
  h_ele_vertexY_->clearVec();
  h_ele_vertexZ_->clearVec();
  h_ele_charge_->clearVec();

// electron matching and ID

  h_ele_EoP_->clearVec();
  h_ele_EoPout_->clearVec();
  h_ele_dEtaSc_propVtx_->clearVec();
  h_ele_dPhiSc_propVtx_->clearVec();
  h_ele_dPhiCl_propOut_->clearVec();
  h_ele_HoE_->clearVec();
  h_ele_PinMnPout_mode_->clearVec();
  h_ele_classes_->clearVec();
  h_ele_eta_golden_->clearVec();
  h_ele_eta_shower_->clearVec();
  h_ele_eta_goldenFrac_->clearVec();
  h_ele_eta_showerFrac_->clearVec();

  h_ele_eta_->clearVec();

// electron track
  h_ele_foundHits_->clearVec();
  h_ele_chi2_->clearVec();

// efficiencies
  h_ele_etaEff_->clearVec();
  h_ele_ptEff_->clearVec();
  h_ele_phiEff_->clearVec();
  h_ele_zEff_->clearVec();

}
