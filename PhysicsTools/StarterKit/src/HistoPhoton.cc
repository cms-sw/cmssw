#include "PhysicsTools/StarterKit/interface/HistoPhoton.h"


//#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"


using namespace std;

// Constructor:

using pat::HistoPhoton;

HistoPhoton::HistoPhoton( std::string dir, std::string role,std::string pre,
			      double pt1, double pt2, double m1, double m2,
			  TFileDirectory * parentDir ) :
  HistoGroup<Photon>( dir, role, pre, pt1, pt2, m1, m2, parentDir)
{
  // book relevant photon histograms

  addHisto( h_trackIso_      =
	    new PhysVarHisto(pre + "TrackIso",       "Photon Track Isolation"    , 100, 0, 100., currDir_, "", "vD")
	    );
  addHisto( h_caloIso_       =
	    new PhysVarHisto(pre + "CaloIso",        "Photon Calo Isolation"     , 100, 0, 1, currDir_, "", "vD")
	    );

/*
// bookProfile
  std::string histname = "nIsoTracks";
  addHisto( p_nTrackIsol_       =
            new PhysVarHisto(pre + histname,        "Avg Number Of Tracks in the Iso Cone"     , 100, 0, 1, currDir_, "", "vD")
            );

  histname = "isoPtSum";
  addHisto( p_trackPtSum_	=
            new PhysVarHisto(pre + histname,        "Avg Tracks Pt Sum in the Iso Cone"     , 100, 0, 1, currDir_, "", "vD")
            );

  histname = "ecalSum";
  addHisto( p_ecalSum_		=
            new PhysVarHisto(pre + histname,        "Avg Ecal Sum in the Iso Cone"     , 100, 0, 1, currDir_, "", "vD")
            );

  histname = "hcalSum";
  addHisto( p_hcalSum_ 		=
            new PhysVarHisto(pre + histname,        "Avg Hcal Sum in the Iso Cone"     , 100, 0, 1, currDir_, "", "vD")
            );
*/



  std::string histname = "nPho";
  addHisto( h_nPho_[0][0]	=
            new PhysVarHisto(pre + histname+"All",        "Number Of Isolated Photon candidates per events: All Ecal  "     , 10, -0.5, 9.5, currDir_, "", "vD")
            );

  addHisto( h_nPho_[0][1]       =
            new PhysVarHisto(pre + histname+"Barrel",        "Number Of Isolated Photon candidates per events: Ecal Barrel  "     , 10, -0.5, 9.5, currDir_, "", "vD")
            );
  addHisto( h_nPho_[0][2]       =
            new PhysVarHisto(pre + histname+"Endcap",        "Number Of Isolated Photon candidates per events: Ecal Endcap "     , 10, -0.5, 9.5, currDir_, "", "vD")
            );

  histname = "scE";
  addHisto(  h_scE_[0][0]           =
            new PhysVarHisto(pre + histname+"All",        "Isolated SC Energy: All Ecal  "     , 100, 0., 100., currDir_, "", "vD")
            );
  addHisto(  h_scE_[0][1]           =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated SC Energy: Barrel "     , 100, 0., 100., currDir_, "", "vD")
            );
  addHisto( h_scE_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated SC Energy: Endcap "     , 100, 0., 100., currDir_, "", "vD")
            );

  histname = "scEt";
  addHisto( h_scEt_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated SC Et: All Ecal "     , 100, 0., 100., currDir_, "", "vD")
            );
  addHisto( h_scEt_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated SC Et: Barrel"     , 100, 0., 100., currDir_, "", "vD")
            );
  addHisto( h_scEt_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated SC Et: Endcap"     , 100, 0., 100., currDir_, "", "vD")
            );

  histname = "r9";
  addHisto( h_r9_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated r9: All Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_r9_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated r9: Barrel "     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_r9_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated r9: Endcap "     , 100, 0, 1, currDir_, "", "vD")
            );

  addHisto( h_scEta_[0]            =
            new PhysVarHisto(pre + "scEta",        "Isolated SC Eta "     , 100, -3.5, 3.5, currDir_, "", "vD")
            );
  addHisto( h_scPhi_[0]            =
            new PhysVarHisto(pre + "scPhi",        "Isolated SC Phi "     , 100, -3.14, 3.14, currDir_, "", "vD")
            );
////////////////// two dimensional histogram
/*  addHisto( h_scEtaPhi_[0]            =
            new PhysVarHisto(pre + "scEtaPhi",        "Isolated SC Phi vs Eta "     , 100, 0, 1, currDir_, "", "vD")
            );
*/


 histname = "phoE";
  addHisto( h_phoE_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated Photon Energy: All ecal "     , 100, 0., 100., currDir_, "", "vD")
            );
  addHisto( h_phoE_[0][1]            =
            new PhysVarHisto(pre + histname+"Barell",        "Isolated Photon Energy: barrel "     , 100, 0., 100., currDir_, "", "vD")
            );
  addHisto( h_phoE_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated Photon Energy: Endcap "     , 100, 0, 100., currDir_, "", "vD")
            );

  histname = "phoEt";
  addHisto( h_phoEt_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated Photon Transverse Energy: All ecal "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_phoEt_[0][1]            =
            new PhysVarHisto(pre + histname+"Barell",        "Isolated Photon Transverse Energy: Barrel "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_phoEt_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated Photon Transverse Energy: Endcap "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_phoEta_[0]            =
            new PhysVarHisto(pre + "phoEta",        "Isolated Photon Eta "     , 100, -3.5, 3.5, currDir_, "", "vD")
            );
  addHisto( h_phoPhi_[0]            =
            new PhysVarHisto(pre + "phoPhi",        "Isolated Photon Phi"     , 100, -3.14, 3.14, currDir_, "", "vD")
            );

  histname="nConv";
  addHisto( h_nConv_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Number Of Conversions per isolated candidates per events: All Ecal  "     , 10, -0.5, 9.5, currDir_, "", "vD")
            );
  addHisto( h_nConv_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Number Of Conversions per isolated candidates per events: Ecal Barrel"     , 10, -0.5, 9.5, currDir_, "", "vD")
            );
  addHisto( h_nConv_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Number Of Conversions per isolated candidates per events: Ecal Endcap"     , 100, -0.5, 9.5, currDir_, "", "vD")
            );
  addHisto( h_convEta_[0]            =
            new PhysVarHisto(pre + "convEta",        "Isolated converted Photon Eta "     , 100, -3.5, 3.5, currDir_, "", "vD")
            );
  addHisto( h_convPhi_[0]            =
            new PhysVarHisto(pre + "convPhi",        "Isolated converted Photon  Phi"     , 100, -3.14, 3.14, currDir_, "", "vD")
            );

///////////////////// two dimensional histogram
/*  histname="r9VsTracks";
  addHisto( h_r9VsNofTracks_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated photons r9 vs nTracks from conversions: All Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_r9VsNofTracks_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated photons r9 vs nTracks from conversions: Barrel Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_r9VsNofTracks_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated photons r9 vs nTracks from conversions: Endcap Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
*/


  histname="EoverPtracks";
  addHisto( h_EoverPTracks_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated photons conversion E/p: all Ecal "     , 100, 0, 5., currDir_, "", "vD")
            );
  addHisto( h_EoverPTracks_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated photons conversion E/p: Barrel Ecal "     , 100, 0, 5., currDir_, "", "vD")
            );
  addHisto( h_EoverPTracks_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated photons conversion E/p: Endcap Ecal "     , 100, 0, 5., currDir_, "", "vD")
            );

/* bookProfile
  histname="pTknHitsVsEta";
  addHisto( p_tk_nHitsVsEta_[0]            =
            new PhysVarHisto(pre + histname,        "Isolated Photons:Tracks from conversions: mean numb of  Hits vs Eta"     , 100, 0, 1, currDir_, "", "vD")
            );
*/

  addHisto( h_tkChi2_[0]            =
            new PhysVarHisto(pre + "tkChi2",        "Isolated Photons:Tracks from conversions: #chi^{2} of tracks"     , 100, 0, 20., currDir_, "", "vD")
            );

  histname="hDPhiTracksAtVtx";
  addHisto( h_DPhiTracksAtVtx_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_DPhiTracksAtVtx_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_DPhiTracksAtVtx_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );

  histname="hDCotTracks";
  addHisto( h_DCotTracks_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Ecal "     , 100, 0, 1, currDir_, "", "vD")
            );

  addHisto( h_DCotTracks_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal "     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_DCotTracks_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Ecal "     , 100, 0, 1, currDir_, "", "vD")
            );

  histname="hInvMass";
  addHisto( h_invMass_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated Photons:Tracks from conversion: Pair invariant mass: all Ecal "     , 100, 0, 1.5, currDir_, "", "vD")
            );
  addHisto( h_invMass_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal "     , 100, 0, 1.5, currDir_, "", "vD")
            );
  addHisto( h_invMass_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal "     , 100, 0, 1.5, currDir_, "", "vD")
            );

  histname="hDPhiTracksAtEcal";
  addHisto( h_DPhiTracksAtEcal_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated Photons:Tracks from conversions:  #delta#phi at Ecal : all Ecal "     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_DPhiTracksAtEcal_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated Photons:Tracks from conversions:  #delta#phi at Ecal : Barrel Ecal "     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_DPhiTracksAtEcal_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated Photons:Tracks from conversions:  #delta#phi at Ecal : Endcap Ecal "     , 100, 0, 1, currDir_, "", "vD")
            );

  histname="hDEtaTracksAtEcal";
  addHisto( h_DEtaTracksAtEcal_[0][0]            =
            new PhysVarHisto(pre + histname+"All",        "Isolated Photons:Tracks from conversions:  #delta#eta at Ecal : all Ecal "     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_DEtaTracksAtEcal_[0][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Isolated Photons:Tracks from conversions:  #delta#eta at Ecal : Barrel Ecal "     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_DEtaTracksAtEcal_[0][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Isolated Photons:Tracks from conversions:  #delta#eta at Ecal : Endcap Ecal "     , 100, 0, 1, currDir_, "", "vD")
            );

////////////////////////two dimensional histogram
/*  addHisto( h_convVtxRvsZ_[0]            =
            new PhysVarHisto(pre + "convVtxRvsZ",        "Isolated Photon Reco conversion vtx position"     , 100, 0, 1, currDir_, "", "vD")
            );
*/


  addHisto( h_zPVFromTracks_[0]            =
            new PhysVarHisto(pre + "zPVFromTracks",        "Isolated Photons: PV z from conversion tracks"     , 100, -25.,  25., currDir_, "", "vD")
            );

  histname = "nPhoNoIs";
  addHisto( h_nPho_[1][0]            =
            new PhysVarHisto(pre + histname+"All",        "Number Of Non Isolated Photon candidates per events: All Ecal  "     , 10, -0.5, 9.5, currDir_, "", "vD")
            );
  addHisto( h_nPho_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Number Of Non Isolated Photon candidates per events: Ecal Barrel "     , 100, -0.5, 9.5, currDir_, "", "vD")
            );
  addHisto( h_nPho_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Number Of Non Isolated Photon candidates per events: Ecal Endcap "     , 100, -0.5, 9.5, currDir_, "", "vD")
            );

  histname = "scENoIs";
  addHisto( h_scE_[1][0]            =
            new PhysVarHisto(pre + histname+"All",        "Non Isolated SC Energy: All Ecal  "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_scE_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Non Isolated SC Energy: Barrel  "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_scE_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Non Isolated SC Energy: Endcap "     , 100, 0, 100., currDir_, "", "vD")
            );

  histname = "scEtNoIs";
  addHisto( h_scEt_[1][0]            =
            new PhysVarHisto(pre + histname+"All",        "Non Isolated SC Et: All Ecal "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_scEt_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Non Isolated SC Et: Barel "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_scEt_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Non Isolated SC Et: Endcap "     , 100, 0, 100., currDir_, "", "vD")
            );

  histname = "r9NoIs";
  addHisto( h_r9_[1][0]            =
            new PhysVarHisto(pre + histname+"All",        "Non Isolated r9: All Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_r9_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Non Isolated r9: Barrel"     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_r9_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Non Isolated r9: Endcap"     , 100, 0, 1, currDir_, "", "vD")
            );


  addHisto( h_scEta_[1]            =
            new PhysVarHisto(pre + "scEtaNoIs",        "Non Isolated SC Eta "     , 100, -3.5, 3.5, currDir_, "", "vD")
            );
  addHisto( h_scPhi_[1]            =
            new PhysVarHisto(pre + "scPhiNoIs",        "Non Isolated SC Phi "     , 100, -3.14, 3.14, currDir_, "", "vD")
            );
/////////////// two dimensional
/*  addHisto( h_scEtaPhi_[1]            =
            new PhysVarHisto(pre + "scEtaPhiNoIs",        "Non Isolated SC Phi vs Eta "     , 100, 0, 1, currDir_, "", "vD")
            );
*/

//
  histname = "phoENoIs";
  addHisto( h_phoE_[1][0]            =
            new PhysVarHisto(pre + histname+"All",        "Non Isolated Photon Energy: All ecal "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_phoE_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel",        "Non Isolated Photon Energy: Barrel  "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_phoE_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap",        "Non Isolated Photon Energy: Endcap "     , 100, 0, 100., currDir_, "", "vD")
            );

  histname = "phoEtNoIs";
  addHisto( h_phoEt_[1][0]            =
            new PhysVarHisto(pre + histname+"All","Non Isolated Photon Transverse Energy: All ecal "      , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_phoEt_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel","Non Isolated Photon Transverse Energy: Barrel "     , 100, 0, 100., currDir_, "", "vD")
            );
  addHisto( h_phoEt_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap","Non Isolated Photon Transverse Energy: Endcap "     , 100, 0, 100., currDir_, "", "vD")
            );

  addHisto( h_phoEta_[1]            =
            new PhysVarHisto(pre + "EtaNoIs","Non Isolated Photon Eta "     , 100, -3.5, 3.5, currDir_, "", "vD")
            );
  addHisto( h_phoPhi_[1]            =
            new PhysVarHisto(pre + "PhiNoIs","Non Isolated Photon  Phi "     , 100, -3.14, 3.14, currDir_, "", "vD")
            );

  histname="nConvNoIs";
  addHisto( h_nConv_[1][0]            =
            new PhysVarHisto(pre + histname+"All","Number Of Conversions per non isolated candidates per events: All Ecal  "     , 10, -0.5, 9.5, currDir_, "", "vD")
            );
  addHisto( h_nConv_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel","Number Of Conversions per non isolated candidates per events: Ecal Barrel  "     , 100, -0.5, 9.5, currDir_, "", "vD")
            );
  addHisto( h_nConv_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap","Number Of Conversions per non isolated candidates per events: Ecal Endcap "     , 100, -0.5, 9.5, currDir_, "", "vD")
            );


  addHisto( h_convEta_[1]            =
            new PhysVarHisto(pre + "convEtaNoIs","Non Isolated converted Photon Eta "     , 100, -3.5, 3.5, currDir_, "", "vD")
            );
  addHisto( h_convPhi_[1]            =
            new PhysVarHisto(pre + "convPhiNoIs","Non Isolated converted Photon  Phi "     , 100, -3.14, 3.14, currDir_, "", "vD")
            );

/////////////////////////////////////two dimensional
/*  histname="r9VsTracksNoIs";
  addHisto( h_r9VsNofTracks_[1][0]            =
            new PhysVarHisto(pre + histname+"All","Non Isolated photons r9 vs nTracks from conversions: All Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_r9VsNofTracks_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel","Non Isolated photons r9 vs nTracks from conversions: Barrel Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
  addHisto( h_r9VsNofTracks_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap","Non Isolated photons r9 vs nTracks from conversions: Endcap Ecal"     , 100, 0, 1, currDir_, "", "vD")
            );
*/


  histname="EoverPtracksNoIs";
  addHisto( h_EoverPTracks_[1][0]            =
            new PhysVarHisto(pre + histname+"All","Non Isolated photons conversion E/p: all Ecal "     , 100, 0, 10., currDir_, "", "vD")
            );
  addHisto( h_EoverPTracks_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel","Non Isolated photons conversion E/p: Barrel Ecal"     , 100, 0, 10., currDir_, "", "vD")
            );
  addHisto( h_EoverPTracks_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap","Non Isolated photons conversion E/p: Endcap Ecal "     , 100, 0, 10., currDir_, "", "vD")
            );

/* bookProfile
  addHisto( p_tk_nHitsVsEta_[1]            =
            new PhysVarHisto(pre + "pTknHitsVsEtaNoIs","Non Isolated Photons:Tracks from conversions: mean numb of  Hits vs Eta"     , 100, 0, 1, currDir_, "", "vD")
            );
*/

  addHisto( h_tkChi2_[1]            =
            new PhysVarHisto(pre + "tkChi2NoIs","NonIsolated Photons:Tracks from conversions: #chi^{2} of tracks"     , 100, 0, 20., currDir_, "", "vD")
            );

  histname="hDPhiTracksAtVtxNoIs";
  addHisto( h_DPhiTracksAtVtx_[1][0]            =
            new PhysVarHisto(pre + histname+"All", "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: all Ecal"     , 100, -2., 2., currDir_, "", "vD")
            );
  addHisto( h_DPhiTracksAtVtx_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel", "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: Barrel Ecal"     , 100, -2., 2., currDir_, "", "vD")
            );
  addHisto( h_DPhiTracksAtVtx_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap", "Isolated Photons:Tracks from conversions: #delta#phi Tracks at vertex: Endcap Ecal"     , 100, -2-2., 2., currDir_, "", "vD")
            );

  histname="hDCotTracksNoIs";
  addHisto( h_DCotTracks_[1][0]            =
            new PhysVarHisto(pre + histname+"All","Non Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: all Eca "     , 100, -1.0, 1.0, currDir_, "", "vD")
            );
  addHisto( h_DCotTracks_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel","Non Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Barrel Ecal "     , 100, -1.0, 1.0, currDir_, "", "vD")
            );
  addHisto( h_DCotTracks_[1][2]            =
            new PhysVarHisto(pre + histname+"Encap","Non Isolated Photons:Tracks from conversions #delta cotg(#Theta) Tracks: Endcap Eca "     , 100, -1.0, 1.0, currDir_, "", "vD")
            );

  histname="hInvMassNoIs";
  addHisto( h_invMass_[1][0]            =
            new PhysVarHisto(pre + histname+"All","Non Isolated Photons:Tracks from conversion: Pair invariant mass: all Ecal "     , 100, 0, 1.5, currDir_, "", "vD")
            );
  addHisto( h_invMass_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel","Non Isolated Photons:Tracks from conversion: Pair invariant mass: Barrel Ecal "     , 100, 0, 1.5, currDir_, "", "vD")
            );
  addHisto( h_invMass_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap","Non Isolated Photons:Tracks from conversion: Pair invariant mass: Endcap Ecal "     , 100, 0, 1.5, currDir_, "", "vD")
            );

  histname="hDPhiTracksAtEcalNoIs";
  addHisto( h_DPhiTracksAtEcal_[1][0]            =
            new PhysVarHisto(pre + histname+"All","Non Isolated Photons:Tracks from conversions: #delta#phi at Ecal : all Ecal "     , 100, -0.2,  0.2, currDir_, "", "vD")
            );
  addHisto( h_DPhiTracksAtEcal_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel","Non Isolated Photons:Tracks from conversions: #delta#phi at Ecal : Barrel Ecal "     , 100, -0.2,  0.2, currDir_, "", "vD")
            );
  addHisto( h_DPhiTracksAtEcal_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap","Non Isolated Photons:Tracks from conversions: #delta#phi at Ecal : Endcap Ecal "     , 100, -0.2,  0.2, currDir_, "", "vD")
            );

  histname="hDEtaTracksAtEcalNoIs";
  addHisto( h_DEtaTracksAtEcal_[1][0]            =
            new PhysVarHisto(pre + histname+"All","Non Isolated Photons:Tracks from conversions: #delta#eta at Ecal : all Ecal "     , 100, -0.2,  0.2, currDir_, "", "vD")
            );
  addHisto( h_DEtaTracksAtEcal_[1][1]            =
            new PhysVarHisto(pre + histname+"Barrel","Non Isolated Photons:Tracks from conversions: #delta#eta at Ecal : Barrel Ecal "     , 100, -0.2,  0.2, currDir_, "", "vD")
            );
  addHisto( h_DEtaTracksAtEcal_[1][2]            =
            new PhysVarHisto(pre + histname+"Endcap","Non Isolated Photons:Tracks from conversions: #delta#eta at Ecal : Endcap Ecal "     , 100, -0.2,  0.2, currDir_, "", "vD")
            );

///////////// two dimensional
/*  addHisto( h_convVtxRvsZ_[1]            =
            new PhysVarHisto(pre + "convVtxRvsZNoIs","Non Isolated Photon Reco conversion vtx position"     , 100, 0, 1, currDir_, "", "vD")
            );
*/

  addHisto( h_zPVFromTracks_[1]            =
            new PhysVarHisto(pre + "zPVFromTracksNoIs","Non Isolated Photons: PV z from conversion tracks"     , 100, -25.,  25., currDir_, "", "vD")
            );


}

HistoPhoton::~HistoPhoton()
{
}


void HistoPhoton::fill( const Photon * photon, uint iE, double weight )
{

  // First fill common 4-vector histograms
  HistoGroup<Photon>::fill( photon, iE, weight );

  // fill relevant photon histograms
  h_trackIso_       ->fill( photon->trackIso(), iE, weight );
  h_caloIso_        ->fill( photon->caloIso(), iE, weight );


/////////////
  using namespace edm;
//  const float etaPhiDistance=0.01;
  // Fiducial region
//  const float TRK_BARL =0.9;
//  const float BARL = 1.4442; // DAQ TDR p.290
//  const float END_LO = 1.566;
//  const float END_HI = 2.5;
  // Electron mass
  const Float_t mElec= 0.000511;


  std::vector<int> nPho(2);
  std::vector<int> nPhoBarrel(2);
  std::vector<int> nPhoEndcap(2);
  for ( unsigned int i=0; i<nPho.size(); i++ ) nPho[i]=0;
  for ( unsigned int i=0; i<nPhoBarrel.size(); i++ ) nPhoBarrel[i]=0;
  for ( unsigned int i=0; i<nPhoEndcap.size(); i++ ) nPhoEndcap[i]=0;





    reco::Photon aPho = reco::Photon(*photon);

    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;

///////////////////a little bit corrections
    float etaPho=aPho.eta();
    if ( fabs(etaPho) <  1.479 ) {
      phoIsInBarrel=true;
    } else {
      phoIsInEndcap=true;
    }


//    float phiClu=aPho.superCluster()->phi();
    float etaClu=aPho.superCluster()->eta();



    bool  scIsInBarrel=false;
    bool  scIsInEndcap=false;
    if ( fabs(etaClu) <  1.479 ) 
      scIsInBarrel=true;
    else
      scIsInEndcap=true;




/*    bookProfile
    p_nTrackIsol_->fill( (*photon).superCluster()->eta(),  float(nTracks));
    p_trackPtSum_->fill((*photon).superCluster()->eta(),  ptSum);
    p_ecalSum_->fill((*photon).superCluster()->eta(),  ecalSum);
    p_hcalSum_->fill((*photon).superCluster()->eta(),  hcalSum);
*/


    bool isIsolated=false;
//    if ( (nTracks < numOfTracksInCone_) && 
//           ( ptSum < trkPtSumCut_) &&
//           ( ecalSum < ecalEtSumCut_ ) &&  ( hcalSum < hcalEtSumCut_ ) ) isIsolated = true;



    int type=0;
    if ( !isIsolated ) type=1;

    nPho[type]++; 
    if (phoIsInBarrel) nPhoBarrel[type]++;
    if (phoIsInEndcap) nPhoEndcap[type]++;



//// e3x3 needs to be calculated
    float e3x3= 0; // EcalClusterTools::e3x3(  *(   (*photon).superCluster()->seed()  ), &ecalRecHitCollection, &(*topology));
    float r9 =e3x3/( (*photon).superCluster()->rawEnergy()+ (*photon).superCluster()->preshowerEnergy());


    h_scEta_[type]->fill( (*photon).superCluster()->eta() ,iE, weight );
    h_scPhi_[type]->fill( (*photon).superCluster()->phi() ,iE, weight);


///////////////////////////////////////////////////////////////two dimensional histogram
//    h_scEtaPhi_[type]->fill( (*photon).superCluster()->eta(), (*photon).superCluster()->phi() );

    h_scE_[type][0]->fill( (*photon).superCluster()->energy() ,iE, weight);
    h_scEt_[type][0]->fill( (*photon).superCluster()->energy()/cosh( (*photon).superCluster()->eta()) ,iE, weight);
    h_r9_[type][0]->fill( r9 ,iE, weight);

    h_phoEta_[type]->fill( (*photon).eta() ,iE, weight);
    h_phoPhi_[type]->fill( (*photon).phi() ,iE, weight);

    h_phoE_[type][0]->fill( (*photon).energy() ,iE, weight);
    h_phoEt_[type][0]->fill( (*photon).energy()/ cosh( (*photon).eta()) ,iE, weight);


    h_nConv_[type][0]->fill(float( (*photon).conversions().size()),iE, weight);

    if ( scIsInBarrel ) {
      h_scE_[type][1]->fill( (*photon).superCluster()->energy() ,iE, weight);
      h_scEt_[type][1]->fill( (*photon).superCluster()->energy()/cosh( (*photon).superCluster()->eta()) ,iE, weight);
      h_r9_[type][1]->fill( r9 ,iE, weight);
     }

    if ( scIsInEndcap ) {
      h_scE_[type][2]->fill( (*photon).superCluster()->energy() ,iE, weight);
      h_scEt_[type][2]->fill( (*photon).superCluster()->energy()/cosh( (*photon).superCluster()->eta()) ,iE, weight);
      h_r9_[type][2]->fill( r9 ,iE, weight);
     }

    if ( phoIsInBarrel ) {
      h_phoE_[type][1]->fill( (*photon).energy() ,iE, weight);
      h_phoEt_[type][1]->fill( (*photon).energy()/ cosh( (*photon).eta()) ,iE, weight);
      h_nConv_[type][1]->fill(float( (*photon).conversions().size()),iE, weight);
     }
    
    if ( phoIsInEndcap ) {
      h_phoE_[type][2]->fill( (*photon).energy() ,iE, weight);
      h_phoEt_[type][2]->fill( (*photon).energy()/ cosh( (*photon).eta()) ,iE, weight);
      h_nConv_[type][2]->fill(float( (*photon).conversions().size()),iE, weight);
     }

   ////////////////// plot quantitied related to conversions
    reco::ConversionRefVector conversions = (*photon).conversions();
 
    for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {

// two dimensional histogram
//      h_r9VsNofTracks_[type][0]->fill( r9, conversions[iConv]->nTracks() ) ; 
 
//      if ( phoIsInBarrel ) h_r9VsNofTracks_[type][1]->fill( r9,  conversions[iConv]->nTracks() ) ; 
 
//      if ( phoIsInEndcap ) h_r9VsNofTracks_[type][2]->fill( r9,  conversions[iConv]->nTracks() ) ; 

      if ( conversions[iConv]->nTracks() <2 ) continue; 
 
      h_convEta_[type]->fill( conversions[iConv]-> caloCluster()[0]->eta() );
      h_convPhi_[type]->fill( conversions[iConv]-> caloCluster()[0]->phi() );
      h_EoverPTracks_[type][0] ->fill( conversions[iConv]->EoverP() ,iE, weight) ;
      if ( phoIsInBarrel ) h_EoverPTracks_[type][1] ->fill( conversions[iConv]->EoverP() ,iE, weight) ;
      if ( phoIsInEndcap ) h_EoverPTracks_[type][2] ->fill( conversions[iConv]->EoverP() ,iE, weight) ;
 
 
/* /// two dimensional histogram      
      if ( conversions[iConv]->conversionVertex().isValid() ) 
        h_convVtxRvsZ_[type] ->fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
*/         
 
       h_zPVFromTracks_[type]->fill ( conversions[iConv]->zOfPrimaryVertexFromTracks() ,iE, weight);

       std::vector<reco::TrackRef> tracks = conversions[iConv]->tracks();

       float px=0;
       float py=0;
       float pz=0;
       float e=0;
       for (unsigned int i=0; i<tracks.size(); i++) {
//	   two dimensional histo
//         p_tk_nHitsVsEta_[type]->fill(  conversions[iConv]->caloCluster()[0]->eta(),   float(tracks[i]->recHitsSize() ) );
         h_tkChi2_[type] ->fill (tracks[i]->normalizedChi2() ,iE, weight); 
         px+= tracks[i]->innerMomentum().x();
         py+= tracks[i]->innerMomentum().y();
         pz+= tracks[i]->innerMomentum().z();
         e +=  sqrt (  tracks[i]->innerMomentum().x()*tracks[i]->innerMomentum().x() +
                       tracks[i]->innerMomentum().y()*tracks[i]->innerMomentum().y() +
                       tracks[i]->innerMomentum().z()*tracks[i]->innerMomentum().z() +
                       +  mElec*mElec ) ;
       }
       float totP = sqrt(px*px +py*py + pz*pz);
       float invM=  (e + totP) * (e-totP) ;
 
       if ( invM> 0.) {
         invM= sqrt( invM);
       } else {
         invM=-1;
       }
 
       h_invMass_[type][0] ->fill( invM,iE, weight);
       if ( phoIsInBarrel ) h_invMass_[type][1] ->fill(invM,iE, weight);
       if ( phoIsInEndcap ) h_invMass_[type][2] ->fill(invM,iE, weight);


       float  dPhiTracksAtVtx = -99;
       
       float phiTk1= tracks[0]->innerMomentum().phi();
       float phiTk2= tracks[1]->innerMomentum().phi();
       dPhiTracksAtVtx = phiTk1-phiTk2;
       dPhiTracksAtVtx = phiNormalization( dPhiTracksAtVtx );
       h_DPhiTracksAtVtx_[type][0]->fill( dPhiTracksAtVtx,iE, weight);
       if ( phoIsInBarrel ) h_DPhiTracksAtVtx_[type][1]->fill( dPhiTracksAtVtx,iE, weight);
       if ( phoIsInEndcap ) h_DPhiTracksAtVtx_[type][2]->fill( dPhiTracksAtVtx,iE, weight);
       h_DCotTracks_[type][0] ->fill ( conversions[iConv]->pairCotThetaSeparation() ,iE, weight);
       if ( phoIsInBarrel ) h_DCotTracks_[type][1] ->fill ( conversions[iConv]->pairCotThetaSeparation() ,iE, weight);
       if ( phoIsInEndcap ) h_DCotTracks_[type][2] ->fill ( conversions[iConv]->pairCotThetaSeparation() ,iE, weight);
       
       
       float  dPhiTracksAtEcal=-99;
       float  dEtaTracksAtEcal=-99;
       if (conversions[iConv]-> bcMatchingWithTracks()[0].isNonnull() && conversions[iConv]->bcMatchingWithTracks()[1].isNonnull() ) {
         
         
         float recoPhi1 = conversions[iConv]->ecalImpactPosition()[0].phi();
         float recoPhi2 = conversions[iConv]->ecalImpactPosition()[1].phi();
         float recoEta1 = conversions[iConv]->ecalImpactPosition()[0].eta();
         float recoEta2 = conversions[iConv]->ecalImpactPosition()[1].eta();
         float bcPhi1 = conversions[iConv]->bcMatchingWithTracks()[0]->phi();
         float bcPhi2 = conversions[iConv]->bcMatchingWithTracks()[1]->phi();
//         float bcEta1 = conversions[iConv]->bcMatchingWithTracks()[0]->eta();
//         float bcEta2 = conversions[iConv]->bcMatchingWithTracks()[1]->eta();
         recoPhi1 = phiNormalization(recoPhi1);
         recoPhi2 = phiNormalization(recoPhi2);
         bcPhi1 = phiNormalization(bcPhi1);
         bcPhi2 = phiNormalization(bcPhi2);
         dPhiTracksAtEcal = recoPhi1 -recoPhi2;
         dPhiTracksAtEcal = phiNormalization( dPhiTracksAtEcal );
         dEtaTracksAtEcal = recoEta1 -recoEta2;
         
         h_DPhiTracksAtEcal_[type][0]->fill( dPhiTracksAtEcal,iE, weight);
         h_DEtaTracksAtEcal_[type][0]->fill( dEtaTracksAtEcal,iE, weight);
         if ( phoIsInBarrel ) {
           h_DPhiTracksAtEcal_[type][1]->fill( dPhiTracksAtEcal,iE, weight);
           h_DEtaTracksAtEcal_[type][1]->fill( dEtaTracksAtEcal,iE, weight);
         }
         if ( phoIsInEndcap ) {
           h_DPhiTracksAtEcal_[type][2]->fill( dPhiTracksAtEcal,iE, weight);
           h_DEtaTracksAtEcal_[type][2]->fill( dEtaTracksAtEcal,iE, weight);
         }
     }

    } // loop over conversions


//  }/// End loop over Reco  particles

  h_nPho_[0][0]-> fill (float(nPho[0]),iE, weight);
  h_nPho_[0][1]-> fill (float(nPhoBarrel[0]),iE, weight);
  h_nPho_[0][2]-> fill (float(nPhoEndcap[0]),iE, weight);
  h_nPho_[1][0]-> fill (float(nPho[1]),iE, weight);
  h_nPho_[1][1]-> fill (float(nPhoBarrel[1]),iE, weight);
  h_nPho_[1][2]-> fill (float(nPhoEndcap[1]),iE, weight);




}



void HistoPhoton::fill( const reco::ShallowClonePtrCandidate * pshallow, uint iE, double weight )
{


  // Get the underlying object that the shallow clone represents
  const pat::Photon * photon = dynamic_cast<const pat::Photon*>(&*(pshallow->masterClonePtr()));

  if ( photon == 0 ) {
    cout << "Error! Was passed a shallow clone that is not at heart a photon" << endl;
    return;
  }

  // first fill common 4-vector histograms
  HistoGroup<Photon>::fill( pshallow, iE, weight );

  // fill relevant photon histograms
  h_trackIso_       ->fill( photon->trackIso(), iE, weight );
  h_caloIso_        ->fill( photon->caloIso(), iE, weight );


/////////////
  using namespace edm;
//  const float etaPhiDistance=0.01;
  // Fiducial region
//  const float TRK_BARL =0.9;
//  const float BARL = 1.4442; // DAQ TDR p.290
//  const float END_LO = 1.566;
//  const float END_HI = 2.5;
  // Electron mass
  const Float_t mElec= 0.000511;


  std::vector<int> nPho(2);
  std::vector<int> nPhoBarrel(2);
  std::vector<int> nPhoEndcap(2);
  for ( unsigned int i=0; i<nPho.size(); i++ ) nPho[i]=0;
  for ( unsigned int i=0; i<nPhoBarrel.size(); i++ ) nPhoBarrel[i]=0;
  for ( unsigned int i=0; i<nPhoEndcap.size(); i++ ) nPhoEndcap[i]=0;



    reco::Photon aPho = reco::Photon(*photon);

    bool  phoIsInBarrel=false;
    bool  phoIsInEndcap=false;

///////////////////a little bit corrections
    float etaPho=aPho.eta();
    if ( fabs(etaPho) <  1.479 ) {
      phoIsInBarrel=true;
    } else {
      phoIsInEndcap=true;
    }


//    float phiClu=aPho.superCluster()->phi();
    float etaClu=aPho.superCluster()->eta();



    bool  scIsInBarrel=false;
    bool  scIsInEndcap=false;
    if ( fabs(etaClu) <  1.479 ) 
      scIsInBarrel=true;
    else
      scIsInEndcap=true;



/*    bookProfile
    p_nTrackIsol_->fill( (*photon).superCluster()->eta(),   float(nTracks));
    p_trackPtSum_->fill((*photon).superCluster()->eta(),  ptSum);
    p_ecalSum_->fill((*photon).superCluster()->eta(),  ecalSum);
    p_hcalSum_->fill((*photon).superCluster()->eta(),  hcalSum);
*/


    bool isIsolated=false;
//    if ( (nTracks < numOfTracksInCone_) && 
//           ( ptSum < trkPtSumCut_) &&
//           ( ecalSum < ecalEtSumCut_ ) &&  ( hcalSum < hcalEtSumCut_ ) ) isIsolated = true;



    int type=0;
    if ( !isIsolated ) type=1;

    nPho[type]++; 
    if (phoIsInBarrel) nPhoBarrel[type]++;
    if (phoIsInEndcap) nPhoEndcap[type]++;



//   e3x3 needs to be calculated
    float e3x3= 0; // EcalClusterTools::e3x3(  *(   (*photon).superCluster()->seed()  ), &ecalRecHitCollection, &(*topology));
    float r9 =e3x3/( (*photon).superCluster()->rawEnergy()+ (*photon).superCluster()->preshowerEnergy());


    h_scEta_[type]->fill( (*photon).superCluster()->eta() ,iE, weight );
    h_scPhi_[type]->fill( (*photon).superCluster()->phi() ,iE, weight);


///////////////////////////////////////////////////////////////two dimensional histogram
//    h_scEtaPhi_[type]->fill( (*photon).superCluster()->eta(), (*photon).superCluster()->phi() );

    h_scE_[type][0]->fill( (*photon).superCluster()->energy() ,iE, weight);
    h_scEt_[type][0]->fill( (*photon).superCluster()->energy()/cosh( (*photon).superCluster()->eta()) ,iE, weight);
    h_r9_[type][0]->fill( r9 ,iE, weight);

    h_phoEta_[type]->fill( (*photon).eta() ,iE, weight);
    h_phoPhi_[type]->fill( (*photon).phi() ,iE, weight);

    h_phoE_[type][0]->fill( (*photon).energy() ,iE, weight);
    h_phoEt_[type][0]->fill( (*photon).energy()/ cosh( (*photon).eta()) ,iE, weight);


    h_nConv_[type][0]->fill(float( (*photon).conversions().size()),iE, weight);

    if ( scIsInBarrel ) {
      h_scE_[type][1]->fill( (*photon).superCluster()->energy() ,iE, weight);
      h_scEt_[type][1]->fill( (*photon).superCluster()->energy()/cosh( (*photon).superCluster()->eta()) ,iE, weight);
      h_r9_[type][1]->fill( r9 ,iE, weight);
     }

    if ( scIsInEndcap ) {
      h_scE_[type][2]->fill( (*photon).superCluster()->energy() ,iE, weight);
      h_scEt_[type][2]->fill( (*photon).superCluster()->energy()/cosh( (*photon).superCluster()->eta()) ,iE, weight);
      h_r9_[type][2]->fill( r9 ,iE, weight);
     }

    if ( phoIsInBarrel ) {
      h_phoE_[type][1]->fill( (*photon).energy() ,iE, weight);
      h_phoEt_[type][1]->fill( (*photon).energy()/ cosh( (*photon).eta()) ,iE, weight);
      h_nConv_[type][1]->fill(float( (*photon).conversions().size()),iE, weight);
     }
    
    if ( phoIsInEndcap ) {
      h_phoE_[type][2]->fill( (*photon).energy() ,iE, weight);
      h_phoEt_[type][2]->fill( (*photon).energy()/ cosh( (*photon).eta()) ,iE, weight);
      h_nConv_[type][2]->fill(float( (*photon).conversions().size()),iE, weight);
     }

   ////////////////// plot quantitied related to conversions
    reco::ConversionRefVector conversions = (*photon).conversions();
 
    for (unsigned int iConv=0; iConv<conversions.size(); iConv++) {

// two dimensional histogram
//      h_r9VsNofTracks_[type][0]->fill( r9, conversions[iConv]->nTracks() ) ; 
 
//      if ( phoIsInBarrel ) h_r9VsNofTracks_[type][1]->fill( r9,  conversions[iConv]->nTracks() ) ; 
 
//      if ( phoIsInEndcap ) h_r9VsNofTracks_[type][2]->fill( r9,  conversions[iConv]->nTracks() ) ; 

      if ( conversions[iConv]->nTracks() <2 ) continue; 
 
 
      h_convEta_[type]->fill( conversions[iConv]-> caloCluster()[0]->eta() );
      h_convPhi_[type]->fill( conversions[iConv]-> caloCluster()[0]->phi() );
      h_EoverPTracks_[type][0] ->fill( conversions[iConv]->EoverP() ,iE, weight) ;
      if ( phoIsInBarrel ) h_EoverPTracks_[type][1] ->fill( conversions[iConv]->EoverP() ,iE, weight) ;
      if ( phoIsInEndcap ) h_EoverPTracks_[type][2] ->fill( conversions[iConv]->EoverP() ,iE, weight) ;
 
 
/* /// two dimensional histogram      
      if ( conversions[iConv]->conversionVertex().isValid() ) 
        h_convVtxRvsZ_[type] ->fill ( fabs (conversions[iConv]->conversionVertex().position().z() ),  sqrt(conversions[iConv]->conversionVertex().position().perp2())  ) ;
*/         
 
       h_zPVFromTracks_[type]->fill ( conversions[iConv]->zOfPrimaryVertexFromTracks() ,iE, weight);

       std::vector<reco::TrackRef> tracks = conversions[iConv]->tracks();

       float px=0;
       float py=0;
       float pz=0;
       float e=0;
       for (unsigned int i=0; i<tracks.size(); i++) {
//	   two dimensional histo
//         p_tk_nHitsVsEta_[type]->fill(  conversions[iConv]->caloCluster()[0]->eta(),   float(tracks[i]->recHitsSize() ) );
         h_tkChi2_[type] ->fill (tracks[i]->normalizedChi2() ,iE, weight); 
         px+= tracks[i]->innerMomentum().x();
         py+= tracks[i]->innerMomentum().y();
         pz+= tracks[i]->innerMomentum().z();
         e +=  sqrt (  tracks[i]->innerMomentum().x()*tracks[i]->innerMomentum().x() +
                       tracks[i]->innerMomentum().y()*tracks[i]->innerMomentum().y() +
                       tracks[i]->innerMomentum().z()*tracks[i]->innerMomentum().z() +
                       +  mElec*mElec ) ;
       }
       float totP = sqrt(px*px +py*py + pz*pz);
       float invM=  (e + totP) * (e-totP) ;
 
       if ( invM> 0.) {
         invM= sqrt( invM);
       } else {
         invM=-1;
       }
 
       h_invMass_[type][0] ->fill( invM,iE, weight);
       if ( phoIsInBarrel ) h_invMass_[type][1] ->fill(invM,iE, weight);
       if ( phoIsInEndcap ) h_invMass_[type][2] ->fill(invM,iE, weight);


       float  dPhiTracksAtVtx = -99;
       
       float phiTk1= tracks[0]->innerMomentum().phi();
       float phiTk2= tracks[1]->innerMomentum().phi();
       dPhiTracksAtVtx = phiTk1-phiTk2;
       dPhiTracksAtVtx = phiNormalization( dPhiTracksAtVtx );
       h_DPhiTracksAtVtx_[type][0]->fill( dPhiTracksAtVtx,iE, weight);
       if ( phoIsInBarrel ) h_DPhiTracksAtVtx_[type][1]->fill( dPhiTracksAtVtx,iE, weight);
       if ( phoIsInEndcap ) h_DPhiTracksAtVtx_[type][2]->fill( dPhiTracksAtVtx,iE, weight);
       h_DCotTracks_[type][0] ->fill ( conversions[iConv]->pairCotThetaSeparation() ,iE, weight);
       if ( phoIsInBarrel ) h_DCotTracks_[type][1] ->fill ( conversions[iConv]->pairCotThetaSeparation() ,iE, weight);
       if ( phoIsInEndcap ) h_DCotTracks_[type][2] ->fill ( conversions[iConv]->pairCotThetaSeparation() ,iE, weight);
       
       
       float  dPhiTracksAtEcal=-99;
       float  dEtaTracksAtEcal=-99;
       if (conversions[iConv]-> bcMatchingWithTracks()[0].isNonnull() && conversions[iConv]->bcMatchingWithTracks()[1].isNonnull() ) {
         
         
         float recoPhi1 = conversions[iConv]->ecalImpactPosition()[0].phi();
         float recoPhi2 = conversions[iConv]->ecalImpactPosition()[1].phi();
         float recoEta1 = conversions[iConv]->ecalImpactPosition()[0].eta();
         float recoEta2 = conversions[iConv]->ecalImpactPosition()[1].eta();
         float bcPhi1 = conversions[iConv]->bcMatchingWithTracks()[0]->phi();
         float bcPhi2 = conversions[iConv]->bcMatchingWithTracks()[1]->phi();
//         float bcEta1 = conversions[iConv]->bcMatchingWithTracks()[0]->eta();
//         float bcEta2 = conversions[iConv]->bcMatchingWithTracks()[1]->eta();
         recoPhi1 = phiNormalization(recoPhi1);
         recoPhi2 = phiNormalization(recoPhi2);
         bcPhi1 = phiNormalization(bcPhi1);
         bcPhi2 = phiNormalization(bcPhi2);
         dPhiTracksAtEcal = recoPhi1 -recoPhi2;
         dPhiTracksAtEcal = phiNormalization( dPhiTracksAtEcal );
         dEtaTracksAtEcal = recoEta1 -recoEta2;
         
         h_DPhiTracksAtEcal_[type][0]->fill( dPhiTracksAtEcal,iE, weight);
         h_DEtaTracksAtEcal_[type][0]->fill( dEtaTracksAtEcal,iE, weight);
         if ( phoIsInBarrel ) {
           h_DPhiTracksAtEcal_[type][1]->fill( dPhiTracksAtEcal,iE, weight);
           h_DEtaTracksAtEcal_[type][1]->fill( dEtaTracksAtEcal,iE, weight);
         }
         if ( phoIsInEndcap ) {
           h_DPhiTracksAtEcal_[type][2]->fill( dPhiTracksAtEcal,iE, weight);
           h_DEtaTracksAtEcal_[type][2]->fill( dEtaTracksAtEcal,iE, weight);
         }
     }

    } // loop over conversions


//  }/// End loop over Reco  particles

  h_nPho_[0][0]-> fill (float(nPho[0]),iE, weight);
  h_nPho_[0][1]-> fill (float(nPhoBarrel[0]),iE, weight);
  h_nPho_[0][2]-> fill (float(nPhoEndcap[0]),iE, weight);
  h_nPho_[1][0]-> fill (float(nPho[1]),iE, weight);
  h_nPho_[1][1]-> fill (float(nPhoBarrel[1]),iE, weight);
  h_nPho_[1][2]-> fill (float(nPhoEndcap[1]),iE, weight);







}


void HistoPhoton::fillCollection( const std::vector<Photon> & coll, double weight ) 
{

  h_size_->fill( coll.size(), 1, weight );     //! Save the size of the collection.

  std::vector<Photon>::const_iterator
    iobj = coll.begin(),
    iend = coll.end();

  uint i = 1;              //! Fortran-style indexing
  for ( ; iobj != iend; ++iobj, ++i ) {
    fill( &*iobj, i, weight);      //! &*iobj dereferences to the pointer to a PHYS_OBJ*
  } 
}

void HistoPhoton::clearVec()
{
  HistoGroup<Photon>::clearVec();

  h_trackIso_->clearVec();
  h_caloIso_->clearVec();


/*
  p_nTrackIsol_->clearVec();
  p_trackPtSum_->clearVec();
  p_ecalSum_->clearVec();
  p_hcalSum_->clearVec();
*/

 for (int i=0; i<2; i++){
    for(int j=0; j<3; j++){
	  h_nPho_[i][j]->clearVec();
	  h_scE_[i][j]->clearVec();
	  h_scEt_[i][j]->clearVec();

	  h_r9_[i][j]->clearVec();
	  h_phoE_[i][j]->clearVec();
	  h_phoEt_[i][j]->clearVec();
	  h_nConv_[i][j]->clearVec();
//	  h_r9VsNofTracks_[i][j]->clearVec();
	  h_EoverPTracks_[i][j]->clearVec();
	  h_DPhiTracksAtVtx_[i][j]->clearVec();
	  h_DCotTracks_[i][j]->clearVec();
	  h_invMass_[i][j]->clearVec();
	  h_DPhiTracksAtEcal_[i][j]->clearVec();
	  h_DEtaTracksAtEcal_[i][j]->clearVec();
     }// loop j
    h_scEta_[i]->clearVec();
    h_scPhi_[i]->clearVec();
//    h_scEtaPhi_[i]->clearVec();

    h_phoEta_[i]->clearVec();
    h_phoPhi_[i]->clearVec();

    h_convEta_[i]->clearVec();
    h_convPhi_[i]->clearVec();
//    p_tk_nHitsVsEta_[i]->clearVec();
    h_tkChi2_[i]->clearVec();

//    h_convVtxRvsZ_[i]->clearVec();
    h_zPVFromTracks_[i]->clearVec();
  } // loop i


}


float HistoPhoton::phiNormalization(float & phi)
{
//---Definitions
  const float PI    = 3.1415927;
  const float TWOPI = 2.0*PI;
 

  if(phi >  PI) {phi = phi - TWOPI;}
  if(phi < -PI) {phi = phi + TWOPI;}
 
  //  cout << " Float_t PHInormalization out " << PHI << endl;
  return phi;
 
}

