// -*- C++ -*-
//
// Package:    RecoEgamma/Examples
// Class:      GsfElectronAnalyzer
// 
/**\class GsfElectronAnalyzer RecoEgamma/Examples/src/GsfElectronAnalyzer.cc

 Description: rereading of GsfElectrons for verification

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: GsfElectronAnalyzer.cc,v 1.6 2008/03/15 14:41:40 charlot Exp $
//
//

// user include files
#include "RecoEgamma/Examples/plugins/GsfElectronAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"

#include <iostream>
#include "TMath.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH1I.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include <iostream>

using namespace reco;
 
GsfElectronAnalyzer::GsfElectronAnalyzer(const edm::ParameterSet& conf)
{

  outputFile_ = conf.getParameter<std::string>("outputFile");
  histfile_ = new TFile(outputFile_.c_str(),"RECREATE");
  electronCollection_=conf.getParameter<edm::InputTag>("electronCollection");
  mcTruthCollection_ = conf.getParameter<edm::InputTag>("mcTruthCollection");
  maxPt_ = conf.getParameter<double>("MaxPt");
  maxAbsEta_ = conf.getParameter<double>("MaxAbsEta");
  deltaR_ = conf.getParameter<double>("DeltaR");
  etamin=conf.getParameter<double>("Etamin");
  etamax=conf.getParameter<double>("Etamax");
  phimin=conf.getParameter<double>("Phimin");
  phimax=conf.getParameter<double>("Phimax");
  ptmax=conf.getParameter<double>("Ptmax");
  pmax=conf.getParameter<double>("Pmax");
  eopmax=conf.getParameter<double>("Eopmax");
  eopmaxsht=conf.getParameter<double>("Eopmaxsht");
  detamin=conf.getParameter<double>("Detamin");
  detamax=conf.getParameter<double>("Detamax");
  dphimin=conf.getParameter<double>("Dphimin");
  dphimax=conf.getParameter<double>("Dphimax");
  detamatchmin=conf.getParameter<double>("Detamatchmin");
  detamatchmax=conf.getParameter<double>("Detamatchmax");
  dphimatchmin=conf.getParameter<double>("Dphimatchmin");
  dphimatchmax=conf.getParameter<double>("Dphimatchmax");
  fhitsmax=conf.getParameter<double>("Fhitsmax");
  lhitsmax=conf.getParameter<double>("Lhitsmax");
  nbineta=conf.getParameter<int>("Nbineta");
  nbineta2D=conf.getParameter<int>("Nbineta2D");
  nbinp=conf.getParameter<int>("Nbinp");
  nbinpt=conf.getParameter<int>("Nbinpt");
  nbinp2D=conf.getParameter<int>("Nbinp2D");
  nbinpt2D=conf.getParameter<int>("Nbinpt2D");
  nbinpteff=conf.getParameter<int>("Nbinpteff");
  nbinphi=conf.getParameter<int>("Nbinphi");
  nbinphi2D=conf.getParameter<int>("Nbinphi2D");
  nbineop=conf.getParameter<int>("Nbineop");
  nbineop2D=conf.getParameter<int>("Nbineop2D");
  nbinfhits=conf.getParameter<int>("Nbinfhits");
  nbinlhits=conf.getParameter<int>("Nbinlhits");
  nbinxyz=conf.getParameter<int>("Nbinxyz");
  nbindeta=conf.getParameter<int>("Nbindeta");
  nbindphi=conf.getParameter<int>("Nbindphi");
  nbindetamatch=conf.getParameter<int>("Nbindetamatch");
  nbindphimatch=conf.getParameter<int>("Nbindphimatch");
  nbindetamatch2D=conf.getParameter<int>("Nbindetamatch2D");
  nbindphimatch2D=conf.getParameter<int>("Nbindphimatch2D");
}  
  
GsfElectronAnalyzer::~GsfElectronAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  histfile_->Write();
  histfile_->Close();
}

void GsfElectronAnalyzer::beginJob(edm::EventSetup const&iSetup){

  histfile_->cd();
  
  
  // mc truth  

  h_mcNum              = new TH1F( "h_mcNum",              "# mc particles",    nbinfhits,0.,fhitsmax );
  h_eleNum             = new TH1F( "h_mcNum_ele",             "# mc electrons",             nbinfhits,0.,fhitsmax);
  h_gamNum             = new TH1F( "h_mcNum_gam",             "# mc gammas",             nbinfhits,0.,fhitsmax);
    
  // rec event
  
  histNum_= new TH1F("h_recEleNum","# rec electrons",20, 0.,20.);
  
  // mc  
  h_simEta             = new TH1F( "h_mc_eta",             "mc #eta",           nbineta,etamin,etamax); 
  h_simAbsEta             = new TH1F( "h_mc_abseta",             "mc |#eta|",           nbineta/2,0.,etamax); 
  h_simP               = new TH1F( "h_mc_P",               "mc p",              nbinp,0.,pmax); 
  h_simPt               = new TH1F( "h_mc_Pt",               "mc pt",            nbinpteff,5.,ptmax); 
  h_simPhi               = new TH1F( "h_mc_phi",               "mc phi",        nbinphi,phimin,phimax); 
  h_simZ      = new TH1F( "h_mc_z",      "mc z ",    50, -25, 25 );

  // ctf tracks
  h_ctf_foundHitsVsEta      = new TH2F( "h_ctf_foundHitsVsEta",      "ctf track # found hits vs eta",  nbineta2D,etamin,etamax,nbinfhits,0.,fhitsmax);
  h_ctf_lostHitsVsEta       = new TH2F( "h_ctf_lostHitsVsEta",       "ctf track # lost hits vs eta",   nbineta2D,etamin,etamax,nbinlhits,0.,lhitsmax);
  
  // all electrons  
  h_ele_EoverP_all       = new TH1F( "h_ele_EoverP_all",       "all ele E/p at vertex",  nbineop,0.,eopmax);
  h_ele_TIP_all       = new TH1F( "h_ele_TIP_all",       "all ele tip at vertex",  100,0.,0.2);
  h_ele_vertexPt_all       = new TH1F( "h_ele_vertexPt_all",       "all ele p_{T} at vertex",  nbinpteff,5.,ptmax);
  h_ele_vertexEta_all      = new TH1F( "h_ele_vertexEta_all",      "all ele #eta at vertex",    nbineta,etamin,etamax);

  // matched electrons
  h_ele_charge         = new TH1F( "h_ele_charge",         "ele charge",             5,-2.,2.);   
  h_ele_chargeVsEta    = new TH2F( "h_ele_chargeVsEta",         "ele charge vs eta", nbineta2D,etamin,etamax,5,-2.,2.);   
  h_ele_chargeVsPhi    = new TH2F( "h_ele_chargeVsPhi",         "ele charge vs phi", nbinphi2D,phimin,phimax,5,-2.,2.);   
  h_ele_chargeVsPt    = new TH2F( "h_ele_chargeVsPt",         "ele charge vs pt", nbinpt,0.,100.,5,-2.,2.);   
  h_ele_vertexP        = new TH1F( "h_ele_vertexP",        "ele p at vertex",       nbinp,0.,pmax);
  h_ele_vertexPt       = new TH1F( "h_ele_vertexPt",       "ele p_{T} at vertex",  nbinpt,0.,ptmax);
  h_ele_vertexPtVsEta   = new TH2F( "h_ele_vertexPtVsEta",       "ele p_{T} at vertex vs eta",nbinpt2D,etamin,etamax,nbinpt2D,0.,ptmax);
  h_ele_vertexPtVsPhi   = new TH2F( "h_ele_vertexPtVsPhi",       "ele p_{T} at vertex vs phi",nbinphi2D,phimin,phimax,nbinpt2D,0.,ptmax);
  h_ele_simPt_matched       = new TH1F( "h_ele_simPt_matched",       "sim p_{T}, matched electrons",  nbinpteff,5.,ptmax);
  h_ele_vertexEta      = new TH1F( "h_ele_vertexEta",      "ele #eta at vertex",    nbineta,etamin,etamax);
  h_ele_vertexEtaVsPhi  = new TH2F( "h_ele_vertexEtaVsPhi",      "ele #eta at vertex vs phi",nbineta2D,etamin,etamax,nbinphi2D,phimin,phimax );
  h_ele_simAbsEta_matched      = new TH1F( "h_ele_simAbsEta_matched",      "sim |#eta|, matched electrons",    nbineta/2,0.,2.5);
  h_ele_simEta_matched      = new TH1F( "h_ele_simEta_matched",      "sim #eta, matched electrons",    nbineta,etamin,etamax);
  h_ele_simPhi_matched               = new TH1F( "h_ele_simPhi_matched",               "sim phi, matched electrons",        nbinphi,phimin,phimax); 
  h_ele_vertexPhi      = new TH1F( "h_ele_vertexPhi",      "ele #phi at vertex",    nbinphi,phimin,phimax);
  h_ele_vertexX      = new TH1F( "h_ele_vertexX",      "ele x at vertex",    nbinxyz,-0.1,0.1 );
  h_ele_vertexY      = new TH1F( "h_ele_vertexY",      "ele y at vertex",    nbinxyz,-0.1,0.1 );
  h_ele_vertexZ      = new TH1F( "h_ele_vertexZ",      "ele z at vertex",    nbinxyz,-25, 25 );
  h_ele_simZ_matched      = new TH1F( "h_ele_simZ_matched",      "sim z matched electrons",    nbinxyz,-25,25);
  h_ele_vertexTIP      = new TH1F( "h_ele_vertexTIP",      "ele TIP",    90,0.,0.15);
  h_ele_vertexTIPVsEta      = new TH2F( "h_ele_vertexTIPVsEta",      "ele TIP vs eta", nbineta2D,etamin,etamax,45,0.,0.15);
  h_ele_vertexTIPVsPhi      = new TH2F( "h_ele_vertexTIPVsPhi",      "ele TIP vs phi", nbinphi2D,phimin,phimax,45,0.,0.15);
  h_ele_vertexTIPVsPt      = new TH2F( "h_ele_vertexTIPVsPt",      "ele TIP vs Pt", nbinpt2D,0.,ptmax,45,0.,0.15);
  h_ele_PoPtrue        = new TH1F( "h_ele_PoPtrue",        "ele P/Ptrue @ vertex", 75,0.,1.5);
  h_ele_PoPtrueVsEta   = new TH2F( "h_ele_PoPtrueVsEta",        "ele P/Ptrue @ vertex vs eta", nbineta2D,etamin,etamax,50,0.,1.5);
  h_ele_PoPtrueVsPhi   = new TH2F( "h_ele_PoPtrueVsPhi",        "ele P/Ptrue @ vertex vs phi", nbinphi2D,phimin,phimax,50,0.,1.5);
  h_ele_PoPtrueVsPt   = new TH2F( "h_ele_PoPtrueVsPt",        "ele P/Ptrue @ vertex vs eta", nbinpt2D,0.,ptmax,50,0.,1.5);
  h_ele_PoPtrue_barrel         = new TH1F( "h_ele_PoPtrue_barrel",        "ele P/Ptrue @ vertex, barrel",75,0.,1.5);
  h_ele_PoPtrue_endcaps        = new TH1F( "h_ele_PoPtrue_endcaps",        "ele P/Ptrue @ vertex, endcaps",75,0.,1.5);
  h_ele_EtaMnEtaTrue   = new TH1F( "h_ele_EtaMnEtaTrue",   "ele #eta_{rec} - #eta_{sim} @ vertex",nbindeta,detamin,detamax);
  h_ele_EtaMnEtaTrueVsEta   = new TH2F( "h_ele_EtaMnEtaTrueVsEta",   "ele #eta_{rec} - #eta_{sim} @ vertex vs eta",nbineta2D,etamin,etamax,nbindeta/2,detamin,detamax);
  h_ele_EtaMnEtaTrueVsPhi   = new TH2F( "h_ele_EtaMnEtaTrueVsPhi",   "ele #eta_{rec} - #eta_{sim} @ vertex vs phi",nbinphi2D,phimin,phimax,nbindeta/2,detamin,detamax);
  h_ele_EtaMnEtaTrueVsPt   = new TH2F( "h_ele_EtaMnEtaTrueVsPt",   "ele #eta_{rec} - #eta_{sim} @ vertex vs pt",nbinpt,0.,ptmax,nbindeta/2,detamin,detamax);
  h_ele_PhiMnPhiTrue   = new TH1F( "h_ele_PhiMnPhiTrue",   "ele #phi_{rec} - #phi_{sim} @ vertex",nbindphi,dphimin,dphimax);
  h_ele_PhiMnPhiTrue2   = new TH1F( "h_ele_PhiMnPhiTrue2",   "ele #phi_{rec} - #phi_{sim} @ vertex",nbindphimatch2D,dphimatchmin,dphimatchmax);
  h_ele_PhiMnPhiTrueVsEta   = new TH2F( "h_ele_PhiMnPhiTrueVsEta",   "ele #phi_{rec} - #phi_{sim} @ vertex vs eta",nbineta2D,etamin,etamax,nbindphi/2,dphimin,dphimax);
  h_ele_PhiMnPhiTrueVsPhi   = new TH2F( "h_ele_PhiMnPhiTrueVsPhi",   "ele #phi_{rec} - #phi_{sim} @ vertex vs phi",nbinphi2D,phimin,phimax,nbindphi/2,dphimin,dphimax);
  h_ele_PhiMnPhiTrueVsPt   = new TH2F( "h_ele_PhiMnPhiTrueVsPt",   "ele #phi_{rec} - #phi_{sim} @ vertex vs pt",nbinpt2D,0.,ptmax,nbindphi/2,dphimin,dphimax);

  // matched electron, superclusters
  histSclEn_ = new TH1F("h_scl_energy","ele supercluster energy",nbinp,0.,pmax);
  histSclEoEtrue_barrel = new TH1F("h_scl_EoEtrue_barrel","ele supercluster energy over true energy, barrel",50,0.2,1.2);
  histSclEoEtrue_endcaps = new TH1F("h_scl_EoEtrue_endcaps","ele supercluster energy over true energy, endcaps",50,0.2,1.2);
  histSclEt_ = new TH1F("h_scl_et","ele supercluster transverse energy",nbinpt,0.,ptmax);
  histSclEtVsEta_ = new TH2F("h_scl_etVsEta","ele supercluster transverse energy vs eta",nbineta2D,etamin,etamax,nbinpt,0.,ptmax);
  histSclEtVsPhi_ = new TH2F("h_scl_etVsPhi","ele supercluster transverse energy vs phi",nbinphi2D,phimin,phimax,nbinpt,0.,ptmax);
  histSclEtaVsPhi_ = new TH2F("h_scl_etaVsPhi","ele supercluster eta vs phi",nbinphi2D,phimin,phimax,nbineta2D,etamin,etamax);
  histSclEta_ = new TH1F("h_scl_eta","ele supercluster eta",nbineta,etamin,etamax);
  histSclPhi_ = new TH1F("h_scl_phi","ele supercluster phi",nbinphi,phimin,phimax);

  // matched electron, gsf tracks
  h_ele_foundHits      = new TH1F( "h_ele_foundHits",      "ele track # found hits",      nbinfhits,0.,fhitsmax);
  h_ele_foundHitsVsEta      = new TH2F( "h_ele_foundHitsVsEta",      "ele track # found hits vs eta",  nbineta2D,etamin,etamax,nbinfhits,0.,fhitsmax);
  h_ele_foundHitsVsPhi      = new TH2F( "h_ele_foundHitsVsPhi",      "ele track # found hits vs phi",  nbinphi2D,phimin,phimax,nbinfhits,0.,fhitsmax);
  h_ele_foundHitsVsPt      = new TH2F( "h_ele_foundHitsVsPt",      "ele track # found hits vs pt",  nbinpt2D,0.,ptmax,nbinfhits,0.,fhitsmax);
  h_ctf_foundHits      = new TH1F( "h_ctf_foundHits",      "ctf track # found hits",      nbinfhits,0.,fhitsmax);
  h_ele_lostHits       = new TH1F( "h_ele_lostHits",       "ele track # lost hits",       5,0.,5.);
  h_ele_lostHitsVsEta       = new TH2F( "h_ele_lostHitsVsEta",       "ele track # lost hits vs eta",   nbineta2D,etamin,etamax,nbinlhits,0.,lhitsmax);
  h_ele_lostHitsVsPhi       = new TH2F( "h_ele_lostHitsVsPhi",       "ele track # lost hits vs eta",   nbinphi2D,phimin,phimax,nbinlhits,0.,lhitsmax);
  h_ele_lostHitsVsPt       = new TH2F( "h_ele_lostHitsVsPt",       "ele track # lost hits vs eta",   nbinpt2D,0.,ptmax,nbinlhits,0.,lhitsmax);
  h_ele_chi2           = new TH1F( "h_ele_chi2",           "ele track #chi^{2}",         100,0.,15.);   
  h_ele_chi2VsEta           = new TH2F( "h_ele_chi2VsEta",           "ele track #chi^{2} vs eta",  nbineta2D,etamin,etamax,50,0.,15.);   
  h_ele_chi2VsPhi           = new TH2F( "h_ele_chi2VsPhi",           "ele track #chi^{2} vs phi",  nbinphi2D,phimin,phimax,50,0.,15.);   
  h_ele_chi2VsPt           = new TH2F( "h_ele_chi2VsPt",           "ele track #chi^{2} vs pt",  nbinpt2D,0.,ptmax,50,0.,15.);   
  h_ele_PinMnPout      = new TH1F( "h_ele_PinMnPout",      "ele track inner p - outer p, mean"   ,nbinp,0.,200.);
  h_ele_PinMnPout_mode      = new TH1F( "h_ele_PinMnPout_mode",      "ele track inner p - outer p, mode"   ,nbinp,0.,100.);
  h_ele_PinMnPoutVsEta_mode = new TH2F( "h_ele_PinMnPoutVsEta_mode",      "ele track inner p - outer p vs eta, mode" ,nbineta2D, etamin,etamax,nbinp2D,0.,100.);
  h_ele_PinMnPoutVsPhi_mode = new TH2F( "h_ele_PinMnPoutVsPhi_mode",      "ele track inner p - outer p vs phi, mode" ,nbinphi2D, phimin,phimax,nbinp2D,0.,100.);
  h_ele_PinMnPoutVsPt_mode = new TH2F( "h_ele_PinMnPoutVsPt_mode",      "ele track inner p - outer p vs pt, mode" ,nbinpt2D, 0.,ptmax,nbinp2D,0.,100.);
  h_ele_PinMnPoutVsE_mode = new TH2F( "h_ele_PinMnPoutVsE_mode",      "ele track inner p - outer p vs E, mode" ,nbinp2D, 0.,200.,nbinp2D,0.,100.);
  h_ele_PinMnPoutVsChi2_mode = new TH2F( "h_ele_PinMnPoutVsChi2_mode",      "ele track inner p - outer p vs track chi2, mode" ,50, 0.,20.,nbinp2D,0.,100.);
  h_ele_outerP         = new TH1F( "h_ele_outerP",         "ele track outer p, mean",          nbinp,0.,pmax);
  h_ele_outerP_mode         = new TH1F( "h_ele_outerP_mode",         "ele track outer p, mode",          nbinp,0.,pmax);
  h_ele_outerPVsEta_mode         = new TH2F( "h_ele_outerPVsEta_mode",         "ele track outer p vs eta mode", nbineta2D,etamin,etamax,50,0.,pmax);
  h_ele_outerPt        = new TH1F( "h_ele_outerPt",        "ele track outer p_{T}, mean",      nbinpt,0.,ptmax);
  h_ele_outerPt_mode        = new TH1F( "h_ele_outerPt_mode",        "ele track outer p_{T}, mode",      nbinpt,0.,ptmax);
  h_ele_outerPtVsEta_mode        = new TH2F( "h_ele_outerPtVsEta_mode", "ele track outer p_{T} vs eta, mode", nbineta2D,etamin,etamax,nbinpt2D,0.,ptmax);
  h_ele_outerPtVsPhi_mode        = new TH2F( "h_ele_outerPtVsPhi_mode", "ele track outer p_{T} vs phi, mode", nbinphi2D,phimin,phimax,nbinpt2D,0.,ptmax);
  h_ele_outerPtVsPt_mode        = new TH2F( "h_ele_outerPtVsPt_mode", "ele track outer p_{T} vs pt, mode", nbinpt2D,0.,100.,nbinpt2D,0.,ptmax);
  
  // matched electrons, matching 
  h_ele_EoP            = new TH1F( "h_ele_EoP",            "ele E/P_{vertex}",        nbineop,0.,eopmax);
  h_ele_EoPVsEta            = new TH2F( "h_ele_EoPVsEta",            "ele E/P_{vertex} vs eta",  nbineta2D,etamin,etamax,nbineop2D,0.,eopmaxsht);
  h_ele_EoPVsPhi            = new TH2F( "h_ele_EoPVsPhi",            "ele E/P_{vertex} vs phi",  nbinphi2D,phimin,phimax,nbineop2D,0.,eopmaxsht);
  h_ele_EoPVsE            = new TH2F( "h_ele_EoPVsE",            "ele E/P_{vertex} vs E",  50,0.,pmax ,50,0.,5.);
  h_ele_EoPout         = new TH1F( "h_ele_EoPout",         "ele E/P_{out}",           nbineop,0.,eopmax);
  h_ele_EoPoutVsEta         = new TH2F( "h_ele_EoPoutVsEta",         "ele E/P_{out} vs eta",    nbineta2D,etamin,etamax,nbineop2D,0.,eopmaxsht);
  h_ele_EoPoutVsPhi         = new TH2F( "h_ele_EoPoutVsPhi",         "ele E/P_{out} vs phi",    nbinphi2D,phimin,phimax,nbineop2D,0.,eopmaxsht);
  h_ele_EoPoutVsE         = new TH2F( "h_ele_EoPoutVsE",         "ele E/P_{out} vs E",    nbinp2D,0.,pmax,nbineop2D,0.,eopmaxsht);
  h_ele_dEtaSc_propVtx = new TH1F( "h_ele_dEtaSc_propVtx", "ele #eta_{sc} - #eta_{tr} - prop from vertex",      nbindetamatch,detamatchmin,detamatchmax);
  h_ele_dEtaScVsEta_propVtx = new TH2F( "h_ele_dEtaScVsEta_propVtx", "ele #eta_{sc} - #eta_{tr} vs eta, prop from vertex", nbineta2D,etamin,etamax,nbindetamatch2D,detamatchmin,detamatchmax);
  h_ele_dEtaScVsPhi_propVtx = new TH2F( "h_ele_dEtaScVsPhi_propVtx", "ele #eta_{sc} - #eta_{tr} vs phi, prop from vertex", nbinphi2D,phimin,phimax,nbindetamatch2D,detamatchmin,detamatchmax);
  h_ele_dEtaScVsPt_propVtx = new TH2F( "h_ele_dEtaScVsPt_propVtx", "ele #eta_{sc} - #eta_{tr} vs pt, prop from vertex", nbinpt2D,0.,ptmax,nbindetamatch2D,detamatchmin,detamatchmax);
  h_ele_dPhiSc_propVtx = new TH1F( "h_ele_dPhiSc_propVtx", "ele #phi_{sc} - #phi_{tr} - prop from vertex",      nbindphimatch,dphimatchmin,dphimatchmax);
  h_ele_dPhiScVsEta_propVtx = new TH2F( "h_ele_dPhiScVsEta_propVtx", "ele #phi_{sc} - #phi_{tr} vs eta, prop from vertex", nbineta2D,etamin,etamax,nbindphimatch2D,dphimatchmin,dphimatchmax);
  h_ele_dPhiScVsPhi_propVtx = new TH2F( "h_ele_dPhiScVsPhi_propVtx", "ele #phi_{sc} - #phi_{tr} vs phi, prop from vertex", nbinphi2D,phimin,phimax,nbindphimatch2D,dphimatchmin,dphimatchmax);
  h_ele_dPhiScVsPt_propVtx = new TH2F( "h_ele_dPhiScVsPt_propVtx", "ele #phi_{sc} - #phi_{tr} vs pt, prop from vertex", nbinpt2D,0.,ptmax,nbindphimatch2D,dphimatchmin,dphimatchmax);
  h_ele_dEtaCl_propOut = new TH1F( "h_ele_dEtaCl_propOut", "ele #eta_{cl} - #eta_{tr} - prop from outermost",   nbindetamatch,detamatchmin,detamatchmax);
  h_ele_dEtaClVsEta_propOut = new TH2F( "h_ele_dEtaClVsEta_propOut", "ele #eta_{cl} - #eta_{tr} vs eta, prop from out", nbineta2D,etamin,etamax,nbindetamatch2D,detamatchmin,detamatchmax);
  h_ele_dEtaClVsPhi_propOut = new TH2F( "h_ele_dEtaClVsPhi_propOut", "ele #eta_{cl} - #eta_{tr} vs phi, prop from out", nbinphi2D,phimin,phimax,nbindetamatch2D,detamatchmin,detamatchmax);
  h_ele_dEtaClVsPt_propOut = new TH2F( "h_ele_dEtaScVsPt_propOut", "ele #eta_{cl} - #eta_{tr} vs pt, prop from out", nbinpt2D,0.,ptmax,nbindetamatch2D,detamatchmin,detamatchmax);
  h_ele_dPhiCl_propOut = new TH1F( "h_ele_dPhiCl_propOut", "ele #phi_{cl} - #phi_{tr} - prop from outermost",   nbindphimatch,dphimatchmin,dphimatchmax);
  h_ele_dPhiClVsEta_propOut = new TH2F( "h_ele_dPhiClVsEta_propOut", "ele #phi_{cl} - #phi_{tr} vs eta, prop from out", nbineta2D,etamin,etamax,nbindphimatch2D,dphimatchmin,dphimatchmax);
  h_ele_dPhiClVsPhi_propOut = new TH2F( "h_ele_dPhiClVsPhi_propOut", "ele #phi_{cl} - #phi_{tr} vs phi, prop from out", nbinphi2D,phimin,phimax,nbindphimatch2D,dphimatchmin,dphimatchmax);
  h_ele_dPhiClVsPt_propOut = new TH2F( "h_ele_dPhiSClsPt_propOut", "ele #phi_{cl} - #phi_{tr} vs pt, prop from out", nbinpt2D,0.,ptmax,nbindphimatch2D,dphimatchmin,dphimatchmax);
  
  h_ele_HoE = new TH1F("h_ele_HoE", "ele H/E", 100,-0.5,0.5) ;
  h_ele_HoEVsEta = new TH2F("h_ele_HoEVsEta", "ele H/E vs eta", nbineta,etamin,etamax,50,-0.5,0.5) ;
  h_ele_HoEVsPhi = new TH2F("h_ele_HoEVsPhi", "ele H/E vs phi", nbinphi2D,phimin,phimax,50,-0.5,0.5) ;
  h_ele_HoEVsE = new TH2F("h_ele_HoEVsE", "ele H/E vs E", nbinp, 0.,300.,50,0.,1.) ;
 
  // classes  
  h_ele_classes = new TH1F( "h_ele_classes", "electron classes",      150,0.0,150.);
  h_ele_eta = new TH1F( "h_ele_eta", "electron eta",  nbineta/2,0.0,etamax);
  h_ele_eta_golden = new TH1F( "h_ele_eta_golden", "electron eta golden",  nbineta/2,0.0,etamax);
  h_ele_eta_bbrem = new TH1F( "h_ele_eta_bbrem", "electron eta bbrem",  nbineta/2,0.0,etamax);
  h_ele_eta_narrow = new TH1F( "h_ele_eta_narrow", "electron eta narrow",  nbineta/2,0.0,etamax);
  h_ele_eta_shower = new TH1F( "h_ele_eta_show", "electron eta showering",  nbineta/2,0.0,etamax);
  h_ele_PinVsPoutGolden_mode = new TH2F( "h_ele_PinVsPoutGolden_mode",      "ele track inner p vs outer p vs eta, golden, mode" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutShowering0_mode = new TH2F( "h_ele_PinVsPoutShowering0_mode",      "ele track inner p vs outer p vs eta, showering0, mode" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutShowering1234_mode = new TH2F( "h_ele_PinVsPoutShowering1234_mode",      "ele track inner p vs outer p vs eta, showering1234, mode" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutGolden_mean = new TH2F( "h_ele_PinVsPoutGolden_mean",      "ele track inner p vs outer p vs eta, golden, mean" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutShowering0_mean = new TH2F( "h_ele_PinVsPoutShowering0_mean",      "ele track inner p vs outer p vs eta, showering0, mean" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutShowering1234_mean = new TH2F( "h_ele_PinVsPoutShowering1234_mean",      "ele track inner p vs outer p vs eta, showering1234, mean" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PtinVsPtoutGolden_mode = new TH2F( "h_ele_PtinVsPtoutGolden_mode",      "ele track inner pt vs outer pt vs eta, golden, mode" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutShowering0_mode = new TH2F( "h_ele_PtinVsPtoutShowering0_mode",      "ele track inner pt vs outer pt vs eta, showering0, mode" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutShowering1234_mode = new TH2F( "h_ele_PtinVsPtoutShowering1234_mode",      "ele track inner pt vs outer pt vs eta, showering1234, mode" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutGolden_mean = new TH2F( "h_ele_PtinVsPtoutGolden_mean",      "ele track inner pt vs outer pt vs eta, golden, mean" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutShowering0_mean = new TH2F( "h_ele_PtinVsPtoutShowering0_mean",      "ele track inner pt vs outer pt vs eta, showering0, mean" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutShowering1234_mean = new TH2F( "h_ele_PtinVsPtoutShowering1234_mean",      "ele track inner pt vs outer pt vs eta, showering1234, mean" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  histSclEoEtrueGolden_barrel = new TH1F("h_scl_EoEtrue golden, barrel","ele supercluster energy over true energy, golden, barrel",100,0.2,1.2);
  histSclEoEtrueGolden_endcaps = new TH1F("h_scl_EoEtrue golden, endcaps","ele supercluster energy over true energy, golden, endcaps",100,0.2,1.2);
  histSclEoEtrueShowering0_barrel = new TH1F("h_scl_EoEtrue showering0, barrel","ele supercluster energy over true energy, showering0, barrel",100,0.2,1.2);
  histSclEoEtrueShowering0_endcaps = new TH1F("h_scl_EoEtrue showering0, endcaps","ele supercluster energy over true energy, showering0, endcaps",100,0.2,1.2);
  histSclEoEtrueShowering1234_barrel = new TH1F("h_scl_EoEtrue showering1234, barrel","ele supercluster energy over true energy, showering1234, barrel",100,0.2,1.2);
  histSclEoEtrueShowering1234_endcaps = new TH1F("h_scl_EoEtrue showering1234, endcaps","ele supercluster energy over true energy, showering1234, endcaps",100,0.2,1.2);

  // fbrem
  h_ele_fbremVsEta_mode = new TProfile( "h_ele_fbremvsEtamode","mean pout/pin vs eta, mode",nbineta2D,etamin,etamax,0.,1.);
  h_ele_fbremVsEta_mean = new TProfile( "h_ele_fbremvsEtamean","mean pout/pin vs eta, mean",nbineta2D,etamin,etamax,0.,1.);
  
  // histos titles
  h_mcNum              -> GetXaxis()-> SetTitle("# true particles");
  h_eleNum             -> GetXaxis()-> SetTitle("# true ele");
  h_gamNum             -> GetXaxis()-> SetTitle("# true gammas");
  h_simEta             -> GetXaxis()-> SetTitle("true #eta");
  h_simP               -> GetXaxis()-> SetTitle("true p (GeV/c)");
  h_ele_foundHits      -> GetXaxis()-> SetTitle("# hits");   
  h_ele_lostHits       -> GetXaxis()-> SetTitle("# lost hits");   
  h_ele_chi2           -> GetXaxis()-> SetTitle("#Chi^{2}");   
  h_ele_charge         -> GetXaxis()-> SetTitle("charge");   
  h_ele_vertexP        -> GetXaxis()-> SetTitle("p_{vertex} (GeV/c)");
  h_ele_vertexPt       -> GetXaxis()-> SetTitle("p_{T vertex} (GeV/c)");
  h_ele_vertexEta      -> GetXaxis()-> SetTitle("#eta");  
  h_ele_vertexPhi      -> GetXaxis()-> SetTitle("#phi");   
  h_ele_PoPtrue        -> GetXaxis()-> SetTitle("P/P_{true}");
  h_ele_EtaMnEtaTrue   -> GetXaxis()-> SetTitle("#eta_{rec} - #eta_{true}");
  h_ele_PhiMnPhiTrue   -> GetXaxis()-> SetTitle("#phi_{rec} - #phi_{true}");
  h_ele_PinMnPout      -> GetXaxis()-> SetTitle("p_{vertex} - p_{out} (GeV)");
  h_ele_PinMnPout_mode      -> GetXaxis()-> SetTitle("p_{vertex} - p_{out}, mode (GeV)");
  h_ele_outerP         -> GetXaxis()-> SetTitle("p_{out} (GeV/c)");
  h_ele_outerP_mode         -> GetXaxis()-> SetTitle("p_{out} (GeV/c)");
  h_ele_outerPt        -> GetXaxis()-> SetTitle("p_{T out} (GeV/c)");
  h_ele_outerPt_mode        -> GetXaxis()-> SetTitle("p_{T out} (GeV/c)");
  h_ele_EoP            -> GetXaxis()-> SetTitle("E/p_{vertex}");
  h_ele_EoPout         -> GetXaxis()-> SetTitle("E/p_{out}");

}     

void
GsfElectronAnalyzer::endJob(){
  
  histfile_->cd();
  std::cout << "efficiency calculation " << std::endl; 
  // efficiency vs eta
  TH1F *h_ele_etaEff = (TH1F*)h_ele_simEta_matched->Clone("h_ele_etaEff");
  h_ele_etaEff->Reset();
  h_ele_etaEff->Divide(h_ele_simEta_matched,h_simEta,1,1);
  h_ele_etaEff->Print();
  h_ele_etaEff->GetXaxis()->SetTitle("#eta");
  h_ele_etaEff->GetYaxis()->SetTitle("eff");
  
  // efficiency vs z
  TH1F *h_ele_zEff = (TH1F*)h_ele_simZ_matched->Clone("h_ele_zEff");
  h_ele_zEff->Reset();
  h_ele_zEff->Divide(h_ele_simZ_matched,h_simZ,1,1);
  h_ele_zEff->Print();
  h_ele_zEff->GetXaxis()->SetTitle("z");
  h_ele_zEff->GetYaxis()->SetTitle("eff");

  // efficiency vs |eta|
  TH1F *h_ele_absetaEff = (TH1F*)h_ele_simAbsEta_matched->Clone("h_ele_absetaEff");
  h_ele_absetaEff->Reset();
  h_ele_absetaEff->Divide(h_ele_simAbsEta_matched,h_simAbsEta,1,1);
  h_ele_absetaEff->GetXaxis()->SetTitle("|#eta|");
  h_ele_absetaEff->GetYaxis()->SetTitle("eff");

  // efficiency vs pt
  TH1F *h_ele_ptEff = (TH1F*)h_ele_simPt_matched->Clone("h_ele_ptEff");
  h_ele_ptEff->Reset();
  h_ele_ptEff->Divide(h_ele_simPt_matched,h_simPt,1,1);
  h_ele_ptEff->GetXaxis()->SetTitle("p_T");
  h_ele_ptEff->GetYaxis()->SetTitle("eff");

  // efficiency vs phi
  TH1F *h_ele_phiEff = (TH1F*)h_ele_simPhi_matched->Clone("h_ele_phiEff");
  h_ele_phiEff->Reset();
  h_ele_phiEff->Divide(h_ele_simPhi_matched,h_simPhi,1,1);
  h_ele_phiEff->GetXaxis()->SetTitle("phi");
  h_ele_phiEff->GetYaxis()->SetTitle("eff");

  // rec/gen all electrons
  TH1F *h_ele_etaEff_all = (TH1F*)h_ele_vertexEta_all->Clone("h_ele_etaEff_all");
  h_ele_etaEff_all->Reset();
  h_ele_etaEff_all->Divide(h_ele_vertexEta_all,h_simEta,1,1);
  h_ele_etaEff_all->Print();
  h_ele_etaEff_all->GetXaxis()->SetTitle("#eta");
  h_ele_etaEff_all->GetYaxis()->SetTitle("rec/gen");
  TH1F *h_ele_ptEff_all = (TH1F*)h_ele_vertexPt_all->Clone("h_ele_ptEff_all");
  h_ele_ptEff_all->Reset();
  h_ele_ptEff_all->Divide(h_ele_vertexPt_all,h_simPt,1,1);
  h_ele_ptEff_all->Print();
  h_ele_ptEff_all->GetXaxis()->SetTitle("p_{T}");
  h_ele_ptEff_all->GetYaxis()->SetTitle("rec/gen");

  // classes
  TH1F *h_ele_eta_goldenFrac = (TH1F*)h_ele_eta_golden->Clone("h_ele_eta_goldenFrac");
  h_ele_eta_goldenFrac->Reset();
  h_ele_eta_goldenFrac->Divide(h_ele_eta_golden,h_ele_eta,1,1);
  TH1F *h_ele_eta_bbremFrac = (TH1F*)h_ele_eta_bbrem->Clone("h_ele_eta_bbremFrac");
  h_ele_eta_bbremFrac->Reset();
  h_ele_eta_bbremFrac->Divide(h_ele_eta_bbrem,h_ele_eta,1,1);
  TH1F *h_ele_eta_narrowFrac = (TH1F*)h_ele_eta_narrow->Clone("h_ele_eta_narrowFrac");
  h_ele_eta_narrowFrac->Reset();
  h_ele_eta_narrowFrac->Divide(h_ele_eta_narrow,h_ele_eta,1,1);
  TH1F *h_ele_eta_showerFrac = (TH1F*)h_ele_eta_shower->Clone("h_ele_eta_showerFrac");
  h_ele_eta_showerFrac->Reset();
  h_ele_eta_showerFrac->Divide(h_ele_eta_shower,h_ele_eta,1,1);
  
  // fbrem
  TH1F *h_ele_xOverX0VsEta = new TH1F( "h_ele_xOverx0VsEta","mean X/X_0 vs eta",nbineta/2,0.0,2.5);
  for (int ibin=1;ibin<h_ele_fbremVsEta_mean->GetNbinsX()+1;ibin++) {
    double xOverX0 = 0.;
    if (h_ele_fbremVsEta_mean->GetBinContent(ibin)>0.) xOverX0 = -log(h_ele_fbremVsEta_mean->GetBinContent(ibin));
    h_ele_xOverX0VsEta->SetBinContent(ibin,xOverX0);
  }
   
  //profiles from 2D histos
  TProfile *p_ele_PoPtrueVsEta = h_ele_PoPtrueVsEta->ProfileX();
  p_ele_PoPtrueVsEta->Write();
  TProfile *p_ele_PoPtrueVsPhi = h_ele_PoPtrueVsPhi->ProfileX();
  p_ele_PoPtrueVsPhi->Write();
  TProfile *p_ele_EtaMnEtaTrueVsEta = h_ele_EtaMnEtaTrueVsEta->ProfileX();
  p_ele_EtaMnEtaTrueVsEta->Write();
  TProfile *p_ele_EtaMnEtaTrueVsPhi = h_ele_EtaMnEtaTrueVsPhi->ProfileX();
  p_ele_EtaMnEtaTrueVsPhi->Write();
  TProfile *p_ele_PhiMnPhiTrueVsEta = h_ele_PhiMnPhiTrueVsEta->ProfileX();
  p_ele_PhiMnPhiTrueVsEta->Write();
  TProfile *p_ele_PhiMnPhiTrueVsPhi = h_ele_PhiMnPhiTrueVsPhi->ProfileX();
  p_ele_PhiMnPhiTrueVsPhi->Write();
  TProfile *p_ele_vertexPtVsEta = h_ele_vertexPtVsEta->ProfileX();
  p_ele_vertexPtVsEta->Write();
  TProfile *p_ele_vertexPtVsPhi = h_ele_vertexPtVsPhi->ProfileX();
  p_ele_vertexPtVsPhi->Write();
  TProfile *p_ele_EoPVsEta = h_ele_EoPVsEta->ProfileX();
  p_ele_EoPVsEta->Write();
  TProfile *p_ele_EoPVsPhi = h_ele_EoPVsPhi->ProfileX();
  p_ele_EoPVsPhi->Write();
  TProfile *p_ele_EoPoutVsEta = h_ele_EoPoutVsEta->ProfileX();
  p_ele_EoPoutVsEta->Write();
  TProfile *p_ele_EoPoutVsPhi = h_ele_EoPoutVsPhi->ProfileX();
  p_ele_EoPoutVsPhi->Write();
  TProfile *p_ele_HoEVsEta = h_ele_HoEVsEta->ProfileX();
  p_ele_HoEVsEta->Write();
  TProfile *p_ele_HoEVsPhi = h_ele_HoEVsPhi->ProfileX();
  p_ele_HoEVsPhi->Write();
  TProfile *p_ele_chi2VsEta = h_ele_chi2VsEta->ProfileX();
  p_ele_chi2VsEta->Write();
  TProfile *p_ele_chi2VsPhi = h_ele_chi2VsPhi->ProfileX();
  p_ele_chi2VsPhi->Write();
  TProfile *p_ele_foundHitsVsEta = h_ele_foundHitsVsEta->ProfileX();
  p_ele_foundHitsVsEta->Write();
  TProfile *p_ele_foundHitsVsPhi = h_ele_foundHitsVsPhi->ProfileX();
  p_ele_foundHitsVsPhi->Write();
  TProfile *p_ele_lostHitsVsEta = h_ele_lostHitsVsEta->ProfileX();
  p_ele_lostHitsVsEta->Write();
  TProfile *p_ele_lostHitsVsPhi = h_ele_lostHitsVsPhi->ProfileX();
  p_ele_lostHitsVsPhi->Write();
  
  // mc truth  

  h_mcNum->Write();
  h_eleNum->Write();
  h_gamNum->Write();
    
  // rec event
  
  histNum_->Write();
  
  // mc  
  h_simEta->Write();
  h_simZ->Write();
  h_simAbsEta->Write();
  h_simP->Write();
  h_simPt->Write();

  // ctf tracks
  h_ctf_foundHitsVsEta->Write();
  h_ctf_lostHitsVsEta->Write();
  
  // all electrons  
  h_ele_EoverP_all->Write();
  h_ele_TIP_all->Write();
  h_ele_vertexPt_all->Write();
  h_ele_vertexEta_all->Write();

  // matched electrons
  h_ele_charge->Write();
  h_ele_chargeVsEta->Write();
  h_ele_chargeVsPhi->Write();
  h_ele_chargeVsPt->Write();
  h_ele_vertexP->Write();
  h_ele_vertexPt->Write();
  h_ele_vertexPtVsEta->Write();
  h_ele_vertexPtVsPhi->Write();
  h_ele_simPt_matched->Write();
  h_ele_vertexEta->Write();
  h_ele_vertexEtaVsPhi->Write();
  h_ele_simAbsEta_matched->Write();
  h_ele_simEta_matched->Write();
  h_ele_simZ_matched->Write();
  h_ele_vertexPhi->Write();
  h_ele_vertexX->Write();
  h_ele_vertexY ->Write();
  h_ele_vertexZ->Write();
  h_ele_vertexTIP->Write();
  h_ele_simZ_matched->Write();
  h_ele_vertexTIPVsEta->Write();
  h_ele_vertexTIPVsPhi->Write();
  h_ele_vertexTIPVsPt->Write();
  h_ele_PoPtrue->Write();
  h_ele_PoPtrueVsEta ->Write();
  h_ele_PoPtrueVsPhi->Write();
  h_ele_PoPtrueVsPt->Write();
  h_ele_PoPtrue_barrel ->Write();
  h_ele_PoPtrue_endcaps->Write();
  h_ele_EtaMnEtaTrue->Write();
  h_ele_EtaMnEtaTrueVsEta ->Write();
  h_ele_EtaMnEtaTrueVsPhi->Write();
  h_ele_EtaMnEtaTrueVsPt->Write();
  h_ele_PhiMnPhiTrue ->Write();
  h_ele_PhiMnPhiTrue2 ->Write();
  h_ele_PhiMnPhiTrueVsEta->Write();
  h_ele_PhiMnPhiTrueVsPhi->Write();
  h_ele_PhiMnPhiTrueVsPt->Write();

  // matched electron, superclusters
  histSclEn_->Write();
  histSclEoEtrue_barrel->Write();
  histSclEoEtrue_endcaps->Write();
  histSclEt_->Write();
  histSclEtVsEta_->Write();
  histSclEtVsPhi_->Write();
  histSclEtaVsPhi_ ->Write();
  histSclEta_->Write();
  histSclPhi_->Write();

  // matched electron, gsf tracks
  h_ele_foundHits->Write();
  h_ele_foundHitsVsEta->Write();
  h_ele_foundHitsVsPhi->Write();
  h_ele_foundHitsVsPt->Write();
  h_ctf_foundHits->Write();
  h_ele_lostHits->Write();
  h_ele_lostHitsVsEta->Write();
  h_ele_lostHitsVsPhi->Write();
  h_ele_lostHitsVsPt->Write();
  h_ele_chi2 ->Write();
  h_ele_chi2VsEta ->Write();
  h_ele_chi2VsPhi ->Write();
  h_ele_chi2VsPt->Write();
  h_ele_PinMnPout->Write();
  h_ele_PinMnPout_mode->Write();
  h_ele_PinMnPoutVsEta_mode->Write();
  h_ele_PinMnPoutVsPhi_mode->Write();
  h_ele_PinMnPoutVsPt_mode->Write();
  h_ele_PinMnPoutVsE_mode->Write();
  h_ele_PinMnPoutVsChi2_mode->Write();
  h_ele_outerP ->Write();
  h_ele_outerP_mode->Write();
  h_ele_outerPVsEta_mode->Write();
  h_ele_outerPt->Write();
  h_ele_outerPt_mode ->Write();
  h_ele_outerPtVsEta_mode->Write();
  h_ele_outerPtVsPhi_mode->Write();
  h_ele_outerPtVsPt_mode->Write();
  
  // matched electrons, matching 
  h_ele_EoP ->Write();
  h_ele_EoPVsEta ->Write();
  h_ele_EoPVsPhi->Write();
  h_ele_EoPVsE->Write();
  h_ele_EoPout->Write();
  h_ele_EoPoutVsEta->Write();
  h_ele_EoPoutVsPhi->Write();
  h_ele_EoPoutVsE ->Write();
  h_ele_dEtaSc_propVtx->Write();
  h_ele_dEtaScVsEta_propVtx->Write();
  h_ele_dEtaScVsPhi_propVtx->Write();
  h_ele_dEtaScVsPt_propVtx ->Write();
  h_ele_dPhiSc_propVtx->Write();
  h_ele_dPhiScVsEta_propVtx ->Write();
  h_ele_dPhiScVsPhi_propVtx->Write();
  h_ele_dPhiScVsPt_propVtx->Write();
  h_ele_dEtaCl_propOut->Write();
  h_ele_dEtaClVsEta_propOut->Write();
  h_ele_dEtaClVsPhi_propOut->Write();
  h_ele_dEtaClVsPt_propOut->Write();
  h_ele_dPhiCl_propOut->Write();
  h_ele_dPhiClVsEta_propOut->Write();
  h_ele_dPhiClVsPhi_propOut->Write();
  h_ele_dPhiClVsPt_propOut->Write();
  
  h_ele_HoE->Write();
  h_ele_HoEVsEta->Write();
  h_ele_HoEVsPhi->Write();
  h_ele_HoEVsE->Write();
 
  // classes  
  h_ele_classes->Write();
  h_ele_eta->Write();
  h_ele_eta_golden->Write();
  h_ele_eta_bbrem->Write();
  h_ele_eta_narrow->Write();
  h_ele_eta_shower->Write();
  h_ele_PinVsPoutGolden_mode->Write();
  h_ele_PinVsPoutShowering0_mode->Write();
  h_ele_PinVsPoutShowering1234_mode->Write();
  h_ele_PinVsPoutGolden_mean->Write();
  h_ele_PinVsPoutShowering0_mean->Write();
  h_ele_PinVsPoutShowering1234_mean->Write();
  h_ele_PtinVsPtoutGolden_mode->Write();
  h_ele_PtinVsPtoutShowering0_mode->Write();
  h_ele_PtinVsPtoutShowering1234_mode->Write();
  h_ele_PtinVsPtoutGolden_mean->Write();
  h_ele_PtinVsPtoutShowering0_mean->Write();
  h_ele_PtinVsPtoutShowering1234_mean->Write();
  histSclEoEtrueGolden_barrel->Write();
  histSclEoEtrueGolden_endcaps->Write();
  histSclEoEtrueShowering0_barrel->Write();
  histSclEoEtrueShowering0_endcaps->Write();
  histSclEoEtrueShowering1234_barrel->Write();
  histSclEoEtrueShowering1234_endcaps->Write();

  // fbrem
  h_ele_fbremVsEta_mode->Write();
  h_ele_fbremVsEta_mean->Write();
  h_ele_etaEff->Write();
  h_ele_zEff->Write();
  h_ele_phiEff->Write();
  h_ele_absetaEff->Write();
  h_ele_ptEff->Write();
  h_ele_etaEff_all->Write();
  h_ele_ptEff_all->Write();
  h_ele_eta_goldenFrac->Write();
  h_ele_eta_bbremFrac->Write();
  h_ele_eta_narrowFrac->Write();
  h_ele_eta_showerFrac->Write();
  h_ele_xOverX0VsEta->Write();
  
}

void
GsfElectronAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::cout << "analyzing new event " << std::endl;
  // get electrons
  
  edm::Handle<GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel(electronCollection_,gsfElectrons); 
  edm::LogInfo("")<<"\n\n =================> Treating event "<<iEvent.id()<<" Number of electrons "<<gsfElectrons.product()->size();

  edm::Handle<edm::HepMCProduct> hepMC;
  iEvent.getByLabel(mcTruthCollection_,hepMC);

  histNum_->Fill((*gsfElectrons).size());
  
  // all rec electrons
  for (reco::GsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin();
   gsfIter!=gsfElectrons->end(); gsfIter++){
    // preselect electrons
    if (gsfIter->pt()>maxPt_ || fabs(gsfIter->eta())>maxAbsEta_) continue;
    double d = gsfIter->gsfTrack()->vertex().x()*gsfIter->gsfTrack()->vertex().x()+
     gsfIter->gsfTrack()->vertex().y()*gsfIter->gsfTrack()->vertex().y();
    h_ele_TIP_all     -> Fill( sqrt(d) );
    h_ele_EoverP_all     -> Fill( gsfIter->eSuperClusterOverP() );
    h_ele_vertexEta_all     -> Fill( gsfIter->eta() );
    h_ele_vertexPt_all      -> Fill( gsfIter->pt() );
  }
   
  // association mc-reco
  HepMC::GenParticle* genPc=0;
  const HepMC::GenEvent *myGenEvent = hepMC->GetEvent();
  int mcNum=0, gamNum=0, eleNum=0;
  HepMC::FourVector pAssSim;
      
  for ( HepMC::GenEvent::particle_const_iterator mcIter=myGenEvent->particles_begin(); mcIter != myGenEvent->particles_end(); mcIter++ ) {
    
    // number of mc particles
    mcNum++;

    // counts photons
    if ((*mcIter)->pdg_id() == 22 ){ gamNum++; }       

    // select electrons
    if ( (*mcIter)->pdg_id() == 11 || (*mcIter)->pdg_id() == -11 ){       

      // single primary electrons or electrons from Zs or Ws
      HepMC::GenParticle* mother = 0;
      if ( (*mcIter)->production_vertex() )  {
       if ( (*mcIter)->production_vertex()->particles_begin(HepMC::parents) != 
           (*mcIter)->production_vertex()->particles_end(HepMC::parents))  
            mother = *((*mcIter)->production_vertex()->particles_begin(HepMC::parents));
      } 
      if ( ((mother == 0) || ((mother != 0) && (mother->pdg_id() == 23))
	                  || ((mother != 0) && (mother->pdg_id() == 32))
	                  || ((mother != 0) && (fabs(mother->pdg_id()) == 24)))) {       
   
      genPc=(*mcIter);
      pAssSim = genPc->momentum();

      if (pAssSim.perp()> maxPt_ || fabs(pAssSim.eta())> maxAbsEta_) continue;
//CC@@
// select z
//      if (fabs((*mcIter)->production_vertex()->position().z()/10.)>10.) continue;
      
      // suppress the endcaps
      //if (fabs(pAssSim.eta()) > 1.5) continue;
      // select central z
      //if ( fabs((*mcIter)->production_vertex()->position().z())>50.) continue;
 
      eleNum++;
      h_simEta -> Fill( pAssSim.eta() );
      h_simAbsEta -> Fill( fabs(pAssSim.eta()) );
      h_simP   -> Fill( pAssSim.t() );
      h_simPt   -> Fill( pAssSim.perp() );
      h_simPhi   -> Fill( pAssSim.phi() );
      h_simZ   -> Fill( (*mcIter)->production_vertex()->position().z()/10. );
     	
      // looking for the best matching gsf electron
      bool okGsfFound = false;
      double gsfOkRatio = 999999.;

      // find best matched electron
      reco::GsfElectron bestGsfElectron;
      for (reco::GsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin();
       gsfIter!=gsfElectrons->end(); gsfIter++){
	
	double deltaR = sqrt(pow((gsfIter->eta()-pAssSim.eta()),2) + pow((gsfIter->phi()-pAssSim.phi()),2));
	if ( deltaR < deltaR_ ){
	if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
	(gsfIter->charge() > 0.) ){
	  double tmpGsfRatio = gsfIter->p()/pAssSim.t();
	  if ( fabs(tmpGsfRatio-1) < fabs(gsfOkRatio-1) ) {
	    gsfOkRatio = tmpGsfRatio;
	    bestGsfElectron=*gsfIter;
	    okGsfFound = true;
	  } 
	} 
	} 
      } // loop over rec ele to look for the best one	

      // analysis when the mc track is found
     if (okGsfFound){

	// electron related distributions
	h_ele_charge        -> Fill( bestGsfElectron.charge() );
	h_ele_chargeVsEta        -> Fill( bestGsfElectron.eta(),bestGsfElectron.charge() );
	h_ele_chargeVsPhi        -> Fill( bestGsfElectron.phi(),bestGsfElectron.charge() );
	h_ele_chargeVsPt        -> Fill( bestGsfElectron.pt(),bestGsfElectron.charge() );
	h_ele_vertexP       -> Fill( bestGsfElectron.p() );
	h_ele_vertexPt      -> Fill( bestGsfElectron.pt() );
	h_ele_vertexPtVsEta      -> Fill(  bestGsfElectron.eta(),bestGsfElectron.pt() );
	h_ele_vertexPtVsPhi      -> Fill(  bestGsfElectron.phi(),bestGsfElectron.pt() );
	h_ele_vertexEta     -> Fill( bestGsfElectron.eta() );
	// generated distributions for matched electrons
	h_ele_simPt_matched      -> Fill( pAssSim.perp() );
        h_ele_simPhi_matched   -> Fill( pAssSim.phi() );
	h_ele_simAbsEta_matched     -> Fill( fabs(pAssSim.eta()) );
	h_ele_simEta_matched     -> Fill( pAssSim.eta() );
	h_ele_vertexEtaVsPhi     -> Fill(  bestGsfElectron.phi(),bestGsfElectron.eta() );
	h_ele_vertexPhi     -> Fill( bestGsfElectron.phi() );
	h_ele_vertexX     -> Fill( bestGsfElectron.vertex().x() );
	h_ele_vertexY     -> Fill( bestGsfElectron.vertex().y() );
	h_ele_vertexZ     -> Fill( bestGsfElectron.vertex().z() );
        h_ele_simZ_matched   -> Fill( (*mcIter)->production_vertex()->position().z()/10. );
	double d = bestGsfElectron.gsfTrack()->vertex().x()*bestGsfElectron.gsfTrack()->vertex().x()+
	 bestGsfElectron.gsfTrack()->vertex().y()*bestGsfElectron.gsfTrack()->vertex().y();
	d = sqrt(d); 
	h_ele_vertexTIP     -> Fill( d );
	h_ele_vertexTIPVsEta     -> Fill(  bestGsfElectron.eta(), d );
	h_ele_vertexTIPVsPhi     -> Fill(  bestGsfElectron.phi(), d );
	h_ele_vertexTIPVsPt     -> Fill(  bestGsfElectron.pt(), d );	
	h_ele_EtaMnEtaTrue  -> Fill( bestGsfElectron.eta()-pAssSim.eta());
	h_ele_EtaMnEtaTrueVsEta  -> Fill( bestGsfElectron.eta(), bestGsfElectron.eta()-pAssSim.eta());
	h_ele_EtaMnEtaTrueVsPhi  -> Fill( bestGsfElectron.phi(), bestGsfElectron.eta()-pAssSim.eta());
	h_ele_EtaMnEtaTrueVsPt  -> Fill( bestGsfElectron.pt(), bestGsfElectron.eta()-pAssSim.eta());
	h_ele_PhiMnPhiTrue  -> Fill( bestGsfElectron.phi()-pAssSim.phi());
	h_ele_PhiMnPhiTrue2  -> Fill( bestGsfElectron.phi()-pAssSim.phi());
	h_ele_PhiMnPhiTrueVsEta  -> Fill( bestGsfElectron.eta(), bestGsfElectron.phi()-pAssSim.phi());
	h_ele_PhiMnPhiTrueVsPhi  -> Fill( bestGsfElectron.phi(), bestGsfElectron.phi()-pAssSim.phi());
	h_ele_PhiMnPhiTrueVsPt  -> Fill( bestGsfElectron.pt(), bestGsfElectron.phi()-pAssSim.phi());
	h_ele_PoPtrue       -> Fill( bestGsfElectron.p()/pAssSim.t());
	h_ele_PoPtrueVsEta       -> Fill( bestGsfElectron.eta(), bestGsfElectron.p()/pAssSim.t());
	h_ele_PoPtrueVsPhi       -> Fill( bestGsfElectron.phi(), bestGsfElectron.p()/pAssSim.t());
	h_ele_PoPtrueVsPt       -> Fill( bestGsfElectron.py(), bestGsfElectron.p()/pAssSim.t());
	if (bestGsfElectron.classification() < 100) h_ele_PoPtrue_barrel       -> Fill( bestGsfElectron.p()/pAssSim.t());
	if (bestGsfElectron.classification() >= 100) h_ele_PoPtrue_endcaps       -> Fill( bestGsfElectron.p()/pAssSim.t());

	// supercluster related distributions
	reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();
        histSclEn_->Fill(sclRef->energy());
        double R=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y() +sclRef->z()*sclRef->z());
        double Rt=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y());
        histSclEt_->Fill(sclRef->energy()*(Rt/R));
        histSclEtVsEta_->Fill(sclRef->eta(),sclRef->energy()*(Rt/R));
        histSclEtVsPhi_->Fill(sclRef->phi(),sclRef->energy()*(Rt/R));
        if (bestGsfElectron.classification() < 100)  histSclEoEtrue_barrel->Fill(sclRef->energy()/pAssSim.t());
        if (bestGsfElectron.classification() >= 100)  histSclEoEtrue_endcaps->Fill(sclRef->energy()/pAssSim.t());
        histSclEta_->Fill(sclRef->eta());
        histSclEtaVsPhi_->Fill(sclRef->phi(),sclRef->eta());
        histSclPhi_->Fill(sclRef->phi());
	if (sclRef->getHitsByDetId()[0].subdetId() != sclRef->seed()->getHitsByDetId()[0].subdetId()) 
	 std::cout << "SC mismatch barrel/endcap " << std::endl;
        for (int i=0;i<sclRef->getHitsByDetId().size();i++) if ( sclRef->getHitsByDetId()[i].subdetId() !=  sclRef->getHitsByDetId()[0].subdetId())
	 std::cout << "SC mismatch barrel/endcap " << std::endl;
	         
//	if (bestGsfElectron.classification()%100 >= 30 && bestGsfElectron.classification()%100 <= 34) {
//	std::cout << "first SC Xtal energy is " << sclRef->getHitsByDetId()[0].energy() << std::endl;
//        std::cout << "first seed cluster Xtal energy is " << sclRef->seed()->getHitsByDetId()[0].energy() << std::endl;
//        }
	 
	// track related distributions
	h_ele_foundHits     -> Fill( bestGsfElectron.gsfTrack()->numberOfValidHits() );
	h_ele_foundHitsVsEta     -> Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
	h_ele_foundHitsVsPhi     -> Fill( bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
	h_ele_foundHitsVsPt     -> Fill( bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfValidHits() );
	h_ele_lostHits      -> Fill( bestGsfElectron.gsfTrack()->numberOfLostHits() );
	h_ele_lostHitsVsEta      -> Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->numberOfLostHits() );
	h_ele_lostHitsVsPhi      -> Fill( bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->numberOfLostHits() );
	h_ele_lostHitsVsPt      -> Fill( bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->numberOfLostHits() );
	h_ele_chi2          -> Fill( bestGsfElectron.gsfTrack()->normalizedChi2() );  
	h_ele_chi2VsEta          -> Fill( bestGsfElectron.eta(), bestGsfElectron.gsfTrack()->normalizedChi2() );  
	h_ele_chi2VsPhi          -> Fill( bestGsfElectron.phi(), bestGsfElectron.gsfTrack()->normalizedChi2() );  
	h_ele_chi2VsPt          -> Fill( bestGsfElectron.pt(), bestGsfElectron.gsfTrack()->normalizedChi2() );  
	// from gsf track interface, hence using mean
	h_ele_PinMnPout     -> Fill( bestGsfElectron.gsfTrack()->innerMomentum().R() - bestGsfElectron.gsfTrack()->outerMomentum().R() );
	h_ele_outerP        -> Fill( bestGsfElectron.gsfTrack()->outerMomentum().R() );
	h_ele_outerPt       -> Fill( bestGsfElectron.gsfTrack()->outerMomentum().Rho() );
        // from electron interface, hence using mode
	h_ele_PinMnPout_mode     -> Fill( bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
	h_ele_PinMnPoutVsEta_mode     -> Fill(  bestGsfElectron.eta(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
	h_ele_PinMnPoutVsPhi_mode     -> Fill(  bestGsfElectron.phi(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
	h_ele_PinMnPoutVsPt_mode     -> Fill(  bestGsfElectron.pt(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
	h_ele_PinMnPoutVsE_mode     -> Fill(  bestGsfElectron.caloEnergy(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
	h_ele_PinMnPoutVsChi2_mode     -> Fill(  bestGsfElectron.gsfTrack()->normalizedChi2(), bestGsfElectron.trackMomentumAtVtx().R() - bestGsfElectron.trackMomentumOut().R() );
	h_ele_outerP_mode        -> Fill( bestGsfElectron.trackMomentumOut().R() );
	h_ele_outerPVsEta_mode        -> Fill(bestGsfElectron.eta(),  bestGsfElectron.trackMomentumOut().R() );
	h_ele_outerPt_mode       -> Fill( bestGsfElectron.trackMomentumOut().Rho() );
	h_ele_outerPtVsEta_mode       -> Fill(bestGsfElectron.eta(),  bestGsfElectron.trackMomentumOut().Rho() );
	h_ele_outerPtVsPhi_mode       -> Fill(bestGsfElectron.phi(),  bestGsfElectron.trackMomentumOut().Rho() );
	h_ele_outerPtVsPt_mode       -> Fill(bestGsfElectron.pt(),  bestGsfElectron.trackMomentumOut().Rho() );
	
	// match distributions 
	h_ele_EoP    -> Fill( bestGsfElectron.eSuperClusterOverP() );
	h_ele_EoPVsEta    -> Fill(bestGsfElectron.eta(),  bestGsfElectron.eSuperClusterOverP() );
	h_ele_EoPVsPhi    -> Fill(bestGsfElectron.phi(),  bestGsfElectron.eSuperClusterOverP() );
	h_ele_EoPVsE    -> Fill(bestGsfElectron.caloEnergy(),  bestGsfElectron.eSuperClusterOverP() );
	h_ele_EoPout -> Fill( bestGsfElectron.eSeedClusterOverPout() );
	h_ele_EoPoutVsEta -> Fill( bestGsfElectron.eta(), bestGsfElectron.eSeedClusterOverPout() );
	h_ele_EoPoutVsPhi -> Fill( bestGsfElectron.phi(), bestGsfElectron.eSeedClusterOverPout() );
	h_ele_EoPoutVsE -> Fill( bestGsfElectron.caloEnergy(), bestGsfElectron.eSeedClusterOverPout() );
	h_ele_dEtaSc_propVtx -> Fill(bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
	h_ele_dEtaScVsEta_propVtx -> Fill( bestGsfElectron.eta(),bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
	h_ele_dEtaScVsPhi_propVtx -> Fill(bestGsfElectron.phi(),bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
	h_ele_dEtaScVsPt_propVtx -> Fill(bestGsfElectron.pt(),bestGsfElectron.deltaEtaSuperClusterTrackAtVtx());
	h_ele_dPhiSc_propVtx -> Fill(bestGsfElectron.deltaPhiSuperClusterTrackAtVtx()); 
	h_ele_dPhiScVsEta_propVtx -> Fill( bestGsfElectron.eta(),bestGsfElectron.deltaPhiSuperClusterTrackAtVtx()); 
	h_ele_dPhiScVsPhi_propVtx -> Fill(bestGsfElectron.phi(),bestGsfElectron.deltaPhiSuperClusterTrackAtVtx()); 
	h_ele_dPhiScVsPt_propVtx -> Fill(bestGsfElectron.pt(),bestGsfElectron.deltaPhiSuperClusterTrackAtVtx()); 
	h_ele_dEtaCl_propOut -> Fill(bestGsfElectron.deltaEtaSeedClusterTrackAtCalo()); 
	h_ele_dEtaClVsEta_propOut -> Fill( bestGsfElectron.eta(),bestGsfElectron.deltaEtaSeedClusterTrackAtCalo()); 
	h_ele_dEtaClVsPhi_propOut -> Fill(bestGsfElectron.phi(),bestGsfElectron.deltaEtaSeedClusterTrackAtCalo()); 
	h_ele_dEtaClVsPt_propOut -> Fill(bestGsfElectron.pt(),bestGsfElectron.deltaEtaSeedClusterTrackAtCalo()); 
	h_ele_dPhiCl_propOut -> Fill(bestGsfElectron.deltaPhiSeedClusterTrackAtCalo()); 
	h_ele_dPhiClVsEta_propOut -> Fill( bestGsfElectron.eta(),bestGsfElectron.deltaPhiSeedClusterTrackAtCalo()); 
	h_ele_dPhiClVsPhi_propOut -> Fill(bestGsfElectron.phi(),bestGsfElectron.deltaPhiSeedClusterTrackAtCalo()); 
	h_ele_dPhiClVsPt_propOut -> Fill(bestGsfElectron.pt(),bestGsfElectron.deltaPhiSeedClusterTrackAtCalo()); 
	h_ele_HoE -> Fill(bestGsfElectron.hadronicOverEm());
	h_ele_HoEVsEta -> Fill( bestGsfElectron.eta(),bestGsfElectron.hadronicOverEm());
	h_ele_HoEVsPhi -> Fill(bestGsfElectron.phi(),bestGsfElectron.hadronicOverEm());
	h_ele_HoEVsE -> Fill(bestGsfElectron.caloEnergy(),bestGsfElectron.hadronicOverEm());
	 
	//classes
	int eleClass = bestGsfElectron.classification();
	h_ele_classes ->Fill(eleClass);	
/*
        if (bestGsfElectron.classification() == 0)  histSclEoEtrueGolden_barrel->Fill(sclRef->energy()/pAssSim.t());
        if (bestGsfElectron.classification() == 100)  histSclEoEtrueGolden_endcaps->Fill(sclRef->energy()/pAssSim.t());
        if (bestGsfElectron.classification() == 30)  histSclEoEtrueShowering0_barrel->Fill(sclRef->energy()/pAssSim.t());
        if (bestGsfElectron.classification() == 130)  histSclEoEtrueShowering0_endcaps->Fill(sclRef->energy()/pAssSim.t());
        if (bestGsfElectron.classification() == 31 || bestGsfElectron.classification() == 32  || bestGsfElectron.classification() == 33 || eleClass == 34 )  histSclEoEtrueShowering1234_barrel->Fill(sclRef->energy()/pAssSim.t());
        if (bestGsfElectron.classification() == 131 || bestGsfElectron.classification() == 132 || bestGsfElectron.classification() == 133 || eleClass == 134 )  histSclEoEtrueShowering1234_endcaps->Fill(sclRef->energy()/pAssSim.t());
*/
	eleClass = eleClass%100; // get rid of barrel/endcap distinction
        h_ele_eta->Fill(fabs(bestGsfElectron.eta()));
        if (eleClass == 0) h_ele_eta_golden ->Fill(fabs(bestGsfElectron.eta()));
        if (eleClass == 10) h_ele_eta_bbrem ->Fill(fabs(bestGsfElectron.eta()));
        if (eleClass == 20) h_ele_eta_narrow ->Fill(fabs(bestGsfElectron.eta()));
        if (eleClass == 30 || eleClass == 31 || eleClass == 32  || eleClass == 33 || eleClass == 34 ) h_ele_eta_shower ->Fill(fabs(bestGsfElectron.eta()));

	//fbrem 
	double fbrem_mean =  bestGsfElectron.gsfTrack()->outerMomentum().R()/bestGsfElectron.gsfTrack()->innerMomentum().R();
	double fbrem_mode =  bestGsfElectron.trackMomentumOut().R()/bestGsfElectron.trackMomentumAtVtx().R();
	h_ele_fbremVsEta_mode->Fill(bestGsfElectron.eta(),fbrem_mode);
	h_ele_fbremVsEta_mean->Fill(bestGsfElectron.eta(),fbrem_mean);
 
        if (eleClass == 0) h_ele_PinVsPoutGolden_mode -> Fill(bestGsfElectron.trackMomentumOut().R(), bestGsfElectron.trackMomentumAtVtx().R());
        if (eleClass == 30)
	 h_ele_PinVsPoutShowering0_mode -> Fill(bestGsfElectron.trackMomentumOut().R(), bestGsfElectron.trackMomentumAtVtx().R());
        if (eleClass == 31 || eleClass == 32  || eleClass == 33 || eleClass == 34 )
	 h_ele_PinVsPoutShowering1234_mode -> Fill(bestGsfElectron.trackMomentumOut().R(), bestGsfElectron.trackMomentumAtVtx().R());
	if (eleClass == 0) h_ele_PinVsPoutGolden_mean -> Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(), bestGsfElectron.gsfTrack()->innerMomentum().R());
        if (eleClass == 30)
	 h_ele_PinVsPoutShowering0_mean ->  Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(), bestGsfElectron.gsfTrack()->innerMomentum().R());
        if (eleClass == 31 || eleClass == 32  || eleClass == 33 || eleClass == 34 )
	 h_ele_PinVsPoutShowering1234_mean ->  Fill(bestGsfElectron.gsfTrack()->outerMomentum().R(), bestGsfElectron.gsfTrack()->innerMomentum().R());
        if (eleClass == 0) h_ele_PtinVsPtoutGolden_mode -> Fill(bestGsfElectron.trackMomentumOut().Rho(), bestGsfElectron.trackMomentumAtVtx().Rho());
        if (eleClass == 30 )
	 h_ele_PtinVsPtoutShowering0_mode -> Fill(bestGsfElectron.trackMomentumOut().Rho(), bestGsfElectron.trackMomentumAtVtx().Rho());
        if (eleClass == 31 || eleClass == 32  || eleClass == 33 || eleClass == 34 )
	 h_ele_PtinVsPtoutShowering1234_mode -> Fill(bestGsfElectron.trackMomentumOut().Rho(), bestGsfElectron.trackMomentumAtVtx().Rho());
	if (eleClass == 0) h_ele_PtinVsPtoutGolden_mean -> Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(), bestGsfElectron.gsfTrack()->innerMomentum().Rho());
        if (eleClass == 30 )
	 h_ele_PtinVsPtoutShowering0_mean ->  Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(), bestGsfElectron.gsfTrack()->innerMomentum().Rho());
        if (eleClass == 31 || eleClass == 32  || eleClass == 33 || eleClass == 34 )
	 h_ele_PtinVsPtoutShowering1234_mean ->  Fill(bestGsfElectron.gsfTrack()->outerMomentum().Rho(), bestGsfElectron.gsfTrack()->innerMomentum().Rho());

      } // gsf electron found

    } // mc particle found

    }

  } // loop over mc particle 
  
  h_mcNum->Fill(mcNum);
  h_eleNum->Fill(eleNum);

}


