// -*- C++ -*-
//
// Package:    RecoEgamma/Examples
// Class:      GsfElectronFakeAnalyzer
// 
/**\class GsfElectronFakeAnalyzer RecoEgamma/Examples/src/GsfElectronFakeAnalyzer.cc

 Description: GsfElectrons fake electrons analyzer using reco data

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: GsfElectronFakeAnalyzer.cc,v 1.6 2008/03/15 14:41:40 charlot Exp $
//
//

// user include files
#include "RecoEgamma/Examples/plugins/GsfElectronFakeAnalyzer.h"
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
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

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
 
GsfElectronFakeAnalyzer::GsfElectronFakeAnalyzer(const edm::ParameterSet& conf)
{

  outputFile_ = conf.getParameter<std::string>("outputFile");
  histfile_ = new TFile(outputFile_.c_str(),"RECREATE");
  electronCollection_=conf.getParameter<edm::InputTag>("electronCollection");
  matchingObjectCollection_ = conf.getParameter<edm::InputTag>("matchingObjectCollection");
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
  
GsfElectronFakeAnalyzer::~GsfElectronFakeAnalyzer()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  histfile_->Write();
  histfile_->Close();
}

void GsfElectronFakeAnalyzer::beginJob(edm::EventSetup const&iSetup){

  histfile_->cd();
  
  
  // matching object
  std::string::size_type locJet = matchingObjectCollection_.label().find( "iterativeCone5CaloJets", 0 );
  std::string type_;
  if ( locJet != std::string::npos ) {
    std::cout << "Matching objects are CaloJets " << std::endl;
    type_ = "CaloJet";
  } else {
    std::cout << "Didn't recognize input matching objects!! " << std::endl; 
  }
  
  std::string htitle, hlabel;
  hlabel="h_"+type_+"Num"; htitle="# "+type_+"s";
  h_matchingObjectNum              = new TH1F( hlabel.c_str(), htitle.c_str(),    nbinfhits,0.,fhitsmax );
    
  // rec event
  
  histNum_= new TH1F("h_recEleNum","# rec electrons",20, 0.,20.);
  
  // matching object distributions 
  hlabel="h_"+type_+"_eta"; htitle=type_+" #eta";
  h_matchingObjectEta             = new TH1F( hlabel.c_str(), htitle.c_str(), nbineta,etamin,etamax); 
  hlabel="h_"+type_+"_ahseta"; htitle=type_+" |#eta|";
  h_matchingObjectAbsEta             = new TH1F( hlabel.c_str(), htitle.c_str(), nbineta/2,0.,etamax); 
  hlabel="h_"+type_+"_P"; htitle=type_+" p";
  h_matchingObjectP               = new TH1F( hlabel.c_str(), htitle.c_str(),              nbinp,0.,pmax); 
  hlabel="h_"+type_+"_Pt"; htitle=type_+" pt";
  h_matchingObjectPt               = new TH1F( hlabel.c_str(),htitle.c_str(),            nbinpteff,5.,ptmax); 
  hlabel="h_"+type_+"_phi"; htitle=type_+" phi";
  h_matchingObjectPhi               = new TH1F( hlabel.c_str(), htitle.c_str(),        nbinphi,phimin,phimax); 
  hlabel="h_"+type_+"_z"; htitle=type_+" z";
  h_matchingObjectZ      = new TH1F( hlabel.c_str(), htitle.c_str(),    50, -25, 25 );

  // ctf tracks
  h_ctf_foundHitsVsEta      = new TH2F( "h_ctf_foundHitsVsEta",      "ctf track # found hits vs eta",  nbineta2D,etamin,etamax,nbinfhits,0.,fhitsmax);
  h_ctf_lostHitsVsEta       = new TH2F( "h_ctf_lostHitsVsEta",       "ctf track # lost hits vs eta",   nbineta2D,etamin,etamax,nbinlhits,0.,lhitsmax);
  
  // all electrons  
  h_ele_EoverP_all       = new TH1F( "h_ele_EoverP_all",       "all reco ele, E/p at vertex",  nbineop,0.,eopmax);
  h_ele_TIP_all       = new TH1F( "h_ele_TIP_all",       "all reco ele, tip at vertex",  100,0.,0.2);
  h_ele_vertexPt_all       = new TH1F( "h_ele_vertexPt_all",       "all reco ele, p_{T} at vertex",  nbinpteff,5.,ptmax);
  h_ele_vertexEta_all      = new TH1F( "h_ele_vertexEta_all",      "all reco ele, #eta at vertex",    nbineta,etamin,etamax);

  // matched electrons
  h_ele_charge         = new TH1F( "h_ele_charge",         "ele charge",             5,-2.,2.);   
  h_ele_chargeVsEta    = new TH2F( "h_ele_chargeVsEta",         "ele charge vs eta", nbineta2D,etamin,etamax,5,-2.,2.);   
  h_ele_chargeVsPhi    = new TH2F( "h_ele_chargeVsPhi",         "ele charge vs phi", nbinphi2D,phimin,phimax,5,-2.,2.);   
  h_ele_chargeVsPt    = new TH2F( "h_ele_chargeVsPt",         "ele charge vs pt", nbinpt,0.,100.,5,-2.,2.);   
  h_ele_vertexP        = new TH1F( "h_ele_vertexP",        "ele p at vertex",       nbinp,0.,pmax);
  h_ele_vertexPt       = new TH1F( "h_ele_vertexPt",       "ele p_{T} at vertex",  nbinpt,0.,ptmax);
  h_ele_vertexPtVsEta   = new TH2F( "h_ele_vertexPtVsEta",       "ele p_{T} at vertex vs eta",nbinpt2D,etamin,etamax,nbinpt2D,0.,ptmax);
  h_ele_vertexPtVsPhi   = new TH2F( "h_ele_vertexPtVsPhi",       "ele p_{T} at vertex vs phi",nbinphi2D,phimin,phimax,nbinpt2D,0.,ptmax);
  h_ele_matchingObjectPt_matched       = new TH1F( "h_ele_matchingObjectPt_matched",       "matching jet p_{T}",  nbinpteff,5.,ptmax);
  h_ele_vertexEta      = new TH1F( "h_ele_vertexEta",      "ele #eta at vertex",    nbineta,etamin,etamax);
  h_ele_vertexEtaVsPhi  = new TH2F( "h_ele_vertexEtaVsPhi",      "ele #eta at vertex vs phi",nbineta2D,etamin,etamax,nbinphi2D,phimin,phimax );
  h_ele_matchingObjectAbsEta_matched      = new TH1F( "h_ele_matchingObjectAbsEta_matched",      "matching jet |#eta|",    nbineta/2,0.,2.5);
  h_ele_matchingObjectEta_matched      = new TH1F( "h_ele_matchingObjectEta_matched",      "matching jet #eta",    nbineta,etamin,etamax);
  h_ele_matchingObjectPhi_matched               = new TH1F( "h_ele_matchingObjectPhi_matched", "matching jet phi",        nbinphi,phimin,phimax); 
  h_ele_vertexPhi      = new TH1F( "h_ele_vertexPhi",      "ele #phi at vertex",    nbinphi,phimin,phimax);
  h_ele_vertexX      = new TH1F( "h_ele_vertexX",      "ele x at vertex",    nbinxyz,-0.1,0.1 );
  h_ele_vertexY      = new TH1F( "h_ele_vertexY",      "ele y at vertex",    nbinxyz,-0.1,0.1 );
  h_ele_vertexZ      = new TH1F( "h_ele_vertexZ",      "ele z at vertex",    nbinxyz,-25, 25 );
  h_ele_matchingObjectZ_matched      = new TH1F( "h_ele_matchingObjectZ_matched",      "matching jet z",    nbinxyz,-25,25);
  h_ele_vertexTIP      = new TH1F( "h_ele_vertexTIP",      "ele TIP",    90,0.,0.15);
  h_ele_vertexTIPVsEta      = new TH2F( "h_ele_vertexTIPVsEta",      "ele TIP vs eta", nbineta2D,etamin,etamax,45,0.,0.15);
  h_ele_vertexTIPVsPhi      = new TH2F( "h_ele_vertexTIPVsPhi",      "ele TIP vs phi", nbinphi2D,phimin,phimax,45,0.,0.15);
  h_ele_vertexTIPVsPt      = new TH2F( "h_ele_vertexTIPVsPt",      "ele TIP vs Pt", nbinpt2D,0.,ptmax,45,0.,0.15);
  h_ele_PoPmatchingObject        = new TH1F( "h_ele_PoPmatchingObject",        "ele P/P_{matching jet} @ vertex", 75,0.,1.5);
  h_ele_PoPmatchingObjectVsEta   = new TH2F( "h_ele_PoPmatchingObjectVsEta",        "ele P/P_{matching jet} @ vertex vs eta", nbineta2D,etamin,etamax,50,0.,1.5);
  h_ele_PoPmatchingObjectVsPhi   = new TH2F( "h_ele_PoPmatchingObjectVsPhi",        "ele P/P_{matching jet} @ vertex vs phi", nbinphi2D,phimin,phimax,50,0.,1.5);
  h_ele_PoPmatchingObjectVsPt   = new TH2F( "h_ele_PoPmatchingObjectVsPt",        "ele P/P_{matching jet} @ vertex vs eta", nbinpt2D,0.,ptmax,50,0.,1.5);
  h_ele_PoPmatchingObject_barrel         = new TH1F( "h_ele_PoPmatchingObject_barrel",        "ele P/P_{matching jet} @ vertex, barrel",75,0.,1.5);
  h_ele_PoPmatchingObject_endcaps        = new TH1F( "h_ele_PoPmatchingObject_endcaps",        "ele P/P_{matching jet} @ vertex, endcaps",75,0.,1.5);
  h_ele_EtaMnEtamatchingObject   = new TH1F( "h_ele_EtaMnEtamatchingObject",   "ele #eta_{rec} - #eta_{matching jet} @ vertex",nbindeta,detamin,detamax);
  h_ele_EtaMnEtamatchingObjectVsEta   = new TH2F( "h_ele_EtaMnEtamatchingObjectVsEta",   "ele #eta_{rec} - #eta_{matching jet} @ vertex vs eta",nbineta2D,etamin,etamax,nbindeta/2,detamin,detamax);
  h_ele_EtaMnEtamatchingObjectVsPhi   = new TH2F( "h_ele_EtaMnEtamatchingObjectVsPhi",   "ele #eta_{rec} - #eta_{matching jet} @ vertex vs phi",nbinphi2D,phimin,phimax,nbindeta/2,detamin,detamax);
  h_ele_EtaMnEtamatchingObjectVsPt   = new TH2F( "h_ele_EtaMnEtamatchingObjectVsPt",   "ele #eta_{rec} - #eta_{matching jet} @ vertex vs pt",nbinpt,0.,ptmax,nbindeta/2,detamin,detamax);
  h_ele_PhiMnPhimatchingObject   = new TH1F( "h_ele_PhiMnPhimatchingObject",   "ele #phi_{rec} - #phi_{matching jet} @ vertex",nbindphi,dphimin,dphimax);
  h_ele_PhiMnPhimatchingObject2   = new TH1F( "h_ele_PhiMnPhimatchingObject2",   "ele #phi_{rec} - #phi_{matching jet} @ vertex",nbindphimatch2D,dphimatchmin,dphimatchmax);
  h_ele_PhiMnPhimatchingObjectVsEta   = new TH2F( "h_ele_PhiMnPhimatchingObjectVsEta",   "ele #phi_{rec} - #phi_{matching jet} @ vertex vs eta",nbineta2D,etamin,etamax,nbindphi/2,dphimin,dphimax);
  h_ele_PhiMnPhimatchingObjectVsPhi   = new TH2F( "h_ele_PhiMnPhimatchingObjectVsPhi",   "ele #phi_{rec} - #phi_{matching jet} @ vertex vs phi",nbinphi2D,phimin,phimax,nbindphi/2,dphimin,dphimax);
  h_ele_PhiMnPhimatchingObjectVsPt   = new TH2F( "h_ele_PhiMnPhimatchingObjectVsPt",   "ele #phi_{rec} - #phi_{matching jet} @ vertex vs pt",nbinpt2D,0.,ptmax,nbindphi/2,dphimin,dphimax);

  // matched electron, superclusters
  histSclEn_ = new TH1F("h_scl_energy","ele supercluster energy",nbinp,0.,pmax);
  histSclEoEmatchingObject_barrel = new TH1F("h_scl_EoEmatchingObject_barrel","ele SC energy over matching jet energy, barrel",50,0.2,1.2);
  histSclEoEmatchingObject_endcaps = new TH1F("h_scl_EoEmatchingObject_endcaps","ele SC energy over matching jet energy, endcaps",50,0.2,1.2);
  histSclEt_ = new TH1F("h_scl_et","ele SC transverse energy",nbinpt,0.,ptmax);
  histSclEtVsEta_ = new TH2F("h_scl_etVsEta","ele SC transverse energy vs eta",nbineta2D,etamin,etamax,nbinpt,0.,ptmax);
  histSclEtVsPhi_ = new TH2F("h_scl_etVsPhi","ele SC transverse energy vs phi",nbinphi2D,phimin,phimax,nbinpt,0.,ptmax);
  histSclEtaVsPhi_ = new TH2F("h_scl_etaVsPhi","ele SC eta vs phi",nbinphi2D,phimin,phimax,nbineta2D,etamin,etamax);
  histSclEta_ = new TH1F("h_scl_eta","ele SC eta",nbineta,etamin,etamax);
  histSclPhi_ = new TH1F("h_scl_phi","ele SC phi",nbinphi,phimin,phimax);

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
  
  // matched electrons,cluster-track  matching 
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
  
  h_ele_HoE = new TH1F("h_ele_HoE", "ele, H/E", 100,-0.5,0.5) ;
  h_ele_HoEVsEta = new TH2F("h_ele_HoEVsEta", "ele, H/E vs eta", nbineta,etamin,etamax,50,-0.5,0.5) ;
  h_ele_HoEVsPhi = new TH2F("h_ele_HoEVsPhi", "ele, H/E vs phi", nbinphi2D,phimin,phimax,50,-0.5,0.5) ;
  h_ele_HoEVsE = new TH2F("h_ele_HoEVsE", "ele, H/E vs E", nbinp, 0.,300.,50,0.,1.) ;
 
  // classes  
  h_ele_classes = new TH1F( "h_ele_classes", "ele, electron classes",      150,0.0,150.);
  h_ele_eta = new TH1F( "h_ele_eta", "ele, electron eta",  nbineta/2,0.0,etamax);
  h_ele_eta_golden = new TH1F( "h_ele_eta_golden", "ele, electron eta golden",  nbineta/2,0.0,etamax);
  h_ele_eta_bbrem = new TH1F( "h_ele_eta_bbrem", "ele, electron eta bbrem",  nbineta/2,0.0,etamax);
  h_ele_eta_narrow = new TH1F( "h_ele_eta_narrow", "ele, electron eta narrow",  nbineta/2,0.0,etamax);
  h_ele_eta_shower = new TH1F( "h_ele_eta_show", "ele, electron eta showering",  nbineta/2,0.0,etamax);
  h_ele_PinVsPoutGolden_mode = new TH2F( "h_ele_PinVsPoutGolden_mode",      "ele track inner p vs outer p, golden, mode" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutShowering0_mode = new TH2F( "h_ele_PinVsPoutShowering0_mode",      "ele track inner p vs outer p vs eta, showering0, mode" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutShowering1234_mode = new TH2F( "h_ele_PinVsPoutShowering1234_mode",      "ele track inner p vs outer p, showering1234, mode" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutGolden_mean = new TH2F( "h_ele_PinVsPoutGolden_mean",      "ele track inner p vs outer p, golden, mean" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutShowering0_mean = new TH2F( "h_ele_PinVsPoutShowering0_mean",      "ele track inner p vs outer p, showering0, mean" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PinVsPoutShowering1234_mean = new TH2F( "h_ele_PinVsPoutShowering1234_mean",      "ele track inner p vs outer p, showering1234, mean" ,nbinp2D,0.,pmax,50,0.,pmax);
  h_ele_PtinVsPtoutGolden_mode = new TH2F( "h_ele_PtinVsPtoutGolden_mode",      "ele track inner pt vs outer pt, golden, mode" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutShowering0_mode = new TH2F( "h_ele_PtinVsPtoutShowering0_mode",      "ele track inner pt vs outer pt, showering0, mode" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutShowering1234_mode = new TH2F( "h_ele_PtinVsPtoutShowering1234_mode",      "ele track inner pt vs outer pt, showering1234, mode" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutGolden_mean = new TH2F( "h_ele_PtinVsPtoutGolden_mean",      "ele track inner pt vs outer pt, golden, mean" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutShowering0_mean = new TH2F( "h_ele_PtinVsPtoutShowering0_mean",      "ele track inner pt vs outer pt, showering0, mean" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  h_ele_PtinVsPtoutShowering1234_mean = new TH2F( "h_ele_PtinVsPtoutShowering1234_mean",      "ele track inner pt vs outer pt, showering1234, mean" ,nbinpt2D,0.,ptmax,50,0.,ptmax);
  histSclEoEmatchingObjectGolden_barrel = new TH1F("h_scl_EoEmatchingObject golden, barrel","ele SC energy over matching jet energy, golden, barrel",100,0.2,1.2);
  histSclEoEmatchingObjectGolden_endcaps = new TH1F("h_scl_EoEmatchingObject golden, endcaps","ele SC energy over matching jet energy, golden, endcaps",100,0.2,1.2);
  histSclEoEmatchingObjectShowering0_barrel = new TH1F("h_scl_EoEmatchingObject showering0, barrel","ele SC energy over matching jet energy, showering0, barrel",100,0.2,1.2);
  histSclEoEmatchingObjectShowering0_endcaps = new TH1F("h_scl_EoEmatchingObject showering0, endcaps","ele SC energy over matching jet energy, showering0, endcaps",100,0.2,1.2);
  histSclEoEmatchingObjectShowering1234_barrel = new TH1F("h_scl_EoEmatchingObject showering1234, barrel","ele SC energy over matching jet energy, showering1234, barrel",100,0.2,1.2);
  histSclEoEmatchingObjectShowering1234_endcaps = new TH1F("h_scl_EoEmatchingObject showering1234, endcaps","ele SC over matchingObject energy, showering1234, endcaps",100,0.2,1.2);

  // fbrem
  h_ele_fbremVsEta_mode = new TProfile( "h_ele_fbremvsEtamode","ele, mean pout/pin vs eta, mode",nbineta2D,etamin,etamax,0.,1.);
  h_ele_fbremVsEta_mean = new TProfile( "h_ele_fbremvsEtamean","ele, mean pout/pin vs eta, mean",nbineta2D,etamin,etamax,0.,1.);
  
  // histos titles
  h_matchingObjectNum              -> GetXaxis()-> SetTitle("# reco jets");
  h_matchingObjectEta             -> GetXaxis()-> SetTitle("jet #eta");
  h_matchingObjectP               -> GetXaxis()-> SetTitle("jet p (GeV/c)");
  h_ele_foundHits      -> GetXaxis()-> SetTitle("# hits");   
  h_ele_lostHits       -> GetXaxis()-> SetTitle("# lost hits");   
  h_ele_chi2           -> GetXaxis()-> SetTitle("#Chi^{2}");   
  h_ele_charge         -> GetXaxis()-> SetTitle("charge");   
  h_ele_vertexP        -> GetXaxis()-> SetTitle("p_{vertex} (GeV/c)");
  h_ele_vertexPt       -> GetXaxis()-> SetTitle("p_{T vertex} (GeV/c)");
  h_ele_vertexEta      -> GetXaxis()-> SetTitle("#eta");  
  h_ele_vertexPhi      -> GetXaxis()-> SetTitle("#phi");   
  h_ele_PoPmatchingObject        -> GetXaxis()-> SetTitle("P/P_{jet}");
  h_ele_EtaMnEtamatchingObject   -> GetXaxis()-> SetTitle("#eta_{rec} - #eta_{jet}");
  h_ele_PhiMnPhimatchingObject   -> GetXaxis()-> SetTitle("#phi_{rec} - #phi_{jet}");
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
GsfElectronFakeAnalyzer::endJob(){
  
  histfile_->cd();
  std::cout << "efficiency calculation " << std::endl; 
  // efficiency vs eta
  TH1F *h_ele_etaEff = (TH1F*)h_ele_matchingObjectEta_matched->Clone("h_ele_etaEff");
  h_ele_etaEff->Reset();
  h_ele_etaEff->Divide(h_ele_matchingObjectEta_matched,h_matchingObjectEta,1,1);
  h_ele_etaEff->Print();
  h_ele_etaEff->GetXaxis()->SetTitle("#eta");
  h_ele_etaEff->GetYaxis()->SetTitle("eff");
  
  // efficiency vs z
  TH1F *h_ele_zEff = (TH1F*)h_ele_matchingObjectZ_matched->Clone("h_ele_zEff");
  h_ele_zEff->Reset();
  h_ele_zEff->Divide(h_ele_matchingObjectZ_matched,h_matchingObjectZ,1,1);
  h_ele_zEff->Print();
  h_ele_zEff->GetXaxis()->SetTitle("z");
  h_ele_zEff->GetYaxis()->SetTitle("eff");

  // efficiency vs |eta|
  TH1F *h_ele_absetaEff = (TH1F*)h_ele_matchingObjectAbsEta_matched->Clone("h_ele_absetaEff");
  h_ele_absetaEff->Reset();
  h_ele_absetaEff->Divide(h_ele_matchingObjectAbsEta_matched,h_matchingObjectAbsEta,1,1);
  h_ele_absetaEff->GetXaxis()->SetTitle("|#eta|");
  h_ele_absetaEff->GetYaxis()->SetTitle("eff");

  // efficiency vs pt
  TH1F *h_ele_ptEff = (TH1F*)h_ele_matchingObjectPt_matched->Clone("h_ele_ptEff");
  h_ele_ptEff->Reset();
  h_ele_ptEff->Divide(h_ele_matchingObjectPt_matched,h_matchingObjectPt,1,1);
  h_ele_ptEff->GetXaxis()->SetTitle("p_T");
  h_ele_ptEff->GetYaxis()->SetTitle("eff");

  // efficiency vs phi
  TH1F *h_ele_phiEff = (TH1F*)h_ele_matchingObjectPhi_matched->Clone("h_ele_phiEff");
  h_ele_phiEff->Reset();
  h_ele_phiEff->Divide(h_ele_matchingObjectPhi_matched,h_matchingObjectPhi,1,1);
  h_ele_phiEff->GetXaxis()->SetTitle("phi");
  h_ele_phiEff->GetYaxis()->SetTitle("eff");

  // rec/matching objects all electrons
  TH1F *h_ele_etaEff_all = (TH1F*)h_ele_vertexEta_all->Clone("h_ele_etaEff_all");
  h_ele_etaEff_all->Reset();
  h_ele_etaEff_all->Divide(h_ele_vertexEta_all,h_matchingObjectEta,1,1);
  h_ele_etaEff_all->Print();
  h_ele_etaEff_all->GetXaxis()->SetTitle("#eta");
  h_ele_etaEff_all->GetYaxis()->SetTitle("# rec/# matching objects");
  TH1F *h_ele_ptEff_all = (TH1F*)h_ele_vertexPt_all->Clone("h_ele_ptEff_all");
  h_ele_ptEff_all->Reset();
  h_ele_ptEff_all->Divide(h_ele_vertexPt_all,h_matchingObjectPt,1,1);
  h_ele_ptEff_all->Print();
  h_ele_ptEff_all->GetXaxis()->SetTitle("p_{T}");
  h_ele_ptEff_all->GetYaxis()->SetTitle("# rec/# matching objects");

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
  TProfile *p_ele_PoPmatchingObjectVsEta = h_ele_PoPmatchingObjectVsEta->ProfileX();
  p_ele_PoPmatchingObjectVsEta->Write();
  TProfile *p_ele_PoPmatchingObjectVsPhi = h_ele_PoPmatchingObjectVsPhi->ProfileX();
  p_ele_PoPmatchingObjectVsPhi->Write();
  TProfile *p_ele_EtaMnEtamatchingObjectVsEta = h_ele_EtaMnEtamatchingObjectVsEta->ProfileX();
  p_ele_EtaMnEtamatchingObjectVsEta->Write();
  TProfile *p_ele_EtaMnEtamatchingObjectVsPhi = h_ele_EtaMnEtamatchingObjectVsPhi->ProfileX();
  p_ele_EtaMnEtamatchingObjectVsPhi->Write();
  TProfile *p_ele_PhiMnPhimatchingObjectVsEta = h_ele_PhiMnPhimatchingObjectVsEta->ProfileX();
  p_ele_PhiMnPhimatchingObjectVsEta->Write();
  TProfile *p_ele_PhiMnPhimatchingObjectVsPhi = h_ele_PhiMnPhimatchingObjectVsPhi->ProfileX();
  p_ele_PhiMnPhimatchingObjectVsPhi->Write();
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

  h_matchingObjectNum->Write();
    
  // rec event
  
  histNum_->Write();
  
  // mc  
  h_matchingObjectEta->Write();
  h_matchingObjectAbsEta->Write();
  h_matchingObjectP->Write();
  h_matchingObjectPt->Write();

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
  h_ele_matchingObjectPt_matched->Write();
  h_ele_vertexEta->Write();
  h_ele_vertexEtaVsPhi->Write();
  h_ele_matchingObjectAbsEta_matched->Write();
  h_ele_matchingObjectEta_matched->Write();
  h_ele_vertexPhi->Write();
  h_ele_vertexX->Write();
  h_ele_vertexY ->Write();
  h_ele_vertexZ->Write();
  h_ele_vertexTIP->Write();
  h_ele_matchingObjectZ_matched->Write();
  h_ele_vertexTIPVsEta->Write();
  h_ele_vertexTIPVsPhi->Write();
  h_ele_vertexTIPVsPt->Write();
  h_ele_PoPmatchingObject->Write();
  h_ele_PoPmatchingObjectVsEta ->Write();
  h_ele_PoPmatchingObjectVsPhi->Write();
  h_ele_PoPmatchingObjectVsPt->Write();
  h_ele_PoPmatchingObject_barrel ->Write();
  h_ele_PoPmatchingObject_endcaps->Write();
  h_ele_EtaMnEtamatchingObject->Write();
  h_ele_EtaMnEtamatchingObjectVsEta ->Write();
  h_ele_EtaMnEtamatchingObjectVsPhi->Write();
  h_ele_EtaMnEtamatchingObjectVsPt->Write();
  h_ele_PhiMnPhimatchingObject ->Write();
  h_ele_PhiMnPhimatchingObject2 ->Write();
  h_ele_PhiMnPhimatchingObjectVsEta->Write();
  h_ele_PhiMnPhimatchingObjectVsPhi->Write();
  h_ele_PhiMnPhimatchingObjectVsPt->Write();

  // matched electron, superclusters
  histSclEn_->Write();
  histSclEoEmatchingObject_barrel->Write();
  histSclEoEmatchingObject_endcaps->Write();
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
  histSclEoEmatchingObjectGolden_barrel->Write();
  histSclEoEmatchingObjectGolden_endcaps->Write();
  histSclEoEmatchingObjectShowering0_barrel->Write();
  histSclEoEmatchingObjectShowering0_endcaps->Write();
  histSclEoEmatchingObjectShowering1234_barrel->Write();
  histSclEoEmatchingObjectShowering1234_endcaps->Write();

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
GsfElectronFakeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::cout << "analyzing new event " << std::endl;

  // get reco electrons  
  edm::Handle<reco::GsfElectronCollection> gsfElectrons;
  iEvent.getByLabel(electronCollection_,gsfElectrons); 
  edm::LogInfo("")<<"\n\n =================> Treating event "<<iEvent.id()<<" Number of electrons "<<gsfElectrons.product()->size();

  // get reco calojet collection
  edm::Handle<reco::CaloJetCollection> recoCaloJets;
  iEvent.getByLabel(matchingObjectCollection_, recoCaloJets); 

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
   
  // association matching object-reco electrons
  int matchingObjectNum=0;
      
  for ( reco::CaloJetCollection::const_iterator moIter=recoCaloJets->begin();
   moIter!=recoCaloJets->end(); moIter++ ) {
    
    // number of matching objects
    matchingObjectNum++;

      if (moIter->energy()/cosh(moIter->eta())> maxPt_ || fabs(moIter->eta())> maxAbsEta_) continue;
      
      // suppress the endcaps
      //if (fabs(moIter->eta()) > 1.5) continue;
      // select central z
      //if ( fabs((*mcIter)->production_vertex()->position().z())>50.) continue;
 
      h_matchingObjectEta -> Fill( moIter->eta() );
      h_matchingObjectAbsEta -> Fill( fabs(moIter->eta()) );
      h_matchingObjectP   -> Fill( moIter->energy() );
      h_matchingObjectPt   -> Fill( moIter->energy()/cosh(moIter->eta()) );
      h_matchingObjectPhi   -> Fill( moIter->phi() );
      h_matchingObjectZ   -> Fill(  moIter->vz() );
     	
      // looking for the best matching gsf electron
      bool okGsfFound = false;
      double gsfOkRatio = 999999.;

      // find best matched electron
      reco::GsfElectron bestGsfElectron;
      for (reco::GsfElectronCollection::const_iterator gsfIter=gsfElectrons->begin();
       gsfIter!=gsfElectrons->end(); gsfIter++){
	
	double deltaR = sqrt(pow((gsfIter->eta()-moIter->eta()),2) + pow((gsfIter->phi()-moIter->phi()),2));
	if ( deltaR < deltaR_ ){
	//if ( (genPc->pdg_id() == 11) && (gsfIter->charge() < 0.) || (genPc->pdg_id() == -11) &&
	//(gsfIter->charge() > 0.) ){
	  double tmpGsfRatio = gsfIter->p()/moIter->energy();
	  if ( fabs(tmpGsfRatio-1) < fabs(gsfOkRatio-1) ) {
	    gsfOkRatio = tmpGsfRatio;
	    bestGsfElectron=*gsfIter;
	    okGsfFound = true;
	  } 
	//} 
	} 
      } // loop over rec ele to look for the best one	

      // analysis when the matching object is matched by a rec electron
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
	h_ele_matchingObjectPt_matched      -> Fill( moIter->energy()/cosh(moIter->eta()) );
        h_ele_matchingObjectPhi_matched   -> Fill( moIter->phi() );
	h_ele_matchingObjectAbsEta_matched     -> Fill( fabs(moIter->eta()) );
	h_ele_matchingObjectEta_matched     -> Fill( moIter->eta() );
	h_ele_vertexEtaVsPhi     -> Fill(  bestGsfElectron.phi(),bestGsfElectron.eta() );
	h_ele_vertexPhi     -> Fill( bestGsfElectron.phi() );
	h_ele_vertexX     -> Fill( bestGsfElectron.vertex().x() );
	h_ele_vertexY     -> Fill( bestGsfElectron.vertex().y() );
	h_ele_vertexZ     -> Fill( bestGsfElectron.vertex().z() );
        h_ele_matchingObjectZ_matched   -> Fill( moIter->vz() );
	double d = bestGsfElectron.gsfTrack()->vertex().x()*bestGsfElectron.gsfTrack()->vertex().x()+
	 bestGsfElectron.gsfTrack()->vertex().y()*bestGsfElectron.gsfTrack()->vertex().y();
	d = sqrt(d); 
	h_ele_vertexTIP     -> Fill( d );
	h_ele_vertexTIPVsEta     -> Fill(  bestGsfElectron.eta(), d );
	h_ele_vertexTIPVsPhi     -> Fill(  bestGsfElectron.phi(), d );
	h_ele_vertexTIPVsPt     -> Fill(  bestGsfElectron.pt(), d );	
	h_ele_EtaMnEtamatchingObject  -> Fill( bestGsfElectron.eta()-moIter->eta());
	h_ele_EtaMnEtamatchingObjectVsEta  -> Fill( bestGsfElectron.eta(), bestGsfElectron.eta()-moIter->eta());
	h_ele_EtaMnEtamatchingObjectVsPhi  -> Fill( bestGsfElectron.phi(), bestGsfElectron.eta()-moIter->eta());
	h_ele_EtaMnEtamatchingObjectVsPt  -> Fill( bestGsfElectron.pt(), bestGsfElectron.eta()-moIter->eta());
	h_ele_PhiMnPhimatchingObject  -> Fill( bestGsfElectron.phi()-moIter->phi());
	h_ele_PhiMnPhimatchingObject2  -> Fill( bestGsfElectron.phi()-moIter->phi());
	h_ele_PhiMnPhimatchingObjectVsEta  -> Fill( bestGsfElectron.eta(), bestGsfElectron.phi()-moIter->phi());
	h_ele_PhiMnPhimatchingObjectVsPhi  -> Fill( bestGsfElectron.phi(), bestGsfElectron.phi()-moIter->phi());
	h_ele_PhiMnPhimatchingObjectVsPt  -> Fill( bestGsfElectron.pt(), bestGsfElectron.phi()-moIter->phi());
	h_ele_PoPmatchingObject       -> Fill( bestGsfElectron.p()/moIter->energy());
	h_ele_PoPmatchingObjectVsEta       -> Fill( bestGsfElectron.eta(), bestGsfElectron.p()/moIter->energy());
	h_ele_PoPmatchingObjectVsPhi       -> Fill( bestGsfElectron.phi(), bestGsfElectron.p()/moIter->energy());
	h_ele_PoPmatchingObjectVsPt       -> Fill( bestGsfElectron.py(), bestGsfElectron.p()/moIter->energy());
	if (bestGsfElectron.classification() < 100) h_ele_PoPmatchingObject_barrel       -> Fill( bestGsfElectron.p()/moIter->energy());
	if (bestGsfElectron.classification() >= 100) h_ele_PoPmatchingObject_endcaps       -> Fill( bestGsfElectron.p()/moIter->energy());

	// supercluster related distributions
	reco::SuperClusterRef sclRef = bestGsfElectron.superCluster();
        histSclEn_->Fill(sclRef->energy());
        double R=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y() +sclRef->z()*sclRef->z());
        double Rt=TMath::Sqrt(sclRef->x()*sclRef->x() + sclRef->y()*sclRef->y());
        histSclEt_->Fill(sclRef->energy()*(Rt/R));
        histSclEtVsEta_->Fill(sclRef->eta(),sclRef->energy()*(Rt/R));
        histSclEtVsPhi_->Fill(sclRef->phi(),sclRef->energy()*(Rt/R));
        if (bestGsfElectron.classification() < 100)  histSclEoEmatchingObject_barrel->Fill(sclRef->energy()/moIter->energy());
        if (bestGsfElectron.classification() >= 100)  histSclEoEmatchingObject_endcaps->Fill(sclRef->energy()/moIter->energy());
        histSclEta_->Fill(sclRef->eta());
        histSclEtaVsPhi_->Fill(sclRef->phi(),sclRef->eta());
        histSclPhi_->Fill(sclRef->phi());

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
        if (bestGsfElectron.classification() == 0)  histSclEoEmatchingObjectGolden_barrel->Fill(sclRef->energy()/moIter->energy());
        if (bestGsfElectron.classification() == 100)  histSclEoEmatchingObjectGolden_endcaps->Fill(sclRef->energy()/moIter->energy());
        if (bestGsfElectron.classification() == 30)  histSclEoEmatchingObjectShowering0_barrel->Fill(sclRef->energy()/moIter->energy());
        if (bestGsfElectron.classification() == 130)  histSclEoEmatchingObjectShowering0_endcaps->Fill(sclRef->energy()/moIter->energy());
        if (bestGsfElectron.classification() == 31 || bestGsfElectron.classification() == 32  || bestGsfElectron.classification() == 33 || eleClass == 34 )  histSclEoEmatchingObjectShowering1234_barrel->Fill(sclRef->energy()/moIter->energy());
        if (bestGsfElectron.classification() == 131 || bestGsfElectron.classification() == 132 || bestGsfElectron.classification() == 133 || eleClass == 134 )  histSclEoEmatchingObjectShowering1234_endcaps->Fill(sclRef->energy()/moIter->energy());
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

//    } // matching object found

//    }

  } // loop overmatching object
  
  h_matchingObjectNum->Fill(matchingObjectNum);

}


