// 
// Original Authors: Nicholas Wardle, Florian Beaudette
//
#include "RecoParticleFlow/PFProducer/interface/PFEGammaFilters.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

using namespace std;
using namespace reco;

PFEGammaFilters::PFEGammaFilters(float ph_Et,
				 float ph_combIso,
				 float ph_loose_hoe,
				 float ph_sietaieta_eb,
				 float ph_sietaieta_ee,
				 const edm::ParameterSet& ph_protectionsForJetMET,
				 float ele_iso_pt,
				 float ele_iso_mva_eb,
				 float ele_iso_mva_ee,
				 float ele_iso_combIso_eb,
				 float ele_iso_combIso_ee,
				 float ele_noniso_mva,
				 unsigned int ele_missinghits,
				 const string& ele_iso_path_mvaWeightFile,
				 const edm::ParameterSet& ele_protectionsForJetMET
				 ):
  ph_Et_(ph_Et),
  ph_combIso_(ph_combIso),
  ph_loose_hoe_(ph_loose_hoe),
  ph_sietaieta_eb_(ph_sietaieta_eb),
  ph_sietaieta_ee_(ph_sietaieta_ee),
  pho_sumPtTrackIso(ph_protectionsForJetMET.getParameter<double>("sumPtTrackIso")), 
  pho_sumPtTrackIsoSlope(ph_protectionsForJetMET.getParameter<double>("sumPtTrackIsoSlope")),
  ele_iso_pt_(ele_iso_pt),
  ele_iso_mva_eb_(ele_iso_mva_eb),
  ele_iso_mva_ee_(ele_iso_mva_ee),
  ele_iso_combIso_eb_(ele_iso_combIso_eb),
  ele_iso_combIso_ee_(ele_iso_combIso_ee),
  ele_noniso_mva_(ele_noniso_mva),
  ele_missinghits_(ele_missinghits),
  ele_maxNtracks(ele_protectionsForJetMET.getParameter<double>("maxNtracks")), 
  ele_maxHcalE(ele_protectionsForJetMET.getParameter<double>("maxHcalE")), 
  ele_maxTrackPOverEele(ele_protectionsForJetMET.getParameter<double>("maxTrackPOverEele")), 
  ele_maxE(ele_protectionsForJetMET.getParameter<double>("maxE")),
  ele_maxEleHcalEOverEcalE(ele_protectionsForJetMET.getParameter<double>("maxEleHcalEOverEcalE")),
  ele_maxEcalEOverPRes(ele_protectionsForJetMET.getParameter<double>("maxEcalEOverPRes")), 
  ele_maxEeleOverPoutRes(ele_protectionsForJetMET.getParameter<double>("maxEeleOverPoutRes")),
  ele_maxHcalEOverP(ele_protectionsForJetMET.getParameter<double>("maxHcalEOverP")), 
  ele_maxHcalEOverEcalE(ele_protectionsForJetMET.getParameter<double>("maxHcalEOverEcalE")), 
  ele_maxEcalEOverP_1(ele_protectionsForJetMET.getParameter<double>("maxEcalEOverP_1")),
  ele_maxEcalEOverP_2(ele_protectionsForJetMET.getParameter<double>("maxEcalEOverP_2")), 
  ele_maxEeleOverPout(ele_protectionsForJetMET.getParameter<double>("maxEeleOverPout")), 
  ele_maxDPhiIN(ele_protectionsForJetMET.getParameter<double>("maxDPhiIN"))
{
}

bool PFEGammaFilters::passPhotonSelection(const reco::Photon & photon) {
  // First simple selection, same as the Run1 to be improved in CMSSW_710


  // Photon ET
  if(photon.pt()  < ph_Et_ ) return false;
//   std::cout<< "Cuts " << ph_combIso_ << " H/E " << ph_loose_hoe_ 
// 	   << " SigmaiEtaiEta_EB " << ph_sietaieta_eb_  
// 	   << " SigmaiEtaiEta_EE " << ph_sietaieta_ee_ << std::endl;

  if (photon.hadTowOverEm() >ph_loose_hoe_ ) return false;
  //Isolation variables in 0.3 cone combined
  if(photon.trkSumPtHollowConeDR03()+photon.ecalRecHitSumEtConeDR03()+photon.hcalTowerSumEtConeDR03() > ph_combIso_)
    return false;		
  
  if(photon.isEB()) {
    if(photon.sigmaIetaIeta() > ph_sietaieta_eb_) 
      return false;
  }
  else {
    if(photon.sigmaIetaIeta() > ph_sietaieta_ee_) 
      return false;
  }


  return true;
}

bool PFEGammaFilters::passElectronSelection(const reco::GsfElectron & electron,
					    const reco::PFCandidate & pfcand, 
					    const int & nVtx) {
  // First simple selection, same as the Run1 to be improved in CMSSW_710
  
  bool passEleSelection = false;
  
  // Electron ET
  float electronPt = electron.pt();
  
  if( electronPt > ele_iso_pt_) {

    double isoDr03 = electron.dr03TkSumPt() + electron.dr03EcalRecHitSumEt() + electron.dr03HcalTowerSumEt();
    double eleEta = fabs(electron.eta());
    if (eleEta <= 1.485 && isoDr03 < ele_iso_combIso_eb_) {
      if( electron.mva_Isolated() > ele_iso_mva_eb_ ) 
	passEleSelection = true;
    }
    else if (eleEta > 1.485  && isoDr03 < ele_iso_combIso_ee_) {
      if( electron.mva_Isolated() > ele_iso_mva_ee_ ) 
	passEleSelection = true;
    }

  }

  //  cout << " My OLD MVA " << pfcand.mva_e_pi() << " MyNEW MVA " << electron.mva() << endl;
  if(electron.mva_e_pi() > ele_noniso_mva_) {
    passEleSelection = true; 
  }
  
  return passEleSelection;
}

bool PFEGammaFilters::isElectron(const reco::GsfElectron & electron) {
 
  unsigned int nmisshits = electron.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
  if(nmisshits > ele_missinghits_)
    return false;

  return true;
}


bool PFEGammaFilters::isElectronSafeForJetMET(const reco::GsfElectron & electron, 
					      const reco::PFCandidate & pfcand,
					      const reco::Vertex & primaryVertex,
					      bool lockTracks) {

  bool debugSafeForJetMET = false;
  bool isSafeForJetMET = true;

//   cout << " ele_maxNtracks " <<  ele_maxNtracks << endl
//        << " ele_maxHcalE " << ele_maxHcalE << endl
//        << " ele_maxTrackPOverEele " << ele_maxTrackPOverEele << endl
//        << " ele_maxE " << ele_maxE << endl
//        << " ele_maxEleHcalEOverEcalE "<< ele_maxEleHcalEOverEcalE << endl
//        << " ele_maxEcalEOverPRes " << ele_maxEcalEOverPRes << endl
//        << " ele_maxEeleOverPoutRes "  << ele_maxEeleOverPoutRes << endl
//        << " ele_maxHcalEOverP " << ele_maxHcalEOverP << endl
//        << " ele_maxHcalEOverEcalE " << ele_maxHcalEOverEcalE << endl
//        << " ele_maxEcalEOverP_1 " << ele_maxEcalEOverP_1 << endl
//        << " ele_maxEcalEOverP_2 " << ele_maxEcalEOverP_2 << endl
//        << " ele_maxEeleOverPout "  << ele_maxEeleOverPout << endl
//        << " ele_maxDPhiIN " << ele_maxDPhiIN << endl;
    
  // loop on the extra-tracks associated to the electron
  PFCandidateEGammaExtraRef pfcandextra = pfcand.egammaExtraRef();
  
  unsigned int iextratrack = 0;
  unsigned int itrackHcalLinked = 0;
  float SumExtraKfP = 0.;
  //float Ene_ecalgsf = 0.;


  // problems here: for now get the electron cluster from the gsf electron
  //  const PFCandidate::ElementsInBlocks& eleCluster = pfcandextra->gsfElectronClusterRef();
  // PFCandidate::ElementsInBlocks::const_iterator iegfirst = eleCluster.begin(); 
  //  float Ene_hcalgsf = pfcandextra->


  float ETtotal = electron.ecalEnergy();

  //NOTE take this from EGammaExtra
  float Ene_ecalgsf = electron.electronCluster()->energy();
  float Ene_hcalgsf = pfcandextra->hadEnergy();
  float HOverHE = Ene_hcalgsf/(Ene_hcalgsf + Ene_ecalgsf);
  float EtotPinMode = electron.eSuperClusterOverP();

  //NOTE take this from EGammaExtra
  float EGsfPoutMode = electron.eEleClusterOverPout();
  float HOverPin = Ene_hcalgsf / electron.gsfTrack()->pMode();
  float dphi_normalsc = electron.deltaPhiSuperClusterTrackAtVtx();


  const PFCandidate::ElementsInBlocks& extraTracks = pfcandextra->extraNonConvTracks();
  for (PFCandidate::ElementsInBlocks::const_iterator itrk = extraTracks.begin(); 
       itrk<extraTracks.end(); ++itrk) {
    const PFBlock& block = *(itrk->first);
    PFBlock::LinkData linkData =  block.linkData();
    const PFBlockElement& pfele = block.elements()[itrk->second];

    if(debugSafeForJetMET) 
      cout << " My track element number " <<  itrk->second << endl;
    if(pfele.type()==reco::PFBlockElement::TRACK) {
      reco::TrackRef trackref = pfele.trackRef();
      unsigned int Algo = whichTrackAlgo(trackref);
      // iter0, iter1, iter2, iter3 = Algo < 3
      // algo 4,5,6,7
      int nexhits = trackref->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS); 
      
      bool trackIsFromPrimaryVertex = false;
      for (Vertex::trackRef_iterator trackIt = primaryVertex.tracks_begin(); trackIt != primaryVertex.tracks_end(); ++trackIt) {
	if ( (*trackIt).castTo<TrackRef>() == trackref ) {
	  trackIsFromPrimaryVertex = true;
	  break;
	}
      }
      
      // probably we could now remove the algo request?? 
      if(Algo < 3 && nexhits == 0 && trackIsFromPrimaryVertex) {


	float p_trk = trackref->p();
	SumExtraKfP += p_trk;
	iextratrack++;
	// Check if these extra tracks are HCAL linked
	std::multimap<double, unsigned int> hcalKfElems;
	block.associatedElements( itrk->second,
				  linkData,
				  hcalKfElems,
				  reco::PFBlockElement::HCAL,
				  reco::PFBlock::LINKTEST_ALL );
	if(hcalKfElems.size() > 0) {
	  itrackHcalLinked++;
	}
	if(debugSafeForJetMET) 
	  cout << " The ecalGsf cluster is not isolated: >0 KF extra with algo < 3" 
	       << " Algo " << Algo
	       << " nexhits " << nexhits
	       << " trackIsFromPrimaryVertex " << trackIsFromPrimaryVertex << endl;
	if(debugSafeForJetMET) 
	  cout << " My track PT " << trackref->pt() << endl;

      }
      else {
	if(debugSafeForJetMET) 
	  cout << " Tracks from PU " 
	       << " Algo " << Algo
	       << " nexhits " << nexhits
	       << " trackIsFromPrimaryVertex " << trackIsFromPrimaryVertex << endl;
	if(debugSafeForJetMET) 
	  cout << " My track PT " << trackref->pt() << endl;
      }
    }
  }
  if( iextratrack > 0) {
    if(iextratrack > ele_maxNtracks || Ene_hcalgsf > ele_maxHcalE || (SumExtraKfP/Ene_ecalgsf) > ele_maxTrackPOverEele 
       || (ETtotal > ele_maxE && iextratrack > 1 && (Ene_hcalgsf/Ene_ecalgsf) > ele_maxEleHcalEOverEcalE) ) {
      if(debugSafeForJetMET) 
	cout << " *****This electron candidate is discarded: Non isolated  # tracks "		
	     << iextratrack << " HOverHE " << HOverHE 
	     << " SumExtraKfP/Ene_ecalgsf " << SumExtraKfP/Ene_ecalgsf 
	     << " SumExtraKfP " << SumExtraKfP 
	     << " Ene_ecalgsf " << Ene_ecalgsf
	     << " ETtotal " << ETtotal
	     << " Ene_hcalgsf/Ene_ecalgsf " << Ene_hcalgsf/Ene_ecalgsf
	     << endl;
      
      isSafeForJetMET = false;
    }
    // the electron is retained and the kf tracks are not locked
    if( (fabs(1.-EtotPinMode) < ele_maxEcalEOverPRes && (fabs(electron.eta()) < 1.0 || fabs(electron.eta()) > 2.0)) ||
	((EtotPinMode < 1.1 && EtotPinMode > 0.6) && (fabs(electron.eta()) >= 1.0 && fabs(electron.eta()) <= 2.0))) {
      if( fabs(1.-EGsfPoutMode) < ele_maxEeleOverPoutRes && 
	  (itrackHcalLinked == iextratrack)) {

	lockTracks = false;
	//	lockExtraKf = false;
	if(debugSafeForJetMET) 
	  cout << " *****This electron is reactivated  # tracks "		
	       << iextratrack << " #tracks hcal linked " << itrackHcalLinked 
	       << " SumExtraKfP/Ene_ecalgsf " << SumExtraKfP/Ene_ecalgsf  
	       << " EtotPinMode " << EtotPinMode << " EGsfPoutMode " << EGsfPoutMode 
	       << " eta gsf " << electron.eta()  <<endl;
      }
    }
  }

  if (HOverPin > ele_maxHcalEOverP && HOverHE > ele_maxHcalEOverEcalE && EtotPinMode < ele_maxEcalEOverP_1) {
    if(debugSafeForJetMET) 
      cout << " *****This electron candidate is discarded  HCAL ENERGY "	
	   << " HOverPin " << HOverPin << " HOverHE " << HOverHE  << " EtotPinMode" << EtotPinMode << endl;
    isSafeForJetMET = false;
  }
  
  // Reject Crazy E/p values... to be understood in the future how to train a 
  // BDT in order to avoid to select this bad electron candidates. 
  
  if( EtotPinMode < ele_maxEcalEOverP_2 && EGsfPoutMode < ele_maxEeleOverPout ) {
    if(debugSafeForJetMET) 
      cout << " *****This electron candidate is discarded  Low ETOTPIN "
	   << " EtotPinMode " << EtotPinMode << " EGsfPoutMode " << EGsfPoutMode << endl;
    isSafeForJetMET = false;
  }
  
  // For not-preselected Gsf Tracks ET > 50 GeV, apply dphi preselection
  if(ETtotal > ele_maxE && fabs(dphi_normalsc) > ele_maxDPhiIN ) {
    if(debugSafeForJetMET) 
      cout << " *****This electron candidate is discarded  Large ANGLE "
	   << " ETtotal " << ETtotal << " EGsfPoutMode " << dphi_normalsc << endl;
    isSafeForJetMET = false;
  }



  return isSafeForJetMET;
}
bool PFEGammaFilters::isPhotonSafeForJetMET(const reco::Photon & photon, const reco::PFCandidate & pfcand) {

  bool isSafeForJetMET = true;
  bool debugSafeForJetMET = false;

//   cout << " pho_sumPtTrackIsoForPhoton " << pho_sumPtTrackIso
//        << " pho_sumPtTrackIsoSlopeForPhoton " << pho_sumPtTrackIsoSlope <<  endl;

  float sum_track_pt = 0.;

  PFCandidateEGammaExtraRef pfcandextra = pfcand.egammaExtraRef();
  const PFCandidate::ElementsInBlocks& extraTracks = pfcandextra->extraNonConvTracks();
  for (PFCandidate::ElementsInBlocks::const_iterator itrk = extraTracks.begin(); 
       itrk<extraTracks.end(); ++itrk) {
    const PFBlock& block = *(itrk->first);
    const PFBlockElement& pfele = block.elements()[itrk->second];
 
    
    if(pfele.type()==reco::PFBlockElement::TRACK) {

     
      reco::TrackRef trackref = pfele.trackRef();
      
      if(debugSafeForJetMET)
	cout << "PFEGammaFilters::isPhotonSafeForJetMET photon track:pt " << trackref->pt() << " SingleLegSize " << pfcandextra->singleLegConvTrackRefMva().size() << endl;
   
      
      //const std::vector<reco::TrackRef>&  mySingleLeg = 
      bool singleLegConv = false;
      for(unsigned int iconv =0; iconv<pfcandextra->singleLegConvTrackRefMva().size(); iconv++) {
	if(debugSafeForJetMET)
	  cout << "PFEGammaFilters::SingleLeg track:pt " << (pfcandextra->singleLegConvTrackRefMva()[iconv].first)->pt() << endl;
	
	if(pfcandextra->singleLegConvTrackRefMva()[iconv].first == trackref) {
	  singleLegConv = true;
	  if(debugSafeForJetMET)
	    cout << "PFEGammaFilters::isPhotonSafeForJetMET: SingleLeg conv track " << endl;
	  break;

	}
      }
      if(singleLegConv)
	continue;
      
      sum_track_pt += trackref->pt();

    }

  }

  if(debugSafeForJetMET)
    cout << " PFEGammaFilters::isPhotonSafeForJetMET: SumPt " << sum_track_pt << endl;

  if(sum_track_pt>(pho_sumPtTrackIso + pho_sumPtTrackIsoSlope * photon.pt())) {
    isSafeForJetMET = false;
    if(debugSafeForJetMET)
      cout << "************************************!!!! PFEGammaFilters::isPhotonSafeForJetMET: Photon Discaded !!! " << endl;
  }

  return isSafeForJetMET;
}
unsigned int PFEGammaFilters::whichTrackAlgo(const reco::TrackRef& trackRef) {
  unsigned int Algo = 0; 
  switch (trackRef->algo()) {
  case TrackBase::ctf:
  case TrackBase::initialStep:
  case TrackBase::lowPtTripletStep:
  case TrackBase::pixelPairStep:
  case TrackBase::jetCoreRegionalStep:
  case TrackBase::muonSeededStepInOut:
  case TrackBase::muonSeededStepOutIn:
    Algo = 0;
    break;
  case TrackBase::detachedTripletStep:
    Algo = 1;
    break;
  case TrackBase::mixedTripletStep:
    Algo = 2;
    break;
  case TrackBase::pixelLessStep:
    Algo = 3;
    break;
  case TrackBase::tobTecStep:
    Algo = 4;
    break;
  default:
    Algo = 5;
    break;
  }
  return Algo;
}
