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
				 std::vector<double> & ph_protectionsForJetMET,
				 float ele_iso_pt,
				 float ele_iso_mva_eb,
				 float ele_iso_mva_ee,
				 float ele_iso_combIso_eb,
				 float ele_iso_combIso_ee,
				 float ele_noniso_mva,
				 unsigned int ele_missinghits,
				 string ele_iso_path_mvaWeightFile,
				 std::vector<double> & ele_protectionsForJetMET
				 ):
  ph_Et_(ph_Et),
  ph_combIso_(ph_combIso),
  ph_loose_hoe_(ph_loose_hoe),
  ph_protectionsForJetMET_(ph_protectionsForJetMET),
  ele_iso_pt_(ele_iso_pt),
  ele_iso_mva_eb_(ele_iso_mva_eb),
  ele_iso_mva_ee_(ele_iso_mva_ee),
  ele_iso_combIso_eb_(ele_iso_combIso_eb),
  ele_iso_combIso_ee_(ele_iso_combIso_ee),
  ele_noniso_mva_(ele_noniso_mva),
  ele_missinghits_(ele_missinghits),
  ele_protectionsForJetMET_(ele_protectionsForJetMET)
{

  ele_iso_mvaID_= new ElectronMVAEstimator(ele_iso_path_mvaWeightFile);

}

bool PFEGammaFilters::passPhotonSelection(const reco::Photon & photon) {
  // First simple selection, same as the Run1 to be improved in CMSSW_710


  // Photon ET
  if(photon.pt()  < ph_Et_ ) return false;
  //std::cout<<"Cuts "<<combIso_<<" H/E "<<loose_hoe_<<std::endl;
  if (photon.hadronicOverEm() >ph_loose_hoe_ ) return false;
  //Isolation variables in 0.3 cone combined
  if(photon.trkSumPtHollowConeDR03()+photon.ecalRecHitSumEtConeDR03()+photon.hcalTowerSumEtConeDR03() > ph_combIso_)
    return false;		
  
  
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
      if( ele_iso_mvaID_->mva( electron, nVtx ) > ele_iso_mva_eb_ ) 
	passEleSelection = true;
    }
    else if (eleEta > 1.485  && isoDr03 < ele_iso_combIso_ee_) {
      if( ele_iso_mvaID_->mva( electron, nVtx ) > ele_iso_mva_ee_ ) 
	passEleSelection = true;
    }

  }

  // TO BE CHANGED by accessing from GsfElectron
  if(pfcand.mva_e_pi() > ele_noniso_mva_) {
    passEleSelection = true; 
  }
  
  return passEleSelection;
}

bool PFEGammaFilters::isElectron(const reco::GsfElectron & electron) {
 
  unsigned int nmisshits = electron.gsfTrack()->trackerExpectedHitsInner().numberOfLostHits();
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

  //MyCuts
  float maxNtracks = ele_protectionsForJetMET_[0];  //3.; 
  float maxHcalE = ele_protectionsForJetMET_[1];  //10.;
  float maxTrackPOverEele = ele_protectionsForJetMET_[2];  //1.;
  float maxE = ele_protectionsForJetMET_[3];  //50.;
  float maxEleHcalEOverEcalE = ele_protectionsForJetMET_[4];  //0.1;
  float maxEcalEOverPRes = ele_protectionsForJetMET_[5];  //0.2;
  float maxEeleOverPoutRes = ele_protectionsForJetMET_[6];  //0.5;
  float maxHcalEOverP = ele_protectionsForJetMET_[7];  //1.0;
  float maxHcalEOverEcalE = ele_protectionsForJetMET_[8];  //0.1;
  float maxEcalEOverP_1 = ele_protectionsForJetMET_[9];  //0.5;
  float maxEcalEOverP_2 = ele_protectionsForJetMET_[10];  //0.2;
  float maxEeleOverPout = ele_protectionsForJetMET_[11];  //0.2;
  float maxDPhiIN = ele_protectionsForJetMET_[12];  //0.1;

  
//   cout << " maxNtracks " <<  maxNtracks << endl
//        << " maxHcalE " << maxHcalE << endl
//        << " maxTrackPOverEele " << maxTrackPOverEele << endl
//        << " maxE " << maxE << endl
//        << " maxEleHcalEOverEcalE "<< maxEleHcalEOverEcalE << endl
//        << " maxEcalEOverPRes " << maxEcalEOverPRes << endl
//        << " maxEeleOverPoutRes "  << maxEeleOverPoutRes << endl
//        << " maxHcalEOverP " << maxHcalEOverP << endl
//        << " maxHcalEOverEcalE " << maxHcalEOverEcalE << endl
//        << " maxEcalEOverP_1 " << maxEcalEOverP_1 << endl
//        << " maxEcalEOverP_2 " << maxEcalEOverP_2 << endl
//        << " maxEeleOverPout "  << maxEeleOverPout << endl
//        << " maxDPhiIN " << maxDPhiIN << endl;
    
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
      int nexhits = trackref->trackerExpectedHitsInner().numberOfLostHits();  
      
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
    if(iextratrack > maxNtracks || Ene_hcalgsf > maxHcalE || (SumExtraKfP/Ene_ecalgsf) > maxTrackPOverEele 
       || (ETtotal > maxE && iextratrack > 1 && (Ene_hcalgsf/Ene_ecalgsf) > maxEleHcalEOverEcalE) ) {
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
    if( (fabs(1.-EtotPinMode) < maxEcalEOverPRes && (fabs(electron.eta()) < 1.0 || fabs(electron.eta()) > 2.0)) ||
	((EtotPinMode < 1.1 && EtotPinMode > 0.6) && (fabs(electron.eta()) >= 1.0 && fabs(electron.eta()) <= 2.0))) {
      if( fabs(1.-EGsfPoutMode) < maxEeleOverPoutRes && 
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

  if (HOverPin > maxHcalEOverP && HOverHE > maxHcalEOverEcalE && EtotPinMode < maxEcalEOverP_1) {
    if(debugSafeForJetMET) 
      cout << " *****This electron candidate is discarded  HCAL ENERGY "	
	   << " HOverPin " << HOverPin << " HOverHE " << HOverHE  << " EtotPinMode" << EtotPinMode << endl;
    isSafeForJetMET = false;
  }
  
  // Reject Crazy E/p values... to be understood in the future how to train a 
  // BDT in order to avoid to select this bad electron candidates. 
  
  if( EtotPinMode < maxEcalEOverP_2 && EGsfPoutMode < maxEeleOverPout ) {
    if(debugSafeForJetMET) 
      cout << " *****This electron candidate is discarded  Low ETOTPIN "
	   << " EtotPinMode " << EtotPinMode << " EGsfPoutMode " << EGsfPoutMode << endl;
    isSafeForJetMET = false;
  }
  
  // For not-preselected Gsf Tracks ET > 50 GeV, apply dphi preselection
  if(ETtotal > maxE && fabs(dphi_normalsc) > maxDPhiIN ) {
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

  float sumPtTrackIso = ph_protectionsForJetMET_[0]; // 2.0
  float sumPtTrackIsoSlope = ph_protectionsForJetMET_[1]; // 0.001

//   cout << " sumPtTrackIsoForPhoton " << sumPtTrackIso
//        << " sumPtTrackIsoSlopeForPhoton " << sumPtTrackIsoSlope <<  endl;

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
	cout << "PFEGammaFilters::isPhotonSafeForJetMET photon track:pt " << trackref->pt() << " SingleLegSize " << pfcandextra->singleLegConvTrackRef().size() << endl;
   
      
      //const std::vector<reco::TrackRef>&  mySingleLeg = 
      bool singleLegConv = false;
      for(unsigned int iconv =0; iconv<pfcandextra->singleLegConvTrackRef().size(); iconv++) {
	if(debugSafeForJetMET)
	  cout << "PFEGammaFilters::SingleLeg track:pt " << (pfcandextra->singleLegConvTrackRef()[iconv])->pt() << endl;
	
	if(pfcandextra->singleLegConvTrackRef()[iconv] == trackref) {
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

  if(sum_track_pt>(sumPtTrackIso + sumPtTrackIsoSlope * photon.pt())) {
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
  case TrackBase::iter0:
  case TrackBase::iter1:
  case TrackBase::iter2:
    Algo = 0;
    break;
  case TrackBase::iter3:
    Algo = 1;
    break;
  case TrackBase::iter4:
    Algo = 2;
    break;
  case TrackBase::iter5:
    Algo = 3;
    break;
  case TrackBase::iter6:
    Algo = 4;
    break;
  default:
    Algo = 5;
    break;
  }
  return Algo;
}
