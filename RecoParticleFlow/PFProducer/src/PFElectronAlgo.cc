//
// Original Author: Daniele Benedetti: daniele.benedetti@cern.ch
//

#include "RecoParticleFlow/PFProducer/interface/PFElectronAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoParticleFlow/PFTracking/interface/PFTrackAlgoTools.h"
//#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h" 
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "RecoParticleFlow/PFProducer/interface/PFElectronExtraEqual.h"
#include "RecoParticleFlow/PFProducer/interface/GsfElectronEqual.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

#include <iomanip>
#include <algorithm>

using namespace std;
using namespace reco;
PFElectronAlgo::PFElectronAlgo(const double mvaEleCut,
			       string mvaWeightFileEleID,
			       const boost::shared_ptr<PFSCEnergyCalibration>& thePFSCEnergyCalibration,
			       const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
			       bool applyCrackCorrections,
			       bool usePFSCEleCalib,
			       bool useEGElectrons,
			       bool useEGammaSupercluster,
			       double sumEtEcalIsoForEgammaSC_barrel,
			       double sumEtEcalIsoForEgammaSC_endcap,
			       double coneEcalIsoForEgammaSC,
			       double sumPtTrackIsoForEgammaSC_barrel,
			       double sumPtTrackIsoForEgammaSC_endcap,
			       unsigned int nTrackIsoForEgammaSC,
			       double coneTrackIsoForEgammaSC):
  mvaEleCut_(mvaEleCut),
  thePFSCEnergyCalibration_(thePFSCEnergyCalibration),
  thePFEnergyCalibration_(thePFEnergyCalibration),
  applyCrackCorrections_(applyCrackCorrections),
  usePFSCEleCalib_(usePFSCEleCalib),
  useEGElectrons_(useEGElectrons),
  useEGammaSupercluster_(useEGammaSupercluster),
  sumEtEcalIsoForEgammaSC_barrel_(sumEtEcalIsoForEgammaSC_barrel),
  sumEtEcalIsoForEgammaSC_endcap_(sumEtEcalIsoForEgammaSC_endcap),
  coneEcalIsoForEgammaSC_(coneEcalIsoForEgammaSC),
  sumPtTrackIsoForEgammaSC_barrel_(sumPtTrackIsoForEgammaSC_barrel),
  sumPtTrackIsoForEgammaSC_endcap_(sumPtTrackIsoForEgammaSC_endcap),
  nTrackIsoForEgammaSC_(nTrackIsoForEgammaSC),
  coneTrackIsoForEgammaSC_(coneTrackIsoForEgammaSC)								   
{
  // Set the tmva reader
  tmvaReader_ = new TMVA::Reader("!Color:Silent");
  tmvaReader_->AddVariable("lnPt_gsf",&lnPt_gsf);
  tmvaReader_->AddVariable("Eta_gsf",&Eta_gsf);
  tmvaReader_->AddVariable("dPtOverPt_gsf",&dPtOverPt_gsf);
  tmvaReader_->AddVariable("DPtOverPt_gsf",&DPtOverPt_gsf);
  //tmvaReader_->AddVariable("nhit_gsf",&nhit_gsf);
  tmvaReader_->AddVariable("chi2_gsf",&chi2_gsf);
  //tmvaReader_->AddVariable("DPtOverPt_kf",&DPtOverPt_kf);
  tmvaReader_->AddVariable("nhit_kf",&nhit_kf);
  tmvaReader_->AddVariable("chi2_kf",&chi2_kf);
  tmvaReader_->AddVariable("EtotPinMode",&EtotPinMode);
  tmvaReader_->AddVariable("EGsfPoutMode",&EGsfPoutMode);
  tmvaReader_->AddVariable("EtotBremPinPoutMode",&EtotBremPinPoutMode);
  tmvaReader_->AddVariable("DEtaGsfEcalClust",&DEtaGsfEcalClust);
  tmvaReader_->AddVariable("SigmaEtaEta",&SigmaEtaEta);
  tmvaReader_->AddVariable("HOverHE",&HOverHE);
//   tmvaReader_->AddVariable("HOverPin",&HOverPin);
  tmvaReader_->AddVariable("lateBrem",&lateBrem);
  tmvaReader_->AddVariable("firstBrem",&firstBrem);
  tmvaReader_->BookMVA("BDT",mvaWeightFileEleID.c_str());
}
void PFElectronAlgo::RunPFElectron(const reco::PFBlockRef&  blockRef,
				   std::vector<bool>& active,
				   const reco::Vertex & primaryVertex) {

  // the maps are initialized 
  AssMap associatedToGsf;
  AssMap associatedToBrems;
  AssMap associatedToEcal;

  // should be cleaned as often as often as possible
  elCandidate_.clear();
  electronExtra_.clear();
  allElCandidate_.clear();
  electronConstituents_.clear();
  fifthStepKfTrack_.clear();
  convGsfTrack_.clear();
  // SetLinks finds all the elements (kf,ecal,ps,hcal,brems) 
  // associated to each gsf track
  bool blockHasGSF = SetLinks(blockRef,associatedToGsf,
			      associatedToBrems,associatedToEcal,
			      active, primaryVertex);
  
  // check if there is at least a gsf track in the block. 
  if (blockHasGSF) {
    
    BDToutput_.clear();

    lockExtraKf_.clear();
    // For each GSF track is initialized a BDT value = -1
    BDToutput_.assign(associatedToGsf.size(),-1.);
    lockExtraKf_.assign(associatedToGsf.size(),true);

    // The FinalID is run and BDToutput values is assigned 
    SetIDOutputs(blockRef,associatedToGsf,associatedToBrems,associatedToEcal,primaryVertex);  

    // For each GSF track that pass the BDT configurable cut a pf candidate electron is created. 
    // This function finds also the best estimation of the initial electron 4-momentum.

    SetCandidates(blockRef,associatedToGsf,associatedToBrems,associatedToEcal);
    if (!elCandidate_.empty() ){
      isvalid_ = true;
      // when a pfelectron candidate is created all the elements associated to the
      // electron are locked. 
      SetActive(blockRef,associatedToGsf,associatedToBrems,associatedToEcal,active);
    }
  } // endif blockHasGSF
}
bool PFElectronAlgo::SetLinks(const reco::PFBlockRef&  blockRef,
			      AssMap& associatedToGsf_,
			      AssMap& associatedToBrems_,
			      AssMap& associatedToEcal_,     
			      std::vector<bool>& active,
				  const reco::Vertex & primaryVertex) {
  unsigned int CutIndex = 100000;
  double CutGSFECAL = 10000. ;  
  // no other cut are not used anymore. We use the default of PFBlockAlgo
  //PFEnergyCalibration pfcalib_;  
  bool DebugSetLinksSummary = false;
  bool DebugSetLinksDetailed = false;

  const reco::PFBlock& block = *blockRef;
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  const PFBlock::LinkData& linkData =  block.linkData();  
  
  bool IsThereAGSFTrack = false;
  bool IsThereAGoodGSFTrack = false;

  vector<unsigned int> trackIs(0);
  vector<unsigned int> gsfIs(0);
  vector<unsigned int> ecalIs(0);

  std::vector<bool> localactive(elements.size(),true);
 

  // Save the elements in shorter vectors like in PFAlgo.
  std::multimap<double, unsigned int> kfElems;
  for(unsigned int iEle=0; iEle<elements.size(); iEle++) {
    localactive[iEle] = active[iEle];
    bool thisIsAMuon = false;
    PFBlockElement::Type type = elements[iEle].type();
    switch( type ) {
    case PFBlockElement::TRACK:
      // Check if the track is already identified as a muon
      thisIsAMuon =  PFMuonAlgo::isMuon(elements[iEle]);
      // Otherwise store index
      if ( !thisIsAMuon && active[iEle] ) { 
	trackIs.push_back( iEle );
	if (DebugSetLinksDetailed) 
	  cout<<"TRACK, stored index, continue "<< iEle << endl;
      }
      continue;
    case PFBlockElement::GSF:
      // Check if the track has a KF partner identified as a muon
      block.associatedElements( iEle,linkData,
				kfElems,
				reco::PFBlockElement::TRACK,
				reco::PFBlock::LINKTEST_ALL );
      thisIsAMuon = !kfElems.empty() ? 
      PFMuonAlgo::isMuon(elements[kfElems.begin()->second]) : false;
      // Otherwise store index
      if ( !thisIsAMuon && active[iEle] ) { 
	IsThereAGSFTrack = true;    
	gsfIs.push_back( iEle );
	if (DebugSetLinksDetailed) 
	  cout<<"GSF, stored index, continue "<< iEle << endl;
      }
      continue;
    case PFBlockElement::ECAL: 
      if ( active[iEle]  ) { 
	ecalIs.push_back( iEle );
  	if (DebugSetLinksDetailed) 
	  cout<<"ECAL, stored index, continue "<< iEle << endl;
      }
      continue;
    default:
      continue;
    }
  }
  // ******************* Start Link *****************************
  // Do something only if a gsf track is found in the block
  if(IsThereAGSFTrack) {
    

    // LocalLock the Elements associated to a Kf tracks and not to a Gsf
    // The clusters associated both to a kf track and to a brem tangend 
    // are then assigned only to the kf track
    // Could be improved doing this after. 

    // 19 Mar 2010 adding the KF track from Gamma Conv. 
    // They are linked to the GSF tracks they are not considered
    // anymore in the following ecal cluster locking 
    if (DebugSetLinksDetailed) {
      cout<<"#########################################################"<<endl;
      cout<<"#####           Process Block:                      #####"<<endl;
      cout<<"#########################################################"<<endl;
      cout<<block<<endl;
    }      

    
    for(unsigned int iEle=0; iEle<trackIs.size(); iEle++) {
      std::multimap<double, unsigned int> gsfElems;
      block.associatedElements( trackIs[iEle],  linkData,
				gsfElems ,
				reco::PFBlockElement::GSF,
				reco::PFBlock::LINKTEST_ALL );
      if(gsfElems.empty()){
	// This means that the considered kf is *not* associated
	// to any gsf track
	std::multimap<double, unsigned int> ecalKfElems;
	block.associatedElements( trackIs[iEle],linkData,
				  ecalKfElems,
				  reco::PFBlockElement::ECAL,
				  reco::PFBlock::LINKTEST_ALL );
	if(!ecalKfElems.empty()) { 
	  unsigned int ecalKf_index = ecalKfElems.begin()->second;
	  if(localactive[ecalKf_index]==true) {
	    // Check if this clusters is however well linked to a primary gsf track
	    // if this the case the cluster is not locked.
	    
	    bool isGsfLinked = false;
	    for(unsigned int iGsf=0; iGsf<gsfIs.size(); iGsf++) {  
	      // if the ecal cluster is associated contemporary to a KF track
	      // and to a GSF track from conv, it is assigned to the KF track 
	      // In this way we can loose some cluster but it is safer for double counting. 
	      const reco::PFBlockElementGsfTrack * GsfEl  =  
		dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[gsfIs[iGsf]]));
	      if(GsfEl->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)) continue;
	      
	      std::multimap<double, unsigned int> ecalGsfElems;
	      block.associatedElements( gsfIs[iGsf],linkData,
					ecalGsfElems,
					reco::PFBlockElement::ECAL,
					reco::PFBlock::LINKTEST_ALL );
	      if(!ecalGsfElems.empty()) {
		if (ecalGsfElems.begin()->second == ecalKf_index) {
		  isGsfLinked = true;
		}
	      }
	    }
	    if(isGsfLinked == false) {
	      // add protection against energy loss because
	      // of the tracking fifth step
	      const reco::PFBlockElementTrack * kfEle =  
		dynamic_cast<const reco::PFBlockElementTrack*>((&elements[(trackIs[iEle])])); 	
	      const reco::TrackRef& refKf = kfEle->trackRef();
	      
	      int nexhits = refKf->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS);
	      
	      bool isGoodTrack=false;
	      if (refKf.isNonnull()) 
		isGoodTrack = PFTrackAlgoTools::isGoodForEGM(refKf->algo());

	      
	      bool trackIsFromPrimaryVertex = false;
	      for (Vertex::trackRef_iterator trackIt = primaryVertex.tracks_begin(); trackIt != primaryVertex.tracks_end(); ++trackIt) {
		if ( (*trackIt).castTo<TrackRef>() == refKf ) {
		  trackIsFromPrimaryVertex = true;
		  break;
		}
	      }
      
	      if(isGoodTrack
	         && nexhits == 0 && trackIsFromPrimaryVertex) {
		localactive[ecalKf_index] = false;
	      } else {
		fifthStepKfTrack_.emplace_back(ecalKf_index,trackIs[iEle]);
	      }
	    }
	  }
	}
      } // gsfElems.size()
    } // loop on kf tracks
    

    // start loop on gsf tracks
    for(unsigned int iEle=0; iEle<gsfIs.size(); iEle++) {  

      if (!localactive[(gsfIs[iEle])]) continue;  

      localactive[gsfIs[iEle]] = false;
      bool ClosestEcalWithKf = false;

      if (DebugSetLinksDetailed) cout << " Gsf Index " << gsfIs[iEle] << endl;

      const reco::PFBlockElementGsfTrack * GsfEl  =  
	dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[(gsfIs[iEle])]));

      // if GsfTrack fron converted bremsstralung continue
      if(GsfEl->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)) continue;
      IsThereAGoodGSFTrack = true;
      float eta_gsf = GsfEl->positionAtECALEntrance().eta();
      float etaOut_gsf = GsfEl->Pout().eta();
      float diffOutEcalEta =  fabs(eta_gsf-etaOut_gsf);
      reco::GsfTrackRef RefGSF = GsfEl->GsftrackRef();
      float Pin_gsf   = 0.01;
      if (RefGSF.isNonnull() ) 
	Pin_gsf = RefGSF->pMode();
      

      // Find Associated Kf Track elements and Ecal to KF elements
      unsigned int KfGsf_index = CutIndex;
      unsigned int KfGsf_secondIndex = CutIndex; 
      std::multimap<double, unsigned int> kfElems;
      block.associatedElements( gsfIs[iEle],linkData,
				kfElems,
				reco::PFBlockElement::TRACK,
				reco::PFBlock::LINKTEST_ALL );
      std::multimap<double, unsigned int> ecalKfElems;
      if (!kfElems.empty()) {
	// 19 Mar 2010 now a loop is needed because > 1 KF track could
	// be associated to the same GSF track

	for(std::multimap<double, unsigned int>::iterator itkf = kfElems.begin();
	    itkf != kfElems.end(); ++itkf) {
	  const reco::PFBlockElementTrack * TrkEl  =  
	    dynamic_cast<const reco::PFBlockElementTrack*>((&elements[itkf->second]));
	  
	  bool isPrim = isPrimaryTrack(*TrkEl,*GsfEl);
	  if(!isPrim) 
	    continue;
	  
	  if(localactive[itkf->second] == true) {

	    KfGsf_index = itkf->second;
	    localactive[KfGsf_index] = false;
	    // Find clusters associated to kftrack using linkbyrechit
	    block.associatedElements( KfGsf_index,  linkData,
				      ecalKfElems ,
				      reco::PFBlockElement::ECAL,
				      reco::PFBlock::LINKTEST_ALL );  
	  }
	  else {	  
	    KfGsf_secondIndex = itkf->second;
	  }
	}
      }
      
      // Find the closest Ecal clusters associated to this Gsf
      std::multimap<double, unsigned int> ecalGsfElems;
      block.associatedElements( gsfIs[iEle],linkData,
				ecalGsfElems,
				reco::PFBlockElement::ECAL,
				reco::PFBlock::LINKTEST_ALL );    
      double ecalGsf_dist = CutGSFECAL;
      unsigned int ClosestEcalGsf_index = CutIndex;
      if (!ecalGsfElems.empty()) {	
	if(localactive[(ecalGsfElems.begin()->second)] == true) {
	  // check energy compatibility for outer eta != ecal entrance, looping tracks
	  bool compatibleEPout = true;
	  if(diffOutEcalEta > 0.3) {
	    reco::PFClusterRef clusterRef = elements[(ecalGsfElems.begin()->second)].clusterRef();	
	    float EoPout = (clusterRef->energy())/(GsfEl->Pout().t());
	    if(EoPout > 5) 
	      compatibleEPout = false;
	  }
	  if(compatibleEPout) {
	    ClosestEcalGsf_index = ecalGsfElems.begin()->second;
	    ecalGsf_dist = block.dist(gsfIs[iEle],ClosestEcalGsf_index,
				      linkData,reco::PFBlock::LINKTEST_ALL);
	    
	    // Check that this cluster is not closer to another primary Gsf track
	    
	    std::multimap<double, unsigned int> ecalOtherGsfElems;
	    block.associatedElements( ClosestEcalGsf_index,linkData,
				      ecalOtherGsfElems,
				      reco::PFBlockElement::GSF,
				      reco::PFBlock::LINKTEST_ALL);
	    
	    if(!ecalOtherGsfElems.empty()) {
	      // get if it is closed to a conv brem gsf tracks
	      const reco::PFBlockElementGsfTrack * gsfCheck  =  
		dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[ecalOtherGsfElems.begin()->second]));
	      
	      if(ecalOtherGsfElems.begin()->second != gsfIs[iEle]&&
		 gsfCheck->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) == false) {	     
		ecalGsf_dist = CutGSFECAL;
		ClosestEcalGsf_index = CutIndex;
	      }
	    }
	  }
	  // do not lock at the moment we need this for the late brem
	}
      }
      // if any cluster is found with the gsf-ecal link, try with kf-ecal
      else if(!ecalKfElems.empty()) {
	if(localactive[(ecalKfElems.begin()->second)] == true) {
	  ClosestEcalGsf_index = ecalKfElems.begin()->second;	  
	  ecalGsf_dist = block.dist(gsfIs[iEle],ClosestEcalGsf_index,
				    linkData,reco::PFBlock::LINKTEST_ALL);
	  ClosestEcalWithKf = true;
	  
	  // Check if this cluster is not closer to another Gsf track
	  std::multimap<double, unsigned int> ecalOtherGsfElems;
	  block.associatedElements( ClosestEcalGsf_index,linkData,
				    ecalOtherGsfElems,
				    reco::PFBlockElement::GSF,
				    reco::PFBlock::LINKTEST_ALL);
	  if(!ecalOtherGsfElems.empty()) {
	    const reco::PFBlockElementGsfTrack * gsfCheck  =  
	      dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[ecalOtherGsfElems.begin()->second]));

	    if(ecalOtherGsfElems.begin()->second != gsfIs[iEle] &&
	       gsfCheck->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) == false) {
	      ecalGsf_dist = CutGSFECAL;
	      ClosestEcalGsf_index = CutIndex;
	      ClosestEcalWithKf = false;
	    }
	  }
	}
      }

      if (DebugSetLinksDetailed) 
	cout << " Closest Ecal to the Gsf/Kf: index " << ClosestEcalGsf_index 
	     << " dist " << ecalGsf_dist << endl;
      
      
      
      //  Find the brems associated to this Gsf
      std::multimap<double, unsigned int> bremElems;
      block.associatedElements( gsfIs[iEle],linkData,
				bremElems,
				reco::PFBlockElement::BREM,
				reco::PFBlock::LINKTEST_ALL );
      
      
      multimap<unsigned int,unsigned int> cleanedEcalBremElems;
      vector<unsigned int> keyBremIndex(0);
      unsigned int latestBrem_trajP = 0;     
      unsigned int latestBrem_index = CutIndex;
      for(std::multimap<double, unsigned int>::iterator ieb = bremElems.begin(); 
	  ieb != bremElems.end(); ++ieb ) {
	unsigned int brem_index = ieb->second;
	if(localactive[brem_index] == false) continue;


	// Find the ecal clusters associated to the brems
	std::multimap<double, unsigned int> ecalBremsElems;

	block.associatedElements( brem_index,  linkData,
				  ecalBremsElems,
				  reco::PFBlockElement::ECAL,
				  reco::PFBlock::LINKTEST_ALL );

	for (std::multimap<double, unsigned int>::iterator ie = ecalBremsElems.begin();
	     ie != ecalBremsElems.end();ie++) {
	  unsigned int ecalBrem_index = ie->second;
	  if(localactive[ecalBrem_index] == false) continue;

	  //to be changed, using the distance
	  float ecalBrem_dist = block.dist(brem_index,ecalBrem_index,
					   linkData,reco::PFBlock::LINKTEST_ALL); 
	  
	  
	  if (ecalBrem_index == ClosestEcalGsf_index && (ecalBrem_dist + 0.0012) > ecalGsf_dist) continue;

	  // Find the closest brem
	  std::multimap<double, unsigned int> sortedBremElems;
	  block.associatedElements( ecalBrem_index,linkData,
				    sortedBremElems,
				    reco::PFBlockElement::BREM,
				    reco::PFBlock::LINKTEST_ALL);
	  // check that this brem is that one coming from the same *primary* gsf
	  bool isGoodBrem = false;
	  unsigned int sortedBrem_index =  CutIndex;
	  for (std::multimap<double, unsigned int>::iterator ibs = sortedBremElems.begin();
	       ibs != sortedBremElems.end();ibs++) {
	    unsigned int temp_sortedBrem_index = ibs->second;
	    std::multimap<double, unsigned int> sortedGsfElems;
	    block.associatedElements( temp_sortedBrem_index,linkData,
				      sortedGsfElems,
				      reco::PFBlockElement::GSF,
				      reco::PFBlock::LINKTEST_ALL);
	    bool enteredInPrimaryGsf = false;
	    for (std::multimap<double, unsigned int>::iterator igs = sortedGsfElems.begin();
		 igs != sortedGsfElems.end();igs++) {
	      const reco::PFBlockElementGsfTrack * gsfCheck  =  
		dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[igs->second]));

	      if(gsfCheck->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) == false) {
		if(igs->second ==  gsfIs[iEle]) {
		  isGoodBrem = true;
		  sortedBrem_index = temp_sortedBrem_index;
		}
		enteredInPrimaryGsf = true;
		break;
	      }
	    }
	    if(enteredInPrimaryGsf)
	      break;
	  }

	  if(isGoodBrem) { 

	    //  Check that this cluster is not closer to another Gsf Track
	    // The check is not performed on KF track because the ecal clusters are aready locked.
	    std::multimap<double, unsigned int> ecalOtherGsfElems;
	    block.associatedElements( ecalBrem_index,linkData,
				      ecalOtherGsfElems,
				      reco::PFBlockElement::GSF,
				      reco::PFBlock::LINKTEST_ALL);
	    if (!ecalOtherGsfElems.empty()) {
	      const reco::PFBlockElementGsfTrack * gsfCheck  =  
		dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[ecalOtherGsfElems.begin()->second]));
	      if(ecalOtherGsfElems.begin()->second != gsfIs[iEle] &&
		 gsfCheck->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) == false) {
		continue;
	      }
	    }

	    const reco::PFBlockElementBrem * BremEl  =  
	      dynamic_cast<const reco::PFBlockElementBrem*>((&elements[sortedBrem_index]));

	    reco::PFClusterRef clusterRef = 
	      elements[ecalBrem_index].clusterRef();
	    

	    float sortedBremEcal_deta = fabs(clusterRef->position().eta() - BremEl->positionAtECALEntrance().eta());
	    // Triangular cut on plan chi2:deta -> OLD
	    //if((0.0075*sortedBremEcal_chi2 + 100.*sortedBremEcal_deta -1.5) < 0.) {
	    if(sortedBremEcal_deta < 0.015) {
	    
	      cleanedEcalBremElems.insert(pair<unsigned int,unsigned int>(sortedBrem_index,ecalBrem_index));
	      
	      unsigned int BremTrajP = BremEl->indTrajPoint();
	      if (BremTrajP > latestBrem_trajP) {
		latestBrem_trajP = BremTrajP;
		latestBrem_index = sortedBrem_index;
	      }
	      if (DebugSetLinksDetailed)
		cout << " brem Index " <<  sortedBrem_index 
		     << " associated cluster " << ecalBrem_index << " BremTrajP " << BremTrajP <<endl;
	      
	      // > 1 ecal clusters could be associated to the same brem twice: allowed N-1 link. 
	      // But the brem need to be stored once. 
	      // locallock the brem and the ecal clusters
	      localactive[ecalBrem_index] = false;  // the cluster
	      bool  alreadyfound = false;
	      for(unsigned int ii=0;ii<keyBremIndex.size();ii++) {
		if (sortedBrem_index == keyBremIndex[ii]) alreadyfound = true;
	      }
	      if (alreadyfound == false) {
		keyBremIndex.push_back(sortedBrem_index);
		localactive[sortedBrem_index] = false;   // the brem
	      }
	    }
	  }
	}
      }

      
      // Find Possible Extra Cluster associated to the gsf/kf
      vector<unsigned int> GsfElemIndex(0);
      vector<unsigned int> EcalIndex(0);

      // locallock the ecal cluster associated to the gsf
      if (ClosestEcalGsf_index < CutIndex) {
	GsfElemIndex.push_back(ClosestEcalGsf_index);
	localactive[ClosestEcalGsf_index] = false;
	for (std::multimap<double, unsigned int>::iterator ii = ecalGsfElems.begin();
	     ii != ecalGsfElems.end();ii++) {	
	  if(localactive[ii->second]) {
	    // Check that this cluster is not closer to another Gsf Track
	    std::multimap<double, unsigned int> ecalOtherGsfElems;
	    block.associatedElements( ii->second,linkData,
				      ecalOtherGsfElems,
				      reco::PFBlockElement::GSF,
				      reco::PFBlock::LINKTEST_ALL);
	    if(!ecalOtherGsfElems.empty()) {
	      if(ecalOtherGsfElems.begin()->second != gsfIs[iEle]) continue;
	    } 
	    
	    // get the cluster only if the deta (ecal-gsf) < 0.05
	    reco::PFClusterRef clusterRef = elements[(ii->second)].clusterRef();
	    float etacl =  clusterRef->eta();
	    if( fabs(eta_gsf-etacl) < 0.05) {	    
	      GsfElemIndex.push_back(ii->second);
	      localactive[ii->second] = false;
	      if (DebugSetLinksDetailed)
		cout << " ExtraCluster From Gsf " << ii->second << endl;
	    }
	  }
	}
      }

      //Add the possibility to link other ecal clusters from kf. 
     
//       for (std::multimap<double, unsigned int>::iterator ii = ecalKfElems.begin();
// 	   ii != ecalKfElems.end();ii++) {
// 	if(localactive[ii->second]) {
//         // Check that this cluster is not closer to another Gsf Track    
// 	  std::multimap<double, unsigned int> ecalOtherGsfElems;
// 	  block.associatedElements( ii->second,linkData,
// 				    ecalOtherGsfElems,
// 				    reco::PFBlockElement::GSF,
// 				    reco::PFBlock::LINKTEST_CHI2);
// 	  if(ecalOtherGsfElems.size()) {
// 	    if(ecalOtherGsfElems.begin()->second != gsfIs[iEle]) continue;
// 	  } 
// 	  GsfElemIndex.push_back(ii->second);
// 	  reco::PFClusterRef clusterRef = elements[(ii->second)].clusterRef();
// 	  float etacl =  clusterRef->eta();
// 	  if( fabs(eta_gsf-etacl) < 0.05) {	    
// 	    localactive[ii->second] = false;
// 	    if (DebugSetLinksDetailed)
// 	      cout << " ExtraCluster From KF " << ii->second << endl;
// 	  }
// 	}
//       }
      
      //****************** Fill Maps *************************

      // The GsfMap    

      // if any clusters have been associated to the gsf track	  
      // use the Ecal clusters associated to the latest brem and associate it to the gsf
       if(GsfElemIndex.empty()){
	if(latestBrem_index < CutIndex) {
	  unsigned int ckey = cleanedEcalBremElems.count(latestBrem_index);
	  if(ckey == 1) {
	    unsigned int temp_cal = 
	      cleanedEcalBremElems.find(latestBrem_index)->second;
	    GsfElemIndex.push_back(temp_cal);
	    if (DebugSetLinksDetailed)
	      cout << "******************** Gsf Cluster From Brem " << temp_cal 
		   << " Latest Brem index " << latestBrem_index 
		   << " ************************* " << endl;
	  }
	  else{
	    pair<multimap<unsigned int,unsigned int>::iterator,multimap<unsigned int,unsigned int>::iterator> ret;
	    ret = cleanedEcalBremElems.equal_range(latestBrem_index);
	    multimap<unsigned int,unsigned int>::iterator it;
	    for(it=ret.first; it!=ret.second; ++it) {
	      GsfElemIndex.push_back((*it).second);
	      if (DebugSetLinksDetailed)
		cout << "******************** Gsf Cluster From Brem " << (*it).second 
		     << " Latest Brem index " << latestBrem_index 
		     << " ************************* " << endl;
	    }
	  }
	  // erase the brem. 
	  unsigned int elToErase = 0;
	  for(unsigned int i = 0; i<keyBremIndex.size();i++) {
	    if(latestBrem_index == keyBremIndex[i]) {
	      elToErase = i;
	    }
	  }
	  keyBremIndex.erase(keyBremIndex.begin()+elToErase);
	}	
      }

      // Get Extra Clusters from converted brem gsf tracks. The locallock method
      // tells me if the ecal cluster has been already assigned to the primary
      // gsf track or to a brem

      for(unsigned int iConv=0; iConv<gsfIs.size(); iConv++) {  
	if(iConv != iEle) {

	  const reco::PFBlockElementGsfTrack * gsfConv  =  
	    dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[(gsfIs[iConv])]));
	  
	  // look at only to secondary gsf tracks
	  if(gsfConv->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)){
	    if (DebugSetLinksDetailed)
	      cout << "  PFElectronAlgo:: I'm running on convGsfBrem " << endl;
	    // check if they are linked to the primary
	    float conv_dist = block.dist(gsfIs[iConv],gsfIs[iEle],
					 linkData,reco::PFBlock::LINKTEST_ALL);
	    if(conv_dist > 0.) {
	      // find the closest ecal cluster associated to conversions

	      std::multimap<double, unsigned int> ecalConvElems;
	      block.associatedElements( gsfIs[iConv],linkData,
					ecalConvElems,
					reco::PFBlockElement::ECAL,
					reco::PFBlock::LINKTEST_ALL );    
	      if(!ecalConvElems.empty()) {
		// the ecal cluster is still active?
		if(localactive[(ecalConvElems.begin()->second)] == true) {
		  if (DebugSetLinksDetailed)
		    cout << "  PFElectronAlgo:: convGsfBrem has a ECAL cluster linked and free" << endl;
		  // Check that this cluster is not closer to another primary Gsf track
		  std::multimap<double, unsigned int> ecalOtherGsfPrimElems;
		  block.associatedElements( ecalConvElems.begin()->second,linkData,
					    ecalOtherGsfPrimElems,
					    reco::PFBlockElement::GSF,
					    reco::PFBlock::LINKTEST_ALL);
		  if(!ecalOtherGsfPrimElems.empty()) {
		    unsigned int gsfprimcheck_index = ecalOtherGsfPrimElems.begin()->second;
		    const reco::PFBlockElementGsfTrack * gsfCheck  =  
		      dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[gsfprimcheck_index]));
		    if(gsfCheck->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) == false) continue;
		    
		    reco::PFClusterRef clusterRef = elements[ecalConvElems.begin()->second].clusterRef();
		    if (DebugSetLinksDetailed)
		      cout << " PFElectronAlgo: !!!!!!! convGsfBrem ECAL cluster has been stored !!!!!!! "
			   << " Energy " << clusterRef->energy() << " eta,phi "  << clusterRef->position().eta()
			   <<", " <<  clusterRef->position().phi() << endl;
		 
		    GsfElemIndex.push_back(ecalConvElems.begin()->second);
		    convGsfTrack_.emplace_back(ecalConvElems.begin()->second,gsfIs[iConv]);
		    localactive[ecalConvElems.begin()->second] = false;
		    
		  }
		}
	      }
	    }
	  }
	}
      }


      
      EcalIndex.insert(EcalIndex.end(),GsfElemIndex.begin(),GsfElemIndex.end());
      
      

      // The BremMap
      for(unsigned int i =0;i<keyBremIndex.size();i++) {
	unsigned int ikey = keyBremIndex[i];
	unsigned int ckey = cleanedEcalBremElems.count(ikey);
	vector<unsigned int> BremElemIndex(0);
	if(ckey == 1) {
	  unsigned int temp_cal = 
	    cleanedEcalBremElems.find(ikey)->second;
	  BremElemIndex.push_back(temp_cal);
	}
	else{
	  pair<multimap<unsigned int,unsigned int>::iterator,multimap<unsigned int,unsigned int>::iterator> ret;
	  ret = cleanedEcalBremElems.equal_range(ikey);
	  multimap<unsigned int,unsigned int>::iterator it;
	  for(it=ret.first; it!=ret.second; ++it) {
	    BremElemIndex.push_back((*it).second);
	  }
	}
	EcalIndex.insert(EcalIndex.end(),BremElemIndex.begin(),BremElemIndex.end());
	associatedToBrems_.insert(pair<unsigned int,vector<unsigned int> >(ikey,BremElemIndex));
      }

      
      // 19 Mar 2010: add KF and ECAL elements from converted brem photons
      vector<unsigned int> convBremKFTrack;
      convBremKFTrack.clear();
      if (!kfElems.empty()) {
	for(std::multimap<double, unsigned int>::iterator itkf = kfElems.begin();
	    itkf != kfElems.end(); ++itkf) {
	  const reco::PFBlockElementTrack * TrkEl  =  
	    dynamic_cast<const reco::PFBlockElementTrack*>((&elements[itkf->second]));
	  bool isPrim = isPrimaryTrack(*TrkEl,*GsfEl);

	  if(!isPrim) {

	    // search for linked ECAL clusters
	    std::multimap<double, unsigned int> ecalConvElems;
	    block.associatedElements( itkf->second,linkData,
				      ecalConvElems,
				      reco::PFBlockElement::ECAL,
				      reco::PFBlock::LINKTEST_ALL );
	    if(!ecalConvElems.empty()) {
	      // Further Cleaning: DANIELE This could be improved!
	      const TrackRef& trkRef =   TrkEl->trackRef();
	      // iter0, iter1, iter2, iter3 = Algo < 3
	      bool isGoodTrack = PFTrackAlgoTools::isGoodForEGM(trkRef->algo());
	      float secpin = trkRef->p();	
	      
	      const reco::PFBlockElementCluster * clust =  
		dynamic_cast<const reco::PFBlockElementCluster*>((&elements[(ecalConvElems.begin()->second)])); 
	      float eneclust  =clust->clusterRef()->energy();

	      //1)  ******* Reject secondary KF tracks linked to also an HCAL cluster with H/(E+H) > 0.1
	      //            This is applied also to KF linked to locked ECAL cluster
	      //            NOTE: trusting the H/(E+H) and not the conv brem selection increse the number
	      //                  of charged hadrons around the electron. DANIELE? re-think about this. 
	      std::multimap<double, unsigned int> hcalConvElems;
	      block.associatedElements( itkf->second,linkData,
					hcalConvElems,
					reco::PFBlockElement::HCAL,
					reco::PFBlock::LINKTEST_ALL );

	      bool isHoHE = false;
	      bool isHoE = false;
	      bool isPoHE = false;

	      float enehcalclust = -1;
	      if(!hcalConvElems.empty()) {
		const reco::PFBlockElementCluster * clusthcal =  
		  dynamic_cast<const reco::PFBlockElementCluster*>((&elements[(hcalConvElems.begin()->second)])); 
		enehcalclust  =clusthcal->clusterRef()->energy();
		// NOTE: DANIELE? Are you sure you want to use the Algo type here? 
		if( (enehcalclust / (enehcalclust+eneclust) ) > 0.1 && isGoodTrack) {
		  isHoHE = true;
		  if(enehcalclust > eneclust) 
		    isHoE = true;
		  if(secpin > (enehcalclust+eneclust) )
		    isPoHE = true;
		}
	      }
	      

	      if(localactive[(ecalConvElems.begin()->second)] == false) {
		
		if(isHoE || isPoHE) {
		  if (DebugSetLinksDetailed)
		    cout << "PFElectronAlgo:: LOCKED ECAL REJECTED TRACK FOR H/E or P/(H+E) "
			 << " H/H+E " << enehcalclust/(enehcalclust+eneclust) 		      
			 << " H/E " <<  enehcalclust/eneclust
			 << " P/(H+E) " << secpin/(enehcalclust+eneclust) 
			 << " HCAL ENE " << enehcalclust 
			 << " ECAL ENE " << eneclust 
			 << " secPIN " << secpin 
			 << " Algo Track " << trkRef->algo() << endl;
		  continue;
		}

		// check if this track has been alread assigned to an ECAL cluster
		for(unsigned int iecal =0; iecal < EcalIndex.size(); iecal++) {
		  // in case this track is already assigned to a locked ECAL cluster
		  // the secondary kf track is also saved for further lock
		  if(EcalIndex[iecal] == ecalConvElems.begin()->second) {
		    if (DebugSetLinksDetailed)
		      cout << " PFElectronAlgo:: Conv Brem Recovery locked cluster and I will lock also the KF track " << endl; 
		    convBremKFTrack.push_back(itkf->second);
		  }
		}
	      }	      
	      else{
		// ECAL cluster free
		
		// 
		if(isHoHE){
		  if (DebugSetLinksDetailed)
		    cout << "PFElectronAlgo:: FREE ECAL REJECTED TRACK FOR H/H+E " 
			 << " H/H+E " <<  (enehcalclust / (enehcalclust+eneclust) ) 
			 << " H/E " <<  enehcalclust/eneclust
			 << " P/(H+E) " << secpin/(enehcalclust+eneclust) 
			 << " HCAL ENE " << enehcalclust 
			 << " ECAL ENE " << eneclust 
			 << " secPIN " << secpin 
			 << " Algo Track " << trkRef->algo() << endl;
		  continue;
		}

		// check that this cluster is not cluser to another KF track (primary)
		std::multimap<double, unsigned int> ecalOtherKFPrimElems;
		block.associatedElements( ecalConvElems.begin()->second,linkData,
					  ecalOtherKFPrimElems,
					  reco::PFBlockElement::TRACK,
					  reco::PFBlock::LINKTEST_ALL);
		if(!ecalOtherKFPrimElems.empty()) {
		  
		  // check that this ECAL clusters is the best associated to at least one of the  KF tracks
		  // linked to the considered GSF track
		  bool isFromGSF = false;
		  for(std::multimap<double, unsigned int>::iterator itclos = kfElems.begin();
		      itclos != kfElems.end(); ++itclos) {
		    if(ecalOtherKFPrimElems.begin()->second == itclos->second) {
		      isFromGSF = true;
		      break;
		    }
		  }
		  if(isFromGSF){

		    // Further Cleaning: DANIELE This could be improved! 		    		    
		  
		   	    		  
		    float Epin = eneclust/secpin;
		    
		    // compute the pfsupercluster energy till now
		    float totenergy = 0.;
		    for(unsigned int ikeyecal = 0; 
			ikeyecal<EcalIndex.size(); ikeyecal++){
		      // EcalIndex can have the same cluster save twice (because of the late brem cluster).
		      bool foundcluster = false;
		      if(ikeyecal > 0) {
			for(unsigned int i2 = 0; i2<ikeyecal-1; i2++) {
			  if(EcalIndex[ikeyecal] == EcalIndex[i2]) 
			    foundcluster = true;
			}
		      }
		      if(foundcluster) continue;
		      const reco::PFBlockElementCluster * clusasso =  
			dynamic_cast<const reco::PFBlockElementCluster*>((&elements[(EcalIndex[ikeyecal])])); 
		      totenergy += clusasso->clusterRef()->energy();
		    }
		    
		    // Further Cleaning: DANIELE This could be improved! 
		    //2) *****  Do not consider secondary tracks if the GSF and brems have failed in finding ECAL clusters
		    if(totenergy == 0.) {
		      if (DebugSetLinksDetailed)
			cout << "PFElectronAlgo:: REJECTED_NULLTOT totenergy " << totenergy << endl;
		      continue;
		    }
		    
		    //3) ****** Reject secondary KF tracks that have an high E/secPin and that make worse the Etot/pin 
		    if(Epin > 3) {
		      double res_before = fabs((totenergy-Pin_gsf)/Pin_gsf);
		      double res_after = fabs(((totenergy+eneclust)-Pin_gsf)/Pin_gsf);
		      
		      if(res_before < res_after) {
			if (DebugSetLinksDetailed)
			  cout << "PFElectronAlgo::REJECTED_RES totenergy " << totenergy << " Pin_gsf " << Pin_gsf << " cluster to secondary " <<  eneclust 
			       << " Res before " <<  res_before << " res_after " << res_after << endl;
			continue;
		      }
		    }
		    
		    if (DebugSetLinksDetailed)
		      cout << "PFElectronAlgo:: conv brem found asso to ECAL linked to a secondary KF " << endl;
		    convBremKFTrack.push_back(itkf->second);
		    GsfElemIndex.push_back(ecalConvElems.begin()->second);
		    EcalIndex.push_back(ecalConvElems.begin()->second);
		    localactive[(ecalConvElems.begin()->second)] = false;
		    localactive[(itkf->second)] = false;
		  }
		}
	      }
	    }
	  }
	}
      }
 
      // 4May import EG supercluster
      if(!EcalIndex.empty() && useEGammaSupercluster_) {
	double sumEtEcalInTheCone  = 0.;
	
	// Position of the first cluster
	const reco::PFBlockElementCluster * clust =  
	  dynamic_cast<const reco::PFBlockElementCluster*>((&elements[EcalIndex[0]])); 
	double PhiFC  = clust->clusterRef()->position().Phi();
	double EtaFC =  clust->clusterRef()->position().Eta();

	// Compute ECAL isolation ->
	for(unsigned int iEcal=0; iEcal<ecalIs.size(); iEcal++) {
	  bool foundcluster = false;
	  for(unsigned int ikeyecal = 0; 
	      ikeyecal<EcalIndex.size(); ikeyecal++){
	    if(ecalIs[iEcal] == EcalIndex[ikeyecal])
	      foundcluster = true;
	  }
	  
	  // -> only for clusters not already in the PFSCCluster
	  if(foundcluster == false) {
	    const reco::PFBlockElementCluster * clustExt =  
	      dynamic_cast<const reco::PFBlockElementCluster*>((&elements[ecalIs[iEcal]])); 
	    double eta_clust =  clustExt->clusterRef()->position().Eta();
	    double phi_clust =  clustExt->clusterRef()->position().Phi();
	    double theta_clust =  clustExt->clusterRef()->position().Theta();
	    double deta_clust = eta_clust - EtaFC;
	    double dphi_clust = phi_clust - PhiFC;
	    if ( dphi_clust < -M_PI ) 
	      dphi_clust = dphi_clust + 2.*M_PI;
	    else if ( dphi_clust > M_PI ) 
	      dphi_clust = dphi_clust - 2.*M_PI;
	    double  DR = sqrt(deta_clust*deta_clust+
			      dphi_clust*dphi_clust);		  
	    
	    //Jurassic veto in deta
	    if(fabs(deta_clust) > 0.05 && DR < coneEcalIsoForEgammaSC_) {
	      vector<double> ps1Ene(0);
	      vector<double> ps2Ene(0);
	      double ps1,ps2;
	      ps1=ps2=0.;
	      double EE_calib = thePFEnergyCalibration_->energyEm(*(clustExt->clusterRef()),ps1Ene,ps2Ene,ps1,ps2,applyCrackCorrections_);
	      double ET_calib = EE_calib*sin(theta_clust);
	      sumEtEcalInTheCone += ET_calib;
	    }
	  }
	} //EndLoop Additional ECAL clusters in the block
	
	// Compute track isolation: number of tracks && sumPt
	unsigned int sumNTracksInTheCone = 0;
	double sumPtTracksInTheCone = 0.;
	for(unsigned int iTrack=0; iTrack<trackIs.size(); iTrack++) {
	  // the track from the electron are already locked at this stage
	  if(localactive[(trackIs[iTrack])]==true) {
	    const reco::PFBlockElementTrack * kfEle =  
	      dynamic_cast<const reco::PFBlockElementTrack*>((&elements[(trackIs[iTrack])])); 	
	    const reco::TrackRef& trkref = kfEle->trackRef();
	    if (trkref.isNonnull()) {
	      double deta_trk =  trkref->eta() - RefGSF->etaMode();
	      double dphi_trk =  trkref->phi() - RefGSF->phiMode();
	      if ( dphi_trk < -M_PI ) 
		dphi_trk = dphi_trk + 2.*M_PI;
	      else if ( dphi_trk > M_PI ) 
		dphi_trk = dphi_trk - 2.*M_PI;
	      double  DR = sqrt(deta_trk*deta_trk+
				dphi_trk*dphi_trk);
	      
	      int NValPixelHit = trkref->hitPattern().numberOfValidPixelHits();
	      
	      if(DR < coneTrackIsoForEgammaSC_ && NValPixelHit >=3) {
		sumNTracksInTheCone++;
		sumPtTracksInTheCone+=trkref->pt();
	      }
	    }
	  }
	}

	
	bool isBarrelIsolated = false;
	if( fabs(EtaFC < 1.478) && 
	    (sumEtEcalInTheCone < sumEtEcalIsoForEgammaSC_barrel_ && 
	     (sumNTracksInTheCone < nTrackIsoForEgammaSC_  || sumPtTracksInTheCone < sumPtTrackIsoForEgammaSC_barrel_)))
	  isBarrelIsolated = true;
	
	
	bool isEndcapIsolated = false;
	if( fabs(EtaFC >= 1.478) && 
	    (sumEtEcalInTheCone < sumEtEcalIsoForEgammaSC_endcap_ &&  
	     (sumNTracksInTheCone < nTrackIsoForEgammaSC_  || sumPtTracksInTheCone < sumPtTrackIsoForEgammaSC_endcap_)))
	  isEndcapIsolated = true;
	

	// only print out
	if(DebugSetLinksDetailed) {
	  if(fabs(EtaFC < 1.478) && isBarrelIsolated == false) {
	    cout << "**** PFElectronAlgo:: SUPERCLUSTER FOUND BUT FAILS ISOLATION:BARREL *** " 
		 << " sumEtEcalInTheCone " <<sumEtEcalInTheCone 
		 << " sumNTracksInTheCone " << sumNTracksInTheCone 
		 << " sumPtTracksInTheCone " << sumPtTracksInTheCone << endl;
	  }
	  if(fabs(EtaFC >= 1.478) && isEndcapIsolated == false) {
	    cout << "**** PFElectronAlgo:: SUPERCLUSTER FOUND BUT FAILS ISOLATION:ENDCAP *** " 
		 << " sumEtEcalInTheCone " <<sumEtEcalInTheCone 
		 << " sumNTracksInTheCone " << sumNTracksInTheCone 
		 << " sumPtTracksInTheCone " << sumPtTracksInTheCone << endl;
	  }
	}



	
	if(isBarrelIsolated || isEndcapIsolated ) {
	  
	  
	  // Compute TotEnergy
	  double totenergy = 0.;
	  for(unsigned int ikeyecal = 0; 
	      ikeyecal<EcalIndex.size(); ikeyecal++){
	    // EcalIndex can have the same cluster save twice (because of the late brem cluster).
	    bool foundcluster = false;
	    if(ikeyecal > 0) {
	      for(unsigned int i2 = 0; i2<ikeyecal-1; i2++) {
		if(EcalIndex[ikeyecal] == EcalIndex[i2]) 
		  foundcluster = true;;
	      }
	    }
	    if(foundcluster) continue;
	    const reco::PFBlockElementCluster * clusasso =  
	      dynamic_cast<const reco::PFBlockElementCluster*>((&elements[(EcalIndex[ikeyecal])])); 
	    totenergy += clusasso->clusterRef()->energy();
	  }
	  // End copute TotEnergy


	  // Find extra cluster from e/g importing
	  for(unsigned int ikeyecal = 0; 
	      ikeyecal<EcalIndex.size(); ikeyecal++){
	    // EcalIndex can have the same cluster save twice (because of the late brem cluster).
	    bool foundcluster = false;
	    if(ikeyecal > 0) {
	      for(unsigned int i2 = 0; i2<ikeyecal-1; i2++) {
		if(EcalIndex[ikeyecal] == EcalIndex[i2]) 
		  foundcluster = true;
	      }
	    }	  
	    if(foundcluster) continue;
	    
	    
	    std::multimap<double, unsigned int> ecalFromSuperClusterElems;
	    block.associatedElements( EcalIndex[ikeyecal],linkData,
				      ecalFromSuperClusterElems,
				      reco::PFBlockElement::ECAL,
				      reco::PFBlock::LINKTEST_ALL);
	    if(!ecalFromSuperClusterElems.empty()) {
	      for(std::multimap<double, unsigned int>::iterator itsc = ecalFromSuperClusterElems.begin();
		  itsc != ecalFromSuperClusterElems.end(); ++itsc) {
		if(localactive[itsc->second] == false) {
		  continue;
		}
		
		std::multimap<double, unsigned int> ecalOtherKFPrimElems;
		block.associatedElements( itsc->second,linkData,
					  ecalOtherKFPrimElems,
					  reco::PFBlockElement::TRACK,
					  reco::PFBlock::LINKTEST_ALL);
		if(!ecalOtherKFPrimElems.empty()) {
		  if(localactive[ecalOtherKFPrimElems.begin()->second] == true) {
		    if (DebugSetLinksDetailed)
		      cout << "**** PFElectronAlgo:: SUPERCLUSTER FOUND BUT FAILS KF VETO *** " << endl;
		    continue;
		  }
		}
		bool isInTheEtaRange = false;
		const reco::PFBlockElementCluster * clustToAdd =  
		  dynamic_cast<const reco::PFBlockElementCluster*>((&elements[itsc->second])); 
		double deta_clustToAdd = clustToAdd->clusterRef()->position().Eta() - EtaFC;
		double ene_clustToAdd = clustToAdd->clusterRef()->energy();
		
		if(fabs(deta_clustToAdd) < 0.05)
		  isInTheEtaRange = true;
		
		// check for both KF and GSF
		bool isBetterEpin = false;
		if(isInTheEtaRange == false ) {
		  if (DebugSetLinksDetailed)
		    cout << "**** PFElectronAlgo:: SUPERCLUSTER FOUND BUT FAILS GAMMA DETA RANGE  *** " 
			 << fabs(deta_clustToAdd) << endl;
		  
		  if(KfGsf_index < CutIndex) {		    
		    //GSF
		    double res_before_gsf = fabs((totenergy-Pin_gsf)/Pin_gsf);
		    double res_after_gsf = fabs(((totenergy+ene_clustToAdd)-Pin_gsf)/Pin_gsf);

		    //KF
		    const reco::PFBlockElementTrack * trackEl =  
		      dynamic_cast<const reco::PFBlockElementTrack*>((&elements[KfGsf_index])); 
		    double Pin_kf = trackEl->trackRef()->p();
		    double res_before_kf = fabs((totenergy-Pin_kf)/Pin_kf);
		    double res_after_kf = fabs(((totenergy+ene_clustToAdd)-Pin_kf)/Pin_kf);			      
		    
		    // The new cluster improve both the E/pin?
		    if(res_after_gsf < res_before_gsf && res_after_kf < res_before_kf ) {
		      isBetterEpin = true;
		    }
		    else {
		      if (DebugSetLinksDetailed)
			cout << "**** PFElectronAlgo:: SUPERCLUSTER FOUND AND FAILS ALSO RES_EPIN" 
			     << " tot energy " << totenergy 
			     << " Pin_gsf " << Pin_gsf 
			     << " Pin_kf " << Pin_kf 
			     << " cluster from SC to ADD " <<  ene_clustToAdd 
			     << " Res before GSF " <<  res_before_gsf << " res_after_gsf " << res_after_gsf 
			     << " Res before KF " <<  res_before_kf << " res_after_kf " << res_after_kf  << endl;
		    }
		  }
		}
		
		if(isInTheEtaRange || isBetterEpin) {		
		  if (DebugSetLinksDetailed)
		    cout << "!!!! PFElectronAlgo:: ECAL from SUPERCLUSTER FOUND !!!!! " << endl;
		  GsfElemIndex.push_back(itsc->second);
		  EcalIndex.push_back(itsc->second);
		  localactive[(itsc->second)] = false;
		}
	      }
	    }
	  }
	} // END ISOLATION IF
      }


      if(KfGsf_index < CutIndex) 
	GsfElemIndex.push_back(KfGsf_index);
      else if(KfGsf_secondIndex < CutIndex) 
	GsfElemIndex.push_back(KfGsf_secondIndex);
      
      // insert the secondary KF tracks
      GsfElemIndex.insert(GsfElemIndex.end(),convBremKFTrack.begin(),convBremKFTrack.end());
      GsfElemIndex.insert(GsfElemIndex.end(),keyBremIndex.begin(),keyBremIndex.end());
      associatedToGsf_.insert(pair<unsigned int, vector<unsigned int> >(gsfIs[iEle],GsfElemIndex));

      // The EcalMap
      for(unsigned int ikeyecal = 0; 
	  ikeyecal<EcalIndex.size(); ikeyecal++){
	

	vector<unsigned int> EcalElemsIndex(0);

	std::multimap<double, unsigned int> PS1Elems;
	block.associatedElements( EcalIndex[ikeyecal],linkData,
				  PS1Elems,
				  reco::PFBlockElement::PS1,
				  reco::PFBlock::LINKTEST_ALL );
	for( std::multimap<double, unsigned int>::iterator it = PS1Elems.begin();
	     it != PS1Elems.end();it++) {
	  unsigned int index = it->second;
	  if(localactive[index] == true) {
	    
	    // Check that this cluster is not closer to another ECAL cluster
	    std::multimap<double, unsigned> sortedECAL;
	    block.associatedElements( index,  linkData,
				      sortedECAL,
				      reco::PFBlockElement::ECAL,
				      reco::PFBlock::LINKTEST_ALL );
	    unsigned jEcal = sortedECAL.begin()->second;
	    if ( jEcal != EcalIndex[ikeyecal]) continue; 


	    EcalElemsIndex.push_back(index);
	    localactive[index] = false;
	  }
	}
	
	std::multimap<double, unsigned int> PS2Elems;
	block.associatedElements( EcalIndex[ikeyecal],linkData,
				  PS2Elems,
				  reco::PFBlockElement::PS2,
				  reco::PFBlock::LINKTEST_ALL );
	for( std::multimap<double, unsigned int>::iterator it = PS2Elems.begin();
	     it != PS2Elems.end();it++) {
	  unsigned int index = it->second;
	  if(localactive[index] == true) {
	    // Check that this cluster is not closer to another ECAL cluster
	    std::multimap<double, unsigned> sortedECAL;
	    block.associatedElements( index,  linkData,
				      sortedECAL,
				      reco::PFBlockElement::ECAL,
				      reco::PFBlock::LINKTEST_ALL );
	    unsigned jEcal = sortedECAL.begin()->second;
	    if ( jEcal != EcalIndex[ikeyecal]) continue; 
	    
	    EcalElemsIndex.push_back(index);
	    localactive[index] = false;
	  }
	}
	if(ikeyecal == 0) {
	  // The first cluster is that one coming from the Gsf. 
	  // Only for this one is found the HCAL cluster using the Track-HCAL link
	  // and not the Ecal-Hcal not well tested yet.
	  std::multimap<double, unsigned int> hcalGsfElems;
	  block.associatedElements( gsfIs[iEle],linkData,
				    hcalGsfElems,
				    reco::PFBlockElement::HCAL,
				    reco::PFBlock::LINKTEST_ALL );	
	  for( std::multimap<double, unsigned int>::iterator it = hcalGsfElems.begin();
	       it != hcalGsfElems.end();it++) {
	    unsigned int index = it->second;
	    //  if(localactive[index] == true) {

	      // Check that this cluster is not closer to another GSF
	      // remove in high energetic jets this is dangerous. 
// 	      std::multimap<double, unsigned> sortedGsf;
// 	      block.associatedElements( index,  linkData,
// 					sortedGsf,
// 					reco::PFBlockElement::GSF,
// 					reco::PFBlock::LINKTEST_ALL );
// 	      unsigned jGsf = sortedGsf.begin()->second;
// 	      if ( jGsf != gsfIs[iEle]) continue; 

	      EcalElemsIndex.push_back(index);
	      localactive[index] = false;
	      
	      // }
	  }
	  // if the closest ecal cluster has been link with the KF, check KF - HCAL link
	  if(hcalGsfElems.empty() && ClosestEcalWithKf == true) {
	    std::multimap<double, unsigned int> hcalKfElems;
	    block.associatedElements( KfGsf_index,linkData,
				      hcalKfElems,
				      reco::PFBlockElement::HCAL,
				      reco::PFBlock::LINKTEST_ALL );	
	    for( std::multimap<double, unsigned int>::iterator it = hcalKfElems.begin();
		 it != hcalKfElems.end();it++) {
	      unsigned int index = it->second;
	      if(localactive[index] == true) {
		
		// Check that this cluster is not closer to another KF
		std::multimap<double, unsigned> sortedKf;
		block.associatedElements( index,  linkData,
					  sortedKf,
					  reco::PFBlockElement::TRACK,
					  reco::PFBlock::LINKTEST_ALL );
		unsigned jKf = sortedKf.begin()->second;
		if ( jKf != KfGsf_index) continue; 
	 	EcalElemsIndex.push_back(index);
 		localactive[index] = false;
	      }
	    }
	  }
	  // Find Other Primary Tracks Associated to the same Gsf Clusters
	  std::multimap<double, unsigned int> kfEtraElems;
	  block.associatedElements( EcalIndex[ikeyecal],linkData,
				    kfEtraElems,
				    reco::PFBlockElement::TRACK,
				    reco::PFBlock::LINKTEST_ALL );
	  if(!kfEtraElems.empty()) {
	    for( std::multimap<double, unsigned int>::iterator it = kfEtraElems.begin();
		 it != kfEtraElems.end();it++) {
	      unsigned int index = it->second;

	      // 19 Mar 2010 do not consider here tracks from gamma conv
	     //  const reco::PFBlockElementTrack * kfTk =  
             //  dynamic_cast<const reco::PFBlockElementTrack*>((&elements[index]));	     
	      // DANIELE ?  It is not need because of the local locking 
	      //   if(kfTk->isLinkedToDisplacedVertex()) continue;

	      bool thisIsAMuon = false;
	      thisIsAMuon =  PFMuonAlgo::isMuon(elements[index]);
	      if (DebugSetLinksDetailed && thisIsAMuon)
		cout << " This is a Muon: index " << index << endl;
	      if(localactive[index] == true && !thisIsAMuon) {
		if(index != KfGsf_index) {
		  // Check that this track is not closer to another ECAL cluster
		  // Not Sure here I need this step
		  std::multimap<double, unsigned> sortedECAL;
		  block.associatedElements( index,  linkData,
					    sortedECAL,
					    reco::PFBlockElement::ECAL,
					    reco::PFBlock::LINKTEST_ALL );
		  unsigned jEcal = sortedECAL.begin()->second;
		  if ( jEcal != EcalIndex[ikeyecal]) continue; 
		  EcalElemsIndex.push_back(index);
		  localactive[index] = false;
		}
	      }
	    }
	  }	  

	}
	associatedToEcal_.insert(pair<unsigned int,vector<unsigned int> >(EcalIndex[ikeyecal],EcalElemsIndex));
      }
    }// end type GSF
  } // endis there a gsf track
  
  // ******************* End Link *****************************

  // 
  // Below is only for debugging printout 
  if (DebugSetLinksSummary) {
    if(IsThereAGoodGSFTrack) {
      if (DebugSetLinksSummary) cout << " -- The Link Summary --" << endl;
      for(map<unsigned int,vector<unsigned int> >::iterator it = associatedToGsf_.begin();
	  it != associatedToGsf_.end(); it++) {
	
	if (DebugSetLinksSummary) cout << " AssoGsf " << it->first << endl;
	vector<unsigned int> eleasso = it->second;
	for(unsigned int i=0;i<eleasso.size();i++) {
	  PFBlockElement::Type type = elements[eleasso[i]].type();
	  if(type == reco::PFBlockElement::BREM) {
	    if (DebugSetLinksSummary) 
	      cout << " AssoGsfElements BREM " <<  eleasso[i] <<  endl;
	  }
	  else if(type == reco::PFBlockElement::ECAL) {
	    if (DebugSetLinksSummary) 
	      cout << " AssoGsfElements ECAL " <<  eleasso[i] <<  endl;
	  }
	  else if(type == reco::PFBlockElement::TRACK) {
	    if (DebugSetLinksSummary) 
	      cout << " AssoGsfElements KF " <<  eleasso[i] <<  endl;
	  }
	  else {
	    if (DebugSetLinksSummary) 
	      cout << " AssoGsfElements ????? " <<  eleasso[i] <<  endl;
	  }
	}
      }
      
      for(map<unsigned int,vector<unsigned int> >::iterator it = associatedToBrems_.begin();
	  it != associatedToBrems_.end(); it++) {
	if (DebugSetLinksSummary) cout << " AssoBrem " << it->first << endl;
	vector<unsigned int> eleasso = it->second;
	for(unsigned int i=0;i<eleasso.size();i++) {
	  PFBlockElement::Type type = elements[eleasso[i]].type();
	  if(type == reco::PFBlockElement::ECAL) {
	    if (DebugSetLinksSummary) 
	      cout << " AssoBremElements ECAL " <<  eleasso[i] <<  endl;
	  }
	  else {
	    if (DebugSetLinksSummary) 
	      cout << " AssoBremElements ????? " <<  eleasso[i] <<  endl;
	  }
	}
      }
      
      for(map<unsigned int,vector<unsigned int> >::iterator it = associatedToEcal_.begin();
	  it != associatedToEcal_.end(); it++) {
	if (DebugSetLinksSummary) cout << " AssoECAL " << it->first << endl;
	vector<unsigned int> eleasso = it->second;
	for(unsigned int i=0;i<eleasso.size();i++) {
	  PFBlockElement::Type type = elements[eleasso[i]].type();
	  if(type == reco::PFBlockElement::PS1) {
	    if (DebugSetLinksSummary) 
	      cout << " AssoECALElements PS1  " <<  eleasso[i] <<  endl;
	  }
	  else if(type == reco::PFBlockElement::PS2) {
	    if (DebugSetLinksSummary) 
	      cout << " AssoECALElements PS2  " <<  eleasso[i] <<  endl;
	  }
	  else if(type == reco::PFBlockElement::HCAL) {
	    if (DebugSetLinksSummary) 
	      cout << " AssoECALElements HCAL  " <<  eleasso[i] <<  endl;
	  }
	  else {
	    if (DebugSetLinksSummary) 
	      cout << " AssoHCALElements ????? " <<  eleasso[i] <<  endl;
	  }
	}
      }
      if (DebugSetLinksSummary) 
	cout << "-- End Summary --" <<  endl;
    }
    
  }
  // EndPrintOut
  return IsThereAGoodGSFTrack;
}
// This function get the associatedToGsf and associatedToBrems maps and run the final electronID. 
void PFElectronAlgo::SetIDOutputs(const reco::PFBlockRef&  blockRef,
					AssMap& associatedToGsf_,
					AssMap& associatedToBrems_,
					AssMap& associatedToEcal_,
					const reco::Vertex & primaryVertex){
  //PFEnergyCalibration pfcalib_;  
  const reco::PFBlock& block = *blockRef;
  const PFBlock::LinkData& linkData =  block.linkData();     
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  bool DebugIDOutputs = false;
  if(DebugIDOutputs) cout << " ######## Enter in SetIDOutputs #########" << endl;
  
  unsigned int cgsf=0;
  for (map<unsigned int,vector<unsigned int> >::iterator igsf = associatedToGsf_.begin();
       igsf != associatedToGsf_.end(); igsf++,cgsf++) {

    float Ene_ecalgsf = 0.;
    float Ene_hcalgsf = 0.;
    double sigmaEtaEta = 0.;
    float deta_gsfecal = 0.;
    float Ene_ecalbrem = 0.;
    float Ene_extraecalgsf = 0.;
    bool LateBrem = false;
    // bool EarlyBrem = false;
    int FirstBrem = 1000;
    unsigned int ecalGsf_index = 100000;
    unsigned int kf_index = 100000;
    // unsigned int nhits_gsf = 0;
    int NumBrem = 0;
    reco::TrackRef RefKF;  
    double posX=0.;
    double posY=0.;
    double posZ=0.;

    unsigned int gsf_index =  igsf->first;
    const reco::PFBlockElementGsfTrack * GsfEl  =  
      dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[gsf_index]));
    reco::GsfTrackRef RefGSF = GsfEl->GsftrackRef();
    float Ein_gsf   = 0.;
    if (RefGSF.isNonnull()) {
      float m_el=0.00051;
      Ein_gsf =sqrt(RefGSF->pMode()*
		    RefGSF->pMode()+m_el*m_el);
      // nhits_gsf = RefGSF->hitPattern().trackerLayersWithMeasurement();
    }
    float Eout_gsf = GsfEl->Pout().t();
    float Etaout_gsf = GsfEl->positionAtECALEntrance().eta();


    if (DebugIDOutputs)  
      cout << " setIdOutput! GSF Track: Ein " <<  Ein_gsf 
	   << " eta,phi " <<  Etaout_gsf
	   <<", " << GsfEl->positionAtECALEntrance().phi() << endl;
    

    vector<unsigned int> assogsf_index = igsf->second;
    bool FirstEcalGsf = true;
    for  (unsigned int ielegsf=0;ielegsf<assogsf_index.size();ielegsf++) {
      PFBlockElement::Type assoele_type = elements[(assogsf_index[ielegsf])].type();
      
      
      // The RefKf is needed to build pure tracking observables
      if(assoele_type == reco::PFBlockElement::TRACK) {
	const reco::PFBlockElementTrack * KfTk =  
	  dynamic_cast<const reco::PFBlockElementTrack*>((&elements[(assogsf_index[ielegsf])]));
	// 19 Mar 2010 do not consider here track from gamma conv
	
	bool isPrim = isPrimaryTrack(*KfTk,*GsfEl);
 	if(!isPrim) continue;
	RefKF = KfTk->trackRef();
	kf_index = assogsf_index[ielegsf];
      }
      
   
      if  (assoele_type == reco::PFBlockElement::ECAL) {
	unsigned int keyecalgsf = assogsf_index[ielegsf];
	vector<unsigned int> assoecalgsf_index = associatedToEcal_.find(keyecalgsf)->second;
	
	vector<double> ps1Ene(0);
	vector<double> ps2Ene(0);
	for(unsigned int ips =0; ips<assoecalgsf_index.size();ips++) {
	  PFBlockElement::Type typeassoecal = elements[(assoecalgsf_index[ips])].type();
	  if  (typeassoecal == reco::PFBlockElement::PS1) {  
	    PFClusterRef  psref = elements[(assoecalgsf_index[ips])].clusterRef();
	    ps1Ene.push_back(psref->energy());
	  }
	  if  (typeassoecal == reco::PFBlockElement::PS2) {  
	    PFClusterRef  psref = elements[(assoecalgsf_index[ips])].clusterRef();
	    ps2Ene.push_back(psref->energy());
	  }
	  if  (typeassoecal == reco::PFBlockElement::HCAL) {
	    const reco::PFBlockElementCluster * clust =  
	      dynamic_cast<const reco::PFBlockElementCluster*>((&elements[(assoecalgsf_index[ips])])); 
	    Ene_hcalgsf+=clust->clusterRef()->energy();
	  }
	}
	if (FirstEcalGsf) {
	  FirstEcalGsf = false;
	  ecalGsf_index = assogsf_index[ielegsf];
	  reco::PFClusterRef clusterRef = elements[(assogsf_index[ielegsf])].clusterRef();	  
	  double ps1,ps2;
	  ps1=ps2=0.;
	  //	  Ene_ecalgsf = pfcalib_.energyEm(*clusterRef,ps1Ene,ps2Ene);	  
	  Ene_ecalgsf = thePFEnergyCalibration_->energyEm(*clusterRef,ps1Ene,ps2Ene,ps1,ps2,applyCrackCorrections_);	  
	  //	  std::cout << "Test " << Ene_ecalgsf <<  " PS1 / PS2 " << ps1 << " " << ps2 << std::endl;
	  posX += Ene_ecalgsf * clusterRef->position().X();
	  posY += Ene_ecalgsf * clusterRef->position().Y();
	  posZ += Ene_ecalgsf * clusterRef->position().Z();

	  if (DebugIDOutputs)  
	    cout << " setIdOutput! GSF ECAL Cluster E " << Ene_ecalgsf
		 << " eta,phi " << clusterRef->position().eta()
		 <<", " <<  clusterRef->position().phi() << endl;
	  
	  deta_gsfecal = clusterRef->position().eta() - Etaout_gsf;
	  
	  vector< const reco::PFCluster * > pfClust_vec(0);
	  pfClust_vec.clear();
	  pfClust_vec.push_back(&(*clusterRef));
	  
	  PFClusterWidthAlgo pfwidth(pfClust_vec);
	  sigmaEtaEta = pfwidth.pflowSigmaEtaEta();


	}
	else {
	  reco::PFClusterRef clusterRef = elements[(assogsf_index[ielegsf])].clusterRef();	  	  
	  float TempClus_energy = thePFEnergyCalibration_->energyEm(*clusterRef,ps1Ene,ps2Ene,applyCrackCorrections_);	 
	  Ene_extraecalgsf += TempClus_energy;
	  posX += TempClus_energy * clusterRef->position().X();
	  posY += TempClus_energy * clusterRef->position().Y();
	  posZ += TempClus_energy * clusterRef->position().Z();	  

	  if (DebugIDOutputs)
	    cout << " setIdOutput! Extra ECAL Cluster E " 
		 << TempClus_energy << " Tot " <<  Ene_extraecalgsf << endl;
	}
      } // end type Ecal

      
      if  (assoele_type == reco::PFBlockElement::BREM) {
	unsigned int brem_index = assogsf_index[ielegsf];
	const reco::PFBlockElementBrem * BremEl  =  
	  dynamic_cast<const reco::PFBlockElementBrem*>((&elements[brem_index]));
	int TrajPos = (BremEl->indTrajPoint())-2;
	//if (TrajPos <= 3) EarlyBrem = true;
	if (TrajPos < FirstBrem) FirstBrem = TrajPos;

	vector<unsigned int> assobrem_index = associatedToBrems_.find(brem_index)->second;
	for (unsigned int ibrem = 0; ibrem < assobrem_index.size(); ibrem++){
	  if (elements[(assobrem_index[ibrem])].type() == reco::PFBlockElement::ECAL) {
	    unsigned int keyecalbrem = assobrem_index[ibrem];
	    vector<unsigned int> assoelebrem_index = associatedToEcal_.find(keyecalbrem)->second;
	    vector<double> ps1EneFromBrem(0);
	    vector<double> ps2EneFromBrem(0);
	    for (unsigned int ielebrem=0; ielebrem<assoelebrem_index.size();ielebrem++) {
	      if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS1) {
		PFClusterRef  psref = elements[(assoelebrem_index[ielebrem])].clusterRef();
		ps1EneFromBrem.push_back(psref->energy());
	      }
	      if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS2) {
		PFClusterRef  psref = elements[(assoelebrem_index[ielebrem])].clusterRef();
		ps2EneFromBrem.push_back(psref->energy());
	      }	       
	    }
	    // check if it is a compatible cluster also with the gsf track
	    if( assobrem_index[ibrem] !=  ecalGsf_index) {
	      reco::PFClusterRef clusterRef = 
		elements[(assobrem_index[ibrem])].clusterRef();
	      float BremClus_energy = thePFEnergyCalibration_->energyEm(*clusterRef,ps1EneFromBrem,ps2EneFromBrem,applyCrackCorrections_);
	      Ene_ecalbrem += BremClus_energy;
	      posX +=  BremClus_energy * clusterRef->position().X();
	      posY +=  BremClus_energy * clusterRef->position().Y();
	      posZ +=  BremClus_energy * clusterRef->position().Z();	  
	      
	      NumBrem++;
	      if (DebugIDOutputs) cout << " setIdOutput::BREM Cluster " 
				       <<  BremClus_energy << " eta,phi " 
				       <<  clusterRef->position().eta()  <<", " 
				       << clusterRef->position().phi() << endl;
	    }
	    else {
	      LateBrem = true;
	    }
	  }
	}
      }	   
    } 
    if (Ene_ecalgsf > 0.) {
      // here build the new BDT observables
      
      //  ***** Normalization observables ****
      if(RefGSF.isNonnull()) {
	PFCandidateElectronExtra myExtra(RefGSF) ;
	myExtra.setGsfTrackPout(GsfEl->Pout());
	myExtra.setKfTrackRef(RefKF);
	float Pt_gsf = RefGSF->ptMode();
	lnPt_gsf = log(Pt_gsf);
	Eta_gsf = RefGSF->etaMode();
	
	// **** Pure tracking observables. 
	if(RefGSF->ptModeError() > 0.)
	  dPtOverPt_gsf = RefGSF->ptModeError()/Pt_gsf;
	
	nhit_gsf= RefGSF->hitPattern().trackerLayersWithMeasurement();
	chi2_gsf = RefGSF->normalizedChi2();
	// change GsfEl->Pout().pt() as soon the PoutMode is on the GsfTrack DataFormat
	DPtOverPt_gsf =  (RefGSF->ptMode() - GsfEl->Pout().pt())/RefGSF->ptMode();
	
	
	nhit_kf = 0;
	chi2_kf = -0.01;
	DPtOverPt_kf = -0.01;
	if (RefKF.isNonnull()) {
	  nhit_kf= RefKF->hitPattern().trackerLayersWithMeasurement();
	  chi2_kf = RefKF->normalizedChi2();
	  // Not used, strange behaviour to be checked. And Kf->OuterPt is 
	  // in track extra. 
	  //DPtOverPt_kf = 
	  //  (RefKF->pt() - RefKF->outerPt())/RefKF->pt();
	  
	}
	
	// **** Tracker-Ecal-Hcal observables 
	EtotPinMode        =  (Ene_ecalgsf + Ene_ecalbrem + Ene_extraecalgsf) / Ein_gsf; 
	EGsfPoutMode       =  Ene_ecalgsf/Eout_gsf;
	EtotBremPinPoutMode = Ene_ecalbrem /(Ein_gsf - Eout_gsf);
	DEtaGsfEcalClust  = fabs(deta_gsfecal);
	myExtra.setSigmaEtaEta(sigmaEtaEta);
	myExtra.setDeltaEta(DEtaGsfEcalClust);
	SigmaEtaEta = log(sigmaEtaEta);
	
	lateBrem = -1;
	firstBrem = -1;
	earlyBrem = -1;
	if(NumBrem > 0) {
	  if (LateBrem) lateBrem = 1;
	  else lateBrem = 0;
	  firstBrem = FirstBrem;
	  if(FirstBrem < 4) earlyBrem = 1;
	  else earlyBrem = 0;
	}      
	
	HOverHE = Ene_hcalgsf/(Ene_hcalgsf + Ene_ecalgsf);
	HOverPin = Ene_hcalgsf / Ein_gsf;
	myExtra.setHadEnergy(Ene_hcalgsf);
	myExtra.setEarlyBrem(earlyBrem);
	myExtra.setLateBrem(lateBrem);
	//	std::cout<< " Inserting in extra " << electronExtra_.size() << std::endl;

	// Put cuts and access the BDT output
	if(DPtOverPt_gsf < -0.2) DPtOverPt_gsf = -0.2;
	if(DPtOverPt_gsf > 1.) DPtOverPt_gsf = 1.;
	
	if(dPtOverPt_gsf > 0.3) dPtOverPt_gsf = 0.3;
	
	if(chi2_gsf > 10.) chi2_gsf = 10.;
	
	if(DPtOverPt_kf < -0.2) DPtOverPt_kf = -0.2;
	if(DPtOverPt_kf > 1.) DPtOverPt_kf = 1.;
	
	if(chi2_kf > 10.) chi2_kf = 10.;
	
	if(EtotPinMode < 0.) EtotPinMode = 0.;
	if(EtotPinMode > 5.) EtotPinMode = 5.;
	
	if(EGsfPoutMode < 0.) EGsfPoutMode = 0.;
	if(EGsfPoutMode > 5.) EGsfPoutMode = 5.;
	
	if(EtotBremPinPoutMode < 0.) EtotBremPinPoutMode = 0.01;
	if(EtotBremPinPoutMode > 5.) EtotBremPinPoutMode = 5.;
	
	if(DEtaGsfEcalClust > 0.1) DEtaGsfEcalClust = 0.1;
	
	if(SigmaEtaEta < -14) SigmaEtaEta = -14;
	
	if(HOverPin < 0.) HOverPin = 0.;
	if(HOverPin > 5.) HOverPin = 5.;
	double mvaValue = tmvaReader_->EvaluateMVA("BDT");
	
	// add output observables 
	BDToutput_[cgsf] = mvaValue;
	myExtra.setMVA(mvaValue);
	electronExtra_.push_back(myExtra);


	// IMPORTANT Additional conditions
	if(mvaValue > mvaEleCut_) {
	  // Check if the ecal cluster is isolated. 
	  //
	  // If there is at least one extra track and H/H+E > 0.05 or SumP(ExtraKf)/EGsf or  
	  // #Tracks > 3 I leave this job to PFAlgo, otherwise I lock those extra tracks. 
	  // Note:
	  // The tracks coming from the 4 step of the iterative tracking are not considered, 
	  // because they can come from secondary converted brems. 
	  // They are locked to avoid double counting.
	  // The lock is done in SetActivate function. 
	  // All this can improved but it is already a good step. 
	  
	  
	  // Find if the cluster is isolated. 
	  unsigned int iextratrack = 0;
	  unsigned int itrackHcalLinked = 0;
	  float SumExtraKfP = 0.;


	  double Etotal = Ene_ecalgsf + Ene_ecalbrem + Ene_extraecalgsf;
	  posX /=Etotal;
	  posY /=Etotal;
	  posZ /=Etotal;
	  math::XYZPoint sc_pflow(posX,posY,posZ);
	  double ETtotal = Etotal*sin(sc_pflow.Theta());
	  double phiTrack = RefGSF->phiMode();
	  double dphi_normalsc = sc_pflow.Phi() - phiTrack;
	  if ( dphi_normalsc < -M_PI ) 
	    dphi_normalsc = dphi_normalsc + 2.*M_PI;
	  else if ( dphi_normalsc > M_PI ) 
	    dphi_normalsc = dphi_normalsc - 2.*M_PI;
	  dphi_normalsc = fabs(dphi_normalsc);
	
	  
	  if(ecalGsf_index < 100000) {
	    vector<unsigned int> assoecalgsf_index = associatedToEcal_.find(ecalGsf_index)->second;
	    for(unsigned int itrk =0; itrk<assoecalgsf_index.size();itrk++) {
	      PFBlockElement::Type typeassoecal = elements[(assoecalgsf_index[itrk])].type();
	      if(typeassoecal == reco::PFBlockElement::TRACK) {
		const reco::PFBlockElementTrack * kfTk =  
		  dynamic_cast<const reco::PFBlockElementTrack*>((&elements[(assoecalgsf_index[itrk])]));
		// 19 Mar 2010 do not consider here tracks from gamma conv
		// This should be not needed because the seleted extra tracks from GammaConv
		// will be locked and can not be associated to the ecal elements 
		//if(kfTk->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)) continue;
		
		
		const reco::TrackRef& trackref =  kfTk->trackRef();
		bool goodTrack = PFTrackAlgoTools::isGoodForEGM(trackref->algo());
		// iter0, iter1, iter2, iter3 = Algo < 3
		// algo 4,5,6,7
		int nexhits = trackref->hitPattern().numberOfLostHits(HitPattern::MISSING_INNER_HITS);
		
		bool trackIsFromPrimaryVertex = false;
		for (Vertex::trackRef_iterator trackIt = primaryVertex.tracks_begin(); trackIt != primaryVertex.tracks_end(); ++trackIt) {
		  if ( (*trackIt).castTo<TrackRef>() == trackref ) {
		    trackIsFromPrimaryVertex = true;
		    break;
		  }
		}
		
		// probably we could now remove the algo request?? 
		if(goodTrack  && nexhits == 0 && trackIsFromPrimaryVertex) {
		  //if(Algo < 3) 
		  if(DebugIDOutputs) 
		    cout << " The ecalGsf cluster is not isolated: >0 KF extra with algo < 3" << endl;
		  
		  float p_trk = trackref->p();
		  
		  // expected number of inner hits
		  
		  if(DebugIDOutputs) 
		    cout << "  p_trk " <<  p_trk
			 << " nexhits " << nexhits << endl;
		  
		  SumExtraKfP += p_trk;
		  iextratrack++;
		  // Check if these extra tracks are HCAL linked
		  std::multimap<double, unsigned int> hcalKfElems;
		  block.associatedElements( assoecalgsf_index[itrk],linkData,
					    hcalKfElems,
					    reco::PFBlockElement::HCAL,
					    reco::PFBlock::LINKTEST_ALL );
		  if(!hcalKfElems.empty()) {
		    itrackHcalLinked++;
		  }
		}
	      } 
	    }
	  }
	  if( iextratrack > 0) {
	    //if(iextratrack > 3 || HOverHE > 0.05 || (SumExtraKfP/Ene_ecalgsf) > 1. 
	    if(iextratrack > 3 || Ene_hcalgsf > 10 || (SumExtraKfP/Ene_ecalgsf) > 1. 
	       || (ETtotal > 50. && iextratrack > 1 && (Ene_hcalgsf/Ene_ecalgsf) > 0.1) ) {
	      if(DebugIDOutputs) 
		cout << " *****This electron candidate is discarded: Non isolated  # tracks "		
		     << iextratrack << " HOverHE " << HOverHE 
		     << " SumExtraKfP/Ene_ecalgsf " << SumExtraKfP/Ene_ecalgsf 
		     << " SumExtraKfP " << SumExtraKfP 
		     << " Ene_ecalgsf " << Ene_ecalgsf
		     << " ETtotal " << ETtotal
		     << " Ene_hcalgsf/Ene_ecalgsf " << Ene_hcalgsf/Ene_ecalgsf
		     << endl;
	      
	      BDToutput_[cgsf] = mvaValue-2.;
	      lockExtraKf_[cgsf] = false;
	    }
	    // if the Ecluster ~ Ptrack and the extra tracks are HCAL linked 
	    // the electron is retained and the kf tracks are not locked
	    if( (fabs(1.-EtotPinMode) < 0.2 && (fabs(Eta_gsf) < 1.0 || fabs(Eta_gsf) > 2.0)) ||
		((EtotPinMode < 1.1 && EtotPinMode > 0.6) && (fabs(Eta_gsf) >= 1.0 && fabs(Eta_gsf) <= 2.0))) {
	      if( fabs(1.-EGsfPoutMode) < 0.5 && 
		  (itrackHcalLinked == iextratrack) && 
		  kf_index < 100000 ) {
		
		BDToutput_[cgsf] = mvaValue;
		lockExtraKf_[cgsf] = false;
		if(DebugIDOutputs)
		  cout << " *****This electron is reactivated  # tracks "		
		       << iextratrack << " #tracks hcal linked " << itrackHcalLinked 
		       << " SumExtraKfP/Ene_ecalgsf " << SumExtraKfP/Ene_ecalgsf  
		       << " EtotPinMode " << EtotPinMode << " EGsfPoutMode " << EGsfPoutMode 
		       << " eta gsf " << fabs(Eta_gsf) << " kf index " << kf_index <<endl;
	      }
	    }
	  }
	  // This is a pion:
	  if (HOverPin > 1. && HOverHE > 0.1 && EtotPinMode < 0.5) {
	    if(DebugIDOutputs) 
	      cout << " *****This electron candidate is discarded  HCAL ENERGY "	
		   << " HOverPin " << HOverPin << " HOverHE " << HOverHE  << " EtotPinMode" << EtotPinMode << endl;
	    BDToutput_[cgsf] = mvaValue-4.;
	    lockExtraKf_[cgsf] = false;
	  }
	  
	  // Reject Crazy E/p values... to be understood in the future how to train a 
	  // BDT in order to avoid to select this bad electron candidates. 
	
	  if( EtotPinMode < 0.2 && EGsfPoutMode < 0.2 ) {
	    if(DebugIDOutputs) 
	      cout << " *****This electron candidate is discarded  Low ETOTPIN "
		   << " EtotPinMode " << EtotPinMode << " EGsfPoutMode " << EGsfPoutMode << endl;
	    BDToutput_[cgsf] = mvaValue-6.;
	  }
	  
	  // For not-preselected Gsf Tracks ET > 50 GeV, apply dphi preselection
	  if(ETtotal > 50. && dphi_normalsc > 0.1 ) {
	    if(DebugIDOutputs) 
	      cout << " *****This electron candidate is discarded  Large ANGLE "
		   << " ETtotal " << ETtotal << " EGsfPoutMode " << dphi_normalsc << endl;
	    BDToutput_[cgsf] =  mvaValue-6.;
	  }
 	}


	if (DebugIDOutputs) {
	  cout << " **** BDT observables ****" << endl;
	  cout << " < Normalization > " << endl;
	  cout << " Pt_gsf " << Pt_gsf << " Pin " << Ein_gsf  << " Pout " << Eout_gsf
	       << " Eta_gsf " << Eta_gsf << endl;
	  cout << " < PureTracking > " << endl;
	  cout << " dPtOverPt_gsf " << dPtOverPt_gsf 
	       << " DPtOverPt_gsf " << DPtOverPt_gsf
	       << " chi2_gsf " << chi2_gsf
	       << " nhit_gsf " << nhit_gsf
	       << " DPtOverPt_kf " << DPtOverPt_kf
	       << " chi2_kf " << chi2_kf 
	       << " nhit_kf " << nhit_kf <<  endl;
	  cout << " < track-ecal-hcal-ps " << endl;
	  cout << " EtotPinMode " << EtotPinMode 
	       << " EGsfPoutMode " << EGsfPoutMode
	       << " EtotBremPinPoutMode " << EtotBremPinPoutMode
	       << " DEtaGsfEcalClust " << DEtaGsfEcalClust 
	       << " SigmaEtaEta " << SigmaEtaEta
	       << " HOverHE " << HOverHE << " Hcal energy " << Ene_hcalgsf
	       << " HOverPin " << HOverPin 
	       << " lateBrem " << lateBrem
	       << " firstBrem " << firstBrem << endl;
	  cout << " !!!!!!!!!!!!!!!! the BDT output !!!!!!!!!!!!!!!!!: direct " << mvaValue 
	       << " corrected " << BDToutput_[cgsf] <<  endl; 
	  
	}
      }
      else {
	if (DebugIDOutputs) 
	  cout << " Gsf Ref isNULL " << endl;
	BDToutput_[cgsf] = -2.;
      }
    } else {
      if (DebugIDOutputs) 
	cout << " No clusters associated to the gsf " << endl;
      BDToutput_[cgsf] = -2.;      
    }  
    DebugIDOutputs = false;
  } // End Loop on Map1   
  return;
}
// This function get the associatedToGsf and associatedToBrems maps and  
// compute the electron 4-mom and set the pf candidate, for
// the gsf track with a BDTcut > mvaEleCut_
void PFElectronAlgo::SetCandidates(const reco::PFBlockRef&  blockRef,
					 AssMap& associatedToGsf_,
					 AssMap& associatedToBrems_,
					 AssMap& associatedToEcal_){
  
  const reco::PFBlock& block = *blockRef;
  const PFBlock::LinkData& linkData =  block.linkData();     
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  PFEnergyResolution pfresol_;
  //PFEnergyCalibration pfcalib_;

  bool DebugIDCandidates = false;
//   vector<reco::PFCluster> pfClust_vec(0);
//   pfClust_vec.clear();

  unsigned int cgsf=0;
  for (map<unsigned int,vector<unsigned int> >::iterator igsf = associatedToGsf_.begin();
       igsf != associatedToGsf_.end(); igsf++,cgsf++) {
    unsigned int gsf_index =  igsf->first;
           
    

    // They should be reset for each gsf track
    int eecal=0;
    int hcal=0;
    int charge =0; 
    // bool goodphi=true;
    math::XYZTLorentzVector momentum_kf,momentum_gsf,momentum,momentum_mean;
    float dpt=0; float dpt_gsf=0;
    float Eene=0; float dene=0; float Hene=0.;
    float RawEene = 0.;
    double posX=0.;
    double posY=0.;
    double posZ=0.;
    std::vector<float> bremEnergyVec;

    std::vector<const PFCluster*> pfSC_Clust_vec; 

    float de_gs = 0., de_me = 0., de_kf = 0.; 
    float m_el=0.00051;
    int nhit_kf=0; int nhit_gsf=0;
    bool has_gsf=false;
    bool has_kf=false;
    math::XYZTLorentzVector newmomentum;
    float ps1TotEne = 0;
    float ps2TotEne = 0;
    vector<unsigned int> elementsToAdd(0);
    reco::TrackRef RefKF;  



    elementsToAdd.push_back(gsf_index);
    const reco::PFBlockElementGsfTrack * GsfEl  =  
      dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[gsf_index]));
    const math::XYZPointF& posGsfEcalEntrance = GsfEl->positionAtECALEntrance();
    reco::GsfTrackRef RefGSF = GsfEl->GsftrackRef();
    if (RefGSF.isNonnull()) {
      
      has_gsf=true;
     
      charge= RefGSF->chargeMode();
      nhit_gsf= RefGSF->hitPattern().trackerLayersWithMeasurement();
      
      momentum_gsf.SetPx(RefGSF->pxMode());
      momentum_gsf.SetPy(RefGSF->pyMode());
      momentum_gsf.SetPz(RefGSF->pzMode());
      float ENE=sqrt(RefGSF->pMode()*
		     RefGSF->pMode()+m_el*m_el);
      
      if( DebugIDCandidates ) 
	cout << "SetCandidates:: GsfTrackRef: Ene " << ENE 
	     << " charge " << charge << " nhits " << nhit_gsf <<endl;
      
      momentum_gsf.SetE(ENE);       
      dpt_gsf=RefGSF->ptModeError()*
	(RefGSF->pMode()/RefGSF->ptMode());
      
      momentum_mean.SetPx(RefGSF->px());
      momentum_mean.SetPy(RefGSF->py());
      momentum_mean.SetPz(RefGSF->pz());
      float ENEm=sqrt(RefGSF->p()*
		      RefGSF->p()+m_el*m_el);
      momentum_mean.SetE(ENEm);       
      //       dpt_mean=RefGSF->ptError()*
      // 	(RefGSF->p()/RefGSF->pt());  
    }
    else {
      if( DebugIDCandidates ) 
	cout <<  "SetCandidates:: !!!!  NULL GSF Track Ref " << endl;	
    } 

    //    vector<unsigned int> assogsf_index =  associatedToGsf_[igsf].second;
    vector<unsigned int> assogsf_index = igsf->second;
    unsigned int ecalGsf_index = 100000;
    bool FirstEcalGsf = true;
    for  (unsigned int ielegsf=0;ielegsf<assogsf_index.size();ielegsf++) {
      PFBlockElement::Type assoele_type = elements[(assogsf_index[ielegsf])].type();
      if  (assoele_type == reco::PFBlockElement::TRACK) {
	elementsToAdd.push_back((assogsf_index[ielegsf])); // Daniele
	const reco::PFBlockElementTrack * KfTk =  
	  dynamic_cast<const reco::PFBlockElementTrack*>((&elements[(assogsf_index[ielegsf])]));
	// 19 Mar 2010 do not consider here track from gamam conv
	bool isPrim = isPrimaryTrack(*KfTk,*GsfEl);
	if(!isPrim) continue;
 	
	RefKF = KfTk->trackRef();
	if (RefKF.isNonnull()) {
	  has_kf = true;
	  // dpt_kf=(RefKF->ptError()*RefKF->ptError());
	  nhit_kf=RefKF->hitPattern().trackerLayersWithMeasurement();
	  momentum_kf.SetPx(RefKF->px());
	  momentum_kf.SetPy(RefKF->py());
	  momentum_kf.SetPz(RefKF->pz());
	  float ENE=sqrt(RefKF->p()*RefKF->p()+m_el*m_el);
	  if( DebugIDCandidates ) 
	    cout << "SetCandidates:: KFTrackRef: Ene " << ENE << " nhits " << nhit_kf << endl;
	  
	  momentum_kf.SetE(ENE);
	}
	else {
	  if( DebugIDCandidates ) 
	    cout <<  "SetCandidates:: !!!! NULL KF Track Ref " << endl;
	}
      } 

      if  (assoele_type == reco::PFBlockElement::ECAL) {
	unsigned int keyecalgsf = assogsf_index[ielegsf];
	vector<unsigned int> assoecalgsf_index = associatedToEcal_.find(keyecalgsf)->second;
	vector<double> ps1Ene(0);
	vector<double> ps2Ene(0);
	// Important is the PS clusters are not saved before the ecal one, these
	// energy are not correctly assigned 
	// For the moment I get only the closest PS clusters: this has to be changed
	for(unsigned int ips =0; ips<assoecalgsf_index.size();ips++) {
	  PFBlockElement::Type typeassoecal = elements[(assoecalgsf_index[ips])].type();
	  if  (typeassoecal == reco::PFBlockElement::PS1) {  
	    PFClusterRef  psref = elements[(assoecalgsf_index[ips])].clusterRef();
	    ps1Ene.push_back(psref->energy());
	    elementsToAdd.push_back((assoecalgsf_index[ips]));
	  }
	  if  (typeassoecal == reco::PFBlockElement::PS2) {  
	    PFClusterRef  psref = elements[(assoecalgsf_index[ips])].clusterRef();
	    ps2Ene.push_back(psref->energy());
	    elementsToAdd.push_back((assoecalgsf_index[ips]));
	  }
	  if  (typeassoecal == reco::PFBlockElement::HCAL) {
	    const reco::PFBlockElementCluster * clust =  
	      dynamic_cast<const reco::PFBlockElementCluster*>((&elements[(assoecalgsf_index[ips])])); 
	    elementsToAdd.push_back((assoecalgsf_index[ips])); 
	    Hene+=clust->clusterRef()->energy();
	    hcal++;
	  }
	}
	elementsToAdd.push_back((assogsf_index[ielegsf]));


	const reco::PFBlockElementCluster * clust =  
	  dynamic_cast<const reco::PFBlockElementCluster*>((&elements[(assogsf_index[ielegsf])]));
	
	eecal++;
	
	const reco::PFCluster& cl(*clust->clusterRef());
	//pfClust_vec.push_back((*clust->clusterRef()));

	// The electron RAW energy is the energy of the corrected GSF cluster	
	double ps1,ps2;
	ps1=ps2=0.;
	//	float EE=pfcalib_.energyEm(cl,ps1Ene,ps2Ene);
	float EE = thePFEnergyCalibration_->energyEm(cl,ps1Ene,ps2Ene,ps1,ps2,applyCrackCorrections_);	  
	//	float RawEE = cl.energy();

	float ceta=cl.position().eta();
	float cphi=cl.position().phi();
	
	/*
	  float mphi=-2.97025;
	  if (ceta<0) mphi+=0.00638;
	  
	  for (int ip=1; ip<19; ip++){
	  float df= cphi - (mphi+(ip*6.283185/18));
	  if (fabs(df)<0.01) goodphi=false;
	  }
	*/

	float dE=pfresol_.getEnergyResolutionEm(EE,cl.position().eta());
	if( DebugIDCandidates ) 
	  cout << "SetCandidates:: EcalCluster: EneNoCalib " << clust->clusterRef()->energy()  
	       << " eta,phi " << ceta << "," << cphi << " Calib " <<  EE << " dE " <<  dE <<endl;

	bool elecCluster=false;
	if (FirstEcalGsf) {
	  FirstEcalGsf = false;
	  elecCluster=true;
	  ecalGsf_index = assogsf_index[ielegsf];
	  //	  std::cout << " PFElectronAlgo / Seed " << EE << std::endl;
	  RawEene += EE;
	}
	
	// create a photon/electron candidate
	math::XYZTLorentzVector clusterMomentum;
	math::XYZPoint direction=cl.position()/cl.position().R();
	clusterMomentum.SetPxPyPzE(EE*direction.x(),
				   EE*direction.y(),
				   EE*direction.z(),
				   EE);
	reco::PFCandidate cluster_Candidate((elecCluster)?charge:0,
					    clusterMomentum, 
					    (elecCluster)? reco::PFCandidate::e : reco::PFCandidate::gamma);
	
	cluster_Candidate.setPs1Energy(ps1);
	cluster_Candidate.setPs2Energy(ps2);
	// The Raw Ecal energy will be the energy of the basic cluster. 
	// It will be the corrected energy without the preshower
	cluster_Candidate.setEcalEnergy(EE-ps1-ps2,EE);
	//	      std::cout << " PFElectronAlgo, adding Brem (1) " << EE << std::endl;
	cluster_Candidate.setPositionAtECALEntrance(math::XYZPointF(cl.position()));
	cluster_Candidate.addElementInBlock(blockRef,assogsf_index[ielegsf]);
	// store the photon candidate
	std::map<unsigned int,std::vector<reco::PFCandidate> >::iterator itcheck=
	  electronConstituents_.find(cgsf);
	if(itcheck==electronConstituents_.end())
	  {		  
	    // beurk
	    std::vector<reco::PFCandidate> tmpVec;
	    tmpVec.push_back(cluster_Candidate);
	    electronConstituents_.insert(std::pair<unsigned int, std::vector<reco::PFCandidate> >
					 (cgsf,tmpVec));
	  }
	else
	  {
	    itcheck->second.push_back(cluster_Candidate);
	  }
	
	Eene+=EE;
	posX +=  EE * cl.position().X();
	posY +=  EE * cl.position().Y();
	posZ +=  EE * cl.position().Z();	  
	ps1TotEne+=ps1;
	ps2TotEne+=ps2;
	dene+=dE*dE;
	
	//MM Add cluster to the vector pfSC_Clust_vec needed for brem corrections
	pfSC_Clust_vec.push_back( &cl );

      }
      


      // Important: Add energy from the brems
      if  (assoele_type == reco::PFBlockElement::BREM) {
	unsigned int brem_index = assogsf_index[ielegsf];
	vector<unsigned int> assobrem_index = associatedToBrems_.find(brem_index)->second;
	elementsToAdd.push_back(brem_index);
	for (unsigned int ibrem = 0; ibrem < assobrem_index.size(); ibrem++){
	  if (elements[(assobrem_index[ibrem])].type() == reco::PFBlockElement::ECAL) {
	    // brem emission is from the considered gsf track
	    if( assobrem_index[ibrem] !=  ecalGsf_index) {
	      unsigned int keyecalbrem = assobrem_index[ibrem];
	      const vector<unsigned int>& assoelebrem_index = associatedToEcal_.find(keyecalbrem)->second;
	      vector<double> ps1EneFromBrem(0);
	      vector<double> ps2EneFromBrem(0);
	      for (unsigned int ielebrem=0; ielebrem<assoelebrem_index.size();ielebrem++) {
		if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS1) {
		  PFClusterRef  psref = elements[(assoelebrem_index[ielebrem])].clusterRef();
		  ps1EneFromBrem.push_back(psref->energy());
		  elementsToAdd.push_back(assoelebrem_index[ielebrem]);
		}
		if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS2) {
		  PFClusterRef  psref = elements[(assoelebrem_index[ielebrem])].clusterRef();
		  ps2EneFromBrem.push_back(psref->energy());
		  elementsToAdd.push_back(assoelebrem_index[ielebrem]);
		}	  
	      }
	      elementsToAdd.push_back(assobrem_index[ibrem]);
	      reco::PFClusterRef clusterRef = elements[(assobrem_index[ibrem])].clusterRef();
	      //pfClust_vec.push_back(*clusterRef);
	      // to get a calibrated PS energy 
	      double ps1=0;
	      double ps2=0;
	      float EE = thePFEnergyCalibration_->energyEm(*clusterRef,ps1EneFromBrem,ps2EneFromBrem,ps1,ps2,applyCrackCorrections_);
	      bremEnergyVec.push_back(EE);
	      // float RawEE  = clusterRef->energy();
	      float ceta = clusterRef->position().eta();
	      // float cphi = clusterRef->position().phi();
	      float dE=pfresol_.getEnergyResolutionEm(EE,ceta);
	      if( DebugIDCandidates ) 
		cout << "SetCandidates:: BremCluster: Ene " << EE << " dE " <<  dE <<endl;	  

	      Eene+=EE;
	      posX +=  EE * clusterRef->position().X();
	      posY +=  EE * clusterRef->position().Y();
	      posZ +=  EE * clusterRef->position().Z();	  
	      ps1TotEne+=ps1;
	      ps2TotEne+=ps2;
	      // Removed 4 March 2009. Florian. The Raw energy is the (corrected) one of the GSF cluster only
	      //	      RawEene += RawEE;
	      dene+=dE*dE;

	      //MM Add cluster to the vector pfSC_Clust_vec needed for brem corrections
	      pfSC_Clust_vec.push_back( clusterRef.get() );

	      // create a PFCandidate out of it. Watch out, it is for the e/gamma and tau only
	      // not to be used by the PFAlgo
	      math::XYZTLorentzVector photonMomentum;
	      math::XYZPoint direction=clusterRef->position()/clusterRef->position().R();
	      
	      photonMomentum.SetPxPyPzE(EE*direction.x(),
					EE*direction.y(),
					EE*direction.z(),
					EE);
	      reco::PFCandidate photon_Candidate(0,photonMomentum, reco::PFCandidate::gamma);
	      
	      photon_Candidate.setPs1Energy(ps1);
	      photon_Candidate.setPs2Energy(ps2);
	      // yes, EE, we want the raw ecal energy of the daugther to have the same definition
	      // as the GSF cluster
	      photon_Candidate.setEcalEnergy(EE-ps1-ps2,EE);
	      //	      std::cout << " PFElectronAlgo, adding Brem " << EE << std::endl;
	      photon_Candidate.setPositionAtECALEntrance(math::XYZPointF(clusterRef->position()));
	      photon_Candidate.addElementInBlock(blockRef,assobrem_index[ibrem]);

	      // store the photon candidate
	      std::map<unsigned int,std::vector<reco::PFCandidate> >::iterator itcheck=
		electronConstituents_.find(cgsf);
	      if(itcheck==electronConstituents_.end())
		{		  
		  // beurk
		  std::vector<reco::PFCandidate> tmpVec;
		  tmpVec.push_back(photon_Candidate);
		  electronConstituents_.insert(std::pair<unsigned int, std::vector<reco::PFCandidate> >
					   (cgsf,tmpVec));
		}
	      else
		{
		  itcheck->second.push_back(photon_Candidate);
		}
	    }
	  } 
	}
      }
    } // End Loop On element associated to the GSF tracks
    if (has_gsf) {
      
      // SuperCluster energy corrections
      double unCorrEene = Eene;
      double absEta = fabs(momentum_gsf.Eta());
      double emTheta = momentum_gsf.Theta();
      PFClusterWidthAlgo pfSCwidth(pfSC_Clust_vec); 
      double brLinear = pfSCwidth.pflowPhiWidth()/pfSCwidth.pflowEtaWidth(); 
      pfSC_Clust_vec.clear();
      
      if( DebugIDCandidates ) 
	cout << "PFEelectronAlgo:: absEta " << absEta  << " theta " << emTheta 
	     << " EneRaw " << Eene << " Err " << dene;
      
      // The calibrations are provided till ET = 200 GeV //No longer a such cut MM
      // Protection on at least 1 GeV energy...avoid possible divergencies at very low energy.
      if(usePFSCEleCalib_ && unCorrEene > 0.) { 
	if( absEta < 1.5) {
	  double Etene = Eene*sin(emTheta);
	  double emBR_e = thePFSCEnergyCalibration_->SCCorrFBremBarrel(Eene, Etene, brLinear); 
	  double emBR_et = emBR_e*sin(emTheta); 
	  double emCorrFull_et = thePFSCEnergyCalibration_->SCCorrEtEtaBarrel(emBR_et, absEta); 
	  Eene = emCorrFull_et/sin(emTheta);
	}
	else {
	  //  double Etene = Eene*sin(emTheta); //not needed anymore for endcaps MM
	  double emBR_e = thePFSCEnergyCalibration_->SCCorrFBremEndcap(Eene, absEta, brLinear); 
	  double emBR_et = emBR_e*sin(emTheta); 
	  double emCorrFull_et = thePFSCEnergyCalibration_->SCCorrEtEtaEndcap(emBR_et, absEta); 
	  Eene = emCorrFull_et/sin(emTheta);
	}
	dene = sqrt(dene)*(Eene/unCorrEene);
	dene = dene*dene;
      }

      if( DebugIDCandidates ) 
	cout << " EneCorrected " << Eene << " Err " << dene  << endl;

      // charge determination with the majority method
      // if the kf track exists: 2 among 3 of supercluster barycenter position
      // gsf track and kf track
      if(has_kf && unCorrEene > 0.) {
	posX /=unCorrEene;
	posY /=unCorrEene;
	posZ /=unCorrEene;
	math::XYZPoint sc_pflow(posX,posY,posZ);

	std::multimap<double, unsigned int> bremElems;
	block.associatedElements( gsf_index,linkData,
				  bremElems,
				  reco::PFBlockElement::BREM,
				  reco::PFBlock::LINKTEST_ALL );

	double phiTrack = RefGSF->phiMode();
	if(!bremElems.empty()) {
	  unsigned int brem_index =  bremElems.begin()->second;
	  const reco::PFBlockElementBrem * BremEl  =  
	    dynamic_cast<const reco::PFBlockElementBrem*>((&elements[brem_index]));
	  phiTrack = BremEl->positionAtECALEntrance().phi();
	}

	double dphi_normalsc = sc_pflow.Phi() - phiTrack;
	if ( dphi_normalsc < -M_PI ) 
	  dphi_normalsc = dphi_normalsc + 2.*M_PI;
	else if ( dphi_normalsc > M_PI ) 
	  dphi_normalsc = dphi_normalsc - 2.*M_PI;
	
	int chargeGsf = RefGSF->chargeMode();
	int chargeKf = RefKF->charge();

	int chargeSC = 0;
	if(dphi_normalsc < 0.) 
	  chargeSC = 1;
	else 
	  chargeSC = -1;
	
	if(chargeKf == chargeGsf) 
	  charge = chargeGsf;
	else if(chargeGsf == chargeSC)
	  charge = chargeGsf;
	else 
	  charge = chargeKf;

	if( DebugIDCandidates ) 
	  cout << "PFElectronAlgo:: charge determination " 
	       << " charge GSF " << chargeGsf 
	       << " charge KF " << chargeKf 
	       << " charge SC " << chargeSC
	       << " Final Charge " << charge << endl;
	
      }
       
      // Think about this... 
      if ((nhit_gsf<8) && (has_kf)){
	
	// Use Hene if some condition.... 
	
	momentum=momentum_kf;
	float Fe=Eene;
	float scale= Fe/momentum.E();
	
	// Daniele Changed
	if (Eene < 0.0001) {
	  Fe = momentum.E();
	  scale = 1.;
	}


	newmomentum.SetPxPyPzE(scale*momentum.Px(),
			       scale*momentum.Py(),
			       scale*momentum.Pz(),Fe);
	if( DebugIDCandidates ) 
	  cout << "SetCandidates:: (nhit_gsf<8) && (has_kf):: pt " << newmomentum.pt() << " Ene " <<  Fe <<endl;

	
      } 
      if ((nhit_gsf>7) || (has_kf==false)){
	if(Eene > 0.0001) {
	  de_gs=1-momentum_gsf.E()/Eene;
	  de_me=1-momentum_mean.E()/Eene;
	  de_kf=1-momentum_kf.E()/Eene;
	}

	momentum=momentum_gsf;
	dpt=1/(dpt_gsf*dpt_gsf);
	
	if(dene > 0.)
	  dene= 1./dene;
	
	float Fe = 0.;
	if(Eene > 0.0001) {
	  Fe =((dene*Eene) +(dpt*momentum.E()))/(dene+dpt);
	}
	else {
	  Fe=momentum.E();
	}
	
	if ((de_gs>0.05)&&(de_kf>0.05)){
	  Fe=Eene;
	}
	if ((de_gs<-0.1)&&(de_me<-0.1) &&(de_kf<0.) && 
	    (momentum.E()/dpt_gsf) > 5. && momentum_gsf.pt() < 30.){
	  Fe=momentum.E();
	}
	float scale= Fe/momentum.E();
	
	newmomentum.SetPxPyPzE(scale*momentum.Px(),
			       scale*momentum.Py(),
			       scale*momentum.Pz(),Fe);
	if( DebugIDCandidates ) 
	  cout << "SetCandidates::(nhit_gsf>7) || (has_kf==false)  " << newmomentum.pt() << " Ene " <<  Fe <<endl;
	
	
      }
      if (newmomentum.pt()>0.5){
	
	// the pf candidate are created: we need to set something more? 
	// IMPORTANT -> We need the gsftrackRef, not only the TrackRef??

	if( DebugIDCandidates )
	  cout << "SetCandidates:: I am before doing candidate " <<endl;
	
	//vector with the cluster energies (for the extra)
	std::vector<float> clusterEnergyVec;
	clusterEnergyVec.push_back(RawEene);
	clusterEnergyVec.insert(clusterEnergyVec.end(),bremEnergyVec.begin(),bremEnergyVec.end());

	// add the information in the extra
	std::vector<reco::PFCandidateElectronExtra>::iterator itextra;
	PFElectronExtraEqual myExtraEqual(RefGSF);
	itextra=find_if(electronExtra_.begin(),electronExtra_.end(),myExtraEqual);
	if(itextra!=electronExtra_.end()) {
	  itextra->setClusterEnergies(clusterEnergyVec);
	}
	else {
	  if(RawEene>0.) 
	    std::cout << " There is a big problem with the electron extra, PFElectronAlgo should crash soon " << RawEene << std::endl;
	}

	reco::PFCandidate::ParticleType particleType 
	  = reco::PFCandidate::e;
	reco::PFCandidate temp_Candidate;
	temp_Candidate = PFCandidate(charge,newmomentum,particleType);
	temp_Candidate.set_mva_e_pi(BDToutput_[cgsf]);
	temp_Candidate.setEcalEnergy(RawEene,Eene);
	// Note the Hcal energy is set but the element is never locked 
	temp_Candidate.setHcalEnergy(Hene,Hene);  
	temp_Candidate.setPs1Energy(ps1TotEne);
	temp_Candidate.setPs2Energy(ps2TotEne);
	temp_Candidate.setTrackRef(RefKF);   
	// This reference could be NULL it is needed a protection? 
	temp_Candidate.setGsfTrackRef(RefGSF);
	temp_Candidate.setPositionAtECALEntrance(posGsfEcalEntrance);
	// Add Vertex
	temp_Candidate.setVertexSource(PFCandidate::kGSFVertex);
	
	// save the superclusterRef when available
	if(RefGSF->extra().isAvailable() && RefGSF->extra()->seedRef().isAvailable()) {
	  reco::ElectronSeedRef seedRef=  RefGSF->extra()->seedRef().castTo<reco::ElectronSeedRef>();
	  if(seedRef.isAvailable() && seedRef->isEcalDriven()) {
	    reco::SuperClusterRef scRef = seedRef->caloCluster().castTo<reco::SuperClusterRef>();
	    if(scRef.isNonnull())  
	      temp_Candidate.setSuperClusterRef(scRef);
	  }
	}

	if( DebugIDCandidates ) 
	  cout << "SetCandidates:: I am after doing candidate " <<endl;
	
	for (unsigned int elad=0; elad<elementsToAdd.size();elad++){
	  temp_Candidate.addElementInBlock(blockRef,elementsToAdd[elad]);
	}

	// now add the photons to this candidate
	std::map<unsigned int, std::vector<reco::PFCandidate> >::const_iterator itcluster=
	  electronConstituents_.find(cgsf);
	if(itcluster!=electronConstituents_.end())
	  {
	    const std::vector<reco::PFCandidate> & theClusters=itcluster->second;
	    unsigned nclus=theClusters.size();
	    //	    std::cout << " PFElectronAlgo " << nclus << " daugthers to add" << std::endl;
	    for(unsigned iclus=0;iclus<nclus;++iclus)
	      {
		temp_Candidate.addDaughter(theClusters[iclus]);
	      }
	  }

	// By-pass the mva is the electron has been pre-selected 
	bool bypassmva=false;
	if(useEGElectrons_) {
	  GsfElectronEqual myEqual(RefGSF);
	  std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(theGsfElectrons_->begin(),theGsfElectrons_->end(),myEqual);
	  if(itcheck!=theGsfElectrons_->end()) {
	    if(BDToutput_[cgsf] >= -1.)  {
	      // bypass the mva only if the reconstruction went fine
	      bypassmva=true;

	      if( DebugIDCandidates ) {
	      	if(BDToutput_[cgsf] < -0.1) {
		  float esceg = itcheck->caloEnergy();		
		  cout << " Attention By pass the mva " << BDToutput_[cgsf] 
		       << " SuperClusterEnergy " << esceg
		       << " PF Energy " << Eene << endl;
		  
		  cout << " hoe " << itcheck->hcalOverEcal()
		       << " tkiso04 " << itcheck->dr04TkSumPt()
		       << " ecaliso04 " << itcheck->dr04EcalRecHitSumEt()
		       << " hcaliso04 " << itcheck->dr04HcalTowerSumEt()
		       << " tkiso03 " << itcheck->dr03TkSumPt()
		       << " ecaliso03 " << itcheck->dr03EcalRecHitSumEt()
		       << " hcaliso03 " << itcheck->dr03HcalTowerSumEt() << endl;
		}
	      } // end DebugIDCandidates
	    }
	  }
	}
	
	bool mvaSelected = (BDToutput_[cgsf] >=  mvaEleCut_);
	if( mvaSelected || bypassmva ) 	  {
	    elCandidate_.push_back(temp_Candidate);
	    if(itextra!=electronExtra_.end()) 
	      itextra->setStatus(PFCandidateElectronExtra::Selected,true);
	  }
	else 	  {
	  if(itextra!=electronExtra_.end()) 
	    itextra->setStatus(PFCandidateElectronExtra::Rejected,true);
	}
	allElCandidate_.push_back(temp_Candidate);
	
	// save the status information
	if(itextra!=electronExtra_.end()) {
	  itextra->setStatus(PFCandidateElectronExtra::ECALDrivenPreselected,bypassmva);
	  itextra->setStatus(PFCandidateElectronExtra::MVASelected,mvaSelected);
	}
	

      }
      else {
	BDToutput_[cgsf] = -1.;   // if the momentum is < 0.5 ID = false, but not sure
	// it could be misleading. 
	if( DebugIDCandidates ) 
	  cout << "SetCandidates:: No Candidate Produced because of Pt cut: 0.5 " <<endl;
      }
    } 
    else {
      BDToutput_[cgsf] = -1.;  // if gsf ref does not exist
      if( DebugIDCandidates ) 
	cout << "SetCandidates:: No Candidate Produced because of No GSF Track Ref " <<endl;
    }
  } // End Loop On GSF tracks
  return;
}
// // This function get the associatedToGsf and associatedToBrems maps and for each pfelectron candidate 
// // the element used are set active == false. 
// // note: the HCAL cluster are not used for the moments and also are not locked. 
void PFElectronAlgo::SetActive(const reco::PFBlockRef&  blockRef,
				     AssMap& associatedToGsf_,
				     AssMap& associatedToBrems_,
				     AssMap& associatedToEcal_,
				     std::vector<bool>& active){
  const reco::PFBlock& block = *blockRef;
  const PFBlock::LinkData& linkData =  block.linkData();  
   
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  
  unsigned int cgsf=0;
  for (map<unsigned int,vector<unsigned int> >::iterator igsf = associatedToGsf_.begin();
       igsf != associatedToGsf_.end(); igsf++,cgsf++) {

    unsigned int gsf_index =  igsf->first;
    const reco::PFBlockElementGsfTrack * GsfEl  =  
      dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[gsf_index]));
    const reco::GsfTrackRef& RefGSF = GsfEl->GsftrackRef();

    // lock only the elements that pass the BDT cut
    bool bypassmva=false;
    if(useEGElectrons_) {
      GsfElectronEqual myEqual(RefGSF);
      std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(theGsfElectrons_->begin(),theGsfElectrons_->end(),myEqual);
      if(itcheck!=theGsfElectrons_->end()) {
	if(BDToutput_[cgsf] >= -1.) 
	  bypassmva=true;
      }
    }

    if(BDToutput_[cgsf] < mvaEleCut_ && bypassmva == false) continue;

    
    active[gsf_index] = false;  // lock the gsf
    vector<unsigned int> assogsf_index = igsf->second;
    for  (unsigned int ielegsf=0;ielegsf<assogsf_index.size();ielegsf++) {
      PFBlockElement::Type assoele_type = elements[(assogsf_index[ielegsf])].type();
      // lock the elements associated to the gsf: ECAL, Brems
      active[(assogsf_index[ielegsf])] = false;  
      if (assoele_type == reco::PFBlockElement::ECAL) {
	unsigned int keyecalgsf = assogsf_index[ielegsf];

	// added protection against fifth step
	if(!fifthStepKfTrack_.empty()) {
	  for(unsigned int itr = 0; itr < fifthStepKfTrack_.size(); itr++) {
	    if(fifthStepKfTrack_[itr].first == keyecalgsf) {
	      active[(fifthStepKfTrack_[itr].second)] = false;
	    }
	  }
	}

	// added locking for conv gsf tracks and kf tracks
	if(!convGsfTrack_.empty()) {
	  for(unsigned int iconv = 0; iconv < convGsfTrack_.size(); iconv++) {
	    if(convGsfTrack_[iconv].first == keyecalgsf) {
	      // lock the GSF track
	      active[(convGsfTrack_[iconv].second)] = false;
	      // lock also the KF track associated
	      std::multimap<double, unsigned> convKf;
	      block.associatedElements( convGsfTrack_[iconv].second,
					linkData,
					convKf,
					reco::PFBlockElement::TRACK,
					reco::PFBlock::LINKTEST_ALL );
	      if(!convKf.empty()) {
		active[convKf.begin()->second] = false;
	      }
	    }
	  }
	}


	vector<unsigned int> assoecalgsf_index = associatedToEcal_.find(keyecalgsf)->second;
	for(unsigned int ips =0; ips<assoecalgsf_index.size();ips++) {
	  // lock the elements associated to ECAL: PS1,PS2, for the moment not HCAL
	  if  (elements[(assoecalgsf_index[ips])].type() == reco::PFBlockElement::PS1) 
	    active[(assoecalgsf_index[ips])] = false;
	  if  (elements[(assoecalgsf_index[ips])].type() == reco::PFBlockElement::PS2) 
	    active[(assoecalgsf_index[ips])] = false;
	  if  (elements[(assoecalgsf_index[ips])].type() == reco::PFBlockElement::TRACK) {
	    if(lockExtraKf_[cgsf] == true) {	      
	      active[(assoecalgsf_index[ips])] = false;
	    }
	  }
	}
      } // End if ECAL
      if (assoele_type == reco::PFBlockElement::BREM) {
	unsigned int brem_index = assogsf_index[ielegsf];
	vector<unsigned int> assobrem_index = associatedToBrems_.find(brem_index)->second;
	for (unsigned int ibrem = 0; ibrem < assobrem_index.size(); ibrem++){
	  if (elements[(assobrem_index[ibrem])].type() == reco::PFBlockElement::ECAL) {
	    unsigned int keyecalbrem = assobrem_index[ibrem];
	    // lock the ecal cluster associated to the brem
	    active[(assobrem_index[ibrem])] = false;

	    // add protection against fifth step
	    if(!fifthStepKfTrack_.empty()) {
	      for(unsigned int itr = 0; itr < fifthStepKfTrack_.size(); itr++) {
		if(fifthStepKfTrack_[itr].first == keyecalbrem) {
		  active[(fifthStepKfTrack_[itr].second)] = false;
		}
	      }
	    }

	    vector<unsigned int> assoelebrem_index = associatedToEcal_.find(keyecalbrem)->second;
	    // lock the elements associated to ECAL: PS1,PS2, for the moment not HCAL
	    for (unsigned int ielebrem=0; ielebrem<assoelebrem_index.size();ielebrem++) {
	      if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS1) 
		active[(assoelebrem_index[ielebrem])] = false;
	      if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS2) 
		active[(assoelebrem_index[ielebrem])] = false;
	    }
	  }
	}
      } // End if BREM	  
    } // End loop on elements from gsf track
  }  // End loop on gsf track  
  return;
}
void PFElectronAlgo::setEGElectronCollection(const reco::GsfElectronCollection & egelectrons) {
  theGsfElectrons_ = & egelectrons;
}

bool PFElectronAlgo::isPrimaryTrack(const reco::PFBlockElementTrack& KfEl,
				    const reco::PFBlockElementGsfTrack& GsfEl) {
  bool isPrimary = false;
  
  const GsfPFRecTrackRef& gsfPfRef = GsfEl.GsftrackRefPF();
  
  if(gsfPfRef.isNonnull()) {
    const PFRecTrackRef&  kfPfRef = KfEl.trackRefPF();
    PFRecTrackRef  kfPfRef_fromGsf = (*gsfPfRef).kfPFRecTrackRef();
    if(kfPfRef.isNonnull() && kfPfRef_fromGsf.isNonnull()) {
      reco::TrackRef kfref= (*kfPfRef).trackRef();
      reco::TrackRef kfref_fromGsf = (*kfPfRef_fromGsf).trackRef();
      if(kfref.isNonnull() && kfref_fromGsf.isNonnull()) {
	if(kfref ==  kfref_fromGsf)
	  isPrimary = true;
      }
    }
  }

  return isPrimary;
}
