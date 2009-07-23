//
// Original Author: Daniele Benedetti: daniele.benedetti@cern.ch
//

#include "RecoParticleFlow/PFProducer/interface/PFElectronAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h" 
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include <iomanip>

using namespace std;
using namespace reco;
PFElectronAlgo::PFElectronAlgo(const double mvaEleCut,
			       string mvaWeightFileEleID):
  mvaEleCut_(mvaEleCut)
{
  // Set the tmva reader
  tmvaReader_ = new TMVA::Reader();
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
				   std::vector<bool>& active){

  // the maps are initialized 
  AssMap associatedToGsf;
  AssMap associatedToBrems;
  AssMap associatedToEcal;

  // should be cleaned as often as often as possible
  elCandidate_.clear();
  allElCandidate_.clear();
  photonCandidates_.clear();
  fifthStepKfTrack_.clear();
  // SetLinks finds all the elements (kf,ecal,ps,hcal,brems) 
  // associated to each gsf track
  bool blockHasGSF = SetLinks(blockRef,associatedToGsf,
			      associatedToBrems,associatedToEcal,
			      active);
  
  // check if there is at least a gsf track in the block. 
  if (blockHasGSF) {
    
    BDToutput_.clear();
    lockExtraKf_.clear();
    // For each GSF track is initialized a BDT value = -1
    BDToutput_.assign(associatedToGsf.size(),-1.);
    lockExtraKf_.assign(associatedToGsf.size(),true);

    // The FinalID is run and BDToutput values is assigned 
    SetIDOutputs(blockRef,associatedToGsf,associatedToBrems,associatedToEcal);  

    // For each GSF track that pass the BDT configurable cut a pf candidate electron is created. 
    // This function finds also the best estimation of the initial electron 4-momentum.

    SetCandidates(blockRef,associatedToGsf,associatedToBrems,associatedToEcal);
    if (elCandidate_.size() > 0 ){
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
			      std::vector<bool>& active){
  unsigned int CutIndex = 100000;
  double CutGSFECAL = 10000. ;  
  // no other cut are not used anymore. We use the default of PFBlockAlgo

  bool DebugSetLinksSummary = false;
  bool DebugSetLinksDetailed = false;

  const reco::PFBlock& block = *blockRef;
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  PFBlock::LinkData linkData =  block.linkData();  
  
  bool IsThereAGSFTrack = false;
  bool IsThereAGoodGSFTrack = false;

  vector<unsigned int> trackIs(0);
  vector<unsigned int> gsfIs(0);


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
      thisIsAMuon = kfElems.size() ? 
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
    

    // LocalLock the Elements associated to a Kf tracks and not to any Gsf
    // The clusters associated both to a kf track and to a brem tangend 
    // are then assigned only to the kf track
    // Could be improved doing this after. 

    for(unsigned int iEle=0; iEle<trackIs.size(); iEle++) {
      std::multimap<double, unsigned int> gsfElems;
      block.associatedElements( trackIs[iEle],  linkData,
				gsfElems ,
				reco::PFBlockElement::GSF,
				reco::PFBlock::LINKTEST_ALL );
      if(gsfElems.size() == 0){
	// This means that the considered kf is *not* associated
        // to any gsf track
	std::multimap<double, unsigned int> ecalKfElems;
	block.associatedElements( trackIs[iEle],linkData,
	 			  ecalKfElems,
				  reco::PFBlockElement::ECAL,
				  reco::PFBlock::LINKTEST_ALL );
	if(ecalKfElems.size() > 0) { 
	  unsigned int ecalKf_index = ecalKfElems.begin()->second;
	  if(localactive[ecalKf_index]==true) {
	    // Check if this clusters is however well linked to a gsf track
	    // if this the case the cluster is not locked.
	    
	    bool isGsfLinked = false;
	    for(unsigned int iGsf=0; iGsf<gsfIs.size(); iGsf++) {  
	      std::multimap<double, unsigned int> ecalGsfElems;
	      block.associatedElements( gsfIs[iGsf],linkData,
					ecalGsfElems,
					reco::PFBlockElement::ECAL,
					reco::PFBlock::LINKTEST_ALL );
	      if(ecalGsfElems.size() > 0) {
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
	      reco::TrackRef refKf = kfEle->trackRef();
	      unsigned int Algo = 0;
	      if (refKf.isNonnull()) 
		Algo = refKf->algo(); 
	      if(Algo < 9) {
		localactive[ecalKf_index] = false;
	      }
	      else {
		fifthStepKfTrack_.push_back(make_pair(ecalKf_index,trackIs[iEle]));
	      }
	    }
	  }
	}
      } // gsfElems.size()
    } // loop on kf tracks


    // start loop on gsf tracks
    for(unsigned int iEle=0; iEle<gsfIs.size(); iEle++) {  

      if (!localactive[(gsfIs[iEle])]) continue;  

      IsThereAGoodGSFTrack = true;
      localactive[gsfIs[iEle]] = false;
      bool ClosestEcalWithKf = false;

      if (DebugSetLinksDetailed) cout << " Gsf Index " << gsfIs[iEle] << endl;

      const reco::PFBlockElementGsfTrack * GsfEl  =  
	dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[(gsfIs[iEle])]));
      float eta_gsf = GsfEl->positionAtECALEntrance().eta();
      



      // Find Associated Kf Track elements and Ecal to KF elements
      unsigned int KfGsf_index = CutIndex;
      unsigned int KfGsf_secondIndex = CutIndex; 
      std::multimap<double, unsigned int> kfElems;
      block.associatedElements( gsfIs[iEle],linkData,
				kfElems,
				reco::PFBlockElement::TRACK,
				reco::PFBlock::LINKTEST_ALL );
      std::multimap<double, unsigned int> ecalKfElems;
      if (kfElems.size() > 0) {
	multimap<double, unsigned int>::iterator itkf = kfElems.begin();
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
      
      // Find the closest Ecal clusters associated to this Gsf
      std::multimap<double, unsigned int> ecalGsfElems;
      block.associatedElements( gsfIs[iEle],linkData,
				ecalGsfElems,
				reco::PFBlockElement::ECAL,
				reco::PFBlock::LINKTEST_ALL );    
      double ecalGsf_dist = CutGSFECAL;
      unsigned int ClosestEcalGsf_index = CutIndex;
      if (ecalGsfElems.size() > 0) {	
	if(localactive[(ecalGsfElems.begin()->second)] == true) {
	  ClosestEcalGsf_index = ecalGsfElems.begin()->second;
	  ecalGsf_dist = block.dist(gsfIs[iEle],ClosestEcalGsf_index,
				    linkData,reco::PFBlock::LINKTEST_ALL);
	
	  // Check that this cluster is not closer to another Gsf track
	   std::multimap<double, unsigned int> ecalOtherGsfElems;
	   block.associatedElements( ClosestEcalGsf_index,linkData,
				     ecalOtherGsfElems,
				     reco::PFBlockElement::GSF,
				     reco::PFBlock::LINKTEST_ALL);
	
	   if(ecalOtherGsfElems.size()>0) {
	     if(ecalOtherGsfElems.begin()->second != gsfIs[iEle]) {	     
	       ecalGsf_dist = CutGSFECAL;
	       ClosestEcalGsf_index = CutIndex;
	     }
	   }
	   // do not lock at the moment we need this for the late brem
	 }
      }
      // if any cluster is found with the gsf-ecal link, try with kf-ecal
      else if(ecalKfElems.size() > 0) {
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
	  if(ecalOtherGsfElems.size() > 0) {
	    if(ecalOtherGsfElems.begin()->second != gsfIs[iEle]) {
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
	  // check that this brem is that one coming from the same gsf
	  float checkBremGsf_dist = block.dist(sortedBremElems.begin()->second,gsfIs[iEle],
					       linkData,reco::PFBlock::LINKTEST_ALL);
	  if(checkBremGsf_dist > 0.) { 

	    //  Check that this cluster is not closer to another Gsf Track
	    // The check is not performed on KF track because the ecal clusters are aready locked.
	    std::multimap<double, unsigned int> ecalOtherGsfElems;
	    block.associatedElements( ecalBrem_index,linkData,
				      ecalOtherGsfElems,
				      reco::PFBlockElement::GSF,
				      reco::PFBlock::LINKTEST_ALL);
	    if (ecalOtherGsfElems.size() > 0) {
	      if(ecalOtherGsfElems.begin()->second != gsfIs[iEle]) continue;
	    }

	    const reco::PFBlockElementBrem * BremEl  =  
	      dynamic_cast<const reco::PFBlockElementBrem*>((&elements[sortedBremElems.begin()->second]));

	    reco::PFClusterRef clusterRef = 
	      elements[ecalBrem_index].clusterRef();
	    

	    float sortedBremEcal_deta = fabs(clusterRef->position().eta() - BremEl->positionAtECALEntrance().eta());
	    // Triangular cut on plan chi2:deta -> OLD
	    //if((0.0075*sortedBremEcal_chi2 + 100.*sortedBremEcal_deta -1.5) < 0.) {
	    if(sortedBremEcal_deta < 0.015) {
	    
	      cleanedEcalBremElems.insert(pair<unsigned int,unsigned int>( sortedBremElems.begin()->second,ecalBrem_index));
	      
	      unsigned int BremTrajP = BremEl->indTrajPoint();
	      if (BremTrajP > latestBrem_trajP) {
		latestBrem_trajP = BremTrajP;
		latestBrem_index = sortedBremElems.begin()->second;
	      }
	      if (DebugSetLinksDetailed)
		cout << " brem Index " <<  sortedBremElems.begin()->second 
		     << " associated cluster " << ecalBrem_index << " BremTrajP " << BremTrajP <<endl;
	      
	      // > 1 ecal clusters could be associated to the same brem twice: allowed N-1 link. 
	      // But the brem need to be stored once. 
	      // locallock the brem and the ecal clusters
	      localactive[ecalBrem_index] = false;  // the cluster
	      bool  alreadyfound = false;
	      for(unsigned int ii=0;ii<keyBremIndex.size();ii++) {
		if (sortedBremElems.begin()->second == keyBremIndex[ii]) alreadyfound = true;
	      }
	      if (alreadyfound == false) {
		keyBremIndex.push_back(sortedBremElems.begin()->second);
		localactive[sortedBremElems.begin()->second] = false;   // the brem
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
      }
      for (std::multimap<double, unsigned int>::iterator ii = ecalGsfElems.begin();
	   ii != ecalGsfElems.end();ii++) {	
	if(localactive[ii->second]) {
	  // Check that this cluster is not closer to another Gsf Track
	  std::multimap<double, unsigned int> ecalOtherGsfElems;
	  block.associatedElements( ii->second,linkData,
				    ecalOtherGsfElems,
				    reco::PFBlockElement::GSF,
				    reco::PFBlock::LINKTEST_ALL);
	  if(ecalOtherGsfElems.size()) {
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
       if(GsfElemIndex.size() == 0){
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

      EcalIndex.insert(EcalIndex.end(),GsfElemIndex.begin(),GsfElemIndex.end());
      if(KfGsf_index < CutIndex) 
	GsfElemIndex.push_back(KfGsf_index);
      else if(KfGsf_secondIndex < CutIndex) 
	GsfElemIndex.push_back(KfGsf_secondIndex);
      
      GsfElemIndex.insert(GsfElemIndex.end(),keyBremIndex.begin(),keyBremIndex.end());
      associatedToGsf_.insert(pair<unsigned int, vector<unsigned int> >(gsfIs[iEle],GsfElemIndex));
          

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
	    if(localactive[index] == true) {

	      // Check that this cluster is not closer to another GSF
	      std::multimap<double, unsigned> sortedGsf;
	      block.associatedElements( index,  linkData,
					sortedGsf,
					reco::PFBlockElement::GSF,
					reco::PFBlock::LINKTEST_ALL );
	      unsigned jGsf = sortedGsf.begin()->second;
	      if ( jGsf != gsfIs[iEle]) continue; 

	      EcalElemsIndex.push_back(index);
	      localactive[index] = false;
	      
	    }
	  }
	  // if the closest ecal cluster has been link with the KF, check KF - HCAL link
	  if(hcalGsfElems.size() == 0 && ClosestEcalWithKf == true) {
	    std::multimap<double, unsigned int> hcalKfElems;
	    block.associatedElements( KfGsf_index,linkData,
				      hcalKfElems,
				      reco::PFBlockElement::HCAL,
				      reco::PFBlock::LINKTEST_ALL );	
	    for( std::multimap<double, unsigned int>::iterator it = hcalKfElems.begin();
		 it != hcalKfElems.end();it++) {
	      unsigned int index = it->second;
	      if(localactive[index] == true) {
		
		// Check that this cluster is not closer to another GSF
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
	  // Find Other Tracks Associated to the same Gsf Clusters
	  std::multimap<double, unsigned int> kfEtraElems;
	  block.associatedElements( EcalIndex[ikeyecal],linkData,
				    kfEtraElems,
				    reco::PFBlockElement::TRACK,
				    reco::PFBlock::LINKTEST_ALL );
	  if(kfEtraElems.size() > 0) {
	    for( std::multimap<double, unsigned int>::iterator it = kfEtraElems.begin();
		 it != kfEtraElems.end();it++) {
	      unsigned int index = it->second;
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
					AssMap& associatedToEcal_){
  PFEnergyCalibration pfcalib_;
  const reco::PFBlock& block = *blockRef;
  PFBlock::LinkData linkData =  block.linkData();     
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  bool DebugIDOutputs = false;
  if(DebugIDOutputs) cout << " ######## Enter in SetIDOutputs #########" << endl;
  
  unsigned int cgsf=0;
  for (map<unsigned int,vector<unsigned int> >::iterator igsf = associatedToGsf_.begin();
       igsf != associatedToGsf_.end(); igsf++,cgsf++) {

    float Ene_ecalgsf = 0.;
    float Ene_hcalgsf = 0.;
    float sigmaEtaEta = 0.;
    float deta_gsfecal = 0.;
    float Ene_ecalbrem = 0.;
    float Ene_extraecalgsf = 0.;
    bool LateBrem = false;
    bool EarlyBrem = false;
    int FirstBrem = 1000;
    unsigned int ecalGsf_index = 100000;
    unsigned int kf_index = 100000;
    unsigned int nhits_gsf = 0;
    int NumBrem = 0;
    reco::TrackRef RefKF;  


    unsigned int gsf_index =  igsf->first;
    const reco::PFBlockElementGsfTrack * GsfEl  =  
      dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[gsf_index]));
    reco::GsfTrackRef RefGSF = GsfEl->GsftrackRef();
    float Ein_gsf   = 0.;
    if (RefGSF.isNonnull()) {
      float m_el=0.00051;
      Ein_gsf =sqrt(RefGSF->pMode()*
		    RefGSF->pMode()+m_el*m_el);
      nhits_gsf = RefGSF->hitPattern().trackerLayersWithMeasurement();
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
	  Ene_ecalgsf = pfcalib_.energyEm(*clusterRef,ps1Ene,ps2Ene,ps1,ps2);	  
	  //	  std::cout << "Test " << Ene_ecalgsf <<  " PS1 / PS2 " << ps1 << " " << ps2 << std::endl;


	  if (DebugIDOutputs)  
	    cout << " setIdOutput! GSF ECAL Cluster E " << Ene_ecalgsf
		 << " eta,phi " << clusterRef->position().eta()
		 <<", " <<  clusterRef->position().phi() << endl;
	  
	  deta_gsfecal = clusterRef->position().eta() - Etaout_gsf;
	  
	  const vector< reco::PFRecHitFraction >& PFRecHits =  clusterRef->recHitFractions();
	  double SeedClusEnergy = -1.;
	  unsigned int SeedDetID = 0;
	  double SeedEta = -1.;
	  double SeedPhi = -1.;
	  for ( vector< reco::PFRecHitFraction >::const_iterator it = PFRecHits.begin(); 
		it != PFRecHits.end(); ++it) {
	    const PFRecHitRef& RefPFRecHit = it->recHitRef(); 
	    double energyRecHit = RefPFRecHit->energy();
	    if (energyRecHit > SeedClusEnergy) {
	      SeedClusEnergy = energyRecHit;
	      SeedEta = RefPFRecHit->position().eta();
	      SeedPhi =  RefPFRecHit->position().phi();
	      SeedDetID = RefPFRecHit->detId();
	    }
	  }
	  
	  for ( vector< reco::PFRecHitFraction >::const_iterator it = PFRecHits.begin(); 
		it != PFRecHits.end(); ++it) {
	    
	    const PFRecHitRef& RefPFRecHit = it->recHitRef(); 
	    double energyRecHit = RefPFRecHit->energy();
	    if (RefPFRecHit->detId() != SeedDetID) {
	      float diffEta =  RefPFRecHit->position().eta() - SeedEta;
	      sigmaEtaEta += (diffEta*diffEta) * (energyRecHit/SeedClusEnergy);
	    }
	  }
	  if (sigmaEtaEta == 0.) sigmaEtaEta = 0.00000001;
	}
	else {
	  reco::PFClusterRef clusterRef = elements[(assogsf_index[ielegsf])].clusterRef();	  	  
	  float TempClus_energy = pfcalib_.energyEm(*clusterRef,ps1Ene,ps2Ene);	 
	  Ene_extraecalgsf += TempClus_energy;
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
	if (TrajPos <= 3) EarlyBrem = true;
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
	      float BremClus_energy = pfcalib_.energyEm(*clusterRef,ps1EneFromBrem,ps2EneFromBrem);
	      Ene_ecalbrem += BremClus_energy;
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
	SigmaEtaEta = log(sigmaEtaEta);
	
	
	lateBrem = -1;
	firstBrem = -1;
	earlyBrem = -1;
	if(NumBrem > 0) {
	  if (LateBrem == true) lateBrem = 1;
	  else lateBrem = 0;
	  firstBrem = FirstBrem;
	  if(FirstBrem < 4) earlyBrem = 1;
	  else earlyBrem = 0;
	}      
	
	HOverHE = Ene_hcalgsf/(Ene_hcalgsf + Ene_ecalgsf);
	HOverPin = Ene_hcalgsf / Ein_gsf;
	
	
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

	  if(ecalGsf_index < 100000) {
	    vector<unsigned int> assoecalgsf_index = associatedToEcal_.find(ecalGsf_index)->second;
	    for(unsigned int itrk =0; itrk<assoecalgsf_index.size();itrk++) {
	      PFBlockElement::Type typeassoecal = elements[(assoecalgsf_index[itrk])].type();
	      if(typeassoecal == reco::PFBlockElement::TRACK) {
		const reco::PFBlockElementTrack * kfTk =  
		  dynamic_cast<const reco::PFBlockElementTrack*>((&elements[(assoecalgsf_index[itrk])]));
		reco::TrackRef trackref =  kfTk->trackRef();
		// iteration 1,2,3,4 correspond to algo = 1/5,6,7,8
		unsigned int Algo = trackref->algo() < 5 ? 
		  trackref->algo()-1 : trackref->algo()-5;
		if(Algo < 3) {
		  if(DebugIDOutputs) 
		    cout << " The ecalGsf cluster is not isolated: >0 KF extra with algo < 3" << endl;

		  float p_trk = trackref->p();
		  SumExtraKfP += p_trk;
		  iextratrack++;
		  // Check if these extra tracks are HCAL linked
		  std::multimap<double, unsigned int> hcalKfElems;
		  block.associatedElements( assoecalgsf_index[itrk],linkData,
					    hcalKfElems,
					    reco::PFBlockElement::HCAL,
					    reco::PFBlock::LINKTEST_ALL );
		  if(hcalKfElems.size() > 0) {
		    itrackHcalLinked++;
		  }
		}
	      } 
	    }
	  }
	  if( iextratrack > 0) {
	    if(iextratrack > 3 || HOverHE > 0.05 || (SumExtraKfP/Ene_ecalgsf) > 1.) {

	      if(DebugIDOutputs) 
		cout << " *****This electron candidate is discarded  # tracks "		
		     << iextratrack << " HOverHE " << HOverHE << " SumExtraKfP/Ene_ecalgsf " << SumExtraKfP/Ene_ecalgsf  << endl;

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
	    BDToutput_[cgsf] = mvaValue-4.;
	    lockExtraKf_[cgsf] = false;
	  }
 	}
		


	if (DebugIDOutputs) {
	  cout << " **** BDT observables ****" << endl;
	  cout << " < Normalization > " << endl;
	  cout << " lnPt_gsf " << lnPt_gsf 
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
	       << " HOverHE " << HOverHE 
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
  PFBlock::LinkData linkData =  block.linkData();     
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  PFEnergyResolution pfresol_;
  PFEnergyCalibration pfcalib_;
  bool DebugIDCandidates = false;

  unsigned int cgsf=0;
  for (map<unsigned int,vector<unsigned int> >::iterator igsf = associatedToGsf_.begin();
       igsf != associatedToGsf_.end(); igsf++,cgsf++) {
    unsigned int gsf_index =  igsf->first;
           
    

    // They should be reset for each gsf track
    int eecal=0;
    int hcal=0;
    int charge =0; 
    bool goodphi=true;
    math::XYZTLorentzVector momentum_kf,momentum_gsf,momentum,momentum_mean;
    float dpt=0; float dpt_kf=0; float dpt_gsf=0; float dpt_mean=0;
    float Eene=0; float dene=0; float Hene=0.;
    float RawEene = 0.;


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
      dpt_mean=RefGSF->ptError()*
	(RefGSF->p()/RefGSF->pt());  
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
	RefKF = KfTk->trackRef();
	if (RefKF.isNonnull()) {
	  has_kf = true;
	  dpt_kf=(RefKF->ptError()*RefKF->ptError());
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
	    float df=fabs(momentum_gsf.phi()-clust->clusterRef()->position().phi());
	    if (df<0.1) {
	      // Not used for the moments
	      // elementsToAdd.push_back((assogsf_index[ielegsf])); 
	      //Eene+=clust->clusterRef()->energy(;)
	      Hene+=clust->clusterRef()->energy();
	      hcal++;
	    }
	  }
	}
	elementsToAdd.push_back((assogsf_index[ielegsf]));


	const reco::PFBlockElementCluster * clust =  
	  dynamic_cast<const reco::PFBlockElementCluster*>((&elements[(assogsf_index[ielegsf])]));
	
	eecal++;
	
	const reco::PFCluster& cl(*clust->clusterRef());

	// The electron RAW energy is the energy of the corrected GSF cluster	
	double ps1,ps2;
	ps1=ps2=0.;
	//	float EE=pfcalib_.energyEm(cl,ps1Ene,ps2Ene);
	float EE = pfcalib_.energyEm(cl,ps1Ene,ps2Ene,ps1,ps2);	  
	//	std::cout << "Test "<< EE << " " <<  " PS1 / PS2 " << ps1 << " " << ps2 << std::endl;
	//	float RawEE = cl.energy();

	float ceta=cl.position().eta();
	float cphi=cl.position().phi();
	
	float mphi=-2.97025;
	if (ceta<0) mphi+=0.00638;
	for (int ip=1; ip<19; ip++){
	  float df= cphi - (mphi+(ip*6.283185/18));
	  if (fabs(df)<0.01) goodphi=false;
	}
	float dE=pfresol_.getEnergyResolutionEm(EE,cl.position().eta());
	if( DebugIDCandidates ) 
	  cout << "SetCandidates:: EcalCluster: EneNoCalib " << clust->clusterRef()->energy()  
	       << " eta,phi " << ceta << "," << cphi << " Calib " <<  EE << " dE " <<  dE <<endl;

	if (FirstEcalGsf) {
	  FirstEcalGsf = false;
	  ecalGsf_index = assogsf_index[ielegsf];
	  //	  std::cout << " PFElectronAlgo / Seed " << EE << std::endl;
	  RawEene += EE;
	}
	else // in this case one should make a photon ! 
	  {
	      math::XYZTLorentzVector photonMomentum;
	      math::XYZPoint direction=cl.position()/cl.position().R();
	      photonMomentum.SetPxPyPzE(EE*direction.x(),
					EE*direction.y(),
					EE*direction.z(),
					EE);
	      reco::PFCandidate photon_Candidate(0,photonMomentum, reco::PFCandidate::gamma);
	      
	      photon_Candidate.setPs1Energy(ps1);
	      photon_Candidate.setPs1Energy(ps2);
	      photon_Candidate.setEcalEnergy(EE);
	      //	      std::cout << " PFElectronAlgo, adding Brem (1) " << EE << std::endl;
	      // yes, EE, we want the raw ecal energy of the daugther to have the same definition
	      // as the GSF cluster
	      photon_Candidate.setRawEcalEnergy(EE);
	      photon_Candidate.setPositionAtECALEntrance(math::XYZPointF(cl.position()));
	      photon_Candidate.addElementInBlock(blockRef,assogsf_index[ielegsf]);
	      // store the photon candidate
	      std::map<unsigned int,std::vector<reco::PFCandidate> >::iterator itcheck=
		photonCandidates_.find(cgsf);
	      if(itcheck==photonCandidates_.end())
		{		  
		  // beurk
		  std::vector<reco::PFCandidate> tmpVec;
		  tmpVec.push_back(photon_Candidate);
		  photonCandidates_.insert(std::pair<unsigned int, std::vector<reco::PFCandidate> >
					   (cgsf,tmpVec));
		}
	      else
		{
		  itcheck->second.push_back(photon_Candidate);
		}
	      
	  }
	Eene+=EE;
	ps1TotEne+=ps1;
	ps2TotEne+=ps2;
	dene+=dE*dE;
      }
      


      // Important: Add energy from the brems
      if  (assoele_type == reco::PFBlockElement::BREM) {
	unsigned int brem_index = assogsf_index[ielegsf];
	vector<unsigned int> assobrem_index = associatedToBrems_.find(brem_index)->second;
	elementsToAdd.push_back(brem_index);
	for (unsigned int ibrem = 0; ibrem < assobrem_index.size(); ibrem++){
	  if (elements[(assobrem_index[ibrem])].type() == reco::PFBlockElement::ECAL) {
	    // brem emission is from the considered gsf track
	    unsigned int keyecalbrem = assobrem_index[ibrem];
	    const vector<unsigned int>& assoelebrem_index = associatedToEcal_.find(keyecalbrem)->second;
	    vector<double> ps1EneFromBrem(0);
	    vector<double> ps2EneFromBrem(0);
	    float ps1EneFromBremTot=0.;
	    float ps2EneFromBremTot=0.;
	    for (unsigned int ielebrem=0; ielebrem<assoelebrem_index.size();ielebrem++) {
	      if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS1) {
		PFClusterRef  psref = elements[(assoelebrem_index[ielebrem])].clusterRef();
		ps1EneFromBrem.push_back(psref->energy());
		ps1EneFromBremTot+=psref->energy();
		elementsToAdd.push_back(assoelebrem_index[ielebrem]);
	      }
	      if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS2) {
		PFClusterRef  psref = elements[(assoelebrem_index[ielebrem])].clusterRef();
		ps2EneFromBrem.push_back(psref->energy());
		ps2EneFromBremTot+=psref->energy();
		elementsToAdd.push_back(assoelebrem_index[ielebrem]);
	      }	  
	      if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::HCAL) {
		reco::PFClusterRef clusterRef = elements[(assoelebrem_index[ielebrem])].clusterRef();
		float df=fabs(momentum_gsf.phi()-clusterRef->position().phi());
		if (df<0.1) {
		  // Not used for the moments
		  // elementsToAdd.push_back(index_assobrem_index[ielebrem]); 
		  //Eene+= clusterRef->energy();
		  Hene+= clusterRef->energy();
		  hcal++;
		}
	      }     
	    }
	    

	    if( assobrem_index[ibrem] !=  ecalGsf_index) {
	      elementsToAdd.push_back(assobrem_index[ibrem]);
	      reco::PFClusterRef clusterRef = elements[(assobrem_index[ibrem])].clusterRef();
	      // to get a calibrated PS energy 
	      double ps1=0;
	      double ps2=0;
	      float EE = pfcalib_.energyEm(*clusterRef,ps1EneFromBrem,ps2EneFromBrem,ps1,ps2);
	      
	      // float RawEE  = clusterRef->energy();
	      float ceta = clusterRef->position().eta();
	      // float cphi = clusterRef->position().phi();
	      float dE=pfresol_.getEnergyResolutionEm(EE,ceta);
	      if( DebugIDCandidates ) 
		cout << "SetCandidates:: BremCluster: Ene " << EE << " dE " <<  dE <<endl;	  
	      Eene+=EE;
	      ps1TotEne+=ps1;
	      ps2TotEne+=ps2;
	      // Removed 4 March 2009. Florian. The Raw energy is the (corrected) one of the GSF cluster only
	      //	      RawEene += RawEE;
	      dene+=dE*dE;

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
	      photon_Candidate.setPs1Energy(ps2);
	      photon_Candidate.setEcalEnergy(EE);
	      //	      std::cout << " PFElectronAlgo, adding Brem " << EE << std::endl;
	      // yes, EE, we want the raw ecal energy of the daugther to have the same definition
	      // as the GSF cluster
	      photon_Candidate.setRawEcalEnergy(EE);
	      photon_Candidate.setPositionAtECALEntrance(math::XYZPointF(clusterRef->position()));
	      photon_Candidate.addElementInBlock(blockRef,assobrem_index[ibrem]);

	      // store the photon candidate
	      std::map<unsigned int,std::vector<reco::PFCandidate> >::iterator itcheck=
		photonCandidates_.find(cgsf);
	      if(itcheck==photonCandidates_.end())
		{		  
		  // beurk
		  std::vector<reco::PFCandidate> tmpVec;
		  tmpVec.push_back(photon_Candidate);
		  photonCandidates_.insert(std::pair<unsigned int, std::vector<reco::PFCandidate> >
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
	
	reco::PFCandidate::ParticleType particleType 
	  = reco::PFCandidate::e;
	reco::PFCandidate temp_Candidate;
	temp_Candidate = PFCandidate(charge,newmomentum,particleType);
	temp_Candidate.set_mva_e_pi(BDToutput_[cgsf]);
	temp_Candidate.setEcalEnergy(Eene);
	temp_Candidate.setRawEcalEnergy(RawEene);
	//	std::cout << " Creating Candidate Total/Raw " << Eene << " " << RawEene << std::endl;
	temp_Candidate.setHcalEnergy(0.);  // Use HCAL energy? 
	temp_Candidate.setPs1Energy(ps1TotEne);
	temp_Candidate.setPs2Energy(ps2TotEne);
	temp_Candidate.setTrackRef(RefKF);   
	// This reference could be NULL it is needed a protection? 
	temp_Candidate.setGsfTrackRef(RefGSF);
	temp_Candidate.setPositionAtECALEntrance(posGsfEcalEntrance);
	// Add Vertex
	temp_Candidate.setVertex(RefGSF->vertex());

	
	if( DebugIDCandidates ) 
	  cout << "SetCandidates:: I am after doing candidate " <<endl;
	
	for (unsigned int elad=0; elad<elementsToAdd.size();elad++){
	  temp_Candidate.addElementInBlock(blockRef,elementsToAdd[elad]);
	}

	if(BDToutput_[cgsf] >=  mvaEleCut_) 
	  elCandidate_.push_back(temp_Candidate);

	// now the special candidate for e/gamma & taus 
	reco::PFCandidate extendedElCandidate(temp_Candidate);
	// now add the photons to this candidate
	std::map<unsigned int, std::vector<reco::PFCandidate> >::const_iterator itcluster=
	  photonCandidates_.find(cgsf);
	if(itcluster!=photonCandidates_.end())
	  {
	    const std::vector<reco::PFCandidate> & theClusters=itcluster->second;
	    unsigned nclus=theClusters.size();
	    //	    std::cout << " PFElectronAlgo " << nclus << " daugthers to add" << std::endl;
	    for(unsigned iclus=0;iclus<nclus;++iclus)
	      {
		extendedElCandidate.addDaughter(theClusters[iclus]);
	      }
	  }
//	else
//	  {
//	    std::cout << " Did not find any photon connected to " <<cgsf << std::endl;
//	  }
	
	allElCandidate_.push_back(extendedElCandidate);
	
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
  PFBlock::LinkData linkData =  block.linkData();  
   
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  
  unsigned int cgsf=0;
  for (map<unsigned int,vector<unsigned int> >::iterator igsf = associatedToGsf_.begin();
       igsf != associatedToGsf_.end(); igsf++,cgsf++) {

    // lock only the elements that pass the BDT cut
    if(BDToutput_[cgsf] < mvaEleCut_) continue;

    unsigned int gsf_index =  igsf->first;
    active[gsf_index] = false;  // lock the gsf
    vector<unsigned int> assogsf_index = igsf->second;
    for  (unsigned int ielegsf=0;ielegsf<assogsf_index.size();ielegsf++) {
      PFBlockElement::Type assoele_type = elements[(assogsf_index[ielegsf])].type();
      // lock the elements associated to the gsf: ECAL, Brems
      active[(assogsf_index[ielegsf])] = false;  
      if (assoele_type == reco::PFBlockElement::ECAL) {
	unsigned int keyecalgsf = assogsf_index[ielegsf];

	// add protection against fifth step
	if(fifthStepKfTrack_.size() > 0) {
	  for(unsigned int itr = 0; itr < fifthStepKfTrack_.size(); itr++) {
	    if(fifthStepKfTrack_[itr].first == keyecalgsf) {
	      active[(fifthStepKfTrack_[itr].second)] = false;
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
	    if(fifthStepKfTrack_.size() > 0) {
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
