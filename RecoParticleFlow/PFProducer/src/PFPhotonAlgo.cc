//
// Original Authors: Fabian Stoeckli: fabian.stoeckli@cern.ch
//                   Nicholas Wardle: nckw@cern.ch
//                   Rishi Patel rpatel@cern.ch
//

#include "RecoParticleFlow/PFProducer/interface/PFPhotonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <iomanip>
#include <algorithm>

using namespace std;
using namespace reco;


PFPhotonAlgo::PFPhotonAlgo(std::string mvaweightfile,  double mvaConvCut, const reco::Vertex& primary) 
  : isvalid_(false), verbosityLevel_(Silent), MVACUT(mvaConvCut) 
{  
    primaryVertex_=primary;  
    //Book MVA  
    tmvaReader_ = new TMVA::Reader("!Color:Silent");  
    tmvaReader_->AddVariable("del_phi",&del_phi);  
    tmvaReader_->AddVariable("nlayers", &nlayers);  
    tmvaReader_->AddVariable("chi2",&chi2);  
    tmvaReader_->AddVariable("EoverPt",&EoverPt);  
    tmvaReader_->AddVariable("HoverPt",&HoverPt);  
    tmvaReader_->AddVariable("track_pt", &track_pt);  
    tmvaReader_->AddVariable("STIP",&STIP);  
    tmvaReader_->AddVariable("nlost", &nlost);  
    tmvaReader_->BookMVA("BDT",mvaweightfile.c_str());  
  }

void PFPhotonAlgo::RunPFPhoton(const reco::PFBlockRef&  blockRef,
			       std::vector<bool>& active,
			       std::auto_ptr<PFCandidateCollection> &pfCandidates){
  
  //std::cout<<" calling RunPFPhoton "<<std::endl;
  
  /*      For now we construct the PhotonCandidate simply from 
	  a) adding the CORRECTED energies of each participating ECAL cluster
	  b) build the energy-weighted direction for the Photon
  */


  // define how much is printed out for debugging.
  // ... will be setable via CFG file parameter
  verbosityLevel_ = Chatty;          // Chatty mode.
  
  // needed for calibration
  PFEnergyCalibration pfcalib_;

  // loop over all elements in the Block
  const edm::OwnVector< reco::PFBlockElement >&          elements         = blockRef->elements();
  edm::OwnVector< reco::PFBlockElement >::const_iterator ele              = elements.begin();
  std::vector<bool>::const_iterator                      actIter          = active.begin();
  PFBlock::LinkData                                      linkData         = blockRef->linkData();
  bool                                                   isActive         = true;

  if(elements.size() != active.size()) {
    // throw excpetion...
    //std::cout<<" WARNING: Size of collection and active-vectro don't agree!"<<std::endl;
    return;
  }
  
  // local vecotr to keep track of the indices of the 'elements' for the Photon candidate
  // once we decide to keep the candidate, the 'active' entriesd for them must be set to false
  std::vector<unsigned int> elemsToLock;
  elemsToLock.resize(0);
  
  for( ; ele != elements.end(); ++ele, ++actIter ) {

    // if it's not a SuperCluster, go to the next element
    if( !( ele->type() == reco::PFBlockElement::SC ) ) continue;
    
    // Photon kienmatics, will be updated for each identified participating element
    float photonEnergy_        =   0.;
    float photonX_             =   0.;
    float photonY_             =   0.;
    float photonZ_             =   0.;
    float RawEcalEne           =   0.;

    // Total pre-shower energy
    float ps1TotEne      = 0.;
    float ps2TotEne      = 0.;
    
    bool hasConvTrack=false;  
    bool hasSingleleg=false;  
    std::vector<unsigned int> AddClusters(0);  
    std::vector<unsigned int> IsoTracks(0);  
    std::multimap<unsigned int, unsigned int>ClusterAddPS1;  
    std::multimap<unsigned int, unsigned int>ClusterAddPS2;

    isActive = *(actIter);
    //cout << " Found a SuperCluster.  Energy " ;
    const reco::PFBlockElementSuperCluster *sc = dynamic_cast<const reco::PFBlockElementSuperCluster*>(&(*ele));
    //std::cout << sc->superClusterRef()->energy () << " Track/Ecal/Hcal Iso " << sc->trackIso()<< " " << sc->ecalIso() ;
    //std::cout << " " << sc->hcalIso() <<std::endl;
    if (!(sc->fromPhoton()))continue;

    // check the status of the SC Element... 
    // ..... I understand it should *always* be active, since PFElectronAlgo does not touch this (yet?) RISHI: YES
    if( !isActive ) {
      //std::cout<<" SuperCluster is NOT active.... "<<std::endl;
      continue;
    }
    elemsToLock.push_back(ele-elements.begin()); //add SC to elements to lock
    // loop over its constituent ECAL cluster
    std::multimap<double, unsigned int> ecalAssoPFClusters;
    blockRef->associatedElements( ele-elements.begin(), 
				  linkData,
				  ecalAssoPFClusters,
				  reco::PFBlockElement::ECAL,
				  reco::PFBlock::LINKTEST_ALL );
    
    // loop over the ECAL clusters linked to the iEle 
    if( ! ecalAssoPFClusters.size() ) {
      // This SC element has NO ECAL elements asigned... *SHOULD NOT HAPPEN*
      //std::cout<<" Found SC element with no ECAL assigned "<<std::endl;
      continue;
    }
    
    // This is basically CASE 2
    // .... we loop over all ECAL cluster linked to each other by this SC
    for(std::multimap<double, unsigned int>::iterator itecal = ecalAssoPFClusters.begin(); 
	itecal != ecalAssoPFClusters.end(); ++itecal) { 
      
      // to get the reference to the PF clusters, this is needed.
      reco::PFClusterRef clusterRef = elements[itecal->second].clusterRef();	
      
      // from the clusterRef get the energy, direction, etc
      //      float ClustRawEnergy = clusterRef->energy();
      //      float ClustEta = clusterRef->position().eta();
      //      float ClustPhi = clusterRef->position().phi();

      // initialize the vectors for the PS energies
      vector<double> ps1Ene(0);
      vector<double> ps2Ene(0);
      double ps1=0;  
      double ps2=0;  
      hasSingleleg=false;  
      hasConvTrack=false;

      /*
      cout << " My cluster index " << itecal->second 
	   << " energy " <<  ClustRawEnergy
	   << " eta " << ClustEta
	   << " phi " << ClustPhi << endl;
      */
      // check if this ECAL element is still active (could have been eaten by PFElectronAlgo)
      // ......for now we give the PFElectron Algo *ALWAYS* Shot-Gun on the ECAL elements to the PFElectronAlgo
      
      if( !( active[itecal->second] ) ) {
	//std::cout<< "  .... this ECAL element is NOT active anymore. Is skipped. "<<std::endl;
	continue;
      }
      
      // ------------------------------------------------------------------------------------------
      // TODO: do some tests on the ECAL cluster itself, deciding to use it or not for the Photons
      // ..... ??? Do we need this?
      if ( false ) {
	// Check if there are a large number tracks that do not pass pre-ID around this ECAL cluster
	bool useIt = true;
	int mva_reject=0;  
	bool isClosest=false;  
	std::multimap<double, unsigned int> Trackscheck;  
	blockRef->associatedElements( itecal->second,  
				      linkData,  
				      Trackscheck,  
				      reco::PFBlockElement::TRACK,  
				      reco::PFBlock::LINKTEST_ALL);  
	for(std::multimap<double, unsigned int>::iterator track = Trackscheck.begin();  
	    track != Trackscheck.end(); ++track) {  
	   
	  // first check if is it's still active  
	  if( ! (active[track->second]) ) continue;  
	  hasSingleleg=EvaluateSingleLegMVA(blockRef,  primaryVertex_, track->second);  
	  //check if it is the closest linked track  
	  std::multimap<double, unsigned int> closecheck;  
	  blockRef->associatedElements(track->second,  
				       linkData,  
				       closecheck,  
				       reco::PFBlockElement::ECAL,  
				       reco::PFBlock::LINKTEST_ALL);  
	  if(closecheck.begin()->second ==itecal->second)isClosest=true;  
	  if(!hasSingleleg)mva_reject++;  
	}  
	 
	if(mva_reject>0 &&  isClosest)useIt=false;  
	//if(mva_reject==1 && isClosest)useIt=false;
	if( !useIt ) continue;    // Go to next ECAL cluster within SC
      }
      // ------------------------------------------------------------------------------------------
      
      // We decided to keep the ECAL cluster for this Photon Candidate ...
      elemsToLock.push_back(itecal->second);
      
      // look for PS in this Block linked to this ECAL cluster      
      std::multimap<double, unsigned int> PS1Elems;
      std::multimap<double, unsigned int> PS2Elems;
      //PS Layer 1 linked to ECAL cluster
      blockRef->associatedElements( itecal->second,
				    linkData,
				    PS1Elems,
				    reco::PFBlockElement::PS1,
				    reco::PFBlock::LINKTEST_ALL );
      //PS Layer 2 linked to the ECAL cluster
      blockRef->associatedElements( itecal->second,
				    linkData,
				    PS2Elems,
				    reco::PFBlockElement::PS2,
				    reco::PFBlock::LINKTEST_ALL );
      
      // loop over all PS1 and compute energy
      for(std::multimap<double, unsigned int>::iterator iteps = PS1Elems.begin();
	  iteps != PS1Elems.end(); ++iteps) {

	// first chekc if it's still active
	if( !(active[iteps->second]) ) continue;
	
	//Check if this PS1 is not closer to another ECAL cluster in this Block          
	std::multimap<double, unsigned int> ECALPS1check;  
	blockRef->associatedElements( iteps->second,  
				      linkData,  
				      ECALPS1check,  
				      reco::PFBlockElement::ECAL,  
				      reco::PFBlock::LINKTEST_ALL );  
	if(itecal->second==ECALPS1check.begin()->second)//then it is closest linked  
	  {
	    reco::PFClusterRef ps1ClusterRef = elements[iteps->second].clusterRef();
	    ps1Ene.push_back( ps1ClusterRef->energy() );
	    ps1=ps1+ps1ClusterRef->energy(); //add to total PS1
	    // incativate this PS1 Element
	    elemsToLock.push_back(iteps->second);
	  }
      }
      for(std::multimap<double, unsigned int>::iterator iteps = PS2Elems.begin();
	  iteps != PS2Elems.end(); ++iteps) {

	// first chekc if it's still active
	if( !(active[iteps->second]) ) continue;
	
	// Check if this PS2 is not closer to another ECAL cluster in this Block:
	std::multimap<double, unsigned int> ECALPS2check;  
	blockRef->associatedElements( iteps->second,  
				      linkData,  
				      ECALPS2check,  
				      reco::PFBlockElement::ECAL,  
				      reco::PFBlock::LINKTEST_ALL );  
	if(itecal->second==ECALPS2check.begin()->second)//is closest linked  
	  {
	    reco::PFClusterRef ps2ClusterRef = elements[iteps->second].clusterRef();
	    ps2Ene.push_back( ps2ClusterRef->energy() );
	    ps2=ps2ClusterRef->energy()+ps2; //add to total PS2
	    // incativate this PS2 Element
	    elemsToLock.push_back(iteps->second);
	  }
      }
            
      // loop over the HCAL Clusters linked to the ECAL cluster (CASE 6)
      std::multimap<double, unsigned int> hcalElems;
      blockRef->associatedElements( itecal->second,linkData,
				    hcalElems,
				    reco::PFBlockElement::HCAL,
				    reco::PFBlock::LINKTEST_ALL );

      for(std::multimap<double, unsigned int>::iterator ithcal = hcalElems.begin();
	  ithcal != hcalElems.end(); ++ithcal) {

	if ( ! (active[ithcal->second] ) ) continue; // HCAL Cluster already used....
	
	// TODO: Decide if this HCAL cluster is to be used
	// .... based on some Physics
	// .... To we need to check if it's closer to any other ECAL/TRACK?

	bool useHcal = false;
	if ( !useHcal ) continue;
	//not locked
	//elemsToLock.push_back(ithcal->second);
      }

      // This is entry point for CASE 3.
      // .... we loop over all Tracks linked to this ECAL and check if it's labeled as conversion
      // This is the part for looping over all 'Conversion' Tracks
      std::multimap<double, unsigned int> convTracks;
      blockRef->associatedElements( itecal->second,
				    linkData,
				    convTracks,
				    reco::PFBlockElement::TRACK,
				    reco::PFBlock::LINKTEST_ALL);
      for(std::multimap<double, unsigned int>::iterator track = convTracks.begin();
	  track != convTracks.end(); ++track) {

	// first check if is it's still active
	if( ! (active[track->second]) ) continue;
	
	// check if it's a CONV track
	const reco::PFBlockElementTrack * trackRef = dynamic_cast<const reco::PFBlockElementTrack*>((&elements[track->second])); 	
	
	//Check if track is a Single leg from a Conversion  
	mvaValue=-999;  
	hasSingleleg=EvaluateSingleLegMVA(blockRef,  primaryVertex_, track->second);  
	//If it is not then it will be used to check Track Isolation at the end  
	if(!hasSingleleg)  
	  {  
	    bool included=false;  
	    //check if this track is already included in the vector so it is linked to an ECAL cluster that is already examined  
	    for(unsigned int i=0; i<IsoTracks.size(); i++)  
	      {if(IsoTracks[i]==track->second)included=true;}  
	    if(!included)IsoTracks.push_back(track->second);  
	  }  
	//For now only Pre-ID tracks that are not already identified as Conversions  
	if(hasSingleleg &&!(trackRef->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)))  
	  {  
	    elemsToLock.push_back(track->second);  
	    //find all the clusters linked to this track  
	    std::multimap<double, unsigned int> moreClusters;  
	    blockRef->associatedElements( track->second,  
					  linkData,  
					  moreClusters,  
					  reco::PFBlockElement::ECAL,  
					  reco::PFBlock::LINKTEST_ALL);  
	     
	    float p_in=sqrt(elements[track->second].trackRef()->innerMomentum().x() * elements[track->second].trackRef()->innerMomentum().x() +  
			    elements[track->second].trackRef()->innerMomentum().y()*elements[track->second].trackRef()->innerMomentum().y()+  
			    elements[track->second].trackRef()->innerMomentum().z()*elements[track->second].trackRef()->innerMomentum().z());  
	    float linked_E=0;  
	    for(std::multimap<double, unsigned int>::iterator clust = moreClusters.begin();  
		clust != moreClusters.end(); ++clust)  
	      {  
		if(!active[clust->second])continue;  
		//running sum of linked energy  
		linked_E=linked_E+elements[clust->second].clusterRef()->energy();  
		//prevent too much energy from being added  
		if(linked_E/p_in>1.5)break;  
		bool included=false;  
		//check if these ecal clusters are already included with the supercluster  
		for(std::multimap<double, unsigned int>::iterator cluscheck = ecalAssoPFClusters.begin();  
		    cluscheck != ecalAssoPFClusters.end(); ++cluscheck)  
		  {  
		    if(cluscheck->second==clust->second)included=true;  
		  }  
		if(!included)AddClusters.push_back(clust->second);//Add to a container of clusters to be Added to the Photon candidate  
	      }  
	  }

	// Possibly need to be more smart about them (CASE 5)
	// .... for now we simply skip non id'ed tracks
	if( ! (trackRef->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) ) ) continue;  
	hasConvTrack=true;  
	elemsToLock.push_back(track->second);  
	//again look at the clusters linked to this track  
	std::multimap<double, unsigned int> moreClusters;  
	blockRef->associatedElements( track->second,  
				      linkData,  
				      moreClusters,  
				      reco::PFBlockElement::ECAL,  
				      reco::PFBlock::LINKTEST_ALL);
	
	float p_in=sqrt(elements[track->second].trackRef()->innerMomentum().x() * elements[track->second].trackRef()->innerMomentum().x() +  
			elements[track->second].trackRef()->innerMomentum().y()*elements[track->second].trackRef()->innerMomentum().y()+  
			elements[track->second].trackRef()->innerMomentum().z()*elements[track->second].trackRef()->innerMomentum().z());  
	float linked_E=0;  
	for(std::multimap<double, unsigned int>::iterator clust = moreClusters.begin();  
	    clust != moreClusters.end(); ++clust)  
	  {  
	    if(!active[clust->second])continue;  
	    linked_E=linked_E+elements[clust->second].clusterRef()->energy();  
	    if(linked_E/p_in>1.5)break;  
	    bool included=false;  
	    for(std::multimap<double, unsigned int>::iterator cluscheck = ecalAssoPFClusters.begin();  
		cluscheck != ecalAssoPFClusters.end(); ++cluscheck)  
	      {  
		if(cluscheck->second==clust->second)included=true;  
	      }  
	    if(!included)AddClusters.push_back(clust->second);//again only add if it is not already included with the supercluster  
	  }

	// we need to check for other TRACKS linked to this conversion track, that point possibly no an ECAL cluster not included in the SC
	// .... This is basically CASE 4.

	std::multimap<double, unsigned int> moreTracks;
	blockRef->associatedElements( track->second,
				      linkData,
				      moreTracks,
				      reco::PFBlockElement::TRACK,
				      reco::PFBlock::LINKTEST_ALL);

	for(std::multimap<double, unsigned int>::iterator track2 = moreTracks.begin();
	    track2 != moreTracks.end(); ++track2) {
	  
	  // first check if is it's still active
	  if( ! (active[track2->second]) ) continue;
	  //skip over the 1st leg already found above  
	  if(track->second==track2->second)continue;
	  // check if it's a CONV track
	  const reco::PFBlockElementTrack * track2Ref = dynamic_cast<const reco::PFBlockElementTrack*>((&elements[track2->second])); 	
	  if( ! (track2Ref->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) ) ) continue;  // Possibly need to be more smart about them (CASE 5)
	  elemsToLock.push_back(track2->second);
	  // so it's another active conversion track, that is in the Block and linked to the conversion track we already found
	  // find the ECAL cluster linked to it...
	  std::multimap<double, unsigned int> convEcal;
	  blockRef->associatedElements( track2->second,
					linkData,
					convEcal,
					reco::PFBlockElement::ECAL,
					reco::PFBlock::LINKTEST_ALL);
	  float p_in=sqrt(elements[track->second].trackRef()->innerMomentum().x() * 
			  elements[track->second].trackRef()->innerMomentum().x() +
			  elements[track->second].trackRef()->innerMomentum().y()*elements[track->second].trackRef()->innerMomentum().y()+  
			  elements[track->second].trackRef()->innerMomentum().z()*elements[track->second].trackRef()->innerMomentum().z());  
	  float linked_E=0;
	  for(std::multimap<double, unsigned int>::iterator itConvEcal = convEcal.begin();
	      itConvEcal != convEcal.end(); ++itConvEcal) {
	    
	    if( ! (active[itConvEcal->second]) ) continue;
	    bool included=false;  
	    for(std::multimap<double, unsigned int>::iterator cluscheck = ecalAssoPFClusters.begin();  
		cluscheck != ecalAssoPFClusters.end(); ++cluscheck)  
	      {  
		if(cluscheck->second==itConvEcal->second)included=true;  
	      }
	    linked_E=linked_E+elements[itConvEcal->second].clusterRef()->energy();
	    if(linked_E/p_in>1.5)break;
	    if(!included){AddClusters.push_back(itConvEcal->second);
	    }
	    
	    // it's still active, so we have to add it.
	    // CAUTION: we don't care here if it's part of the SC or not, we include it anyways
	    
	    // loop over the HCAL Clusters linked to the ECAL cluster (CASE 6)
	    std::multimap<double, unsigned int> hcalElems_conv;
	    blockRef->associatedElements( itecal->second,linkData,
					  hcalElems_conv,
					  reco::PFBlockElement::HCAL,
					  reco::PFBlock::LINKTEST_ALL );
	    
	    for(std::multimap<double, unsigned int>::iterator ithcal2 = hcalElems_conv.begin();
		ithcal2 != hcalElems_conv.end(); ++ithcal2) {
	      
	      if ( ! (active[ithcal2->second] ) ) continue; // HCAL Cluster already used....
	      
	      // TODO: Decide if this HCAL cluster is to be used
	      // .... based on some Physics
	      // .... To we need to check if it's closer to any other ECAL/TRACK?
	      
	      bool useHcal = true;
	      if ( !useHcal ) continue;
	      
	      //elemsToLock.push_back(ithcal2->second);

	    } // end of loop over HCAL clusters linked to the ECAL cluster from second CONVERSION leg
	    
	  } // end of loop over ECALs linked to second T_FROM_GAMMACONV
	  
	} // end of loop over SECOND conversion leg

	// TODO: Do we need to check separatly if there are HCAL cluster linked to the track?
	
      } // end of loop over tracks
      
            
      // Calibrate the Added ECAL energy
      float addedCalibEne=0;
      float addedRawEne=0;
      std::vector<double>AddedPS1(0);
      std::vector<double>AddedPS2(0);  
      double addedps1=0;  
      double addedps2=0;  
      for(unsigned int i=0; i<AddClusters.size(); i++)  
	{  
	  std::multimap<double, unsigned int> PS1Elems_conv;  
	  std::multimap<double, unsigned int> PS2Elems_conv;  
	  blockRef->associatedElements(AddClusters[i],  
				       linkData,  
				       PS1Elems_conv,  
				       reco::PFBlockElement::PS1,  
				       reco::PFBlock::LINKTEST_ALL );  
	  blockRef->associatedElements( AddClusters[i],  
					linkData,  
					PS2Elems_conv,  
					reco::PFBlockElement::PS2,  
					reco::PFBlock::LINKTEST_ALL );  
	   
	  for(std::multimap<double, unsigned int>::iterator iteps = PS1Elems_conv.begin();  
	      iteps != PS1Elems_conv.end(); ++iteps)  
	    {  
	      if(!active[iteps->second])continue;  
	      std::multimap<double, unsigned int> PS1Elems_check;  
	      blockRef->associatedElements(iteps->second,  
					   linkData,  
					   PS1Elems_check,  
					   reco::PFBlockElement::ECAL,  
					   reco::PFBlock::LINKTEST_ALL );  
	      if(PS1Elems_check.begin()->second==AddClusters[i])  
		{  
		   
		  reco::PFClusterRef ps1ClusterRef = elements[iteps->second].clusterRef();  
		  AddedPS1.push_back(ps1ClusterRef->energy());  
		  addedps1=addedps1+ps1ClusterRef->energy();  
		  elemsToLock.push_back(iteps->second);  
		}  
	    }  
	   
	  for(std::multimap<double, unsigned int>::iterator iteps = PS1Elems_conv.begin();  
	      iteps != PS1Elems_conv.end(); ++iteps) {  
	    if(!active[iteps->second])continue;  
	    std::multimap<double, unsigned int> PS2Elems_check;  
	    blockRef->associatedElements(iteps->second,  
					 linkData,  
					 PS2Elems_check,  
					 reco::PFBlockElement::ECAL,  
					 reco::PFBlock::LINKTEST_ALL );  
	     
	    if(PS2Elems_check.begin()->second==AddClusters[i])  
	      {  
		reco::PFClusterRef ps2ClusterRef = elements[iteps->second].clusterRef();  
		AddedPS2.push_back(ps2ClusterRef->energy());  
		addedps2=addedps2+ps2ClusterRef->energy();  
		elemsToLock.push_back(iteps->second);  
	      }  
	  }  
	  reco::PFClusterRef AddclusterRef = elements[AddClusters[i]].clusterRef();  
	  addedRawEne=AddclusterRef->energy()+addedRawEne;  
	  addedCalibEne=pfcalib_.energyEm(*AddclusterRef,AddedPS1,AddedPS2,false)+addedCalibEne;  
	  AddedPS2.clear(); AddedPS1.clear();  
	  elemsToLock.push_back(AddClusters[i]);  
	}  
      AddClusters.clear();
      float EE=pfcalib_.energyEm(*clusterRef,ps1Ene,ps2Ene,false)+addedCalibEne;  
      //cout<<"Original Energy "<<EE<<"Added Energy "<<addedCalibEne<<endl;
      
      photonEnergy_ +=  EE;
      RawEcalEne    +=  clusterRef->energy()+addedRawEne;
      photonX_      +=  EE * clusterRef->position().X();
      photonY_      +=  EE * clusterRef->position().Y();
      photonZ_      +=  EE * clusterRef->position().Z();	        
      ps1TotEne     +=  ps1+addedps1;
      ps2TotEne     +=  ps2+addedps2;
    } // end of loop over all ECAL cluster within this SC
    
    // we've looped over all ECAL clusters, ready to generate PhotonCandidate
    if( ! (photonEnergy_ > 0.) ) continue;    // This SC is not a Photon Candidate
    float sum_track_pt=0;
    //Now check if there are tracks failing isolation outside of the Jurassic isolation region  
    for(unsigned int i=0; i<IsoTracks.size(); i++)sum_track_pt=sum_track_pt+elements[IsoTracks[i]].trackRef()->pt();  
    float Et=sqrt(photonX_ * photonX_ + photonY_ * photonY_);  
    if(sum_track_pt>(2+ 0.001* Et))continue;//THIS SC is not a Photon it fails track Isolation

    math::XYZVector photonPosition(photonX_,
				   photonY_,
				   photonZ_);

    math::XYZVector photonDirection=photonPosition.Unit();
    
    math::XYZTLorentzVector photonMomentum(photonEnergy_* photonDirection.X(),
					   photonEnergy_* photonDirection.Y(),
					   photonEnergy_* photonDirection.Z(),
					   photonEnergy_           );
    /*
    std::cout<<" Created Photon with energy = "<<photonEnergy_<<std::endl;
    std::cout<<"                         pT = "<<photonMomentum.pt()<<std::endl;
    std::cout<<"                       Mass = "<<photonMomentum.mag()<<std::endl;
    std::cout<<"                          E = "<<photonMomentum.e()<<std::endl;
    */
    reco::PFCandidate photonCand(0,photonMomentum, reco::PFCandidate::gamma);
    
    photonCand.setPs1Energy(ps1TotEne);
    photonCand.setPs2Energy(ps2TotEne);
    photonCand.setEcalEnergy(RawEcalEne,photonEnergy_);
    photonCand.setHcalEnergy(0.,0.);
    photonCand.set_mva_nothing_gamma(1.);  
    photonCand.setSuperClusterRef(sc->superClusterRef());
    if(hasConvTrack || hasSingleleg)photonCand.setFlag( reco::PFCandidate::GAMMA_TO_GAMMACONV, true);

    //photonCand.setPositionAtECALEntrance(math::XYZPointF(photonMom_.position()));

    
    // set isvalid_ to TRUE since we've found at least one photon candidate
    isvalid_ = true;
    // push back the candidate into the collection ...

    // ... and lock all elemts used
    for(std::vector<unsigned int>::const_iterator it = elemsToLock.begin();
	it != elemsToLock.end(); ++it)
      {
	if(active[*it])photonCand.addElementInBlock(blockRef,*it);
	active[*it] = false;
      }
    pfCandidates->push_back(photonCand);
    // ... and reset the vector
    elemsToLock.resize(0);
    hasConvTrack=false;
    hasSingleleg=false;
  } // end of loops over all elements in block
  
  return;

}

bool PFPhotonAlgo::EvaluateSingleLegMVA(const reco::PFBlockRef& blockref, const reco::Vertex& primaryvtx, unsigned int track_index)  
{  
  bool convtkfound=false;  
  const reco::PFBlock& block = *blockref;  
  const edm::OwnVector< reco::PFBlockElement >& elements = block.elements();  
  //use this to store linkdata in the associatedElements function below  
  PFBlock::LinkData linkData =  block.linkData();  
  //calculate MVA Variables  
  chi2=elements[track_index].trackRef()->chi2()/elements[track_index].trackRef()->ndof();  
  nlost=elements[track_index].trackRef()->trackerExpectedHitsInner().numberOfLostHits();  
  nlayers=elements[track_index].trackRef()->hitPattern().trackerLayersWithMeasurement();  
  track_pt=elements[track_index].trackRef()->pt();  
  STIP=elements[track_index].trackRefPF()->STIP();  
   
  float linked_e=0;  
  float linked_h=0;  
  std::multimap<double, unsigned int> ecalAssoTrack;  
  block.associatedElements( track_index,linkData,  
			    ecalAssoTrack,  
			    reco::PFBlockElement::ECAL,  
			    reco::PFBlock::LINKTEST_ALL );  
  std::multimap<double, unsigned int> hcalAssoTrack;  
  block.associatedElements( track_index,linkData,  
			    hcalAssoTrack,  
			    reco::PFBlockElement::HCAL,  
			    reco::PFBlock::LINKTEST_ALL );  
  if(ecalAssoTrack.size() > 0) {  
    for(std::multimap<double, unsigned int>::iterator itecal = ecalAssoTrack.begin();  
	itecal != ecalAssoTrack.end(); ++itecal) {  
      linked_e=linked_e+elements[itecal->second].clusterRef()->energy();  
    }  
  }  
  if(hcalAssoTrack.size() > 0) {  
    for(std::multimap<double, unsigned int>::iterator ithcal = hcalAssoTrack.begin();  
	ithcal != hcalAssoTrack.end(); ++ithcal) {  
      linked_h=linked_h+elements[ithcal->second].clusterRef()->energy();  
    }  
  }  
  EoverPt=linked_e/elements[track_index].trackRef()->pt();  
  HoverPt=linked_h/elements[track_index].trackRef()->pt();  
  GlobalVector rvtx(elements[track_index].trackRef()->innerPosition().X()-primaryvtx.x(),  
		    elements[track_index].trackRef()->innerPosition().Y()-primaryvtx.y(),  
		    elements[track_index].trackRef()->innerPosition().Z()-primaryvtx.z());  
  double vtx_phi=rvtx.phi();  
  //delta Phi between conversion vertex and track  
  del_phi=fabs(deltaPhi(vtx_phi, elements[track_index].trackRef()->innerMomentum().Phi()));  
  mvaValue = tmvaReader_->EvaluateMVA("BDT");  
  if(mvaValue > MVACUT)convtkfound=true;  
  return convtkfound;  
}
