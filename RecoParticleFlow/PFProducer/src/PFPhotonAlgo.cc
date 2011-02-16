//
// Original Authors: Fabian Stoeckli: fabian.stoeckli@cern.ch
//                   Nicholas Wardle: nckw@cern.ch
//

#include "RecoParticleFlow/PFProducer/interface/PFPhotonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include <iomanip>
#include <algorithm>

using namespace std;
using namespace reco;


PFPhotonAlgo::PFPhotonAlgo() : isvalid_(false), verbosityLevel_(Silent) {
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
    std::cout<<" WARNING: Size of collection and active-vectro don't agree!"<<std::endl;
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
    
    // Total pre-shower energy?
    float ps1TotEne      = 0.;
    float ps2TotEne      = 0.;
    
    isActive = *(actIter);
    cout << " Found a SuperCluster.  Energy " ;
    const reco::PFBlockElementSuperCluster *sc = dynamic_cast<const reco::PFBlockElementSuperCluster*>(&(*ele));
    std::cout << sc->superClusterRef()->energy () << " Track/Ecal/Hcal Iso " << sc->trackIso()<< " " << sc->ecalIso() ;
    std::cout << " " << sc->hcalIso() <<std::endl;
    
    // check the status of the SC Element... 
    // ..... I understand it should *always* be active, since PFElectronAlgo does not touch this (yet?)
    if( !isActive ) {
      std::cout<<" SuperCluster is NOT active.... "<<std::endl;
      continue;
    }
    
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
      std::cout<<" Found SC element with no ECAL assigned "<<std::endl;
      continue;
    }
    
    // This is basically CASE 2
    // .... we loop over all ECAL cluster linked to each other by this SC
    for(std::multimap<double, unsigned int>::iterator itecal = ecalAssoPFClusters.begin(); 
	itecal != ecalAssoPFClusters.end(); ++itecal) { 
      
      // to get the reference to the PF clusters, this is needed.
      reco::PFClusterRef clusterRef = elements[itecal->second].clusterRef();	
      
      // from the clusterRef get the energy, direction, etc
      float ClustRawEnergy = clusterRef->energy();
      float ClustEta = clusterRef->position().eta();
      float ClustPhi = clusterRef->position().phi();

      // initialize the vectors for the PS energies
      vector<double> ps1Ene(0);
      vector<double> ps2Ene(0);

      cout << " My cluster index " << itecal->second 
	   << " energy " <<  ClustRawEnergy
	   << " eta " << ClustEta
	   << " phi " << ClustPhi << endl;
      
      // check if this ECAL element is still active (could have been eaten by PFElectronAlgo)
      // ......for now we give the PFElectron Algo *ALWAYS* Shot-Gun on the ECAL elements to the PFElectronAlgo
      
      if( !( active[itecal->second] ) ) {
	std::cout<< "  .... this ECAL element is NOT active anymore. Is skipped. "<<std::endl;
	continue;
      }
      
      // ------------------------------------------------------------------------------------------
      // TODO: do some tests on the ECAL cluster itself, deciding to use it or not for the Photons
      // ..... ??? Do we need this?
      if ( false ) {
	// This is DUMMY
	bool useIt = true;
	if( !useIt ) continue;    // Go to next ECAL cluster within SC
      }
      // ------------------------------------------------------------------------------------------
      
      // We decided to keep the ECAL cluster for this Photon Candidate ...
      elemsToLock.push_back(itecal->second);
      
      // look for PS in this Block linked to this ECAL cluster      
      std::multimap<double, unsigned int> PS1Elems;
      std::multimap<double, unsigned int> PS2Elems;
      blockRef->associatedElements( itecal->second,
				    linkData,
				    PS1Elems,
				    reco::PFBlockElement::PS1,
				    reco::PFBlock::LINKTEST_ALL );
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
	
	// TODO: Check if this PS1 is not closer to another ECAL cluster in this Block
	
	reco::PFClusterRef ps1ClusterRef = elements[iteps->second].clusterRef();
	ps1Ene.push_back( ps1ClusterRef->energy() );
	// incativate this PS1 Element
	elemsToLock.push_back(iteps->second);
      }
      for(std::multimap<double, unsigned int>::iterator iteps = PS2Elems.begin();
	  iteps != PS2Elems.end(); ++iteps) {

	// first chekc if it's still active
	if( !(active[iteps->second]) ) continue;
	
	// TODO: Check if this PS2 is not closer to another ECAL cluster in this Block
	
	reco::PFClusterRef ps2ClusterRef = elements[iteps->second].clusterRef();
	ps2Ene.push_back( ps2ClusterRef->energy() );
	// incativate this PS1 Element
	elemsToLock.push_back(iteps->second);
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

	bool useHcal = true;
	if ( !useHcal ) continue;
	
	elemsToLock.push_back(ithcal->second);
	// how do we 'add' the energy of the HCAL...???
	// for now we just lock it, but don't use it's energy

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

	// Possibly need to be more smart about them (CASE 5)
	// .... for now we simply skip non id'ed tracks
	if( ! (trackRef->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) ) ) continue;  
	
	
	// we need to check for other TRACKS linked to this conversion track, that point possibly no an ECAL cluster not included in the SC
	// .... This is basically CASE 4.
	// .... However, for now we don't check if the ECAL is actually part of the SC or not, since we anyways want to include it.
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
	  // check if it's a CONV track
	  const reco::PFBlockElementTrack * track2Ref = dynamic_cast<const reco::PFBlockElementTrack*>((&elements[track2->second])); 	
	  if( ! (track2Ref->trackType(reco::PFBlockElement::T_FROM_GAMMACONV) ) ) continue;  // Possibly need to be more smart about them (CASE 5)
	  
	  // so it's another active conversion track, that is in the Block and linked to the conversion track we already found
	  // find the ECAL cluster linked to it...
	  std::multimap<double, unsigned int> convEcal;
	  blockRef->associatedElements( track2->second,
					linkData,
					convEcal,
					reco::PFBlockElement::ECAL,
					reco::PFBlock::LINKTEST_ALL);
	  
    
	  for(std::multimap<double, unsigned int>::iterator itConvEcal = convEcal.begin();
	      itConvEcal != convEcal.end(); ++itConvEcal) {
	    
	    reco::PFClusterRef clusterRef = elements[itConvEcal->second].clusterRef();	
	    if( ! (active[itConvEcal->second]) ) continue;
	    
	    elemsToLock.push_back(itConvEcal->second);
	    
	    // it's still active, so we have to add it.
	    // CAUTION: we don't care here if it's part of the SC or not, we include it anyways
	    // TODO: Do we want the option of NOT include the second T_FROM_GAMMACONV leg?
	    
	    // look for PS in this Block linked to this ECAL cluster      
	    std::multimap<double, unsigned int> PS1Elems_conv;
	    std::multimap<double, unsigned int> PS2Elems_conv;
	    blockRef->associatedElements( itConvEcal->second,
					  linkData,
					  PS1Elems_conv,
					  reco::PFBlockElement::PS1,
					  reco::PFBlock::LINKTEST_ALL );
	    blockRef->associatedElements( itConvEcal->second,
					  linkData,
					  PS2Elems_conv,
					  reco::PFBlockElement::PS2,
					  reco::PFBlock::LINKTEST_ALL );
      
	    for(std::multimap<double, unsigned int>::iterator iteps = PS1Elems_conv.begin();
		iteps != PS1Elems_conv.end(); ++iteps) {
	      
	      // first chekc if it's still active
	      if( !(active[iteps->second]) ) continue;
	      
	      // TODO: Check if this PS1 is not closer to another ECAL cluster in this Block
	
	      reco::PFClusterRef ps1ClusterRef = elements[iteps->second].clusterRef();
	      ps1Ene.push_back( ps1ClusterRef->energy() );
	      // incativate this PS1 Element
	      elemsToLock.push_back(iteps->second);
	    } // end of loop over PS1
	    
	    for(std::multimap<double, unsigned int>::iterator iteps = PS2Elems_conv.begin();
		iteps != PS2Elems_conv.end(); ++iteps) {

	      // first chekc if it's still active
	      if( !(active[iteps->second]) ) continue;
	      
	      // TODO: Check if this PS2 is not closer to another ECAL cluster in this Block
	      
	      reco::PFClusterRef ps2ClusterRef = elements[iteps->second].clusterRef();
	      ps2Ene.push_back( ps2ClusterRef->energy() );
	      // incativate this PS1 Element
	      elemsToLock.push_back(iteps->second);
	    } // end of loop over PS2
	    
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
	      
	      elemsToLock.push_back(ithcal2->second);

	    } // end of loop over HCAL clusters linked to the ECAL cluster from second CONVERSION leg
	    
	  } // end of loop over ECALs linked to second T_FROM_GAMMACONV
	  
	} // end of loop over SECOND conversion leg

	// TODO: Do we need to check separatly if there are HCAL cluster linked to the track?
	
      } // end of loop over tracks
      
      // Are we using the correct calibration  here ???
      double ps1=0;
      double ps2=0;
      float  EE = pfcalib_.energyEm(*clusterRef,ps1Ene,ps2Ene,ps1,ps2,true); //applyCrackCorrections_); ADD THIS VARIABLE
      
      photonEnergy_ +=  EE;
      photonX_      +=  EE * clusterRef->position().X();
      photonY_      +=  EE * clusterRef->position().Y();
      photonZ_      +=  EE * clusterRef->position().Z();	  
      
      ps1TotEne     +=  ps1;
      ps2TotEne     +=  ps2;
    } // end of loop over all ECAL cluster within this SC
    
    // we've looped over all ECAL clusters, ready to generate PhotonCandidate
    if( ! (photonEnergy_ > 0.) ) continue;    // This SC is not a Photon Candidate

    math::XYZVector photonPosition(photonX_,
				   photonY_,
				   photonZ_);

    math::XYZVector photonDirection=photonPosition.Unit();
    
    math::XYZTLorentzVector photonMomentum(photonEnergy_* photonDirection.X(),
					   photonEnergy_* photonDirection.Y(),
					   photonEnergy_* photonDirection.Z(),
					   photonEnergy_           );
    
    std::cout<<" Created Photon with energy = "<<photonEnergy_<<std::endl;
    std::cout<<"                         pT = "<<photonMomentum.pt()<<std::endl;
    std::cout<<"                       Mass = "<<photonMomentum.mag()<<std::endl;
    std::cout<<"                          E = "<<photonMomentum.e()<<std::endl;
    
    reco::PFCandidate photonCand(0,photonMomentum, reco::PFCandidate::gamma);
    
    photonCand.setPs1Energy(ps1TotEne);
    photonCand.setPs2Energy(ps2TotEne);
    photonCand.setEcalEnergy(photonEnergy_);
    photonCand.setRawEcalEnergy(photonEnergy_-ps1TotEne-ps2TotEne);
    //photonCand.setPositionAtECALEntrance(math::XYZPointF(photonMom_.position()));
    //photonCand.addElementInBlock(blockRef,assobrem_index[ibrem]);
    
    // set isvalid_ to TRUE since we've found at least one photon candidate
    isvalid_ = true;
    // push back the candidate into the collection ...
    pfCandidates->push_back(photonCand);
    // ... and lock all elemts used
    for(std::vector<unsigned int>::const_iterator it = elemsToLock.begin();
	it != elemsToLock.end(); ++it)
      active[*it] = false;
    // ... and reset the vector
    elemsToLock.resize(0);
    
  } // end of loops over all elements in block
  
  return;

}
