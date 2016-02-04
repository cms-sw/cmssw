#include "RecoParticleFlow/PFProducer/interface/PFConversionAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h" 
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "RecoParticleFlow/PFProducer/interface/Utils.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

using namespace std;
using namespace reco;
PFConversionAlgo::PFConversionAlgo( ) {

}

void PFConversionAlgo::runPFConversion(const reco::PFBlockRef&  blockRef, 
				       std::vector<bool>& active) {

  //  std::cout << " PFConversionAlgo::RunPFConversion " << std::endl;

  AssMap elemAssociatedToConv;
  
  bool blockHasConversion = setLinks(blockRef,elemAssociatedToConv, active );
  
  if (  blockHasConversion ) {
    conversionCandidate_.clear();
    setCandidates(blockRef,elemAssociatedToConv);
    if (conversionCandidate_.size() > 0 ){
      isvalid_ = true;
      //cout << " There is a candidate " << endl;
      // if there is at least a candidate the active vector is modified
      // setting = false all the elements used to build the candidate
      setActive(blockRef,elemAssociatedToConv, active);
      // this is just debug to check that all is going fine. Will take it out later      
      std::vector<reco::PFCandidate>::iterator it;
      for ( it = conversionCandidate_.begin(); it != conversionCandidate_.end(); ++it )  {
	reco::PFCandidate::ParticleType type = (*it).particleId();
	bool isConverted =  (*it).flag( reco::PFCandidate::GAMMA_TO_GAMMACONV );
	if ( type == reco::PFCandidate::gamma && isConverted ) {
	  //	  cout<<"Conversion PFCandidate!"<<  *it <<  endl;
	}
      }
    }
    
  } // conversion was found in the block

}


bool PFConversionAlgo::setLinks(const reco::PFBlockRef& blockRef,  
				AssMap&  elemAssociatedToConv,  
				std::vector<bool>& active ) {

  bool conversionFound = false;
  typedef std::multimap<double, unsigned>::iterator IE;
  const reco::PFBlock& block = *blockRef;
  //  std::cout << " PFConversionAlgo::setLinks block " << block << std::endl;
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  PFBlock::LinkData linkData =  block.linkData();    

  // this method looks in all elements in a block, idenitifies those 
  // blonging to a conversions and stores them in a local map 
  // so that in the further code the search needs not be done again

  unsigned convTrack1Ind=100;
  unsigned convTrack2Ind=100;
  unsigned convEcal1Ind=200;
  unsigned convEcal2Ind=200;
  unsigned convHcal1Ind=200;
  unsigned convHcal2Ind=200;

  for(unsigned iElem=0; iElem<elements.size(); iElem++) {
    bool trackFromConv = elements[iElem].trackType( PFBlockElement::T_FROM_GAMMACONV);

    if (!active[iElem]) continue;  
    if ( !trackFromConv ) continue;
    conversionFound = true;    

    //    std::cout << " Track " <<  iElem << " is from conversion" << std::endl;
    convTrack1Ind= iElem;

    std::multimap<unsigned, std::vector<unsigned> >::iterator found = elemAssociatedToConv.find(iElem);
    if ( found!= elemAssociatedToConv.end()) {
      //      std::cout << " Track " <<  iElem << " has already been included " << std::endl;
      continue;
    }
    
    
    bool alreadyTaken=false;
    for (  std::multimap<unsigned, std::vector<unsigned> >::iterator i=elemAssociatedToConv.begin(); 
	   i!=  elemAssociatedToConv.end(); ++i) {
      for (unsigned int j=0; j< (i->second).size(); ++j ) {
	if ( iElem == (i->second)[j] )  alreadyTaken=true;
      }
    }
    if ( alreadyTaken ) {
      //      std::cout << " iElem " << iElem << " already taken" <<  std::endl;
      continue;
    }

 
    vector<unsigned> assElements(0);     
    vector<unsigned>::iterator iVec;

    std::multimap<double, unsigned> ecalElems;
    block.associatedElements(iElem ,  linkData,
			      ecalElems ,
			      reco::PFBlockElement::ECAL );

    std::multimap<double, unsigned> hcalElems;
    block.associatedElements( iElem ,  linkData,
                              hcalElems,
                              reco::PFBlockElement::HCAL,
                              reco::PFBlock::LINKTEST_ALL );
    
    std::multimap<double, unsigned> trackElems;
    block.associatedElements( iElem,  linkData,
                              trackElems ,
                              reco::PFBlockElement::TRACK,
                              reco::PFBlock::LINKTEST_RECHIT);


    if(trackElems.empty() ) {
      //      std::cout<<"PFConversionAlgo::setLinks no track element connected to track "<<iElem<<std::endl;
    }
    
    if(ecalElems.empty() ) {
      //  std::cout<<"PFConversionAlgo::setLinks no ecal element connected to track "<<iElem<<std::endl;
    }
    
    if(hcalElems.empty() ) {
      //  std::cout<<"PFConversionAlgo::setLinks no hcal element connected to track "<<iElem<<std::endl;
    }
    
    //std::cout<<"PFConversionAlgo::setLinks now looping on elements associated to the track"<<std::endl;



    //    std::cout<<"  look at linked hcal clusters"<<std::endl;
    for(IE iTk = hcalElems.begin(); iTk != hcalElems.end(); ++iTk ) {
      unsigned index = iTk->second;
      PFBlockElement::Type type = elements[index].type();
      if ( type ==  reco::PFBlockElement::HCAL) {
	// link track-ecal is found 
        convHcal1Ind=index;
	//	std::cout << " Hcal-Track link found with " << convHcal1Ind << std::endl;
	//	if ( index< 100)  assElements.push_back(index);
      }
    }



    //    std::cout<<"  look at linked ecal clusters"<<std::endl;
    for(IE iTk = ecalElems.begin(); iTk != ecalElems.end(); ++iTk ) {
      unsigned index = iTk->second;
      PFBlockElement::Type type = elements[index].type();
      if ( type ==  reco::PFBlockElement::ECAL) {
	// link track-ecal is found 
        convEcal1Ind=index;
	//std::cout << " Ecal-Track link found with " << convEcal1Ind << std::endl;
	iVec = find ( assElements.begin(), assElements.end(), index) ;        
	if ( index< 100 && iVec == assElements.end() )  assElements.push_back(index);
      }
    }


    //    std::cout<<"PFConversionAlgo::setLinks  look at linked tracks"<<std::endl;    
    for(IE iTk = trackElems.begin(); iTk != trackElems.end(); ++iTk ) {
      unsigned index = iTk->second;
      //PFBlockElement::Type type = elements[index].type();
	// link track-track is found 
        convTrack2Ind=index;
	if ( index< 100)  assElements.push_back(index);
	//std::cout << " Track-Track link found with " << convTrack2Ind << std::endl;
	std::multimap<double, unsigned> ecalElems2;
	block.associatedElements(convTrack2Ind ,  linkData,
				 ecalElems2 ,
				 reco::PFBlockElement::ECAL );


	for(IE iTk = ecalElems2.begin(); iTk != ecalElems2.end(); ++iTk ) {
	  unsigned index = iTk->second;
	  PFBlockElement::Type type = elements[index].type();
	  if ( type ==  reco::PFBlockElement::ECAL) {
	    convEcal2Ind=index;
	    //	    std::cout << " 2nd ecal track link found betwtenn track " << convTrack2Ind  << " and Ecal " << convEcal2Ind << std::endl;
	    iVec = find ( assElements.begin(), assElements.end(), index) ;        
	    if ( index< 100 && iVec== assElements.end() )  assElements.push_back(index);
	  
	  }
	}

	std::multimap<double, unsigned> hcalElems2;
	block.associatedElements(convTrack2Ind ,  linkData,
				 hcalElems2 ,
				 reco::PFBlockElement::HCAL );
	for(IE iTk = hcalElems.begin(); iTk != hcalElems.end(); ++iTk ) {
	  unsigned index = iTk->second;
	  PFBlockElement::Type type = elements[index].type();
	  if ( type ==  reco::PFBlockElement::HCAL) {
	    // link track-ecal is found 
	    convHcal2Ind=index;
	    std::cout << " Hcal-Track link found with " << convHcal2Ind << std::endl;
	    //   if ( index< 100)  assElements.push_back(index);
	  }
	}

    }


    elemAssociatedToConv.insert(make_pair(convTrack1Ind, assElements));

    // This is just for debug
    //std::cout << " PFConversionAlgo::setLink map size " << elemAssociatedToConv.size() << std::endl;
    // for (  std::multimap<unsigned, std::vector<unsigned> >::iterator i=elemAssociatedToConv.begin(); 
	   //   i!=  elemAssociatedToConv.end(); ++i) {
      //std::cout << " links found for " << i->first << std::endl;
      //std::cout << " elements " << (i->second).size() << std::endl;
    // for (unsigned int j=0; j< (i->second).size(); ++j ) {
	//	unsigned int iEl = (i->second)[j];
	//std::cout  << " ass element " << iEl << std::endl;
    // }
    // }



  } // end loop over elements of the block looking for conversion track


  return conversionFound;

}


void PFConversionAlgo::setCandidates(const reco::PFBlockRef& blockRef,  
				     AssMap&  elemAssociatedToConv ) {

  
  vector<unsigned int> elementsToAdd(0);
  const reco::PFBlock& block = *blockRef;
  PFBlock::LinkData linkData =  block.linkData();     
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();

  //// Loop over the element of the block and fill in the vector elementsToAdd to the candidate
  //////////////////////////////
  float EcalEne=0;
  float pairPx=0;
  float pairPy=0;
  float pairPz=0;
  const reco::PFBlockElementTrack*  elTrack=0;
  reco::TrackRef convTrackRef;
  for (  std::multimap<unsigned, std::vector<unsigned> >::iterator i=elemAssociatedToConv.begin(); 
	 i!=  elemAssociatedToConv.end(); ++i) {

    unsigned int iTrack =  i->first;
    elementsToAdd.push_back(iTrack);
    //    std::cout << " setCandidates adding track " << iTrack << " to block in PFCandiate " << std::endl;
    elTrack = dynamic_cast<const reco::PFBlockElementTrack*>((&elements[iTrack]));
    convTrackRef= elTrack->trackRef();     
    pairPx+=convTrackRef->innerMomentum().x();
    pairPy+=convTrackRef->innerMomentum().y();
    pairPz+=convTrackRef->innerMomentum().z();
    
    ConversionRef origConv = elements[iTrack].convRef();
    //    std::cout << " Ref to original conversions: track size " << origConv->tracks().size() << " SC energy " << origConv->caloCluster()[0]->energy() << std::endl;
    // std::cout  << " SC Et " <<  origConv->caloCluster()[0]->energy()/cosh(origConv->caloCluster()[0]->eta()) <<" eta " << origConv->caloCluster()[0]->eta() << " phi " << origConv->caloCluster()[0]->phi() <<  std::endl;

    unsigned int nEl=  (i->second).size();
    //    std::cout << " Number of elements connected " << nEl << std::endl;
    for (unsigned int j=0; j< nEl; ++j ) {
      unsigned int iEl = (i->second)[j];
      //std::cout  << " Adding element " << iEl << std::endl;
      PFBlockElement::Type typeassCalo = elements[iEl].type();

      /// Get the momentum of the parent track pair
      if ( typeassCalo == reco::PFBlockElement::TRACK) {
	const reco::PFBlockElementTrack * elTrack = dynamic_cast<const reco::PFBlockElementTrack*>((&elements[iEl]));	
	elTrack = dynamic_cast<const reco::PFBlockElementTrack*>((&elements[iEl]));
	convTrackRef= elTrack->trackRef();     
	pairPx+=convTrackRef->innerMomentum().x();
	pairPy+=convTrackRef->innerMomentum().y();
	pairPz+=convTrackRef->innerMomentum().z();
      }

      if ( typeassCalo == reco::PFBlockElement::ECAL) {
	const reco::PFBlockElementCluster * clu = dynamic_cast<const reco::PFBlockElementCluster*>((&elements[iEl]));
	reco::PFCluster cl=*clu->clusterRef();
        EcalEne+= cl.energy();
      }

      elementsToAdd.push_back(iEl);
    }

    //    std::cout << " setCandidates EcalEne " << EcalEne << std::endl;
    reco::PFCandidate::ParticleType particleType = reco::PFCandidate::gamma;

    // define energy and momentum of the conversion. Momentum from the track pairs and Energy from 
    // the sum of the ecal cluster(s)
    math::XYZTLorentzVector  momentum;
    momentum.SetPxPyPzE(pairPx , pairPy, pairPz, EcalEne);

    //// inputs for ID 

    float deltaCotTheta=origConv->pairCotThetaSeparation();
    float phiTk1=  origConv->tracks()[0]->innerMomentum().phi();
    float phiTk2=  origConv->tracks()[1]->innerMomentum().phi();
    float deltaPhi = phiTk1-phiTk2;
    if(deltaPhi >  pi) {deltaPhi = deltaPhi - twopi;}
    if(deltaPhi < -pi) {deltaPhi = deltaPhi + twopi;}

    /// for a first try just simple cuts
    if (  fabs(deltaCotTheta) < 0.05 && abs(deltaPhi<0.1) )  {
      
      
      /// Build candidate  
      reco::PFCandidate aCandidate = PFCandidate(0, momentum,particleType);
      for (unsigned int elad=0; elad<elementsToAdd.size();elad++){
	aCandidate.addElementInBlock(blockRef,elementsToAdd[elad]);
      }
      
      
      aCandidate.setFlag( reco::PFCandidate::GAMMA_TO_GAMMACONV, true);
      aCandidate.setConversionRef(origConv);
      aCandidate.setEcalEnergy(EcalEne);
      aCandidate.setHcalEnergy(0.); 
      aCandidate.setPs1Energy(0.); 
      aCandidate.setPs2Energy(0.); 
      
      
      conversionCandidate_.push_back( aCandidate);       
    }  
    

  }



  return;
}




void PFConversionAlgo::setActive(const reco::PFBlockRef& blockRef,  
				 AssMap&  elemAssociatedToConv, std::vector<bool>& active ) {

  // Lock tracks and clusters belonging to the conversion
  for (  std::multimap<unsigned, std::vector<unsigned> >::iterator i=elemAssociatedToConv.begin(); 
	 i!=  elemAssociatedToConv.end(); ++i) {
    unsigned int iConvTrk =  i->first;
    active[iConvTrk]=false;
    //std::cout << "  PFConversionAlgo::setActive locking all elements linked to a conversion " << std::endl;
    for (unsigned int j=0; j< (i->second).size(); ++j ) {
      active[(i->second)[j]]=false;
      //std::cout << " Locking element " << (i->second)[j] << std::endl;
    }
  }

  
  return;
}

