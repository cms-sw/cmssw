//
// Original Authors: Fabian Stoeckli: fabian.stoeckli@cern.ch
//                   Nicholas Wardle: nckw@cern.ch
//                   Rishi Patel rpatel@cern.ch(ongoing developer and maintainer)
//

#include "RecoParticleFlow/PFProducer/interface/PFPhotonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFPhotonClusters.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <TFile.h>
#include <iomanip>
#include <algorithm>
#include <TMath.h>
using namespace std;
using namespace reco;


PFPhotonAlgo::PFPhotonAlgo(std::string mvaweightfile,  
			   double mvaConvCut, 
			   bool useReg,
			   std::string X0_Map,
			   const reco::Vertex& primary,
			   const boost::shared_ptr<PFEnergyCalibration>& thePFEnergyCalibration,
                           double sumPtTrackIsoForPhoton,
                           double sumPtTrackIsoSlopeForPhoton
			   ) : 
  isvalid_(false), 
  verbosityLevel_(Silent), 
  MVACUT(mvaConvCut),
  useReg_(useReg),
  thePFEnergyCalibration_(thePFEnergyCalibration),
  sumPtTrackIsoForPhoton_(sumPtTrackIsoForPhoton),
  sumPtTrackIsoSlopeForPhoton_(sumPtTrackIsoSlopeForPhoton),
  nlost(0.0), nlayers(0.0),
  chi2(0.0), STIP(0.0), del_phi(0.0),HoverPt(0.0), EoverPt(0.0), track_pt(0.0),
  mvaValue(0.0),
  CrysPhi_(0.0), CrysEta_(0.0),  VtxZ_(0.0), ClusPhi_(0.0), ClusEta_(0.0),
  ClusR9_(0.0), Clus5x5ratio_(0.0),  PFCrysEtaCrack_(0.0), logPFClusE_(0.0), e3x3_(0.0),
  CrysIPhi_(0), CrysIEta_(0),
  CrysX_(0.0), CrysY_(0.0),
  EB(0.0),
  eSeed_(0.0), e1x3_(0.0),e3x1_(0.0), e1x5_(0.0), e2x5Top_(0.0),  e2x5Bottom_(0.0), e2x5Left_(0.0),  e2x5Right_(0.0),
  etop_(0.0), ebottom_(0.0), eleft_(0.0), eright_(0.0),
  e2x5Max_(0.0),
  PFPhoEta_(0.0), PFPhoPhi_(0.0), PFPhoR9_(0.0), PFPhoR9Corr_(0.0), SCPhiWidth_(0.0), SCEtaWidth_(0.0), 
  PFPhoEt_(0.0), RConv_(0.0), PFPhoEtCorr_(0.0), PFPhoE_(0.0), PFPhoECorr_(0.0), MustE_(0.0), E3x3_(0.0),
  dEta_(0.0), dPhi_(0.0), LowClusE_(0.0), RMSAll_(0.0), RMSMust_(0.0), nPFClus_(0.0),
  TotPS1_(0.0), TotPS2_(0.0),
  nVtx_(0.0),
  x0inner_(0.0), x0middle_(0.0), x0outer_(0.0),
  excluded_(0.0), Mustache_EtRatio_(0.0), Mustache_Et_out_(0.0)
{  

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

    //Material Map
    TFile *XO_File = new TFile(X0_Map.c_str(),"READ");
    X0_sum=(TH2D*)XO_File->Get("TrackerSum");
    X0_inner = (TH2D*)XO_File->Get("Inner");
    X0_middle = (TH2D*)XO_File->Get("Middle");
    X0_outer = (TH2D*)XO_File->Get("Outer");
    
}

void PFPhotonAlgo::RunPFPhoton(const reco::PFBlockRef&  blockRef,
			       std::vector<bool>& active,
			       std::auto_ptr<PFCandidateCollection> &pfCandidates,
			       std::vector<reco::PFCandidatePhotonExtra>& pfPhotonExtraCandidates,
			       std::vector<reco::PFCandidate> 
			       &tempElectronCandidates
){
  
  //std::cout<<" calling RunPFPhoton "<<std::endl;
  
  /*      For now we construct the PhotonCandidate simply from 
	  a) adding the CORRECTED energies of each participating ECAL cluster
	  b) build the energy-weighted direction for the Photon
  */


  // define how much is printed out for debugging.
  // ... will be setable via CFG file parameter
  verbosityLevel_ = Chatty;          // Chatty mode.
  

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
    std::vector<reco::TrackRef>singleLegRef;
    std::vector<float>MVA_values(0);
    std::vector<float>MVALCorr;
    std::vector<CaloCluster>PFClusters;
    reco::ConversionRefVector ConversionsRef_;
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
    //R9 of SuperCluster and RawE
    PFPhoR9_=sc->photonRef()->r9();
    E3x3_=PFPhoR9_*(sc->superClusterRef()->rawEnergy());
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
	  hasSingleleg=EvaluateSingleLegMVA(blockRef,  *primaryVertex_, track->second);  
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
	hasSingleleg=EvaluateSingleLegMVA(blockRef,  *primaryVertex_, track->second);  

	// Daniele; example for mvaValues, do the same for single leg trackRef and convRef
	//          
	// 	if(hasSingleleg)
	// 	  mvaValues.push_back(mvaValue);

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
	    
	    reco::TrackRef t_ref=elements[track->second].trackRef();
	    bool matched=false;
	    for(unsigned int ic=0; ic<singleLegRef.size(); ic++)
	      if(singleLegRef[ic]==t_ref)matched=true;
	    
	    if(!matched){
	      singleLegRef.push_back(t_ref);
	      MVA_values.push_back(mvaValue);
	    }
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
	//if(elements[track->second].convRef().isNonnull())
	//{	    
	//  ConversionsRef_.push_back(elements[track->second].convRef());
	//}
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
	  float p_in=sqrt(elements[track->second].trackRef()->innerMomentum().x()*elements[track->second].trackRef()->innerMomentum().x()+
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
	   
	  for(std::multimap<double, unsigned int>::iterator iteps = PS2Elems_conv.begin();  
	      iteps != PS2Elems_conv.end(); ++iteps) {  
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
	  addedRawEne = AddclusterRef->energy()+addedRawEne;  
	  addedCalibEne = thePFEnergyCalibration_->energyEm(*AddclusterRef,AddedPS1,AddedPS2,false)+addedCalibEne;  
	  AddedPS2.clear(); 
	  AddedPS1.clear();  
	  elemsToLock.push_back(AddClusters[i]);  
	}  
      AddClusters.clear();
      float EE=thePFEnergyCalibration_->energyEm(*clusterRef,ps1Ene,ps2Ene,false)+addedCalibEne;
      PFClusters.push_back(*clusterRef);
      if(useReg_){
	float LocCorr=EvaluateLCorrMVA(clusterRef);
	EE=LocCorr*clusterRef->energy()+addedCalibEne;
      }
      else{
	float LocCorr=EvaluateLCorrMVA(clusterRef);
	MVALCorr.push_back(LocCorr*clusterRef->energy());
      }
      
      //cout<<"Original Energy "<<EE<<"Added Energy "<<addedCalibEne<<endl;
      
      photonEnergy_ +=  EE;
      RawEcalEne    +=  clusterRef->energy()+addedRawEne;
      photonX_      +=  EE * clusterRef->position().X();
      photonY_      +=  EE * clusterRef->position().Y();
      photonZ_      +=  EE * clusterRef->position().Z();	        
      ps1TotEne     +=  ps1+addedps1;
      ps2TotEne     +=  ps2+addedps2;
    } // end of loop over all ECAL cluster within this SC
    AddFromElectron_.clear();
    float Elec_energy=0;
    float Elec_rawEcal=0;
    float Elec_totPs1=0;
    float Elec_totPs2=0;
    float ElectronX=0;
    float ElectronY=0;
    float ElectronZ=0;  
    std::vector<double>AddedPS1(0);
    std::vector<double>AddedPS2(0);
    
    EarlyConversion(    
		    tempElectronCandidates,
		    sc
		    );   
    
    if(AddFromElectron_.size()>0)
      {	
	//collect elements from early Conversions that are reconstructed as Electrons
	
	for(std::vector<unsigned int>::const_iterator it = 
	      AddFromElectron_.begin();
	    it != AddFromElectron_.end(); ++it)
	  {
	    
	    if(elements[*it].type()== reco::PFBlockElement::ECAL)
	      {
	        //cout<<"Cluster ind "<<*it<<endl;
		AddedPS1.clear();
		AddedPS2.clear();
		unsigned int index=*it;
		reco::PFClusterRef clusterRef = 
		elements[index].clusterRef();
		//match to PS1 and PS2 to this cluster for calibration
		Elec_rawEcal=Elec_rawEcal+
		  elements[index].clusterRef()->energy();
		std::multimap<double, unsigned int> PS1Elems;  
		std::multimap<double, unsigned int> PS2Elems;  
		
		blockRef->associatedElements(index,  
					     linkData,  
					     PS1Elems,  					     reco::PFBlockElement::PS1,  
					     reco::PFBlock::LINKTEST_ALL );  
		blockRef->associatedElements( index,  
					      linkData,  
					      PS2Elems,  
					      reco::PFBlockElement::PS2,  
					      reco::PFBlock::LINKTEST_ALL );
		
		
		for(std::multimap<double, unsigned int>::iterator iteps = 
		      PS1Elems.begin();  
		    iteps != PS1Elems.end(); ++iteps) 
		  {  
		    std::multimap<double, unsigned int> Clustcheck;  		    	    blockRef->associatedElements( iteps->second,  								   linkData,  
															  Clustcheck,  
															  reco::PFBlockElement::ECAL,  
															  reco::PFBlock::LINKTEST_ALL );
		    if(Clustcheck.begin()->second==index)
		      {
			AddedPS1.push_back(elements[iteps->second].clusterRef()->energy());
			Elec_totPs1=Elec_totPs1+elements[iteps->second].clusterRef()->energy();
		      }
		  }
		
		for(std::multimap<double, unsigned int>::iterator iteps = 
		      PS2Elems.begin();  
		    iteps != PS2Elems.end(); ++iteps) 
		  {  
		    std::multimap<double, unsigned int> Clustcheck;  		    	    blockRef->associatedElements( iteps->second,  								   linkData,  
															  Clustcheck,  
															  reco::PFBlockElement::ECAL,  
															  reco::PFBlock::LINKTEST_ALL );
		    if(Clustcheck.begin()->second==index)
		      {
			AddedPS2.push_back(elements[iteps->second].clusterRef()->energy());
			Elec_totPs2=Elec_totPs2+elements[iteps->second].clusterRef()->energy();
		      }
		  }
		
		//energy calibration 
		float EE=thePFEnergyCalibration_->
		  energyEm(*clusterRef,AddedPS1,AddedPS2,false);
		PFClusters.push_back(*clusterRef);
		if(useReg_){
		  float LocCorr=EvaluateLCorrMVA(clusterRef);
		  EE=LocCorr*clusterRef->energy();	  
		  MVALCorr.push_back(LocCorr*clusterRef->energy());
		  
		}
		else{
		  float LocCorr=EvaluateLCorrMVA(clusterRef);
		  MVALCorr.push_back(LocCorr*clusterRef->energy());
		}
		
		Elec_energy    += EE;
		ElectronX      +=  EE * clusterRef->position().X();
		ElectronY      +=  EE * clusterRef->position().Y();
		ElectronZ      +=  EE * clusterRef->position().Z();
		
	      }
	    if(elements[*it].type()==reco::PFBlockElement::TRACK){
	      reco::TrackRef t_ref=elements[*it].trackRef();
	      singleLegRef.push_back(t_ref);
	      EvaluateSingleLegMVA(blockRef,  *primaryVertex_, *it);  
	      MVA_values.push_back(mvaValue);	      
	    }
	  }
	
      }
    
    //std::cout<<"Added Energy to Photon "<<Elec_energy<<" to "<<photonEnergy_<<std::endl;   
    photonEnergy_ +=  Elec_energy;
    RawEcalEne    +=  Elec_rawEcal;
    photonX_      +=  ElectronX;
    photonY_      +=  ElectronY;
    photonZ_      +=  ElectronZ;	        
    ps1TotEne     +=  Elec_totPs1;
    ps2TotEne     +=  Elec_totPs2;
    
    // we've looped over all ECAL clusters, ready to generate PhotonCandidate
    if( ! (photonEnergy_ > 0.) ) continue;    // This SC is not a Photon Candidate
    float sum_track_pt=0;
    //Now check if there are tracks failing isolation outside of the Jurassic isolation region  
    for(unsigned int i=0; i<IsoTracks.size(); i++)sum_track_pt=sum_track_pt+elements[IsoTracks[i]].trackRef()->pt();  
    


    math::XYZVector photonPosition(photonX_,
				   photonY_,
				   photonZ_);
    math::XYZVector photonPositionwrtVtx(
					 photonX_- primaryVertex_->x(),
					 photonY_-primaryVertex_->y(),
					 photonZ_-primaryVertex_->z()
					 );
    math::XYZVector photonDirection=photonPositionwrtVtx.Unit();
    
    math::XYZTLorentzVector photonMomentum(photonEnergy_* photonDirection.X(),
					   photonEnergy_* photonDirection.Y(),
					   photonEnergy_* photonDirection.Z(),
					   photonEnergy_           );

    if(sum_track_pt>(sumPtTrackIsoForPhoton_ + sumPtTrackIsoSlopeForPhoton_ * photonMomentum.pt()) && AddFromElectron_.size()==0)
      {
	elemsToLock.resize(0);
	continue;
	
      }

	//THIS SC is not a Photon it fails track Isolation
    //if(sum_track_pt>(2+ 0.001* photonMomentum.pt()))
    //continue;//THIS SC is not a Photon it fails track Isolation

    /*
    std::cout<<" Created Photon with energy = "<<photonEnergy_<<std::endl;
    std::cout<<"                         pT = "<<photonMomentum.pt()<<std::endl;
    std::cout<<"                     RawEne = "<<RawEcalEne<<std::endl;
    std::cout<<"                          E = "<<photonMomentum.e()<<std::endl;
    std::cout<<"                        eta = "<<photonMomentum.eta()<<std::endl;
    std::cout<<"             TrackIsolation = "<< sum_track_pt <<std::endl;
    */

    reco::PFCandidate photonCand(0,photonMomentum, reco::PFCandidate::gamma);
    photonCand.setPs1Energy(ps1TotEne);
    photonCand.setPs2Energy(ps2TotEne);
    photonCand.setEcalEnergy(RawEcalEne,photonEnergy_);
    photonCand.setHcalEnergy(0.,0.);
    photonCand.set_mva_nothing_gamma(1.);  
    photonCand.setSuperClusterRef(sc->superClusterRef());
    math::XYZPoint v(primaryVertex_->x(), primaryVertex_->y(), primaryVertex_->z());
    photonCand.setVertex( v  );
    if(hasConvTrack || hasSingleleg)photonCand.setFlag( reco::PFCandidate::GAMMA_TO_GAMMACONV, true);
    int matches=match_ind.size();
    int count=0;
    for ( std::vector<reco::PFCandidate>::const_iterator ec=tempElectronCandidates.begin();   ec != tempElectronCandidates.end(); ++ec ){
      for(int i=0; i<matches; i++)
	{
	  if(count==match_ind[i])photonCand.addDaughter(*ec);
	  count++;
	}
    }
    // set isvalid_ to TRUE since we've found at least one photon candidate
    isvalid_ = true;
    // push back the candidate into the collection ...
    //Add Elements from Electron
    for(std::vector<unsigned int>::const_iterator it = 
	  AddFromElectron_.begin();
	it != AddFromElectron_.end(); ++it)photonCand.addElementInBlock(blockRef,*it);
    
    // ... and lock all elemts used
    for(std::vector<unsigned int>::const_iterator it = elemsToLock.begin();
	it != elemsToLock.end(); ++it)
      {
	if(active[*it])
	  {
	    photonCand.addElementInBlock(blockRef,*it);
	    if( elements[*it].type() == reco::PFBlockElement::TRACK  )
	      {
		if(elements[*it].convRef().isNonnull())
		  {
		    //make sure it is not stored already as the partner track
		    bool matched=false;
		    for(unsigned int ic = 0; ic < ConversionsRef_.size(); ic++)
		      {
			if(ConversionsRef_[ic]==elements[*it].convRef())matched=true;
		      }
		    if(!matched)ConversionsRef_.push_back(elements[*it].convRef());
		  }
	      }
	  }
	active[*it] = false;	
      }
    PFPhoECorr_=0;
    // here add the extra information
    PFCandidatePhotonExtra myExtra(sc->superClusterRef());
    //Store Locally Contained PF Cluster regressed energy
    for(unsigned int l=0; l<MVALCorr.size(); ++l)
      {
	myExtra.addLCorrClusEnergy(MVALCorr[l]);
	PFPhoECorr_=PFPhoECorr_+MVALCorr[l];//total Locally corrected energy
      }
    TotPS1_=ps1TotEne;
    TotPS2_=ps2TotEne;
    //Do Global Corrections here:
    float GCorr=EvaluateGCorrMVA(photonCand, PFClusters);
    if(useReg_){
      math::XYZTLorentzVector photonCorrMomentum(GCorr*PFPhoECorr_* photonDirection.X(),
						 GCorr*PFPhoECorr_* photonDirection.Y(),
						 GCorr*PFPhoECorr_* photonDirection.Z(),
						 GCorr * photonEnergy_           );
      photonCand.setP4(photonCorrMomentum);
    }
    
    std::multimap<float, unsigned int>OrderedClust;
    for(unsigned int i=0; i<PFClusters.size(); ++i){  
      float et=PFClusters[i].energy()*sin(PFClusters[i].position().theta());
      OrderedClust.insert(make_pair(et, i));
    }
    std::multimap<float, unsigned int>::reverse_iterator rit;
    rit=OrderedClust.rbegin();
    unsigned int highEindex=(*rit).second;
    //store Position at ECAL Entrance as Position of Max Et PFCluster
    photonCand.setPositionAtECALEntrance(math::XYZPointF(PFClusters[highEindex].position()));
    
    //Mustache ID variables
    Mustache Must;
    Must.FillMustacheVar(PFClusters);
    int excluded= Must.OutsideMust();
    float MustacheEt=Must.MustacheEtOut();
    myExtra.setMustache_Et(MustacheEt);
    myExtra.setExcludedClust(excluded);
    if(fabs(photonCand.eta()<1.4446))
      myExtra.setMVAGlobalCorrE(GCorr * PFPhoECorr_);
    else if(PFPhoR9_>0.94)
      myExtra.setMVAGlobalCorrE(GCorr * PFPhoECorr_);
    else myExtra.setMVAGlobalCorrE(GCorr * photonEnergy_);
    float Res=EvaluateResMVA(photonCand, PFClusters);
    myExtra.SetPFPhotonRes(Res);
    
    //    Daniele example for mvaValues
    //    do the same for single leg trackRef and convRef
    for(unsigned int ic = 0; ic < MVA_values.size(); ic++)
      {
	myExtra.addSingleLegConvMva(MVA_values[ic]);
	myExtra.addSingleLegConvTrackRef(singleLegRef[ic]);
	//cout<<"Single Leg Tracks "<<singleLegRef[ic]->pt()<<" MVA "<<MVA_values[ic]<<endl;
      }
    for(unsigned int ic = 0; ic < ConversionsRef_.size(); ic++)
      {
	myExtra.addConversionRef(ConversionsRef_[ic]);
	//cout<<"Conversion Pairs "<<ConversionsRef_[ic]->pairMomentum()<<endl;
      }
    pfPhotonExtraCandidates.push_back(myExtra);
    pfCandidates->push_back(photonCand);
    // ... and reset the vector
    elemsToLock.resize(0);
    hasConvTrack=false;
    hasSingleleg=false;
  } // end of loops over all elements in block
  
  return;
}

float PFPhotonAlgo::EvaluateResMVA(reco::PFCandidate photon, std::vector<reco::CaloCluster>PFClusters){
  float BDTG=1;
  PFPhoEta_=photon.eta();
  PFPhoPhi_=photon.phi();
  PFPhoE_=photon.energy();
  //fill Material Map:
  int ix = X0_sum->GetXaxis()->FindBin(PFPhoEta_);
  int iy = X0_sum->GetYaxis()->FindBin(PFPhoPhi_);
  x0inner_= X0_inner->GetBinContent(ix,iy);
  x0middle_=X0_middle->GetBinContent(ix,iy);
  x0outer_=X0_outer->GetBinContent(ix,iy);
  SCPhiWidth_=photon.superClusterRef()->phiWidth();
  SCEtaWidth_=photon.superClusterRef()->etaWidth();
  Mustache Must;
  std::vector<unsigned int>insideMust;
  std::vector<unsigned int>outsideMust;
  std::multimap<float, unsigned int>OrderedClust;
  Must.FillMustacheVar(PFClusters);
  MustE_=Must.MustacheE();
  LowClusE_=Must.LowestMustClust();
  PFPhoR9Corr_=E3x3_/MustE_;
  Must.MustacheClust(PFClusters,insideMust, outsideMust );
  for(unsigned int i=0; i<insideMust.size(); ++i){
    int index=insideMust[i];
    OrderedClust.insert(make_pair(PFClusters[index].energy(),index));
  }
  std::multimap<float, unsigned int>::iterator it;
  it=OrderedClust.begin();
  unsigned int lowEindex=(*it).second;
  std::multimap<float, unsigned int>::reverse_iterator rit;
  rit=OrderedClust.rbegin();
  unsigned int highEindex=(*rit).second;
  if(insideMust.size()>1){
    dEta_=fabs(PFClusters[highEindex].eta()-PFClusters[lowEindex].eta());
    dPhi_=asin(PFClusters[highEindex].phi()-PFClusters[lowEindex].phi());
  }
  else{
    dEta_=0;
    dPhi_=0;
    LowClusE_=0;
  }
  //calculate RMS for All clusters and up until the Next to Lowest inside the Mustache
  RMSAll_=ClustersPhiRMS(PFClusters, PFPhoPhi_);
  std::vector<reco::CaloCluster>PFMustClusters;
  if(insideMust.size()>2){
    for(unsigned int i=0; i<insideMust.size(); ++i){
      unsigned int index=insideMust[i];
      if(index==lowEindex)continue;
      PFMustClusters.push_back(PFClusters[index]);
    }
  }
  else{
    for(unsigned int i=0; i<insideMust.size(); ++i){
      unsigned int index=insideMust[i];
      PFMustClusters.push_back(PFClusters[index]);
    }    
  }
  RMSMust_=ClustersPhiRMS(PFMustClusters, PFPhoPhi_);
  //then use cluster Width for just one PFCluster
  RConv_=310;
  PFCandidate::ElementsInBlocks eleInBlocks = photon.elementsInBlocks();
  for(unsigned i=0; i<eleInBlocks.size(); i++)
    {
      PFBlockRef blockRef = eleInBlocks[i].first;
      unsigned indexInBlock = eleInBlocks[i].second;
      const edm::OwnVector< reco::PFBlockElement >&  elements=eleInBlocks[i].first->elements();
      const reco::PFBlockElement& element = elements[indexInBlock];
      if(element.type()==reco::PFBlockElement::TRACK){
	float R=sqrt(element.trackRef()->innerPosition().X()*element.trackRef()->innerPosition().X()+element.trackRef()->innerPosition().Y()*element.trackRef()->innerPosition().Y());
	if(RConv_>R)RConv_=R;
      }
      else continue;
    }
  float GC_Var[17];
  GC_Var[0]=PFPhoEta_;
  GC_Var[1]=PFPhoEt_;
  GC_Var[2]=PFPhoR9Corr_;
  GC_Var[3]=PFPhoPhi_;
  GC_Var[4]=SCEtaWidth_;
  GC_Var[5]=SCPhiWidth_;
  GC_Var[6]=x0inner_;  
  GC_Var[7]=x0middle_;
  GC_Var[8]=x0outer_;
  GC_Var[9]=RConv_;
  GC_Var[10]=LowClusE_;
  GC_Var[11]=RMSMust_;
  GC_Var[12]=RMSAll_;
  GC_Var[13]=dEta_;
  GC_Var[14]=dPhi_;
  GC_Var[15]=nVtx_;
  GC_Var[16]=MustE_;
  
  BDTG=ReaderRes_->GetResponse(GC_Var);
  //  cout<<"Res "<<BDTG<<endl;
  
  //  cout<<"BDTG Parameters X0"<<x0inner_<<", "<<x0middle_<<", "<<x0outer_<<endl;
  //  cout<<"Et, Eta, Phi "<<PFPhoEt_<<", "<<PFPhoEta_<<", "<<PFPhoPhi_<<endl;
  // cout<<"PFPhoR9 "<<PFPhoR9_<<endl;
  // cout<<"R "<<RConv_<<endl;
  
  return BDTG;
   
}

float PFPhotonAlgo::EvaluateGCorrMVA(reco::PFCandidate photon, std::vector<CaloCluster>PFClusters){
  float BDTG=1;
  PFPhoEta_=photon.eta();
  PFPhoPhi_=photon.phi();
  PFPhoE_=photon.energy();
    //fill Material Map:
  int ix = X0_sum->GetXaxis()->FindBin(PFPhoEta_);
  int iy = X0_sum->GetYaxis()->FindBin(PFPhoPhi_);
  x0inner_= X0_inner->GetBinContent(ix,iy);
  x0middle_=X0_middle->GetBinContent(ix,iy);
  x0outer_=X0_outer->GetBinContent(ix,iy);
  SCPhiWidth_=photon.superClusterRef()->phiWidth();
  SCEtaWidth_=photon.superClusterRef()->etaWidth();
  Mustache Must;
  std::vector<unsigned int>insideMust;
  std::vector<unsigned int>outsideMust;
  std::multimap<float, unsigned int>OrderedClust;
  Must.FillMustacheVar(PFClusters);
  MustE_=Must.MustacheE();
  LowClusE_=Must.LowestMustClust();
  PFPhoR9Corr_=E3x3_/MustE_;
  Must.MustacheClust(PFClusters,insideMust, outsideMust );
  for(unsigned int i=0; i<insideMust.size(); ++i){
    int index=insideMust[i];
    OrderedClust.insert(make_pair(PFClusters[index].energy(),index));
  }
  std::multimap<float, unsigned int>::iterator it;
  it=OrderedClust.begin();
  unsigned int lowEindex=(*it).second;
  std::multimap<float, unsigned int>::reverse_iterator rit;
  rit=OrderedClust.rbegin();
  unsigned int highEindex=(*rit).second;
  if(insideMust.size()>1){
    dEta_=fabs(PFClusters[highEindex].eta()-PFClusters[lowEindex].eta());
    dPhi_=asin(PFClusters[highEindex].phi()-PFClusters[lowEindex].phi());
  }
  else{
    dEta_=0;
    dPhi_=0;
    LowClusE_=0;
  }
  //calculate RMS for All clusters and up until the Next to Lowest inside the Mustache
  RMSAll_=ClustersPhiRMS(PFClusters, PFPhoPhi_);
  std::vector<reco::CaloCluster>PFMustClusters;
  if(insideMust.size()>2){
    for(unsigned int i=0; i<insideMust.size(); ++i){
      unsigned int index=insideMust[i];
      if(index==lowEindex)continue;
      PFMustClusters.push_back(PFClusters[index]);
    }
  }
  else{
    for(unsigned int i=0; i<insideMust.size(); ++i){
      unsigned int index=insideMust[i];
      PFMustClusters.push_back(PFClusters[index]);
    }    
  }
  RMSMust_=ClustersPhiRMS(PFMustClusters, PFPhoPhi_);
  //then use cluster Width for just one PFCluster
  RConv_=310;
  PFCandidate::ElementsInBlocks eleInBlocks = photon.elementsInBlocks();
  for(unsigned i=0; i<eleInBlocks.size(); i++)
    {
      PFBlockRef blockRef = eleInBlocks[i].first;
      unsigned indexInBlock = eleInBlocks[i].second;
      const edm::OwnVector< reco::PFBlockElement >&  elements=eleInBlocks[i].first->elements();
      const reco::PFBlockElement& element = elements[indexInBlock];
      if(element.type()==reco::PFBlockElement::TRACK){
	float R=sqrt(element.trackRef()->innerPosition().X()*element.trackRef()->innerPosition().X()+element.trackRef()->innerPosition().Y()*element.trackRef()->innerPosition().Y());
	if(RConv_>R)RConv_=R;
      }
      else continue;
    }
  //cout<<"Nvtx "<<nVtx_<<endl;
  if(fabs(PFPhoEta_)<1.4446){
    float GC_Var[17];
    GC_Var[0]=PFPhoEta_;
    GC_Var[1]=PFPhoECorr_;
    GC_Var[2]=PFPhoR9Corr_;
    GC_Var[3]=SCEtaWidth_;
    GC_Var[4]=SCPhiWidth_;
    GC_Var[5]=PFPhoPhi_;
    GC_Var[6]=x0inner_;
    GC_Var[7]=x0middle_;
    GC_Var[8]=x0outer_;
    GC_Var[9]=RConv_;
    GC_Var[10]=LowClusE_;
    GC_Var[11]=RMSMust_;
    GC_Var[12]=RMSAll_;
    GC_Var[13]=dEta_;
    GC_Var[14]=dPhi_;
    GC_Var[15]=nVtx_;
    GC_Var[16]=MustE_;
    BDTG=ReaderGCEB_->GetResponse(GC_Var);
  }
  else if(PFPhoR9_>0.94){
    float GC_Var[19];
    GC_Var[0]=PFPhoEta_;
    GC_Var[1]=PFPhoECorr_;
    GC_Var[2]=PFPhoR9Corr_;
    GC_Var[3]=SCEtaWidth_;
    GC_Var[4]=SCPhiWidth_;
    GC_Var[5]=PFPhoPhi_;
    GC_Var[6]=x0inner_;
    GC_Var[7]=x0middle_;
    GC_Var[8]=x0outer_;
    GC_Var[9]=RConv_;
    GC_Var[10]=LowClusE_;
    GC_Var[11]=RMSMust_;
    GC_Var[12]=RMSAll_;
    GC_Var[13]=dEta_;
    GC_Var[14]=dPhi_;
    GC_Var[15]=nVtx_;
    GC_Var[16]=TotPS1_;
    GC_Var[17]=TotPS2_;
    GC_Var[18]=MustE_;
    BDTG=ReaderGCEEhR9_->GetResponse(GC_Var);
  }
  
  else{
    float GC_Var[19];
    GC_Var[0]=PFPhoEta_;
    GC_Var[1]=PFPhoE_;
    GC_Var[2]=PFPhoR9Corr_;
    GC_Var[3]=SCEtaWidth_;
    GC_Var[4]=SCPhiWidth_;
    GC_Var[5]=PFPhoPhi_;
    GC_Var[6]=x0inner_;
    GC_Var[7]=x0middle_;
    GC_Var[8]=x0outer_;
    GC_Var[9]=RConv_;
    GC_Var[10]=LowClusE_;
    GC_Var[11]=RMSMust_;
    GC_Var[12]=RMSAll_;
    GC_Var[13]=dEta_;
    GC_Var[14]=dPhi_;
    GC_Var[15]=nVtx_;
    GC_Var[16]=TotPS1_;
    GC_Var[17]=TotPS2_;
    GC_Var[18]=MustE_;
    BDTG=ReaderGCEElR9_->GetResponse(GC_Var);
  }
  //cout<<"GC "<<BDTG<<endl;

  return BDTG;
  
}

double PFPhotonAlgo::ClustersPhiRMS(std::vector<reco::CaloCluster>PFClusters, float PFPhoPhi){
  double PFClustPhiRMS=0;
  double delPhi2=0;
  double delPhiSum=0;
  double ClusSum=0;
  for(unsigned int c=0; c<PFClusters.size(); ++c){
    delPhi2=(acos(cos(PFPhoPhi-PFClusters[c].phi()))* acos(cos(PFPhoPhi-PFClusters[c].phi())) )+delPhi2;
    delPhiSum=delPhiSum+ acos(cos(PFPhoPhi-PFClusters[c].phi()))*PFClusters[c].energy();
    ClusSum=ClusSum+PFClusters[c].energy();
  }
  double meandPhi=delPhiSum/ClusSum;
  PFClustPhiRMS=sqrt(fabs(delPhi2/ClusSum - (meandPhi*meandPhi)));
  
  return PFClustPhiRMS;
}

float PFPhotonAlgo::EvaluateLCorrMVA(reco::PFClusterRef clusterRef ){
  float BDTG=1;
  PFPhotonClusters ClusterVar(clusterRef);
  std::pair<double, double>ClusCoor=ClusterVar.GetCrysCoor();
  std::pair<int, int>ClusIndex=ClusterVar.GetCrysIndex();
  //Local Coordinates:
  if(clusterRef->layer()==PFLayer:: ECAL_BARREL ){//is Barrel
    PFCrysEtaCrack_=ClusterVar.EtaCrack();
    CrysEta_=ClusCoor.first;
    CrysPhi_=ClusCoor.second;
    CrysIEta_=ClusIndex.first;
    CrysIPhi_=ClusIndex.second;
  }
  else{
    CrysX_=ClusCoor.first;
    CrysY_=ClusCoor.second;
  }
  //Shower Shape Variables:
  eSeed_= ClusterVar.E5x5Element(0, 0)/clusterRef->energy();
  etop_=ClusterVar.E5x5Element(0,1)/clusterRef->energy();
  ebottom_=ClusterVar.E5x5Element(0,-1)/clusterRef->energy();
  eleft_=ClusterVar.E5x5Element(-1,0)/clusterRef->energy();
  eright_=ClusterVar.E5x5Element(1,0)/clusterRef->energy();
  e1x3_=(ClusterVar.E5x5Element(0,0)+ClusterVar.E5x5Element(0,1)+ClusterVar.E5x5Element(0,-1))/clusterRef->energy();
  e3x1_=(ClusterVar.E5x5Element(0,0)+ClusterVar.E5x5Element(-1,0)+ClusterVar.E5x5Element(1,0))/clusterRef->energy();
  e1x5_=ClusterVar.E5x5Element(0,0)+ClusterVar.E5x5Element(0,-2)+ClusterVar.E5x5Element(0,-1)+ClusterVar.E5x5Element(0,1)+ClusterVar.E5x5Element(0,2);
  
  e2x5Top_=(ClusterVar.E5x5Element(-2,2)+ClusterVar.E5x5Element(-1, 2)+ClusterVar.E5x5Element(0, 2)
	    +ClusterVar.E5x5Element(1, 2)+ClusterVar.E5x5Element(2, 2)
	    +ClusterVar.E5x5Element(-2,1)+ClusterVar.E5x5Element(-1,1)+ClusterVar.E5x5Element(0,1)
	    +ClusterVar.E5x5Element(1,1)+ClusterVar.E5x5Element(2,1))/clusterRef->energy();
  e2x5Bottom_=(ClusterVar.E5x5Element(-2,-2)+ClusterVar.E5x5Element(-1,-2)+ClusterVar.E5x5Element(0,-2)
	       +ClusterVar.E5x5Element(1,-2)+ClusterVar.E5x5Element(2,-2)
	       +ClusterVar.E5x5Element(-2,1)+ClusterVar.E5x5Element(-1,1)
	       +ClusterVar.E5x5Element(0,1)+ClusterVar.E5x5Element(1,1)+ClusterVar.E5x5Element(2,1))/clusterRef->energy();
  e2x5Left_= (ClusterVar.E5x5Element(-2,-2)+ClusterVar.E5x5Element(-2,-1)
	      +ClusterVar.E5x5Element(-2,0)
	       +ClusterVar.E5x5Element(-2,1)+ClusterVar.E5x5Element(-2,2)
	      +ClusterVar.E5x5Element(-1,-2)+ClusterVar.E5x5Element(-1,-1)+ClusterVar.E5x5Element(-1,0)
	      +ClusterVar.E5x5Element(-1,1)+ClusterVar.E5x5Element(-1,2))/clusterRef->energy();
  
  e2x5Right_ =(ClusterVar.E5x5Element(2,-2)+ClusterVar.E5x5Element(2,-1)
	       +ClusterVar.E5x5Element(2,0)+ClusterVar.E5x5Element(2,1)+ClusterVar.E5x5Element(2,2)
	       +ClusterVar.E5x5Element(1,-2)+ClusterVar.E5x5Element(1,-1)+ClusterVar.E5x5Element(1,0)
	       +ClusterVar.E5x5Element(1,1)+ClusterVar.E5x5Element(1,2))/clusterRef->energy();
  float centerstrip=ClusterVar.E5x5Element(0,0)+ClusterVar.E5x5Element(0, -2)
    +ClusterVar.E5x5Element(0,-1)+ClusterVar.E5x5Element(0,1)+ClusterVar.E5x5Element(0,2);
  float rightstrip=ClusterVar.E5x5Element(1, 0)+ClusterVar.E5x5Element(1,1)
    +ClusterVar.E5x5Element(1,2)+ClusterVar.E5x5Element(1,-1)+ClusterVar.E5x5Element(1,-2);
  float leftstrip=ClusterVar.E5x5Element(-1,0)+ClusterVar.E5x5Element(-1,-1)+ClusterVar.E5x5Element(-1,2)
    +ClusterVar.E5x5Element(-1,1)+ClusterVar.E5x5Element(-1,2);
  
  if(rightstrip>leftstrip)e2x5Max_=rightstrip+centerstrip;
  else e2x5Max_=leftstrip+centerstrip;
  e2x5Max_=e2x5Max_/clusterRef->energy();
  //GetCrysCoordinates(clusterRef);
  //fill5x5Map(clusterRef);
  VtxZ_=primaryVertex_->z();
  ClusPhi_=clusterRef->position().phi(); 
  ClusEta_=fabs(clusterRef->position().eta());
  EB=fabs(clusterRef->position().eta())/clusterRef->position().eta();
  logPFClusE_=log(clusterRef->energy());
  if(ClusEta_<1.4446){
    float LC_Var[26];
    LC_Var[0]=VtxZ_;
    LC_Var[1]=EB;
    LC_Var[2]=ClusEta_;
    LC_Var[3]=ClusPhi_;
    LC_Var[4]=logPFClusE_;
    LC_Var[5]=eSeed_;
    //top bottom left right
    LC_Var[6]=etop_;
    LC_Var[7]=ebottom_;
    LC_Var[8]=eleft_;
    LC_Var[9]=eright_;
    LC_Var[10]=ClusR9_;
    LC_Var[11]=e1x3_;
    LC_Var[12]=e3x1_;
    LC_Var[13]=Clus5x5ratio_;
    LC_Var[14]=e1x5_;
    LC_Var[15]=e2x5Max_;
    LC_Var[16]=e2x5Top_;
    LC_Var[17]=e2x5Bottom_;
    LC_Var[18]=e2x5Left_;
    LC_Var[19]=e2x5Right_;
    LC_Var[20]=CrysEta_;
    LC_Var[21]=CrysPhi_;
    float CrysIphiMod2=CrysIPhi_%2;
    float CrysIetaMod5=CrysIEta_%5;
    float CrysIphiMod20=CrysIPhi_%20;
    LC_Var[22]=CrysIphiMod2;
    LC_Var[23]=CrysIetaMod5;
    LC_Var[24]=CrysIphiMod20;   
    LC_Var[25]=PFCrysEtaCrack_;
    BDTG=ReaderLCEB_->GetResponse(LC_Var);   
    //cout<<"LC "<<BDTG<<endl;  
  }
  else{
    float LC_Var[22];
    LC_Var[0]=VtxZ_;
    LC_Var[1]=EB;
    LC_Var[2]=ClusEta_;
    LC_Var[3]=ClusPhi_;
    LC_Var[4]=logPFClusE_;
    LC_Var[5]=eSeed_;
    //top bottom left right
    LC_Var[6]=etop_;
    LC_Var[7]=ebottom_;
    LC_Var[8]=eleft_;
    LC_Var[9]=eright_;
    LC_Var[10]=ClusR9_;
    LC_Var[11]=e1x3_;
    LC_Var[12]=e3x1_;
    LC_Var[13]=Clus5x5ratio_;
    LC_Var[14]=e1x5_;
    LC_Var[15]=e2x5Max_;
    LC_Var[16]=e2x5Top_;
    LC_Var[17]=e2x5Bottom_;
    LC_Var[18]=e2x5Left_;
    LC_Var[19]=e2x5Right_;
    LC_Var[20]=CrysX_;
    LC_Var[21]=CrysY_;
    BDTG=ReaderLCEE_->GetResponse(LC_Var);   
    //cout<<"LC "<<BDTG<<endl;  
  }
   return BDTG;
  
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

//Recover Early Conversions reconstructed as PFelectrons
void PFPhotonAlgo::EarlyConversion(    
				   //std::auto_ptr< reco::PFCandidateCollection > 
				   //&pfElectronCandidates_,
				   std::vector<reco::PFCandidate>& 
				   tempElectronCandidates,
				   const reco::PFBlockElementSuperCluster* sc
				   ){
  //step 1 check temp electrons for clusters that match Photon Supercluster:
  // permElectronCandidates->clear();
  int count=0;
  for ( std::vector<reco::PFCandidate>::const_iterator ec=tempElectronCandidates.begin();   ec != tempElectronCandidates.end(); ++ec ) 
    {
      //      bool matched=false;
      int mh=ec->gsfTrackRef()->trackerExpectedHitsInner().numberOfLostHits();
      //if(mh==0)continue;//Case where missing hits greater than zero
      
      reco::GsfTrackRef gsf=ec->gsfTrackRef();
      //some hoopla to get Electron SC ref
      
      if(gsf->extra().isAvailable() && gsf->extra()->seedRef().isAvailable() && mh>0) 
	{
	  reco::ElectronSeedRef seedRef=  gsf->extra()->seedRef().castTo<reco::ElectronSeedRef>();
	  if(seedRef.isAvailable() && seedRef->isEcalDriven()) 
	    {
	      reco::SuperClusterRef ElecscRef = seedRef->caloCluster().castTo<reco::SuperClusterRef>();
	      
	      if(ElecscRef.isNonnull()){
		//finally see if it matches:
		reco::SuperClusterRef PhotscRef=sc->superClusterRef();
		if(PhotscRef==ElecscRef)
		  {
		    match_ind.push_back(count);
		    //  matched=true; 
		    //cout<<"Matched Electron with Index "<<count<<" This is the electron "<<*ec<<endl;
		    //find that they have the same SC footprint start to collect Clusters and tracks and these will be passed to PFPhoton
		    reco::PFCandidate::ElementsInBlocks eleInBlocks = ec->elementsInBlocks();
		    for(unsigned i=0; i<eleInBlocks.size(); i++) 
		      {
			reco::PFBlockRef blockRef = eleInBlocks[i].first;
			unsigned indexInBlock = eleInBlocks[i].second;	 
			//const edm::OwnVector< reco::PFBlockElement >&  elements=eleInBlocks[i].first->elements();
			//const reco::PFBlockElement& element = elements[indexInBlock];  		
			
			AddFromElectron_.push_back(indexInBlock);	       	
		      }		    
		  }		
	      }
	    }	  
	}           
      count++;
    }
}
