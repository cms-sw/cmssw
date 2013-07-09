//
// Original Authors: Fabian Stoeckli: fabian.stoeckli@cern.ch
//                   Nicholas Wardle: nckw@cern.ch
//                   Rishi Patel rpatel@cern.ch(ongoing developer and maintainer)
//

#include "RecoParticleFlow/PFProducer/interface/PFEGammaAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h" 
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
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFSCEnergyCalibration.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyResolution.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFClusterWidthAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFElectronExtraEqual.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include <TFile.h>
#include <iomanip>
#include <algorithm>
#include <TMath.h>
using namespace std;
using namespace reco;


PFEGammaAlgo::PFEGammaAlgo(const double mvaEleCut,
			   std::string  mvaWeightFileEleID,
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
			   double coneTrackIsoForEgammaSC,
			   std::string mvaweightfile,  
			   double mvaConvCut, 
			   bool useReg, 
			   std::string X0_Map,
			   const reco::Vertex& primary,
			   double sumPtTrackIsoForPhoton,
			   double sumPtTrackIsoSlopeForPhoton
			   ) : 
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
  coneTrackIsoForEgammaSC_(coneTrackIsoForEgammaSC),  
  isvalid_(false), 
  verbosityLevel_(Silent), 
  MVACUT(mvaConvCut),
  useReg_(useReg),
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

  
  // Set the tmva reader for electrons
  tmvaReaderEle_ = new TMVA::Reader("!Color:Silent");
  tmvaReaderEle_->AddVariable("lnPt_gsf",&lnPt_gsf);
  tmvaReaderEle_->AddVariable("Eta_gsf",&Eta_gsf);
  tmvaReaderEle_->AddVariable("dPtOverPt_gsf",&dPtOverPt_gsf);
  tmvaReaderEle_->AddVariable("DPtOverPt_gsf",&DPtOverPt_gsf);
  //tmvaReaderEle_->AddVariable("nhit_gsf",&nhit_gsf);
  tmvaReaderEle_->AddVariable("chi2_gsf",&chi2_gsf);
  //tmvaReaderEle_->AddVariable("DPtOverPt_kf",&DPtOverPt_kf);
  tmvaReaderEle_->AddVariable("nhit_kf",&nhit_kf);
  tmvaReaderEle_->AddVariable("chi2_kf",&chi2_kf);
  tmvaReaderEle_->AddVariable("EtotPinMode",&EtotPinMode);
  tmvaReaderEle_->AddVariable("EGsfPoutMode",&EGsfPoutMode);
  tmvaReaderEle_->AddVariable("EtotBremPinPoutMode",&EtotBremPinPoutMode);
  tmvaReaderEle_->AddVariable("DEtaGsfEcalClust",&DEtaGsfEcalClust);
  tmvaReaderEle_->AddVariable("SigmaEtaEta",&SigmaEtaEta);
  tmvaReaderEle_->AddVariable("HOverHE",&HOverHE);
//   tmvaReaderEle_->AddVariable("HOverPin",&HOverPin);
  tmvaReaderEle_->AddVariable("lateBrem",&lateBrem);
  tmvaReaderEle_->AddVariable("firstBrem",&firstBrem);
  tmvaReaderEle_->BookMVA("BDT",mvaWeightFileEleID.c_str());
  
  
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

void PFEGammaAlgo::RunPFEG(const reco::PFBlockRef&  blockRef,
			       std::vector<bool>& active
){
  
  // should be cleaned as often as often as possible
//   elCandidate_.clear();
//   electronExtra_.clear();
//   allElCandidate_.clear();
  //electronConstituents_.clear();
  fifthStepKfTrack_.clear();
  convGsfTrack_.clear();
  
  egCandidate_.clear();
  egExtra_.clear();
  
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
  
  //build large set of association maps between gsf/kf tracks, PFClusters and SuperClusters
  //this function originally came from the PFElectronAlgo
  // the maps are initialized 
  AssMap associatedToGsf;
  AssMap associatedToBrems;
  AssMap associatedToEcal;  
  
  //bool blockHasGSF =  
  SetLinks(blockRef,associatedToGsf,
			    associatedToBrems,associatedToEcal,
			    active, *primaryVertex_);
  
  //printf("blockHasGsf = %i\n",int(blockHasGSF));
  
  
  // local vecotr to keep track of the indices of the 'elements' for the Photon candidate
  // once we decide to keep the candidate, the 'active' entriesd for them must be set to false
  std::vector<unsigned int> elemsToLock;
  elemsToLock.resize(0);
  
  for( ; ele != elements.end(); ++ele, ++actIter ) {

    // if it's not a SuperCluster, go to the next element
    if( !( ele->type() == reco::PFBlockElement::SC ) ) continue;
    
    //printf("supercluster\n");
    
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
    //if (!(sc->fromPhoton()))continue;
    
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
    //PFPhoR9_=sc->photonRef()->r9();
    PFPhoR9_=1.0;
    E3x3_=PFPhoR9_*(sc->superClusterRef()->rawEnergy());
    // loop over the ECAL clusters linked to the iEle 
    if( ! ecalAssoPFClusters.size() ) {
      // This SC element has NO ECAL elements asigned... *SHOULD NOT HAPPEN*
      //std::cout<<" Found SC element with no ECAL assigned "<<std::endl;
      continue;
    }
    
    //list of matched gsf tracks associated to this supercluster through
    //a shared PFCluster
    //std::set<unsigned int> matchedGsf;
    std::vector<unsigned int> matchedGsf;
    for (map<unsigned int,vector<unsigned int> >::iterator igsf = associatedToGsf.begin();
       igsf != associatedToGsf.end(); igsf++) {    
      
      bool matched = false;
      if( !( active[igsf->first] ) ) continue;
    
      vector<unsigned int> assogsf_index = igsf->second;
      for  (unsigned int ielegsf=0;ielegsf<assogsf_index.size();ielegsf++) {
	unsigned int associdx = assogsf_index[ielegsf];
	
	if( !( active[associdx] ) ) continue;
	
	PFBlockElement::Type assoele_type = elements[associdx].type();
	
	if(assoele_type == reco::PFBlockElement::ECAL) {
	  for(std::multimap<double, unsigned int>::iterator itecal = ecalAssoPFClusters.begin(); 
	    itecal != ecalAssoPFClusters.end(); ++itecal) {
	    
	    if (itecal->second==associdx) {
	      matchedGsf.push_back(igsf->first);
	      matched = true;
	      break;
	    }
	  }
	}
	  
	if (matched) break;
      }
      
    }
    
   //printf("matchedGsf size = %i\n",int(matchedGsf.size()));

      
    // This is basically CASE 2
    // .... we loop over all ECAL cluster linked to each other by this SC
    for(std::multimap<double, unsigned int>::iterator itecal = ecalAssoPFClusters.begin(); 
	itecal != ecalAssoPFClusters.end(); ++itecal) { 
      
      //printf("ecal cluster\n");
      
      //loop over associated elements to check for gsf
      
//       std::map<unsigned int, std::vector<unsigned int> >::const_iterator assoc_ecal_it = associatedToEcal.find(itecal->second);
//       if (assoc_ecal_it!=associatedToEcal.end()) {
// 	const std::vector<unsigned int> &assoecal_index = assoc_ecal_it->second;
// 	for  (unsigned int iecalassoc=0;iecalassoc<assoecal_index.size();iecalassoc++) {
// 	  int associdx = assoecal_index[iecalassoc];
// 	  PFBlockElement::Type assoele_type = elements[associdx].type();
// 	  // lock the elements associated to the gsf: ECAL, Brems
// 	  //active[(assogsf_index[ielegsf])] = false;  
// 	  printf("matched element to ecal cluster, type = %i\n",int(assoele_type));
// 	  
// 	  if (assoele_type == reco::PFBlockElement::GSF) {      
// 	    printf("matched gsf\n");
// 	    if (!matchedGsf.count(associdx)) {
// 	      matchedGsf.insert(associdx);
// 	    }
// 	  }
// 	}
//       }
      

      
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
	  std::multimap<double, unsigned int> convEcalAll;
	  blockRef->associatedElements( track2->second,
					linkData,
					convEcalAll,
					reco::PFBlockElement::ECAL,
					reco::PFBlock::LINKTEST_ALL);
	  
	  //create cleaned collection of associated ecal clusters restricted to subdetector of the seeding supercluster
	  //This cleaning is needed since poorly reconstructed conversions can occasionally have the second track pointing
	  //to the wrong subdetector
	  std::multimap<double, unsigned int> convEcal;
	  for(std::multimap<double, unsigned int>::iterator itecal = convEcalAll.begin(); 
	      itecal != convEcalAll.end(); ++itecal) { 
		  
		  // to get the reference to the PF clusters, this is needed.
	    reco::PFClusterRef clusterRef = elements[itecal->second].clusterRef();
	    
	    if (clusterRef->hitsAndFractions().at(0).first.subdetId()==sc->superClusterRef()->seed()->hitsAndFractions().at(0).first.subdetId()) {
	      convEcal.insert(*itecal);
	    }
	  }
	  
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

    
    //add elements from electron candidates
    bool goodelectron = false;
    if (matchedGsf.size()>0) {
      //printf("making electron, candsize = %i\n",int(egCandidate_.size()));
      int eleidx = matchedGsf.front();
      AddElectronElements(eleidx, elemsToLock, blockRef, associatedToGsf, associatedToBrems, associatedToEcal);
      goodelectron = AddElectronCandidate(eleidx, sc->superClusterRef(), elemsToLock, blockRef, associatedToGsf, associatedToBrems, associatedToEcal, active);
      //printf("goodelectron = %i, candsize = %i\n",int(goodelectron),int(egCandidate_.size()));
    }    
    
    if (goodelectron) continue;
    
    //printf("making photon\n");
    
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

//     if(sum_track_pt>(sumPtTrackIsoForPhoton_ + sumPtTrackIsoSlopeForPhoton_ * photonMomentum.pt()) && AddFromElectron_.size()==0)
//       {
// 	elemsToLock.resize(0);
// 	continue;
// 	
//       }

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
//     int matches=match_ind.size();
//     int count=0;
//     for ( std::vector<reco::PFCandidate>::const_iterator ec=tempElectronCandidates.begin();   ec != tempElectronCandidates.end(); ++ec ){
//       for(int i=0; i<matches; i++)
// 	{
// 	  if(count==match_ind[i])photonCand.addDaughter(*ec);
// 	  count++;
// 	}
//     }
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
    //PFCandidateEGammaExtra myExtra(sc->superClusterRef());
    PFCandidateEGammaExtra myExtra;
    //myExtra.setSuperClusterRef(sc->superClusterRef());
    myExtra.setSuperClusterBoxRef(sc->superClusterRef());
    //myExtra.setClusterEnergies(MVALCorr);
    //Store Locally Contained PF Cluster regressed energy
    for(unsigned int l=0; l<MVALCorr.size(); ++l)
      {
	//myExtra.addLCorrClusEnergy(MVALCorr[l]);
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
//     Mustache Must;
//     Must.FillMustacheVar(PFClusters);
//     int excluded= Must.OutsideMust();
//     float MustacheEt=Must.MustacheEtOut();
    //myExtra.setMustache_Et(MustacheEt);
    //myExtra.setExcludedClust(excluded);
//     if(fabs(photonCand.eta()<1.4446))
//       myExtra.setMVAGlobalCorrE(GCorr * PFPhoECorr_);
//     else if(PFPhoR9_>0.94)
//       myExtra.setMVAGlobalCorrE(GCorr * PFPhoECorr_);
//     else myExtra.setMVAGlobalCorrE(GCorr * photonEnergy_);
//     float Res=EvaluateResMVA(photonCand, PFClusters);
//     myExtra.SetPFPhotonRes(Res);
    
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
    egExtra_.push_back(myExtra);
    egCandidate_.push_back(photonCand);
    // ... and reset the vector
    elemsToLock.resize(0);
    hasConvTrack=false;
    hasSingleleg=false;
  } // end of loops over all elements in block
  
  return;
}

float PFEGammaAlgo::EvaluateResMVA(reco::PFCandidate photon, std::vector<reco::CaloCluster>PFClusters){
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

float PFEGammaAlgo::EvaluateGCorrMVA(reco::PFCandidate photon, std::vector<CaloCluster>PFClusters){
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

double PFEGammaAlgo::ClustersPhiRMS(std::vector<reco::CaloCluster>PFClusters, float PFPhoPhi){
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

float PFEGammaAlgo::EvaluateLCorrMVA(reco::PFClusterRef clusterRef ){
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

bool PFEGammaAlgo::EvaluateSingleLegMVA(const reco::PFBlockRef& blockref, const reco::Vertex& primaryvtx, unsigned int track_index)  
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
void PFEGammaAlgo::EarlyConversion(    
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

bool PFEGammaAlgo::SetLinks(const reco::PFBlockRef&  blockRef,
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
  PFBlock::LinkData linkData =  block.linkData();  
  
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
	      
	      int nexhits = refKf->trackerExpectedHitsInner().numberOfLostHits();  
	      
	      unsigned int Algo = 0;
	      if (refKf.isNonnull()) 
		Algo = refKf->algo(); 
	      
	      bool trackIsFromPrimaryVertex = false;
	      for (Vertex::trackRef_iterator trackIt = primaryVertex.tracks_begin(); trackIt != primaryVertex.tracks_end(); ++trackIt) {
		if ( (*trackIt).castTo<TrackRef>() == refKf ) {
		  trackIsFromPrimaryVertex = true;
		  break;
		}
	      }
	      
	      if(Algo < 9 && nexhits == 0 && trackIsFromPrimaryVertex) {
		localactive[ecalKf_index] = false;
	      } else {
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
      if (kfElems.size() > 0) {
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
      if (ecalGsfElems.size() > 0) {	
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
	    
	    if(ecalOtherGsfElems.size()>0) {
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
	    if (ecalOtherGsfElems.size() > 0) {
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
	      if(ecalConvElems.size() > 0) {
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
		  if(ecalOtherGsfPrimElems.size()>0) {
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
		    convGsfTrack_.push_back(make_pair(ecalConvElems.begin()->second,gsfIs[iConv]));
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
      if (kfElems.size() > 0) {
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
	    if(ecalConvElems.size() > 0) {
	      // Further Cleaning: DANIELE This could be improved!
	      TrackRef trkRef =   TrkEl->trackRef();
	      // iter0, iter1, iter2, iter3 = Algo < 3
	      unsigned int Algo = whichTrackAlgo(trkRef);

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
	      if(hcalConvElems.size() > 0) {
		const reco::PFBlockElementCluster * clusthcal =  
		  dynamic_cast<const reco::PFBlockElementCluster*>((&elements[(hcalConvElems.begin()->second)])); 
		enehcalclust  =clusthcal->clusterRef()->energy();
		// NOTE: DANIELE? Are you sure you want to use the Algo type here? 
		if( (enehcalclust / (enehcalclust+eneclust) ) > 0.1 && Algo < 3) {
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
			 << " Algo Track " << Algo << endl;
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
			 << " Algo Track " << Algo << endl;
		  continue;
		}

		// check that this cluster is not cluser to another KF track (primary)
		std::multimap<double, unsigned int> ecalOtherKFPrimElems;
		block.associatedElements( ecalConvElems.begin()->second,linkData,
					  ecalOtherKFPrimElems,
					  reco::PFBlockElement::TRACK,
					  reco::PFBlock::LINKTEST_ALL);
		if(ecalOtherKFPrimElems.size() > 0) {
		  
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
      if(EcalIndex.size() > 0 && useEGammaSupercluster_) {
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
	    reco::TrackRef trkref = kfEle->trackRef();
	    if (trkref.isNonnull()) {
	      double deta_trk =  trkref->eta() - RefGSF->etaMode();
	      double dphi_trk =  trkref->phi() - RefGSF->phiMode();
	      if ( dphi_trk < -M_PI ) 
		dphi_trk = dphi_trk + 2.*M_PI;
	      else if ( dphi_trk > M_PI ) 
		dphi_trk = dphi_trk - 2.*M_PI;
	      double  DR = sqrt(deta_trk*deta_trk+
				dphi_trk*dphi_trk);
	      
	      reco::HitPattern kfHitPattern = trkref->hitPattern();
	      int NValPixelHit = kfHitPattern.numberOfValidPixelHits();
	      
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
	    if(ecalFromSuperClusterElems.size() > 0) {
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
		if(ecalOtherKFPrimElems.size() > 0) {
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
	  if(kfEtraElems.size() > 0) {
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

unsigned int PFEGammaAlgo::whichTrackAlgo(const reco::TrackRef& trackRef) {
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
bool PFEGammaAlgo::isPrimaryTrack(const reco::PFBlockElementTrack& KfEl,
				    const reco::PFBlockElementGsfTrack& GsfEl) {
  bool isPrimary = false;
  
  GsfPFRecTrackRef gsfPfRef = GsfEl.GsftrackRefPF();
  
  if(gsfPfRef.isNonnull()) {
    PFRecTrackRef  kfPfRef = KfEl.trackRefPF();
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

void PFEGammaAlgo::AddElectronElements(unsigned int gsf_index,
			             std::vector<unsigned int> &elemsToLock,
				     const reco::PFBlockRef&  blockRef,
				     AssMap& associatedToGsf_,
				     AssMap& associatedToBrems_,
				     AssMap& associatedToEcal_){
  const reco::PFBlock& block = *blockRef;
  PFBlock::LinkData linkData =  block.linkData();  
   
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  
  const reco::PFBlockElementGsfTrack * GsfEl  =  
    dynamic_cast<const reco::PFBlockElementGsfTrack*>((&elements[gsf_index]));
  reco::GsfTrackRef RefGSF = GsfEl->GsftrackRef();

  // lock only the elements that pass the BDT cut
//   bool bypassmva=false;
//   if(useEGElectrons_) {
//     GsfElectronEqual myEqual(RefGSF);
//     std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(theGsfElectrons_->begin(),theGsfElectrons_->end(),myEqual);
//     if(itcheck!=theGsfElectrons_->end()) {
//       if(BDToutput_[cgsf] >= -1.) 
// 	bypassmva=true;
//     }
//   }

  //if(BDToutput_[cgsf] < mvaEleCut_ && bypassmva == false) continue;

  
  elemsToLock.push_back(gsf_index);
  vector<unsigned int> &assogsf_index = associatedToGsf_[gsf_index];
  for  (unsigned int ielegsf=0;ielegsf<assogsf_index.size();ielegsf++) {
    PFBlockElement::Type assoele_type = elements[(assogsf_index[ielegsf])].type();
    // lock the elements associated to the gsf: ECAL, Brems
    elemsToLock.push_back((assogsf_index[ielegsf]));
    if (assoele_type == reco::PFBlockElement::ECAL) {
      unsigned int keyecalgsf = assogsf_index[ielegsf];

      // added protection against fifth step
      if(fifthStepKfTrack_.size() > 0) {
	for(unsigned int itr = 0; itr < fifthStepKfTrack_.size(); itr++) {
	  if(fifthStepKfTrack_[itr].first == keyecalgsf) {
	    elemsToLock.push_back((fifthStepKfTrack_[itr].second));
	  }
	}
      }

      // added locking for conv gsf tracks and kf tracks
      if(convGsfTrack_.size() > 0) {
	for(unsigned int iconv = 0; iconv < convGsfTrack_.size(); iconv++) {
	  if(convGsfTrack_[iconv].first == keyecalgsf) {
	    // lock the GSF track
	    elemsToLock.push_back(convGsfTrack_[iconv].second);
	    // lock also the KF track associated
	    std::multimap<double, unsigned> convKf;
	    block.associatedElements( convGsfTrack_[iconv].second,
				      linkData,
				      convKf,
				      reco::PFBlockElement::TRACK,
				      reco::PFBlock::LINKTEST_ALL );
	    if(convKf.size() > 0) {
	      elemsToLock.push_back(convKf.begin()->second);
	    }
	  }
	}
      }


      vector<unsigned int> assoecalgsf_index = associatedToEcal_.find(keyecalgsf)->second;
      for(unsigned int ips =0; ips<assoecalgsf_index.size();ips++) {
	// lock the elements associated to ECAL: PS1,PS2, for the moment not HCAL
	if  (elements[(assoecalgsf_index[ips])].type() == reco::PFBlockElement::PS1) 
	  elemsToLock.push_back((assoecalgsf_index[ips]));
	if  (elements[(assoecalgsf_index[ips])].type() == reco::PFBlockElement::PS2) 
	  elemsToLock.push_back(assoecalgsf_index[ips]);
	if  (elements[(assoecalgsf_index[ips])].type() == reco::PFBlockElement::TRACK) {
	  //FIXME: some extra input needed here which is not available yet
// 	  if(lockExtraKf_[cgsf] == true) {	      
// 	    elemsToLock.push_back(assoecalgsf_index[ips])
// 	  }
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
	  elemsToLock.push_back(assobrem_index[ibrem]);

	  // add protection against fifth step
	  if(fifthStepKfTrack_.size() > 0) {
	    for(unsigned int itr = 0; itr < fifthStepKfTrack_.size(); itr++) {
	      if(fifthStepKfTrack_[itr].first == keyecalbrem) {
		elemsToLock.push_back(fifthStepKfTrack_[itr].second);
	      }
	    }
	  }

	  vector<unsigned int> assoelebrem_index = associatedToEcal_.find(keyecalbrem)->second;
	  // lock the elements associated to ECAL: PS1,PS2, for the moment not HCAL
	  for (unsigned int ielebrem=0; ielebrem<assoelebrem_index.size();ielebrem++) {
	    if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS1) 
	      elemsToLock.push_back(assoelebrem_index[ielebrem]);
	    if (elements[(assoelebrem_index[ielebrem])].type() == reco::PFBlockElement::PS2) 
	      elemsToLock.push_back(assoelebrem_index[ielebrem]);
	  }
	}
      }
    } // End if BREM	  
  } // End loop on elements from gsf track
  return;
}

// This function get the associatedToGsf and associatedToBrems maps and  
// compute the electron 4-mom and set the pf candidate, for
// the gsf track with a BDTcut > mvaEleCut_
bool PFEGammaAlgo::AddElectronCandidate(unsigned int gsf_index,
					reco::SuperClusterRef scref,
					 std::vector<unsigned int> &elemsToLock,
					 const reco::PFBlockRef&  blockRef,
					 AssMap& associatedToGsf_,
					 AssMap& associatedToBrems_,
					 AssMap& associatedToEcal_,
					 std::vector<bool>& active) {
  
  const reco::PFBlock& block = *blockRef;
  PFBlock::LinkData linkData =  block.linkData();     
  const edm::OwnVector< reco::PFBlockElement >&  elements = block.elements();
  PFEnergyResolution pfresol_;
  //PFEnergyCalibration pfcalib_;

  bool DebugIDCandidates = false;
//   vector<reco::PFCluster> pfClust_vec(0);
//   pfClust_vec.clear();

	  
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
  vector<unsigned int> &assogsf_index = associatedToGsf_[gsf_index];
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
//       std::map<unsigned int,std::vector<reco::PFCandidate> >::iterator itcheck=
// 	electronConstituents_.find(cgsf);
//       if(itcheck==electronConstituents_.end())
// 	{		  
// 	  // beurk
// 	  std::vector<reco::PFCandidate> tmpVec;
// 	  tmpVec.push_back(cluster_Candidate);
// 	  electronConstituents_.insert(std::pair<unsigned int, std::vector<reco::PFCandidate> >
// 					(cgsf,tmpVec));
// 	}
//       else
// 	{
// 	  itcheck->second.push_back(cluster_Candidate);
// 	}
      
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
	    //FIXME: constituents needed?
// 	    std::map<unsigned int,std::vector<reco::PFCandidate> >::iterator itcheck=
// 	      electronConstituents_.find(cgsf);
// 	    if(itcheck==electronConstituents_.end())
// 	      {		  
// 		// beurk
// 		std::vector<reco::PFCandidate> tmpVec;
// 		tmpVec.push_back(photon_Candidate);
// 		electronConstituents_.insert(std::pair<unsigned int, std::vector<reco::PFCandidate> >
// 					  (cgsf,tmpVec));
// 	      }
// 	    else
// 	      {
// 		itcheck->second.push_back(photon_Candidate);
// 	      }
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
      if(bremElems.size()>0) {
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
      //std::vector<reco::PFCandidateElectronExtra>::iterator itextra;
      //PFElectronExtraEqual myExtraEqual(RefGSF);
      PFCandidateEGammaExtra myExtraEqual(RefGSF);
      //myExtraEqual.setSuperClusterRef(scref);
      myExtraEqual.setSuperClusterBoxRef(scref);
      myExtraEqual.setClusterEnergies(clusterEnergyVec);
      //itextra=find_if(electronExtra_.begin(),electronExtra_.end(),myExtraEqual);
      //if(itextra!=electronExtra_.end()) {
	//itextra->setClusterEnergies(clusterEnergyVec);
//       else {
// 	if(RawEene>0.) 
// 	  std::cout << " There is a big problem with the electron extra, PFElectronAlgo should crash soon " << RawEene << std::endl;
//       }

      reco::PFCandidate::ParticleType particleType 
	= reco::PFCandidate::e;
      //reco::PFCandidate temp_Candidate;
      reco::PFCandidate temp_Candidate(charge,newmomentum,particleType);
      //FIXME: need bdt output
      //temp_Candidate.set_mva_e_pi(BDToutput_[cgsf]);
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
      
      //supercluster ref is always available now and points to ecal-drive box/mustache supercluster
      temp_Candidate.setSuperClusterRef(scref);
      
      // save the superclusterRef when available
      //FIXME: Point back to ecal-driven supercluster ref, which is now always available
//       if(RefGSF->extra().isAvailable() && RefGSF->extra()->seedRef().isAvailable()) {
// 	reco::ElectronSeedRef seedRef=  RefGSF->extra()->seedRef().castTo<reco::ElectronSeedRef>();
// 	if(seedRef.isAvailable() && seedRef->isEcalDriven()) {
// 	  reco::SuperClusterRef scRef = seedRef->caloCluster().castTo<reco::SuperClusterRef>();
// 	  if(scRef.isNonnull())  
// 	    temp_Candidate.setSuperClusterRef(scRef);
// 	}
//       }

      if( DebugIDCandidates ) 
	cout << "SetCandidates:: I am after doing candidate " <<endl;
      
//       for (unsigned int elad=0; elad<elementsToAdd.size();elad++){
// 	temp_Candidate.addElementInBlock(blockRef,elementsToAdd[elad]);
//       }
// 
//       // now add the photons to this candidate
//       std::map<unsigned int, std::vector<reco::PFCandidate> >::const_iterator itcluster=
// 	electronConstituents_.find(cgsf);
//       if(itcluster!=electronConstituents_.end())
// 	{
// 	  const std::vector<reco::PFCandidate> & theClusters=itcluster->second;
// 	  unsigned nclus=theClusters.size();
// 	  //	    std::cout << " PFElectronAlgo " << nclus << " daugthers to add" << std::endl;
// 	  for(unsigned iclus=0;iclus<nclus;++iclus)
// 	    {
// 	      temp_Candidate.addDaughter(theClusters[iclus]);
// 	    }
// 	}

      // By-pass the mva is the electron has been pre-selected 
//       bool bypassmva=false;
//       if(useEGElectrons_) {
// 	GsfElectronEqual myEqual(RefGSF);
// 	std::vector<reco::GsfElectron>::const_iterator itcheck=find_if(theGsfElectrons_->begin(),theGsfElectrons_->end(),myEqual);
// 	if(itcheck!=theGsfElectrons_->end()) {
// 	  if(BDToutput_[cgsf] >= -1.)  {
// 	    // bypass the mva only if the reconstruction went fine
// 	    bypassmva=true;
// 
// 	    if( DebugIDCandidates ) {
// 	      if(BDToutput_[cgsf] < -0.1) {
// 		float esceg = itcheck->caloEnergy();		
// 		cout << " Attention By pass the mva " << BDToutput_[cgsf] 
// 		      << " SuperClusterEnergy " << esceg
// 		      << " PF Energy " << Eene << endl;
// 		
// 		cout << " hoe " << itcheck->hcalOverEcal()
// 		      << " tkiso04 " << itcheck->dr04TkSumPt()
// 		      << " ecaliso04 " << itcheck->dr04EcalRecHitSumEt()
// 		      << " hcaliso04 " << itcheck->dr04HcalTowerSumEt()
// 		      << " tkiso03 " << itcheck->dr03TkSumPt()
// 		      << " ecaliso03 " << itcheck->dr03EcalRecHitSumEt()
// 		      << " hcaliso03 " << itcheck->dr03HcalTowerSumEt() << endl;
// 	      }
// 	    } // end DebugIDCandidates
// 	  }
// 	}
//       }
      
      myExtraEqual.setStatus(PFCandidateEGammaExtra::Selected,true);
      
      // ... and lock all elemts used
      for(std::vector<unsigned int>::const_iterator it = elemsToLock.begin();
	  it != elemsToLock.end(); ++it)
	{
	  if(active[*it])
	    {
	      temp_Candidate.addElementInBlock(blockRef,*it);
	    }
	  active[*it] = false;	
	}      
      
      egCandidate_.push_back(temp_Candidate);
      egExtra_.push_back(myExtraEqual);
      
      return true;
      
//       bool mvaSelected = (BDToutput_[cgsf] >=  mvaEleCut_);
//       if( mvaSelected || bypassmva ) 	  {
// 	  elCandidate_.push_back(temp_Candidate);
// 	  if(itextra!=electronExtra_.end()) 
// 	    itextra->setStatus(PFCandidateElectronExtra::Selected,true);
// 	}
//       else 	  {
// 	if(itextra!=electronExtra_.end()) 
// 	  itextra->setStatus(PFCandidateElectronExtra::Rejected,true);
//       }
//       allElCandidate_.push_back(temp_Candidate);
//       
//       // save the status information
//       if(itextra!=electronExtra_.end()) {
// 	itextra->setStatus(PFCandidateElectronExtra::ECALDrivenPreselected,bypassmva);
// 	itextra->setStatus(PFCandidateElectronExtra::MVASelected,mvaSelected);
//       }
      

    }
    else {
      //BDToutput_[cgsf] = -1.;   // if the momentum is < 0.5 ID = false, but not sure
      // it could be misleading. 
      if( DebugIDCandidates ) 
	cout << "SetCandidates:: No Candidate Produced because of Pt cut: 0.5 " <<endl;
      return false;
    }
  } 
  else {
    //BDToutput_[cgsf] = -1.;  // if gsf ref does not exist
    if( DebugIDCandidates ) 
      cout << "SetCandidates:: No Candidate Produced because of No GSF Track Ref " <<endl;
    return false;
  }
  return false;
}
