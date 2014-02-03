#ifndef RecoParticleFlow_PFProducer_PFBlockAlgo_h
#define RecoParticleFlow_PFProducer_PFBlockAlgo_h 

#include <set>
#include <vector>
#include <iostream>

// #include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/Handle.h"
// #include "FWCore/Framework/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"


#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedTrackerVertex.h" // gouzevitch
#include "DataFormats/ParticleFlowReco/interface/PFConversionFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFConversion.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0Fwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFV0.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrack.h"
#include "DataFormats/ParticleFlowReco/interface/GsfPFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFTrajectoryPoint.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockFwd.h"

// Glowinski & Gouzevitch
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"             
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTrackEcal.h" 
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerTrackHcal.h" 
#include "RecoParticleFlow/PFProducer/interface/KDTreeLinkerPSEcal.h" 
// !Glowinski & Gouzevitch

// #include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PhotonSelectorAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/PFBlockElementSCEqual.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"

#include "RecoParticleFlow/PFClusterTools/interface/ClusterClusterMapping.h"

#include "RecoParticleFlow/PFProducer/interface/PFBlockLink.h"

#include <map>


/// \brief Particle Flow Algorithm
/*!
  \author Colin Bernet
  \date January 2006
*/

class PFBlockAlgo {

 public:

  PFBlockAlgo();

  ~PFBlockAlgo();
  

  void setParameters( std::vector<double>& DPtovPtCut, 
		      std::vector<unsigned>& NHitCut,
		      bool useConvBremPFRecTracks,
		      bool useIterTracking,
		      int nuclearInteractionsPurity,
		      bool useEGPhotons,
		      std::vector<double> & photonSelectionCuts,
		      bool useSuperClusters
		      );
  
  // Glowinski & Gouzevitch
  void setUseOptimization(bool useKDTreeTrackEcalLinker);
  // ! Glowinski & Gouzevitch

  typedef std::vector<bool> Mask;

  /// set input collections of tracks and clusters
  template< template<typename> class T>
    void  setInput(const T<reco::PFRecTrackCollection>&    trackh,
		   const T<reco::GsfPFRecTrackCollection>&    gsftrackh,
		   const T<reco::GsfPFRecTrackCollection>&    convbremgsftrackh,
		   const T<reco::MuonCollection>&    muonh,
		   const T<reco::PFDisplacedTrackerVertexCollection>&  nuclearh,
		   const T<reco::PFRecTrackCollection>&    nucleartrackh,
		   const T<reco::PFConversionCollection>&  conv,
		   const T<reco::PFV0Collection>&  v0,
		   const T<reco::PFClusterCollection>&  ecalh,
		   const T<reco::PFClusterCollection>&  hcalh,
		   const T<reco::PFClusterCollection>&  hoh,
		   const T<reco::PFClusterCollection>&  hfemh,
		   const T<reco::PFClusterCollection>&  hfhadh,
		   const T<reco::PFClusterCollection>&  psh,
		   const T<reco::PhotonCollection>&  egphh,
		   const T<reco::SuperClusterCollection>&  sceb,
		   const T<reco::SuperClusterCollection>&  scee,		   
		   const Mask& trackMask = dummyMask_,
		   const Mask& gsftrackMask = dummyMask_,
		   const Mask& ecalMask = dummyMask_,
		   const Mask& hcalMask = dummyMask_,
		   const Mask& hoMask = dummyMask_, 
		   const Mask& hfemMask = dummyMask_,		   
		   const Mask& hfhadMask = dummyMask_,
		   const Mask& psMask = dummyMask_,
		   const Mask& phMask = dummyMask_,
		   const Mask& scMask = dummyMask_);
  
  ///COLIN: I think this is for particle flow at HLT...
  template< template<typename> class T >
    void setInput(const T<reco::PFRecTrackCollection>&    trackh,
		  const T<reco::MuonCollection>&    muonh,
		  const T<reco::PFClusterCollection>&  ecalh,
		  const T<reco::PFClusterCollection>&  hcalh,
 		  const T<reco::PFClusterCollection>&  hoh,
		  const T<reco::PFClusterCollection>&  hfemh,
		  const T<reco::PFClusterCollection>&  hfhadh,
		  const T<reco::PFClusterCollection>&  psh,
		  const Mask& trackMask = dummyMask_,
		  const Mask& ecalMask = dummyMask_,
		  const Mask& hcalMask = dummyMask_,
		  const Mask& hoMask = dummyMask_,
		  const Mask& psMask = dummyMask_) {
    T<reco::GsfPFRecTrackCollection> gsftrackh;
    T<reco::GsfPFRecTrackCollection> convbremgsftrackh;
    //T<reco::MuonCollection> muonh;
    T<reco::PFDisplacedTrackerVertexCollection> nuclearh;
    T<reco::PFRecTrackCollection>    nucleartrackh;
    T<reco::PFConversionCollection> convh;
    T<reco::PFV0Collection> v0;
    T<reco::PhotonCollection> phh;
    T<reco::SuperClusterCollection> scebh;
    T<reco::SuperClusterCollection> sceeh;    
    setInput<T>( trackh, gsftrackh, convbremgsftrackh, muonh, nuclearh, nucleartrackh, convh, v0, 
		 ecalh, hcalh, hoh, hfemh, hfhadh, psh, phh, scebh, sceeh,
		 trackMask, ecalMask, hcalMask, hoMask, psMask); 
  }
  
  ///COLIN: what is this setinput function for? can it be removed?
  template< template<typename> class T >
    void setInput(const T<reco::PFRecTrackCollection>&    trackh,
		  const T<reco::GsfPFRecTrackCollection>&    gsftrackh,
		  const T<reco::PFClusterCollection>&  ecalh,
		  const T<reco::PFClusterCollection>&  hcalh,
		  const T<reco::PFClusterCollection>&  hoh,
		  const T<reco::PFClusterCollection>&  psh,
		  const Mask& trackMask = dummyMask_,
		  const Mask& gsftrackMask = dummyMask_,
		  const Mask& ecalMask = dummyMask_,
		  const Mask& hcalMask = dummyMask_,
		  const Mask& hoMask = dummyMask_,
		  const Mask& psMask = dummyMask_) {
    T<reco::GsfPFRecTrackCollection> convbremgsftrackh;
    T<reco::MuonCollection> muonh;
    T<reco::PFDisplacedTrackerVertexCollection>  nuclearh;
    T<reco::PFRecTrackCollection>    nucleartrackh;
    T<reco::PFConversionCollection> convh;
    T<reco::PFV0Collection> v0;
    T<reco::PhotonCollection> egphh;
    setInput<T>( trackh, gsftrackh, convbremgsftrackh, muonh, nuclearh, nucleartrackh, convh, v0, ecalh, hcalh, hoh, psh, egphh,
		 trackMask, gsftrackMask,ecalMask, hcalMask, hoMask, psMask); 
  }
  
  
  /// sets debug printout flag
  void setDebug( bool debug ) {debug_ = debug;}
  
  /// build blocks
  void findBlocks();
  

  /// \return collection of blocks
  /*   const  reco::PFBlockCollection& blocks() const {return *blocks_;} */
  const std::auto_ptr< reco::PFBlockCollection >& blocks() const 
    {return blocks_;}
  
  /// \return auto_ptr to collection of blocks
  std::auto_ptr< reco::PFBlockCollection > transferBlocks() {return blocks_;}

  /// define these in *Fwd files in DataFormats/ParticleFlowReco?
  typedef std::list< reco::PFBlockElement* >::iterator IE;
  typedef std::list< reco::PFBlockElement* >::const_iterator IEC;  
  typedef reco::PFBlockCollection::const_iterator IBC;
  
  void setHOTag(bool ho) { useHO_ = ho;}

 private:
  
  /// recursive procedure which adds elements from 
  /// elements_ to the current block, ie blocks_->back().
  /// the resulting links between elements are stored in links, 
  /// not in the block. afterwards, 
  /// packLinks( reco::PFBlock& block, const vector<PFBlockLink>& links)
  /// has to be called in order to pack the link information in the block.
  IE associate(IE next, IE last, std::vector<PFBlockLink>& links);

  /// compute missing links in the blocks 
  /// (the recursive procedure does not build all links)  
  void packLinks(reco::PFBlock& block, 
		 const std::vector<PFBlockLink>& links) const; 

  /// remove extra links between primary track and clusters
  void checkDisplacedVertexLinks( reco::PFBlock& block ) const;
  
  ///COLIN: not used. Could be removed.
  /// Could also be implemented, to produce a graph of a block,
  /// Showing the connections between the elements
  void buildGraph(); 

  /// Avoid to check links when not useful
  inline bool linkPrefilter(const reco::PFBlockElement* last, const reco::PFBlockElement* next) const;

  /// check whether 2 elements are linked. Returns distance and linktype
  void link( const reco::PFBlockElement* el1, 
	     const reco::PFBlockElement* el2, 
	     PFBlockLink::Type& linktype, 
	     reco::PFBlock::LinkTest& linktest,
	     double& dist) const;
		 
  /// tests association between a track and a PS cluster
  /// returns distance
  double testTrackAndPS(const reco::PFRecTrack& track,
			const reco::PFCluster& ps) const;
  
  /// tests association between an ECAL and an HCAL cluster
  /// \returns distance
  double testECALAndHCAL(const reco::PFCluster& ecal, 
			 const reco::PFCluster& hcal) const;

  /// tests association between an HCAL and an HO cluster
  /// \returns distance
  double testHCALAndHO(const reco::PFCluster& hcal, 
		       const reco::PFCluster& ho) const;

			 
  /// tests association between a PS1 v cluster and a PS2 h cluster
  /// returns distance
  double testPS1AndPS2(const reco::PFCluster& ps1,
		       const reco::PFCluster& ps2) const;

 /// test association by Supercluster between two ECAL
  double testLinkBySuperCluster(const reco::PFClusterRef & elt1,
				const reco::PFClusterRef & elt2) const;   

  /// test association between SuperClusters and ECAL
  double testSuperClusterPFCluster(const reco::SuperClusterRef & sct1,
				   const reco::PFClusterRef & elt2) const;

  /// checks size of the masks with respect to the vectors
  /// they refer to. throws std::length_error if one of the
  /// masks has the wrong size
  void checkMaskSize( const reco::PFRecTrackCollection& tracks,
		      const reco::GsfPFRecTrackCollection& gsftracks,
		      const reco::PFClusterCollection&  ecals,
		      const reco::PFClusterCollection&  hcals,
		      const reco::PFClusterCollection&  hos, 
		      const reco::PFClusterCollection&  hfems,
		      const reco::PFClusterCollection&  hfhads,
		      const reco::PFClusterCollection&  pss, 
		      const reco::PhotonCollection&  egphh, 
		      const reco::SuperClusterCollection&  sceb, 
		      const reco::SuperClusterCollection&  scee, 		      
		      const Mask& trackMask,
		      const Mask& gsftrackMask, 
		      const Mask& ecalMask, 
		      const Mask& hcalMask,
		      const Mask& hoMask,
		      const Mask& hfemMask,
		      const Mask& hfhadMask,		      
		      const Mask& psMask,
		      const Mask& phMask,
		      const Mask& scMask) const;

  /// open a resolution map
  // PFResolutionMap* openResolutionMap(const char* resMapName);

  /// check the Pt resolution 
  bool goodPtResolution( const reco::TrackRef& trackref);

  double testLinkByVertex(const reco::PFBlockElement* elt1,
			  const reco::PFBlockElement* elt2) const;

  int muAssocToTrack( const reco::TrackRef& trackref,
		      const edm::Handle<reco::MuonCollection>& muonh) const;
  int muAssocToTrack( const reco::TrackRef& trackref,
		      const edm::OrphanHandle<reco::MuonCollection>& muonh) const;

  template< template<typename> class T>
    void fillFromPhoton(const T<reco::PhotonCollection> &, unsigned isc, reco::PFBlockElementSuperCluster * pfbe);

  std::auto_ptr< reco::PFBlockCollection >    blocks_;
  
  /// actually, particles will be created by a separate producer
  // std::vector< reco::PFCandidate >   particles_;

  // the test elements will be transferred to the blocks
  std::list< reco::PFBlockElement* >     elements_;

  // Glowinski & Gouzevitch
  bool useKDTreeTrackEcalLinker_;
  KDTreeLinkerTrackEcal TELinker_;
  KDTreeLinkerTrackHcal THLinker_;
  KDTreeLinkerPSEcal	PSELinker_;
  // !Glowinski & Gouzevitch

  static const Mask                      dummyMask_;

  /// DPt/Pt cut for creating atrack element
  std::vector<double> DPtovPtCut_;
  
  /// Number of layers crossed cut for creating atrack element
  std::vector<unsigned> NHitCut_;
  
  /// Flag to turn off quality cuts which require iterative tracking (for heavy-ions)
  bool useIterTracking_;

  /// Flag to turn off the import of EG Photons
  bool useEGPhotons_;
  
  /// Flag to turn off the import of SuperCluster collections
  bool useSuperClusters_;

  // This parameters defines the level of purity of
  // nuclear interactions choosen.
  // Level 1 is only high Purity sample labeled as isNucl.
  // Level 2 isNucl + isNucl_Loose (2 secondary tracks vertices)
  // Level 3 isNucl + isNucl_Loose + isNucl_Kink
  //         (low purity sample made of 1 primary and 1 secondary track)
  // By default the level 1 is teh safest one.
  int nuclearInteractionsPurity_;

  /// switch on/off Conversions Brem Recovery with KF Tracks
  bool  useConvBremPFRecTracks_;

  /// PhotonSelector
  const PhotonSelectorAlgo * photonSelector_;
   /// list of superclusters 
  std::vector<reco::SuperClusterRef > superClusters_;

  /// SC corresponding to the PF cluster
  //  std::map<reco::PFClusterRef,int>  pfcRefSCMap_;
  std::vector<int> pfcSCVec_;

  // A boolean to avoid to compare ECAL and ECAl if there i no superclusters in the event
  bool bNoSuperclus_;

  /// PF clusters corresponding to a given SC
  std::vector<std::vector<reco::PFClusterRef> > scpfcRefs_;
  /// if true, debug printouts activated
  bool   debug_;
  
  friend std::ostream& operator<<(std::ostream&, const PFBlockAlgo&);

  // Create links between tracks or HCAL clusters, and HO clusters
  bool useHO_;

};

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

template< template<typename> class T >
void
PFBlockAlgo::setInput(const T<reco::PFRecTrackCollection>&    trackh,
		      const T<reco::GsfPFRecTrackCollection>&    gsftrackh, 
		      const T<reco::GsfPFRecTrackCollection>&    convbremgsftrackh,
		      const T<reco::MuonCollection>&    muonh,
		      const T<reco::PFDisplacedTrackerVertexCollection>&  nuclearh,
		      const T<reco::PFRecTrackCollection>&    nucleartrackh,
		      const T<reco::PFConversionCollection>&  convh,
		      const T<reco::PFV0Collection>&  v0,
                      const T<reco::PFClusterCollection>&  ecalh,
                      const T<reco::PFClusterCollection>&  hcalh,
		      const T<reco::PFClusterCollection>&  hoh,
                      const T<reco::PFClusterCollection>&  hfemh,
                      const T<reco::PFClusterCollection>&  hfhadh,
                      const T<reco::PFClusterCollection>&  psh,
		      const T<reco::PhotonCollection>& egphh,
		      const T<reco::SuperClusterCollection>&  sceb,
		      const T<reco::SuperClusterCollection>&  scee,		      
                      const Mask& trackMask,
		      const Mask& gsftrackMask,
                      const Mask& ecalMask,
                      const Mask& hcalMask,
		      const Mask& hoMask,
                      const Mask& hfemMask,
                      const Mask& hfhadMask,
                      const Mask& psMask,
		      const Mask& phMask,
		      const Mask& scMask) {


  checkMaskSize( *trackh,
		 *gsftrackh,
                 *ecalh,
                 *hcalh,
		 *hoh,
		 *hfemh,
		 *hfhadh,
                 *psh,
		 *egphh,
		 *sceb,
		 *scee,		 
                 trackMask,
		 gsftrackMask,
                 ecalMask,
                 hcalMask,
		 hoMask,
		 hfemMask,
		 hfhadMask,
                 psMask,
		 phMask,
		 scMask);

  /*
  if (nucleartrackh.isValid()){
    for(unsigned i=0;i<nucleartrackh->size(); i++) {
      reco::PFRecTrackRef trackRef(nucleartrackh,i);
      std::cout << *trackRef << std::endl;
    }
  }
  */

  /// -------------- GSF Primary tracks and brems ---------------------
  std::vector<reco::PFRecTrackRef> convBremPFRecTracks;
  convBremPFRecTracks.clear();
  // Super cluster mapping
  superClusters_.clear();
  scpfcRefs_.clear();
  pfcSCVec_.clear();

  if(gsftrackh.isValid() ) {
    const  reco::GsfPFRecTrackCollection PFGsfProd = *(gsftrackh.product());
    for(unsigned i=0;i<gsftrackh->size(); ++i) {
      if( !gsftrackMask.empty() &&
          !gsftrackMask[i] ) continue;
      reco::GsfPFRecTrackRef refgsf(gsftrackh,i );   
   
      if((refgsf).isNull()) continue;
      reco::GsfTrackRef gsf=refgsf->gsfTrackRef();

      // retrieve and save the SC if ECAL-driven - Florian
      if(gsf->extra().isAvailable() && gsf->extra()->seedRef().isAvailable()) {
	reco::ElectronSeedRef seedRef=  gsf->extra()->seedRef().castTo<reco::ElectronSeedRef>();
	// check that the seed is valid
	if(seedRef.isAvailable() && seedRef->isEcalDriven()) {
	  reco::SuperClusterRef scRef = seedRef->caloCluster().castTo<reco::SuperClusterRef>();
	  if(scRef.isNonnull())   {	     
	    std::vector<reco::SuperClusterRef>::iterator itcheck=find(superClusters_.begin(),superClusters_.end(),scRef);
	    // not present, add it
	    if(itcheck==superClusters_.end())
	      {
		superClusters_.push_back(scRef);
		reco::PFBlockElementSuperCluster * sce =
		  new reco::PFBlockElementSuperCluster(scRef);
		sce->setFromGsfElectron(true);
		elements_.push_back(sce);
	      }
	    else // it is already present, update the PFBE
	      {
		PFBlockElementSCEqual myEqual(scRef);
		std::list<reco::PFBlockElement*>::iterator itcheck=find_if(elements_.begin(),elements_.end(),myEqual);
		if(itcheck!=elements_.end())
		  {
		    reco::PFBlockElementSuperCluster* thePFBE=dynamic_cast<reco::PFBlockElementSuperCluster*>(*itcheck);
		    thePFBE->setFromGsfElectron(true);
		    //		    std::cout << " Updating element to add electron information" << std::endl;
		  }
//		else
//		  {
//		    std::cout << " Missing element " << std::endl;
//		  }
	      }
	  }
	}
      }

      reco::PFBlockElement* gsfEl;
      
      const  std::vector<reco::PFTrajectoryPoint> 
	PfGsfPoint =  PFGsfProd[i].trajectoryPoints();
  
      unsigned int c_gsf=0;
      bool PassTracker = false;
      bool GetPout = false;
      unsigned int IndexPout = 0;
      
      typedef std::vector<reco::PFTrajectoryPoint>::const_iterator IP;
      for(IP itPfGsfPoint =  PfGsfPoint.begin();  
	  itPfGsfPoint!= PfGsfPoint.end();++itPfGsfPoint) {
	
	if (itPfGsfPoint->isValid()){
	  int layGsfP = itPfGsfPoint->layer();
	  if (layGsfP == -1) PassTracker = true;
	  if (PassTracker && layGsfP > 0 && GetPout == false) {
	    IndexPout = c_gsf-1;
	    GetPout = true;
	  }
	  //const math::XYZTLorentzVector GsfMoment = itPfGsfPoint->momentum();
	  ++c_gsf;
	}
      }
      math::XYZTLorentzVector pin = PfGsfPoint[0].momentum();      
      math::XYZTLorentzVector pout = PfGsfPoint[IndexPout].momentum();

      /// get tracks from converted brems
      if(useConvBremPFRecTracks_) {
	const std::vector<reco::PFRecTrackRef>& temp_convBremPFRecTracks(refgsf->convBremPFRecTrackRef());
	if(temp_convBremPFRecTracks.size() > 0) {
	  for(unsigned int iconv = 0; iconv <temp_convBremPFRecTracks.size(); ++iconv) {
	    convBremPFRecTracks.push_back(temp_convBremPFRecTracks[iconv]);
	  }
	}
      }
      
      gsfEl = new reco::PFBlockElementGsfTrack(refgsf, pin, pout);
      
      elements_.push_back( gsfEl);

      std::vector<reco::PFBrem> pfbrem = refgsf->PFRecBrem();
      
      for (unsigned i2=0;i2<pfbrem.size(); ++i2) {
	const double DP = pfbrem[i2].DeltaP();
	const double SigmaDP =  pfbrem[i2].SigmaDeltaP(); 
	const unsigned int TrajP = pfbrem[i2].indTrajPoint();
	if(TrajP == 99) continue;

	reco::PFBlockElement* bremEl;
	bremEl = new reco::PFBlockElementBrem(refgsf,DP,SigmaDP,TrajP);
	elements_.push_back(bremEl);
	
      }
    }

  }

  /// Loop over the photons
  if(useEGPhotons_ && egphh.isValid()) {
    unsigned size=egphh->size();
    for(unsigned isc=0; isc<size; ++isc) {
      if(!phMask.empty() && !(phMask)[isc] ) continue;
      if(!photonSelector_->passPhotonSelection((*egphh)[isc])) continue;
      //      std::cout << " Selected a supercluster" << std::endl;
      // Add only the super clusters not already included 
      reco::SuperClusterRef scRef((*egphh)[isc].superCluster());
      std::vector<reco::SuperClusterRef>::iterator itcheck=find(superClusters_.begin(),superClusters_.end(),(*egphh)[isc].superCluster());
      if(itcheck==superClusters_.end())
	{
	  superClusters_.push_back(scRef);
	  reco::PFBlockElementSuperCluster * sce =
	    new reco::PFBlockElementSuperCluster((*egphh)[isc].superCluster());
	  fillFromPhoton(egphh,isc,sce);
	  elements_.push_back(sce);
	}
      else
	{
	  PFBlockElementSCEqual myEqual(scRef);
	  std::list<reco::PFBlockElement*>::iterator itcheck=find_if(elements_.begin(),elements_.end(),myEqual);
	  if(itcheck!=elements_.end())
	    {
	      reco::PFBlockElementSuperCluster* thePFBE=dynamic_cast<reco::PFBlockElementSuperCluster*>(*itcheck);
	      fillFromPhoton(egphh,isc,thePFBE);
	      thePFBE->setFromPhoton(true);
	      //	      std::cout << " Updating element to add Photon information " << photonSelector_->passPhotonSelection((*egphh)[isc]) << std::endl;

	    }
//	  else
//	    {
//	      std::cout << " Missing element " << std::endl;
//	    }
	}
    }
  }
  
  /// Loop over the SuperClusters
  if(useSuperClusters_ && sceb.isValid() && scee.isValid()) {
    std::vector<reco::SuperClusterRef> screfs;
    screfs.reserve(sceb->size()+scee->size());
    
    for (unsigned int isc=0; isc<sceb->size(); ++isc) {
      reco::SuperClusterRef scRef(sceb,isc);
      screfs.push_back(scRef);
    }

    for (unsigned int isc=0; isc<scee->size(); ++isc) {
      reco::SuperClusterRef scRef(scee,isc);
      screfs.push_back(scRef);
    }    
    
    unsigned size=screfs.size();
    for(unsigned isc=0; isc<size; ++isc) {
      if(!scMask.empty() && !(scMask)[isc] ) continue;
      //if(!photonSelector_->passPhotonSelection((*egphh)[isc])) continue;
      
      //      std::cout << " Selected a supercluster" << std::endl;
      // Add only the super clusters not already included 
      reco::SuperClusterRef &scRef = screfs[isc];
      std::vector<reco::SuperClusterRef>::iterator itcheck=find(superClusters_.begin(),superClusters_.end(),scRef);
      if(itcheck==superClusters_.end())
	{
	  superClusters_.push_back(scRef);
	  reco::PFBlockElementSuperCluster * sce =
	    new reco::PFBlockElementSuperCluster(scRef);
	  //fillFromPhoton(egphh,isc,sce);
	  //sce->setFromPhoton(true);
	  elements_.push_back(sce);
	}
//       else
// 	{
// 	  PFBlockElementSCEqual myEqual(scRef);
// 	  std::list<reco::PFBlockElement*>::iterator itcheck=find_if(elements_.begin(),elements_.end(),myEqual);
// 	  if(itcheck!=elements_.end())
// 	    {
// 	      reco::PFBlockElementSuperCluster* thePFBE=dynamic_cast<reco::PFBlockElementSuperCluster*>(*itcheck);
// 	      //fillFromPhoton(egphh,isc,thePFBE);
// 	      //thePFBE->setFromPhoton(true);
// 	      //	      std::cout << " Updating element to add Photon information " << photonSelector_->passPhotonSelection((*egphh)[isc]) << std::endl;
// 
// 	    }
// //	  else
// //	    {
// //	      std::cout << " Missing element " << std::endl;
// //	    }
// 	}
    }
  }  
  
  // set the vector to the right size so to allow random access
  scpfcRefs_.resize(superClusters_.size());

  /// -------------- conversions ---------------------

  /// The tracks from conversions are filled into the elements collection

  if(convh.isValid() ) {
    reco::PFBlockElement* trkFromConversionElement;
    for(unsigned i=0;i<convh->size(); ++i) {
      reco::PFConversionRef convRef(convh,i);

      unsigned int trackSize=(convRef->pfTracks()).size();
      if ( convRef->pfTracks().size() < 2) continue;
      for(unsigned iTk=0;iTk<trackSize; ++iTk) {
	
	reco::PFRecTrackRef compPFTkRef = convRef->pfTracks()[iTk];	
	trkFromConversionElement = new reco::PFBlockElementTrack(convRef->pfTracks()[iTk]);
	trkFromConversionElement->setConversionRef( convRef->originalConversion(), reco::PFBlockElement::T_FROM_GAMMACONV);

	elements_.push_back( trkFromConversionElement );


	if (debug_){
	  std::cout << "PF Block Element from Conversion electron " << 
	    (*trkFromConversionElement).trackRef().key() << std::endl;
	  std::cout << *trkFromConversionElement << std::endl;
	}
       
      }     
    }  
  }
  
  
  /// -------------- V0 ---------------------
  
  /// The tracks from V0 are filled into the elements collection
  
  if(v0.isValid() ) {
    reco::PFBlockElement* trkFromV0Element = 0;
    for(unsigned i=0;i<v0->size(); ++i) {
      reco::PFV0Ref v0Ref( v0, i );
      unsigned int trackSize=(v0Ref->pfTracks()).size();
      for(unsigned iTk=0;iTk<trackSize; ++iTk) {

	reco::PFRecTrackRef newPFRecTrackRef = (v0Ref->pfTracks())[iTk]; 
	reco::TrackBaseRef newTrackBaseRef(newPFRecTrackRef->trackRef());
	bool bNew = true;
	
	/// One need to cross check if those tracks was not already filled
	/// from the conversion collection
	for(IE iel = elements_.begin(); iel != elements_.end(); ++iel){
	  reco::TrackBaseRef elemTrackBaseRef((*iel)->trackRef());
	  if (newTrackBaseRef == elemTrackBaseRef){	    
	    trkFromV0Element = *iel;
	    bNew = false;
	    continue;
	  }
	} 

	/// This is a new track not yet included into the elements collection
	if (bNew) {
	  trkFromV0Element = new reco::PFBlockElementTrack(v0Ref->pfTracks()[iTk]);
	  elements_.push_back( trkFromV0Element );
	}

	trkFromV0Element->setV0Ref( v0Ref->originalV0(),
				    reco::PFBlockElement::T_FROM_V0 );
	  
	if (debug_){
	  std::cout << "PF Block Element from V0 track New = " << bNew 
		    << (*trkFromV0Element).trackRef().key() << std::endl;
	  std::cout << *trkFromV0Element << std::endl;
	}

	
      }
    }
  }
  
  /// -------------- Displaced Vertices ---------------------

  /// The tracks from Displaced Vertices are filled into the elements collection

  if(nuclearh.isValid()) {
    reco::PFBlockElement* trkFromDisplacedVertexElement = 0;
    for(unsigned i=0;i<nuclearh->size(); ++i) {

      const reco::PFDisplacedTrackerVertexRef dispacedVertexRef( nuclearh, i );

      //      std::cout << "Nuclear Interactions Purity " <<  nuclearInteractionsPurity_ << std::endl;
      //     dispacedVertexRef->displacedVertexRef()->Dump();
      //bool bIncludeVertices = true;
      // We add a cut at rho > 2.7 since this corresponds to the lower edge of the beam pipe
      // This cut have to be changer when a new beam pipe would be installed
      
      bool bIncludeVertices = false; 
      bool bNucl = dispacedVertexRef->displacedVertexRef()->isNucl()
	&& dispacedVertexRef->displacedVertexRef()->position().rho()>  2.7;
      bool bNucl_Loose = dispacedVertexRef->displacedVertexRef()->isNucl_Loose();
      bool bNucl_Kink = dispacedVertexRef->displacedVertexRef()->isNucl_Kink();

      if (nuclearInteractionsPurity_ >= 1) bIncludeVertices = bNucl;
      if (nuclearInteractionsPurity_ >= 2) bIncludeVertices = bIncludeVertices || bNucl_Loose;
      if (nuclearInteractionsPurity_ >= 3) bIncludeVertices = bIncludeVertices || bNucl_Kink;

      if (bIncludeVertices){

	unsigned int trackSize= dispacedVertexRef->pfRecTracks().size();
	if (debug_){
	  std::cout << "" << std::endl;
	  std::cout << "Displaced Vertex " << i << std::endl;
	  dispacedVertexRef->displacedVertexRef()->Dump();
	}
	for(unsigned iTk=0;iTk < trackSize; ++iTk) {


	  // This peace of code looks weired at first but it seems to be necessary to let 
	  // PFRooTEvent work properly. Since the track position called REPPoint is transient 
	  // it has to be calculated and the PFRecTrack collection associted to Displaced Vertex is not
	  // anymore the original collection. So here we match both collections if possible
	  reco::PFRecTrackRef newPFRecTrackRef = dispacedVertexRef->pfRecTracks()[iTk]; 
	  reco::TrackBaseRef constTrackBaseRef(newPFRecTrackRef->trackRef());

	  
	  if (nucleartrackh.isValid()){
	    for(unsigned i=0;i<nucleartrackh->size(); ++i) {
	      reco::PFRecTrackRef transientPFRecTrackRef(nucleartrackh,i);
	      reco::TrackBaseRef transientTrackBaseRef(transientPFRecTrackRef->trackRef());
	      if (constTrackBaseRef==transientTrackBaseRef){
		newPFRecTrackRef = transientPFRecTrackRef;
		break;
	      }
	    }
	  }
	  reco::TrackBaseRef newTrackBaseRef(newPFRecTrackRef->trackRef());
	  


	  bool bNew = true;
	  reco::PFBlockElement::TrackType blockType;

	  /// One need to cross check if those tracks was not already filled
	  /// from the conversion or V0 collections
	  for(IE iel = elements_.begin(); iel != elements_.end(); ++iel){
	    reco::TrackBaseRef elemTrackBaseRef((*iel)->trackRef());
	    if (newTrackBaseRef == elemTrackBaseRef){
	      trkFromDisplacedVertexElement = *iel;
	      bNew = false;
	      continue;
	    }
	  }


	  /// This is a new track not yet included into the elements collection
	  if (bNew) { 
	    


	    trkFromDisplacedVertexElement = new reco::PFBlockElementTrack(newPFRecTrackRef);
	    elements_.push_back( trkFromDisplacedVertexElement );
	  }

	  if (dispacedVertexRef->isIncomingTrack(newPFRecTrackRef)) 
	    blockType = reco::PFBlockElement::T_TO_DISP;
	  else if (dispacedVertexRef->isOutgoingTrack(newPFRecTrackRef)) 
	    blockType = reco::PFBlockElement::T_FROM_DISP;
	  else 
	    blockType = reco::PFBlockElement::DEFAULT;

	  /// Fill the displaced vertex ref
	  trkFromDisplacedVertexElement->setDisplacedVertexRef( dispacedVertexRef, blockType );


	  if (debug_){
	    std::cout << "PF Block Element from DisplacedTrackingVertex track New = " << bNew
		      << (*trkFromDisplacedVertexElement).trackRef().key() << std::endl;
	    std::cout << *trkFromDisplacedVertexElement << std::endl;
	  }
	
	
	}
      }
    }  

    if (debug_) std::cout << "" << std::endl;

  }

  /// -------------- Tracks ---------------------

  /// Mask the tracks in trackh collection already included from Conversions
  /// V0 and Displaced Vertices. Take care that all those collections come
  /// from the same "generalTracks" collection.

  if(trackh.isValid() ) {

    if (debug_) std::cout << "Tracks already in from Displaced Vertices " << std::endl;

    Mask trackMaskVertex;

    for(unsigned i=0;i<trackh->size(); ++i) {
      reco::PFRecTrackRef pfRefTrack( trackh,i );
      reco::TrackRef trackRef = pfRefTrack->trackRef();

      bool bMask = true;
      for(IE iel = elements_.begin(); iel != elements_.end(); ++iel){
	reco::TrackRef elemTrackRef = (*iel)->trackRef();
	if( trackRef == elemTrackRef ) {
	  if (debug_) std::cout << " " << trackRef.key();
	  bMask = false; continue;
	}
      }
    
      trackMaskVertex.push_back(bMask);
    }

    if (debug_) std::cout << "" << std::endl;

    if (debug_) std::cout << "Additionnal tracks from main collection " << std::endl;

    for(unsigned i=0;i<trackh->size(); ++i) {


      // this track has been disabled
      if( !trackMask.empty() && !trackMask[i] ) continue;
      
      reco::PFRecTrackRef ref( trackh,i );

      if (debug_) std::cout << " " << ref->trackRef().key();

      // Get the eventual muon associated to this track
      int muId_ = muAssocToTrack( ref->trackRef(), muonh );
      bool thisIsAPotentialMuon = false;
      if( muId_ != -1 ) {
	reco::MuonRef muonref( muonh, muId_ );
	thisIsAPotentialMuon = 
	  PFMuonAlgo::isLooseMuon(muonref) || 
	  PFMuonAlgo::isMuon(muonref);
      }
      // Reject bad tracks (except if identified as muon
      if( !thisIsAPotentialMuon && !goodPtResolution( ref->trackRef() ) ) continue;

      if (thisIsAPotentialMuon && debug_) std::cout << "Potential Muon P " <<  ref->trackRef()->p() 
						    << " pt " << ref->trackRef()->p() << std::endl; 



      reco::PFBlockElement* primaryElement = new reco::PFBlockElementTrack( ref );

      if( muId_ != -1 ) {
	// if a muon has been found
	reco::MuonRef muonref( muonh, muId_ );

	// If this track was already added to the collection, we just need to find the associated element and 
	// attach to it the reference
	if (!trackMaskVertex.empty() && !trackMaskVertex[i]){
	  reco::TrackRef primaryTrackRef = ref->trackRef();
	  for(IE iel = elements_.begin(); iel != elements_.end(); ++iel){
	    reco::TrackRef elemTrackRef = (*iel)->trackRef();
	    if( primaryTrackRef == elemTrackRef ) {
	      (*iel)->setMuonRef( muonref );
	      if (debug_) std::cout << "One of the tracks identified in displaced vertices collections was spotted as muon" <<std:: endl;
	    }
	  }
	} else primaryElement->setMuonRef( muonref );

      }

      if (!trackMaskVertex.empty() && !trackMaskVertex[i]) continue;

      
      // set track type T_FROM_GAMMA for pfrectracks associated to conv brems
      if(useConvBremPFRecTracks_) {
	if(convBremPFRecTracks.size() > 0.) {
	  for(unsigned int iconv = 0; iconv < convBremPFRecTracks.size(); ++iconv) {
	    if((*ref).trackRef() == (*convBremPFRecTracks[iconv]).trackRef()) {
	      bool value = true;
	      primaryElement->setTrackType(reco::PFBlockElement::T_FROM_GAMMACONV, value);
	    }
	  }
	}
      }
      elements_.push_back( primaryElement );
    }

    if (debug_) std::cout << " " << std::endl;

  }

 
  // -------------- GSF tracks and brems for Conversion Recovery ----------
   
  if(convbremgsftrackh.isValid() ) {
    
 
    const  reco::GsfPFRecTrackCollection ConvPFGsfProd = *(convbremgsftrackh.product());
    for(unsigned i=0;i<convbremgsftrackh->size(); ++i) {

      reco::GsfPFRecTrackRef refgsf(convbremgsftrackh,i );   
      
      if((refgsf).isNull()) continue;
      
      reco::PFBlockElement* gsfEl;
      
      const  std::vector<reco::PFTrajectoryPoint> 
	PfGsfPoint =  ConvPFGsfProd[i].trajectoryPoints();
      
      unsigned int c_gsf=0;
      bool PassTracker = false;
      bool GetPout = false;
      unsigned int IndexPout = -1;
      
      typedef std::vector<reco::PFTrajectoryPoint>::const_iterator IP;
      for(IP itPfGsfPoint =  PfGsfPoint.begin();  
	  itPfGsfPoint!= PfGsfPoint.end();++itPfGsfPoint) {
	
	if (itPfGsfPoint->isValid()){
	  int layGsfP = itPfGsfPoint->layer();
	  if (layGsfP == -1) PassTracker = true;
	  if (PassTracker && layGsfP > 0 && GetPout == false) {
	    IndexPout = c_gsf-1;
	    GetPout = true;
	  }
	  //const math::XYZTLorentzVector GsfMoment = itPfGsfPoint->momentum();
	  ++c_gsf;
	}
      }
      math::XYZTLorentzVector pin = PfGsfPoint[0].momentum();      
      math::XYZTLorentzVector pout = PfGsfPoint[IndexPout].momentum();
      
      
    
      gsfEl = new reco::PFBlockElementGsfTrack(refgsf, pin, pout);
      
      bool valuegsf = true;
      // IMPORTANT SET T_FROM_GAMMACONV trackType() FOR CONVERSIONS
      gsfEl->setTrackType(reco::PFBlockElement::T_FROM_GAMMACONV, valuegsf);

      

      elements_.push_back( gsfEl);
      std::vector<reco::PFBrem> pfbrem = refgsf->PFRecBrem();
      
      for (unsigned i2=0;i2<pfbrem.size(); ++i2) {
	const double DP = pfbrem[i2].DeltaP();
	const double SigmaDP =  pfbrem[i2].SigmaDeltaP(); 
	const unsigned int TrajP = pfbrem[i2].indTrajPoint();
	if(TrajP == 99) continue;

	reco::PFBlockElement* bremEl;
	bremEl = new reco::PFBlockElementBrem(refgsf,DP,SigmaDP,TrajP);
	elements_.push_back(bremEl);
	
      }
    }
  }

  
  // -------------- ECAL clusters ---------------------


  if(ecalh.isValid() ) {
    //  pfcSCVec_.resize(ecalh->size(),-1);
 
    bNoSuperclus_  = (superClusters_.size() == 0);
    if (!bNoSuperclus_) pfcSCVec_.resize(ecalh->size(),-1);


    for(unsigned i=0;i<ecalh->size(); ++i)  {

      // this ecal cluster has been disabled
      if( !ecalMask.empty() &&
          !ecalMask[i] ) continue;

      reco::PFClusterRef ref( ecalh,i );
      reco::PFBlockElement* te
        = new reco::PFBlockElementCluster( ref,
					   reco::PFBlockElement::ECAL);
      elements_.push_back( te );

      if (!bNoSuperclus_) {

	// Now mapping with Superclusters
	int scindex= ClusterClusterMapping::checkOverlap(*ref,superClusters_);

	if(scindex>=0) 	{
	  pfcSCVec_[ref.key()]=scindex;
	  scpfcRefs_[scindex].push_back(ref);
	}
      }

    }

    bNoSuperclus_ = (scpfcRefs_.size() == 0);

  }

  // -------------- HCAL clusters ---------------------

  if(hcalh.isValid() ) {
    
    for(unsigned i=0;i<hcalh->size(); ++i)  {
      
      // this hcal cluster has been disabled
      if( !hcalMask.empty() &&
          !hcalMask[i] ) continue;
      
      reco::PFClusterRef ref( hcalh,i );
      reco::PFBlockElement* th
        = new reco::PFBlockElementCluster( ref,
					   reco::PFBlockElement::HCAL );
      elements_.push_back( th );
    }
  }

  // -------------- HO clusters ---------------------

  if(useHO_ && hoh.isValid() ) {
    
    for(unsigned i=0;i<hoh->size(); ++i)  {
      
      // this hcal cluster has been disabled
      if( !hoMask.empty() &&
          !hoMask[i] ) continue;
      
      reco::PFClusterRef ref( hoh,i );
      reco::PFBlockElement* th
        = new reco::PFBlockElementCluster( ref,
					   reco::PFBlockElement::HO );
      elements_.push_back( th );
    }
  }

  // -------------- HFEM clusters ---------------------

  if(hfemh.isValid() ) {
    
    for(unsigned i=0;i<hfemh->size(); ++i)  {
      
      // this hfem cluster has been disabled
      if( !hfemMask.empty() &&
          !hfemMask[i] ) continue;
      
      reco::PFClusterRef ref( hfemh,i );
      reco::PFBlockElement* th
        = new reco::PFBlockElementCluster( ref,
					   reco::PFBlockElement::HFEM );
      elements_.push_back( th );
    }
  }


  // -------------- HFHAD clusters ---------------------

  if(hfhadh.isValid() ) {
    
    for(unsigned i=0;i<hfhadh->size(); ++i)  {
      
      // this hfhad cluster has been disabled
      if( !hfhadMask.empty() &&
          !hfhadMask[i] ) continue;
      
      reco::PFClusterRef ref( hfhadh,i );
      reco::PFBlockElement* th
        = new reco::PFBlockElementCluster( ref,
					   reco::PFBlockElement::HFHAD );
      elements_.push_back( th );
    }
  }




  // -------------- PS clusters ---------------------

  if(psh.isValid() ) {
    for(unsigned i=0;i<psh->size(); ++i)  {

      // this ps cluster has been disabled
      if( !psMask.empty() &&
          !psMask[i] ) continue;
      reco::PFBlockElement::Type type = reco::PFBlockElement::NONE;
      reco::PFClusterRef ref( psh,i );
      // two types of elements:  PS1 (V) and PS2 (H) 
      // depending on layer:  PS1 or PS2
      switch(ref->layer()){
      case PFLayer::PS1:
        type = reco::PFBlockElement::PS1;
        break;
      case PFLayer::PS2:
        type = reco::PFBlockElement::PS2;
        break;
      default:
        break;
      }
      reco::PFBlockElement* tp
        = new reco::PFBlockElementCluster( ref,
					   type );
      elements_.push_back( tp );
    }
  }


  // -------------- Loop over block elements ---------------------

  // Here we provide to all KDTree linkers the collections to link.
  // Glowinski & Gouzevitch
  
  for (std::list< reco::PFBlockElement* >::iterator it = elements_.begin();
       it != elements_.end(); ++it) {
    switch ((*it)->type()){
	
    case reco::PFBlockElement::TRACK:
      if (useKDTreeTrackEcalLinker_) {
	if ( (*it)->trackRefPF()->extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax ).isValid() )
	  TELinker_.insertTargetElt(*it);
	if ( (*it)->trackRefPF()->extrapolatedPoint( reco::PFTrajectoryPoint::HCALEntrance ).isValid() )
	  THLinker_.insertTargetElt(*it);
      }
      
      break;

    case reco::PFBlockElement::PS1:
    case reco::PFBlockElement::PS2:
      if (useKDTreeTrackEcalLinker_)
	PSELinker_.insertTargetElt(*it);
      break;

    case reco::PFBlockElement::HCAL:
      if (useKDTreeTrackEcalLinker_)
	THLinker_.insertFieldClusterElt(*it);
      break;

    case reco::PFBlockElement::HO: 
      if (useHO_ && useKDTreeTrackEcalLinker_) {
	// THLinker_.insertFieldClusterElt(*it);
      }
      break;

	
    case reco::PFBlockElement::ECAL:
      if (useKDTreeTrackEcalLinker_) {
	TELinker_.insertFieldClusterElt(*it);
	PSELinker_.insertFieldClusterElt(*it);
      }
      break;

    default:
      break;
    }
  }
}


template< template<typename> class T>
  void PFBlockAlgo::fillFromPhoton(const T<reco::PhotonCollection> & egh, unsigned isc, reco::PFBlockElementSuperCluster * pfbe) {
  reco::PhotonRef photonRef(egh,isc);
    pfbe->setTrackIso(photonRef->trkSumPtHollowConeDR04());
    pfbe->setEcalIso(photonRef->ecalRecHitSumEtConeDR04());
    pfbe->setHcalIso(photonRef->hcalTowerSumEtConeDR04());
    pfbe->setHoE(photonRef->hadronicOverEm());
    pfbe->setPhotonRef(photonRef);
    pfbe->setFromPhoton(true);
  }

#endif


