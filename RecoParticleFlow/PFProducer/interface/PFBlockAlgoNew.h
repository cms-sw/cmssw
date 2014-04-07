#ifndef RecoParticleFlow_PFProducer_PFBlockAlgoNew_h
#define RecoParticleFlow_PFProducer_PFBlockAlgoNew_h 

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

#include "RecoParticleFlow/PFProducer/interface/BlockElementLinkerBase.h"
#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"

#include <map>
#include <unordered_map>

/// \brief Particle Flow Algorithm
/*!
  \author Colin Bernet
  \date January 2006
*/

class PFBlockAlgoNew {

 public:

  PFBlockAlgoNew();

  ~PFBlockAlgoNew();
  

  void setParameters( std::vector<double>& DPtovPtCut, 
		      std::vector<unsigned>& NHitCut,
		      bool useConvBremPFRecTracks,
		      bool useIterTracking,
		      int nuclearInteractionsPurity,
		      bool useEGPhotons,
		      std::vector<double> & photonSelectionCuts,
		      bool useSuperClusters,
                      bool superClusterMatchByRef
		      );
  
  void setLinkers(const std::vector<edm::ParameterSet>&);

  void setImporters(const std::vector<edm::ParameterSet>&,
		    edm::ConsumesCollector&);

  // Glowinski & Gouzevitch
  void setUseOptimization(bool useKDTreeTrackEcalLinker);
  // ! Glowinski & Gouzevitch

  typedef std::vector<bool> Mask;

  /// set input collections of tracks and clusters
  template< template<typename> class T>
    void  setInput(const edm::Event& evt,
		   const T<reco::PFRecTrackCollection>&    trackh,
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
                   const T<edm::ValueMap<reco::CaloClusterPtr> >& pfclusterassoc,
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
    void setInput(const edm::Event& evt,
		  const T<reco::PFRecTrackCollection>&    trackh,
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
    T<edm::ValueMap<reco::CaloClusterPtr> > pfclusterassoc;
    setInput<T>( evt, trackh, gsftrackh, convbremgsftrackh, muonh, nuclearh, nucleartrackh, convh, v0, 
		 ecalh, hcalh, hoh, hfemh, hfhadh, psh, phh, scebh, sceeh, pfclusterassoc,
		 trackMask, ecalMask, hcalMask, hoMask, psMask); 
  }
  
  ///COLIN: what is this setinput function for? can it be removed?
  template< template<typename> class T >
    void setInput(const edm::Event& evt,
		  const T<reco::PFRecTrackCollection>&    trackh,
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
    setInput<T>( evt, trackh, gsftrackh, convbremgsftrackh, muonh, nuclearh, nucleartrackh, convh, v0, ecalh, hcalh, hoh, psh, egphh,
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

  // the element list should **always** be a list of (smart) pointers
  typedef std::vector<std::unique_ptr<reco::PFBlockElement> > ElementList;
  /// define these in *Fwd files in DataFormats/ParticleFlowReco?
  typedef ElementList::iterator IE;
  typedef ElementList::const_iterator IEC;  
  typedef reco::PFBlockCollection::const_iterator IBC;
  
  void setHOTag(bool ho) { useHO_ = ho;}

 private:
  typedef std::unique_ptr<const BlockElementLinkerBase> LinkTestPtr;
  typedef std::unique_ptr<const BlockElementImporterBase> ImporterPtr;


  // flattened version of topological
  // association of block elements
  IE associate( ElementList& elems,
		std::vector<PFBlockLink>& links,
		reco::PFBlock& );

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
  inline bool linkPrefilter(const reco::PFBlockElement* last, 
			    const reco::PFBlockElement* next) const;

  /// check whether 2 elements are linked. Returns distance and linktype
  void link( const reco::PFBlockElement* el1, 
	     const reco::PFBlockElement* el2, 
	     PFBlockLink::Type& linktype, 
	     reco::PFBlock::LinkTest& linktest,
	     double& dist) const;
  
  /// tests association between an ECAL and an HCAL cluster
  /// \returns distance
  double testECALAndHCAL(const reco::PFCluster& ecal, 
			 const reco::PFCluster& hcal) const;

  /// tests association between an HCAL and an HO cluster
  /// \returns distance
  double testHCALAndHO(const reco::PFCluster& hcal, 
		       const reco::PFCluster& ho) const;
  /// test association by Supercluster between two ECAL
  double testLinkBySuperCluster(const reco::PFClusterRef & elt1,
				const reco::PFClusterRef & elt2) const;   

  /// test association between SuperClusters and ECAL
  double testSuperClusterPFCluster(const reco::SuperClusterRef & sct1,
				   const reco::PFClusterRef & elt2) const;

  double testLinkByVertex(const reco::PFBlockElement* elt1,
			  const reco::PFBlockElement* elt2) const;

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
  ElementList     elements_;

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

  //flag to control whether superclusters are matched to ecal pfclusters by reference instead of det id overlap
  //(more robust, but requires that the SuperClusters were produced from PFClusters together with the 
  //appropriate ValueMap
  bool superClusterMatchByRef_;
  
  const edm::ValueMap<reco::CaloClusterPtr> *pfclusterassoc_;
  
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
  
  friend std::ostream& operator<<(std::ostream&, const PFBlockAlgoNew&);

  // Create links between tracks or HCAL clusters, and HO clusters
  bool useHO_;

  std::vector<ImporterPtr> _importers;

  const std::unordered_map<std::string,reco::PFBlockElement::Type> 
    _elementTypes;
  std::vector<LinkTestPtr> _linkTests;
};

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"

template< template<typename> class T >
void
PFBlockAlgoNew::setInput(const edm::Event& evt,
			 const T<reco::PFRecTrackCollection>&    trackh,
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
                      const T<edm::ValueMap<reco::CaloClusterPtr> >& pfclusterassoc,
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

  if (superClusterMatchByRef_) {
    pfclusterassoc_ = pfclusterassoc.product();
  }
  
  // import block elements as defined in python configuration
  for( const auto& importer : _importers ) {
    importer->importToBlock(evt,elements_);
  }

  // sort to regularize access patterns
  std::sort(elements_.begin(),elements_.end(),
	    [](const ElementList::value_type& a,
	       const ElementList::value_type& b) {
	      return a->type() < b->type();
	    });

  // -------------- Loop over block elements ---------------------

  // Here we provide to all KDTree linkers the collections to link.
  // Glowinski & Gouzevitch
  
  for (ElementList::iterator it = elements_.begin();
       it != elements_.end(); ++it) {
    switch ((*it)->type()){
	
    case reco::PFBlockElement::TRACK:
      if (useKDTreeTrackEcalLinker_) {
	if ( (*it)->trackRefPF()->extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax ).isValid() )
	  TELinker_.insertTargetElt(it->get());
	if ( (*it)->trackRefPF()->extrapolatedPoint( reco::PFTrajectoryPoint::HCALEntrance ).isValid() )
	  THLinker_.insertTargetElt(it->get());
      }
      
      break;

    case reco::PFBlockElement::PS1:
    case reco::PFBlockElement::PS2:
      if (useKDTreeTrackEcalLinker_)
	PSELinker_.insertTargetElt(it->get());
      break;

    case reco::PFBlockElement::HCAL:
      if (useKDTreeTrackEcalLinker_)
	THLinker_.insertFieldClusterElt(it->get());
      break;

    case reco::PFBlockElement::HO: 
      if (useHO_ && useKDTreeTrackEcalLinker_) {
	// THLinker_.insertFieldClusterElt(*it);
      }
      break;

	
    case reco::PFBlockElement::ECAL:
      if (useKDTreeTrackEcalLinker_) {
	TELinker_.insertFieldClusterElt(it->get());
	PSELinker_.insertFieldClusterElt(it->get());
      }
      break;

    default:
      break;
    }
  }
}


template< template<typename> class T>
  void PFBlockAlgoNew::fillFromPhoton(const T<reco::PhotonCollection> & egh, unsigned isc, reco::PFBlockElementSuperCluster * pfbe) {
  reco::PhotonRef photonRef(egh,isc);
    pfbe->setTrackIso(photonRef->trkSumPtHollowConeDR04());
    pfbe->setEcalIso(photonRef->ecalRecHitSumEtConeDR04());
    pfbe->setHcalIso(photonRef->hcalTowerSumEtConeDR04());
    pfbe->setHoE(photonRef->hadronicOverEm());
    pfbe->setPhotonRef(photonRef);
    pfbe->setFromPhoton(true);
  }

#endif


