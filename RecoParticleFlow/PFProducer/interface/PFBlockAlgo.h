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
#include "DataFormats/ParticleFlowReco/interface/PFNuclearInteraction.h"
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

// #include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "RecoParticleFlow/PFProducer/interface/PFMuonAlgo.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFResolutionMap.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

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
		      std::vector<unsigned>& NHitCut );
  
  typedef std::vector<bool> Mask;

  /// set input collections of tracks and clusters
  template< template<typename> class T>
    void  setInput(const T<reco::PFRecTrackCollection>&    trackh,
		   const T<reco::GsfPFRecTrackCollection>&    gsftrackh,
		   const T<reco::MuonCollection>&    muonh,
		   const T<reco::PFNuclearInteractionCollection>&  nuclh,
		   const T<reco::PFConversionCollection>&  conv,
		   const T<reco::PFV0Collection>&  v0,
		   const T<reco::PFClusterCollection>&  ecalh,
		   const T<reco::PFClusterCollection>&  hcalh,
		   const T<reco::PFClusterCollection>&  hfemh,
		   const T<reco::PFClusterCollection>&  hfhadh,
		   const T<reco::PFClusterCollection>&  psh,
		   const Mask& trackMask = dummyMask_,
		   const Mask& gsftrackMask = dummyMask_,
		   const Mask& ecalMask = dummyMask_,
		   const Mask& hcalMask = dummyMask_,
		   const Mask& hfemMask = dummyMask_,		   
		   const Mask& hfhadMask = dummyMask_,
		   const Mask& psMask = dummyMask_  ); 
  
  ///COLIN: I think this is for particle flow at HLT...
  template< template<typename> class T >
    void setInput(const T<reco::PFRecTrackCollection>&    trackh,
		  const T<reco::PFClusterCollection>&  ecalh,
		  const T<reco::PFClusterCollection>&  hcalh,
		  const T<reco::PFClusterCollection>&  hfemh,
		  const T<reco::PFClusterCollection>&  hfhadh,
		  const T<reco::PFClusterCollection>&  psh,
		  const Mask& trackMask = dummyMask_,
		  const Mask& ecalMask = dummyMask_,
		  const Mask& hcalMask = dummyMask_,
		  const Mask& psMask = dummyMask_ ) {
    T<reco::GsfPFRecTrackCollection> gsftrackh;
    T<reco::MuonCollection> muonh;
    T<reco::PFNuclearInteractionCollection> nuclh;
    T<reco::PFConversionCollection> convh;
    T<reco::PFV0Collection> v0;
    setInput<T>( trackh, gsftrackh, muonh, nuclh, convh, v0, 
		 ecalh, hcalh, hfemh, hfhadh, psh, 
		 trackMask, ecalMask, hcalMask, psMask); 
  }
  
  ///COLIN: what is this setinput function for? can it be removed?
  template< template<typename> class T >
    void setInput(const T<reco::PFRecTrackCollection>&    trackh,
		  const T<reco::GsfPFRecTrackCollection>&    gsftrackh,
		  const T<reco::PFClusterCollection>&  ecalh,
		  const T<reco::PFClusterCollection>&  hcalh,
		  const T<reco::PFClusterCollection>&  psh,
		  const Mask& trackMask = dummyMask_,
		  const Mask& gsftrackMask = dummyMask_,
		  const Mask& ecalMask = dummyMask_,
		  const Mask& hcalMask = dummyMask_,
		  const Mask& psMask = dummyMask_ ) {
    T<reco::MuonCollection> muonh;
    T<reco::PFNuclearInteractionCollection> nuclh;
    T<reco::PFConversionCollection> convh;
    T<reco::PFV0Collection> v0;
    setInput<T>( trackh, gsftrackh, muonh, nuclh, convh, v0, ecalh, hcalh, psh, 
		 trackMask, gsftrackMask,ecalMask, hcalMask, psMask); 
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
  void checkNuclearLinks( reco::PFBlock& block ) const;
  
  ///COLIN: not used. Could be removed.
  /// Could also be implemented, to produce a graph of a block,
  /// Showing the connections between the elements
  void buildGraph(); 

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
			 
 /// tests association between a PS1 v cluster and a PS2 h cluster
 /// returns distance
 double testPS1AndPS2(const reco::PFCluster& ps1,
		      const reco::PFCluster& ps2) const;

  //tests association between a track and a cluster by rechit
 double testTrackAndClusterByRecHit( const reco::PFRecTrack& track, 
				     const reco::PFCluster& cluster,
				     bool isBrem = false) const;  

  //tests association between ECAL and PS clusters by rechit
 double testECALAndPSByRecHit( const reco::PFCluster& clusterECAL, 
			       const reco::PFCluster& clusterPS)  const;

  /// test association between HFEM and HFHAD, by rechit
  double testHFEMAndHFHADByRecHit( const reco::PFCluster& clusterHFEM, 
				   const reco::PFCluster& clusterHFHAD)  const;
  

  /// computes a chisquare
  double computeDist( double eta1, double phi1, 
					double eta2, double phi2 ) const;

  /// checks size of the masks with respect to the vectors
  /// they refer to. throws std::length_error if one of the
  /// masks has the wrong size
  void checkMaskSize( const reco::PFRecTrackCollection& tracks,
		      const reco::GsfPFRecTrackCollection& gsftracks,
		      const reco::PFClusterCollection&  ecals,
		      const reco::PFClusterCollection&  hcals,
		      const reco::PFClusterCollection&  hfems,
		      const reco::PFClusterCollection&  hfhads,
		      const reco::PFClusterCollection&  pss, 
		      const Mask& trackMask,
		      const Mask& gsftrackMask, 
		      const Mask& ecalMask, 
		      const Mask& hcalMask,
		      const Mask& hfemMask,
		      const Mask& hfhadMask,		      
		      const Mask& psMask ) const;

  /// open a resolution map
  // PFResolutionMap* openResolutionMap(const char* resMapName);

  /// check the Pt resolution 
  bool goodPtResolution( const reco::TrackRef& trackref);

  /// find index of the nuclear interaction associated to primTkRef.
  /// if no nuclear interaction have been found then return -1.
  int niAssocToTrack( const reco::TrackRef& primTkRef,
       const edm::Handle<reco::PFNuclearInteractionCollection>& good_ni) const;
  int niAssocToTrack( const reco::TrackRef& primTkRef,
       const edm::OrphanHandle<reco::PFNuclearInteractionCollection>& good_ni) const;
  /// find index of the muon associated to trackref.
  /// if no muon have been found then return -1.
  int muAssocToTrack( const reco::TrackRef& trackref,
       const edm::Handle<reco::MuonCollection>& muonh) const;
  int muAssocToTrack( const reco::TrackRef& trackref,
       const edm::OrphanHandle<reco::MuonCollection>& muonh) const;
  /// find index of the V0 track associated to trackref.
  /// if no V0 tracks have been found then return -1.
  int v0AssocToTrack( const reco::TrackRef& trackref,
       const edm::Handle<reco::PFV0Collection>& v0) const;
  int v0AssocToTrack( const reco::TrackRef& trackref,
       const edm::OrphanHandle<reco::PFV0Collection>& v0) const;

  // fill secondary tracks of a nuclear interaction
  void fillSecondaries( const reco::PFNuclearInteractionRef& nuclref );

  double testLinkByVertex(const reco::PFBlockElement* elt1,
			  const reco::PFBlockElement* elt2) const;

  std::auto_ptr< reco::PFBlockCollection >    blocks_;
  
  /// actually, particles will be created by a separate producer
  // std::vector< reco::PFCandidate >   particles_;

  // the test elements will be transferred to the blocks
  std::list< reco::PFBlockElement* >     elements_;

  static const Mask                      dummyMask_;

  /// DPt/Pt cut for creating atrack element
  std::vector<double> DPtovPtCut_;
  
  /// Number of layers crossed cut for creating atrack element
  std::vector<unsigned> NHitCut_;
  
  /// PS strip resolution
  double resPSpitch_;
  
  /// PS resolution along strip
  double resPSlength_;
 
  /// if true, debug printouts activated
  bool   debug_;
  
  friend std::ostream& operator<<(std::ostream&, const PFBlockAlgo&);

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
		      const T<reco::MuonCollection>&    muonh,
                      const T<reco::PFNuclearInteractionCollection>&  nuclh,
		      const T<reco::PFConversionCollection>&  convh,
		      const T<reco::PFV0Collection>&  v0,
                      const T<reco::PFClusterCollection>&  ecalh,
                      const T<reco::PFClusterCollection>&  hcalh,
                      const T<reco::PFClusterCollection>&  hfemh,
                      const T<reco::PFClusterCollection>&  hfhadh,
                      const T<reco::PFClusterCollection>&  psh,
                      const Mask& trackMask,
		      const Mask& gsftrackMask,
                      const Mask& ecalMask,
                      const Mask& hcalMask,
                      const Mask& hfemMask,
                      const Mask& hfhadMask,
                      const Mask& psMask  ) {


  checkMaskSize( *trackh,
		 *gsftrackh,
                 *ecalh,
                 *hcalh,
		 *hfemh,
		 *hfhadh,
                 *psh,
                 trackMask,
		 gsftrackMask,
                 ecalMask,
                 hcalMask,
		 hfemMask,
		 hfhadMask,
                 psMask  );


  // -------------- conversions ---------------------

  if(convh.isValid() ) {
    reco::PFBlockElement* trkFromConversionElement;
    for(unsigned i=0;i<convh->size(); i++) {
      reco::PFConversionRef convRef(convh,i);

      unsigned int trackSize=(convRef->pfTracks()).size();
      if ( convRef->pfTracks().size() < 2) continue;
      for(unsigned iTk=0;iTk<trackSize; iTk++) {
	if (debug_) 
	  std::cout<<" PFBlockAlgo setInput building element for track charge" 
		   <<convRef->pfTracks()[iTk]->charge() 
		   <<" pt "<<convRef->pfTracks()[iTk]->trackRef()->pt()
		   <<std::endl;
       if (debug_) 
	  std::cout<<" # of traj points "
		   <<convRef->pfTracks()[iTk]->nTrajectoryPoints() 
		   <<" # of traj measurements "
		   <<convRef->pfTracks()[iTk]->nTrajectoryMeasurements()
		   <<std::endl;
	
       for ( unsigned int iP=0; 
	     iP<convRef->pfTracks()[iTk]->nTrajectoryPoints(); iP++) {
	 if (debug_) 
	   std::cout<<" Trajectory point "
		    <<iP<<" x,y,z "
		    <<convRef->pfTracks()[iTk]->trajectoryPoint(iP).position() 
		    <<" r, eta, phi " 
		    <<convRef->pfTracks()[iTk]->trajectoryPoint(iP).positionREP()<<std::endl;
	 
       }

       trkFromConversionElement = new reco::PFBlockElementTrack(convRef->pfTracks()[iTk]);
       trkFromConversionElement->setConversionRef( convRef->originalConversion(), reco::PFBlockElement::T_FROM_GAMMACONV);
       elements_.push_back( trkFromConversionElement );
       
      }     
    }  
  }
  
  // -------------- tracks ---------------------

  if(trackh.isValid() ) {
    for(unsigned i=0;i<trackh->size(); i++) {

      // this track has been disabled
      if( !trackMask.empty() &&
          !trackMask[i] ) continue;
      reco::PFRecTrackRef ref( trackh,i );

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

      // get the eventual nuclear interaction associated to this track 
      int niId_ = niAssocToTrack( ref->trackRef(), nuclh );

      // get the eventual v0Track associated to this track
      int v0Id_ = v0AssocToTrack( ref->trackRef(), v0 );

      reco::PFBlockElement* primaryElement = new reco::PFBlockElementTrack( ref );

      if( niId_ != -1 ) {
	  // if a nuclear interaction has been found 
          reco::PFNuclearInteractionRef ni_(nuclh, niId_);
          primaryElement->setNuclearRef( ni_->nuclInterRef(), 
					 reco::PFBlockElement::T_TO_NUCL );
          fillSecondaries( ni_ );
      }
      if( muId_ != -1 ) {
          // if a muon has been found
          reco::MuonRef muonref( muonh, muId_ );
          primaryElement->setMuonRef( muonref );
      }
      if( v0Id_ != -1 ) {
	// if a V0 has been found
	reco::PFV0Ref v0ref( v0, v0Id_ );
	primaryElement->setV0Ref( v0ref->originalV0(),
				  reco::PFBlockElement::T_FROM_V0 );
      }

      elements_.push_back( primaryElement );
    }
  }

  // -------------- GSF tracks and brems ---------------------

  if(gsftrackh.isValid() ) {
    const  reco::GsfPFRecTrackCollection PFGsfProd = *(gsftrackh.product());
    for(unsigned i=0;i<gsftrackh->size(); i++) {
      if( !gsftrackMask.empty() &&
          !gsftrackMask[i] ) continue;
      reco::GsfPFRecTrackRef refgsf(gsftrackh,i );   
   
      if((refgsf).isNull()) continue;

      reco::PFBlockElement* gsfEl;
        
      const  std::vector<reco::PFTrajectoryPoint> 
	PfGsfPoint =  PFGsfProd[i].trajectoryPoints();
  
      uint c_gsf=0;
      bool PassTracker = false;
      bool GetPout = false;
      uint IndexPout = -1;
      
      typedef std::vector<reco::PFTrajectoryPoint>::const_iterator IP;
      for(IP itPfGsfPoint =  PfGsfPoint.begin();  
	  itPfGsfPoint!= PfGsfPoint.end();itPfGsfPoint++) {
	
	if (itPfGsfPoint->isValid()){
	  int layGsfP = itPfGsfPoint->layer();
	  if (layGsfP == -1) PassTracker = true;
	  if (PassTracker && layGsfP > 0 && GetPout == false) {
	    IndexPout = c_gsf-1;
	    GetPout = true;
	  }
	  //const math::XYZTLorentzVector GsfMoment = itPfGsfPoint->momentum();
	  c_gsf++;
	}
      }
      math::XYZTLorentzVector pin = PfGsfPoint[0].momentum();      
      math::XYZTLorentzVector pout = PfGsfPoint[IndexPout].momentum();
      
      
      gsfEl = new reco::PFBlockElementGsfTrack(refgsf, pin, pout);
      
      elements_.push_back( gsfEl);

      std::vector<reco::PFBrem> pfbrem = refgsf->PFRecBrem();
      
      for (unsigned i2=0;i2<pfbrem.size(); i2++) {
	const double DP = pfbrem[i2].DeltaP();
	const double SigmaDP =  pfbrem[i2].SigmaDeltaP(); 
	const uint TrajP = pfbrem[i2].indTrajPoint();
	if(TrajP == 99) continue;

	reco::PFBlockElement* bremEl;
	bremEl = new reco::PFBlockElementBrem(refgsf,DP,SigmaDP,TrajP);
	elements_.push_back(bremEl);
	
      }
    }
  }


  
  // -------------- ECAL clusters ---------------------


  if(ecalh.isValid() ) {
    for(unsigned i=0;i<ecalh->size(); i++)  {

      // this ecal cluster has been disabled
      if( !ecalMask.empty() &&
          !ecalMask[i] ) continue;

      reco::PFClusterRef ref( ecalh,i );
      reco::PFBlockElement* te
        = new reco::PFBlockElementCluster( ref,
					   reco::PFBlockElement::ECAL);
      elements_.push_back( te );
    }
  }

  // -------------- HCAL clusters ---------------------

  if(hcalh.isValid() ) {
    
    for(unsigned i=0;i<hcalh->size(); i++)  {
      
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


  // -------------- HFEM clusters ---------------------

  if(hfemh.isValid() ) {
    
    for(unsigned i=0;i<hfemh->size(); i++)  {
      
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
    
    for(unsigned i=0;i<hfhadh->size(); i++)  {
      
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
    for(unsigned i=0;i<psh->size(); i++)  {

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
}



#endif


