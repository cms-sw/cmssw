#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgoNew.h"
#include "RecoParticleFlow/PFProducer/interface/Utils.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h" // gouzevitch

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include <stdexcept>
#include "TMath.h"

using namespace std;
using namespace reco;

#define INIT_ENTRY(name) {#name,name}


//for debug only 
//#define PFLOW_DEBUG

const PFBlockAlgoNew::Mask PFBlockAlgoNew::dummyMask_;

PFBlockAlgoNew::PFBlockAlgoNew() : 
  blocks_( new reco::PFBlockCollection ),
  DPtovPtCut_(std::vector<double>(4,static_cast<double>(999.))),
  NHitCut_(std::vector<unsigned int>(4,static_cast<unsigned>(0))), 
  useIterTracking_(true),
  photonSelector_(0),
  debug_(false),
  _elementTypes( {
        INIT_ENTRY(PFBlockElement::TRACK),
	INIT_ENTRY(PFBlockElement::PS1),
	INIT_ENTRY(PFBlockElement::PS2),
	INIT_ENTRY(PFBlockElement::ECAL),
	INIT_ENTRY(PFBlockElement::HCAL),
	INIT_ENTRY(PFBlockElement::GSF),
	INIT_ENTRY(PFBlockElement::BREM),
	INIT_ENTRY(PFBlockElement::HFEM),
	INIT_ENTRY(PFBlockElement::HFHAD),
	INIT_ENTRY(PFBlockElement::SC),
	INIT_ENTRY(PFBlockElement::HO) 
	  } ) {}


void PFBlockAlgoNew::setLinkers(const std::vector<edm::ParameterSet>& confs) {
   constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  _linkTests.resize(rowsize*rowsize);
  const std::string prefix("PFBlockElement::");
  for( const auto& conf : confs ) {
    const std::string& linkerName = 
      conf.getParameter<std::string>("linkerName");
    const std::string&  linkTypeStr =
      conf.getParameter<std::string>("linkType");
    size_t split = linkTypeStr.find(':');
    if( split == std::string::npos ) {
      throw cms::Exception("MalformedLinkType")
	<< "\"" << linkTypeStr << "\" is not a valid link type definition."
	<< " This string should have the form \"linkFrom:linkTo\"";
    }
    std::string link1(prefix+linkTypeStr.substr(0,split));
    std::string link2(prefix+linkTypeStr.substr(split+1,std::string::npos));
    if( !(_elementTypes.count(link1) && _elementTypes.count(link2) ) ) {
      throw cms::Exception("InvalidBlockElementType")
	<< "One of \"" << link1 << "\" or \"" << link2 
	<< "\" are invalid block element types!";
    }
    const PFBlockElement::Type type1 = _elementTypes.at(link1);
    const PFBlockElement::Type type2 = _elementTypes.at(link2);    
    const unsigned index = rowsize*std::max(type1,type2)+std::min(type1,type2);
    const BlockElementLinkerBase * linker =
      BlockElementLinkerFactory::get()->create(linkerName,conf);
    _linkTests[index].reset(linker);
  }
}

void PFBlockAlgoNew::setParameters( std::vector<double>& DPtovPtCut,
				 std::vector<unsigned int>& NHitCut,
				 bool useConvBremPFRecTracks,
				 bool useIterTracking,
				 int nuclearInteractionsPurity,
				 bool useEGPhotons,
				 std::vector<double>& photonSelectionCuts,
				 bool useSuperClusters,
                                 bool superClusterMatchByRef
                               ) {
  
  DPtovPtCut_    = DPtovPtCut;
  NHitCut_       = NHitCut;
  useIterTracking_ = useIterTracking;
  useConvBremPFRecTracks_ = useConvBremPFRecTracks;
  nuclearInteractionsPurity_ = nuclearInteractionsPurity;
  useEGPhotons_ = useEGPhotons;
  // Pt cut; Track iso (constant + slope), Ecal iso (constant + slope), HCAL iso (constant+slope), H/E
  if(useEGPhotons_)
    photonSelector_ = new PhotonSelectorAlgo(photonSelectionCuts[0],   
					     photonSelectionCuts[1], photonSelectionCuts[2],   
					     photonSelectionCuts[3], photonSelectionCuts[4],    
					     photonSelectionCuts[5], photonSelectionCuts[6],    
					     photonSelectionCuts[7],
					     photonSelectionCuts[8],
					     photonSelectionCuts[9],
					     photonSelectionCuts[10]
					     );


  useSuperClusters_ = useSuperClusters;
  superClusterMatchByRef_ = superClusterMatchByRef;
}

// Glowinski & Gouzevitch
void PFBlockAlgoNew::setUseOptimization(bool useKDTreeTrackEcalLinker)
{
  useKDTreeTrackEcalLinker_ = useKDTreeTrackEcalLinker;
}
// !Glowinski & Gouzevitch


PFBlockAlgoNew::~PFBlockAlgoNew() {

#ifdef PFLOW_DEBUG
  if(debug_)
    cout<<"~PFBlockAlgoNew - number of remaining elements: "
	<<elements_.size()<<endl;
#endif

  if(photonSelector_) delete photonSelector_;

}

void 
PFBlockAlgoNew::findBlocks() {

  // Glowinski & Gouzevitch
  if (useKDTreeTrackEcalLinker_) {
    TELinker_.process();
    THLinker_.process();
    PSELinker_.process();
  }
  // !Glowinski & Gouzevitch

  // the blocks have not been passed to the event, and need to be cleared
  if(blocks_.get() )blocks_->clear();
  else 
    blocks_.reset( new reco::PFBlockCollection );

  blocks_->reserve(elements_.size());
  for(IE ie = elements_.begin(); 
      ie != elements_.end();) {
    
#ifdef PFLOW_DEBUG
    if(debug_) {
      cout<<" PFBlockAlgoNew::findBlocks() ----------------------"<<endl;
      cout<<" element "<<**ie<<endl;
      cout<<" creating new block"<<endl;
    }
#endif

    blocks_->push_back( reco::PFBlock() );
    
    std::vector< PFBlockLink > links;
    
    ie = associate(elements_, links, blocks_->back());    
    
    packLinks( blocks_->back(), links );
  }
  std::cout << "(new) Found " << blocks_->size() << " PFBlocks!" << std::endl;
}

// start from first element in elements_
// partition elements until block grows no further
// return the start of the new block
PFBlockAlgoNew::IE
PFBlockAlgoNew::associate( PFBlockAlgoNew::ElementList& elems,
			   std::vector<PFBlockLink>& links,
			   reco::PFBlock& block) {
  if( elems.size() == 0 ) return elems.begin();
  ElementList::iterator scan_upper(elems.begin()), search_lower(elems.begin()), 
    scan_lower(elems.begin());
  ++scan_upper; ++search_lower;
  double dist = -1;
  PFBlockLink::Type linktype = PFBlockLink::NONE;
  PFBlock::LinkTest linktest = PFBlock::LINKTEST_RECHIT;
  block.addElement(scan_lower->get()); // seed the block
  do {     
    scan_upper = search_lower;
    for( auto comp = scan_lower; comp != scan_upper; ++comp ) {
      search_lower = // group everything that's linked to the current element
	std::partition(search_lower,elems.end(),
		       [&](ElementList::value_type& a){
			 // check if link is somehow possible
			 if( !linkPrefilter(comp->get(), a.get()) ) {
			   return false;
			 }
			 dist = -1.0;
			 linktype = PFBlockLink::NONE;
			 linktest = PFBlock::LINKTEST_RECHIT;
			 // compute linking info
			 link( comp->get(), a.get(), 
			       linktype, linktest, dist ); 
			 if( dist >= -0.5 ) { 
			   block.addElement( a.get() ); 
			   links.emplace_back( linktype, linktest, dist,
					       (*comp)->index(), 
					       a->index() );
			   return true;
			 } else {
			   return false;
			 }
		       });
    }
    scan_lower = scan_upper;      
  } while( search_lower != scan_upper );  
  return elems.erase(elems.begin(),scan_upper);
}

void 
PFBlockAlgoNew::packLinks( reco::PFBlock& block, 
			   const std::vector<PFBlockLink>& links ) const {
  
  
  const edm::OwnVector< reco::PFBlockElement >& els = block.elements();
  
  block.bookLinkData();
  unsigned elsize = els.size();
  unsigned ilStart = 0;
  //First Loop: update all link data
  for( unsigned i1=0; i1<elsize; ++i1 ) {
    for( unsigned i2=0; i2<i1; ++i2 ) {
      
      // no reflexive link
      //if( i1==i2 ) continue;
      
      double dist = -1;
      
      bool linked = false;
      PFBlock::LinkTest linktest 
	= PFBlock::LINKTEST_RECHIT; 

      // are these elements already linked ?
      // this can be optimized
      unsigned linksize = links.size();
      for( unsigned il = ilStart; il<linksize; ++il ) {
	// The following three lines exploits the increasing-element2 ordering of links.
	if ( links[il].element2() < i1 ) ilStart = il;
	if ( links[il].element2() > i1 ) break;
	if( (links[il].element1() == i2 &&
             links[il].element2() == i1) ) {  // yes
	  
	  dist = links[il].dist();
	  linked = true;

	  //modif-beg	  
	  //retrieve type of test used to get distance
	  linktest = links[il].test();
#ifdef PFLOW_DEBUG
	  if( debug_ )
	    cout << "Reading link vector: linktest used=" 
		 << linktest 
		 << " distance = " << dist 
		 << endl; 
#endif
	  //modif-end
	  
	  break;
	} 
      }
      
      if(!linked) {
	PFBlockLink::Type linktype = PFBlockLink::NONE;
	bool bTestLink = linkPrefilter(&els[i1], &els[i2]);
	if (bTestLink) link( & els[i1], & els[i2], linktype, linktest, dist);
      }

      //loading link data according to link test used: RECHIT 
      //block.setLink( i1, i2, chi2, block.linkData() );
#ifdef PFLOW_DEBUG
      if( debug_ )
	cout << "Setting link between elements " << i1 << " and " << i2
	     << " of dist =" << dist << " computed from link test "
	     << linktest << endl;
#endif
      block.setLink( i1, i2, dist, block.linkData(), linktest );
    }
  }

}



void 
PFBlockAlgoNew::buildGraph() {
  // loop on all blocks and create a big graph
}



void 
PFBlockAlgoNew::link( const reco::PFBlockElement* el1, 
		      const reco::PFBlockElement* el2, 
		      PFBlockLink::Type& linktype, 
		      reco::PFBlock::LinkTest& linktest,
		      double& dist) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  // ACHTUNG!!!! If you introduce new links check that they are not 
  // desabled in linkPrefilter!!!!


  dist=-1.0;
  linktest = PFBlock::LINKTEST_RECHIT; //rechit by default 

  PFBlockElement::Type type1 = el1->type();
  PFBlockElement::Type type2 = el2->type();
  const unsigned index = rowsize*std::max(type1,type2) + std::min(type1,type2);

  /*
  linktype = static_cast<PFBlockLink::Type>((1<<(type1-1))|(1<< (type2-1)));
  */
  if(debug_ ) { 
    std::cout << " PFBlockAlgoNew links type1 " << type1 
	      << " type2 " << type2 << std::endl;
  }

  if( index < _linkTests.size() && _linkTests[index] ) {
    dist = _linkTests[index]->testLink(el1,el2);
  }  
}

double
PFBlockAlgoNew::testECALAndHCAL(const PFCluster& ecal, 
			     const PFCluster& hcal)  const {
  
  //   cout<<"entering testECALAndHCAL"<<endl;
  
  double dist = fabs(ecal.positionREP().Eta()) > 2.5 ?
    LinkByRecHit::computeDist( ecal.positionREP().Eta(),
			       ecal.positionREP().Phi(), 
			       hcal.positionREP().Eta(), 
			       hcal.positionREP().Phi() )
    : 
    -1.;

#ifdef PFLOW_DEBUG
  if(debug_) cout<<"testECALAndHCAL "<< dist <<" "<<endl;
  if(debug_){
    cout<<" ecaleta " << ecal.positionREP().Eta()
	<<" ecalphi " << ecal.positionREP().Phi()
	<<" hcaleta " << hcal.positionREP().Eta()
	<<" hcalphi " << hcal.positionREP().Phi()
  }
#endif

  if ( dist < 0.2 ) return dist; 
 
  // Need to implement a link by RecHit
  return -1.;
}

double
PFBlockAlgoNew::testHCALAndHO(const PFCluster& hcal, 
			     const PFCluster& ho)  const {
  
  double dist = fabs(hcal.positionREP().Eta()) < 1.5 ?
    LinkByRecHit::computeDist( hcal.positionREP().Eta(),
			       hcal.positionREP().Phi(), 
			       ho.positionREP().Eta(), 
			       ho.positionREP().Phi() )
    : 
    -1.;

#ifdef PFLOW_DEBUG
  if(debug_) cout<<"testHCALAndHO "<< dist <<" "<<endl;
  if(debug_){
    cout<<" hcaleta " << hcal.positionREP().Eta()
	<<" hcalphi " << hcal.positionREP().Phi()
	<<" hoeta " << ho.positionREP().Eta()
	<<" hophi " << ho.positionREP().Phi()
	<<" dist " << dist<<endl;
  }
#endif

  if ( dist < 0.20 ) return dist; 
 
  // Need to implement a link by RecHit
  return -1.;
}



double
PFBlockAlgoNew::testLinkBySuperCluster(const PFClusterRef& ecal1, 
				    const PFClusterRef& ecal2)  const {
  
  //  cout<<"entering testECALAndECAL "<< pfcRefSCMap_.size() << endl;
  
  double dist = -1;
  
  // the first one is not in any super cluster
  int testindex=pfcSCVec_[ecal1.key()];
  if(testindex == -1.) return dist;
  //  if(itcheck==pfcRefSCMap_.end()) return dist;
  // now retrieve the of PFclusters in this super cluster  

  const std::vector<reco::PFClusterRef> & thePFClusters(scpfcRefs_[testindex]);
  
  unsigned npf=thePFClusters.size();
  for(unsigned i=0;i<npf;++i)
    {
      if(thePFClusters[i]==ecal2) // yes they are in the same SC 
	{
	  dist=LinkByRecHit::computeDist( ecal1->positionREP().Eta(),
					  ecal1->positionREP().Phi(), 
					  ecal2->positionREP().Eta(), 
					  ecal2->positionREP().Phi() );
//	  std::cout << " DETA " << fabs(ecal1->positionREP().Eta()-ecal2->positionREP().Eta()) << std::endl;
//	  if(fabs(ecal1->positionREP().Eta()-ecal2->positionREP().Eta())>0.2)
//	    {
//	      std::cout <<  " Super Cluster " <<  *(superClusters_[testindex]) << std::endl;
//	      std::cout <<  " Cluster1 " <<  *ecal1 << std::endl;
//	      std::cout <<  " Cluster2 " <<  *ecal2 << std::endl;
//	      ClusterClusterMapping::checkOverlap(*ecal1,superClusters_,0.01,true);
//	      ClusterClusterMapping::checkOverlap(*ecal2,superClusters_,0.01,true);
//	    }
	  return dist;
	}
    }
  return dist;
}


double
PFBlockAlgoNew::testSuperClusterPFCluster(const SuperClusterRef& ecal1, 
				       const PFClusterRef& ecal2)  const {
  
  //  cout<<"entering testECALAndECAL "<< pfcRefSCMap_.size() << endl;
  
  double dist = -1;
  
  if(superClusterMatchByRef_) {
    //loop over supercluster CaloClusters, look up PFCluster ptrs in value map, and match by ref
    for (reco::CaloCluster_iterator caloclus = ecal1->clustersBegin(); caloclus!=ecal1->clustersEnd(); ++caloclus) {
      bool overlap = ClusterClusterMapping::overlap(ecal2, *ecal1,*pfclusterassoc_);
      if (overlap) dist = 0.001;
    }
  }
  else {
    bool overlap=ClusterClusterMapping::overlap(*ecal1,*ecal2);
    
    if(overlap) 	{
      dist=LinkByRecHit::computeDist( ecal1->position().eta(),
                                      ecal1->position().phi(), 
                                      ecal2->positionREP().Eta(), 
                                      ecal2->positionREP().Phi() );
    }
  }
  return dist;
}

double
PFBlockAlgoNew::testLinkByVertex( const reco::PFBlockElement* elt1, 
			       const reco::PFBlockElement* elt2) const {

  //  cout << "Test link by vertex between" << endl << *elt1 << endl << " and " << endl << *elt2 << endl;

  double result=-1.;

  reco::PFBlockElement::TrackType T_TO_DISP = reco::PFBlockElement::T_TO_DISP;
  reco::PFBlockElement::TrackType T_FROM_DISP = reco::PFBlockElement::T_FROM_DISP;
  PFDisplacedTrackerVertexRef ni1_TO_DISP = elt1->displacedVertexRef(T_TO_DISP);
  PFDisplacedTrackerVertexRef ni2_TO_DISP = elt2->displacedVertexRef(T_TO_DISP);
  PFDisplacedTrackerVertexRef ni1_FROM_DISP = elt1->displacedVertexRef(T_FROM_DISP);
  PFDisplacedTrackerVertexRef ni2_FROM_DISP = elt2->displacedVertexRef(T_FROM_DISP);
  
  if( ni1_TO_DISP.isNonnull() && ni2_FROM_DISP.isNonnull())
    if( ni1_TO_DISP == ni2_FROM_DISP ) { result = 1.0; return result; }

  if( ni1_FROM_DISP.isNonnull() && ni2_TO_DISP.isNonnull())
    if( ni1_FROM_DISP == ni2_TO_DISP ) { result = 1.0; return result; }

  if( ni1_FROM_DISP.isNonnull() && ni2_FROM_DISP.isNonnull())
    if( ni1_FROM_DISP == ni2_FROM_DISP ) { result = 1.0; return result; }
    
  
  if (  elt1->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)  &&
	     elt2->trackType(reco::PFBlockElement::T_FROM_GAMMACONV)  ) {
    
    if(debug_ ) std::cout << " testLinkByVertex On Conversions " << std::endl;
    
    if ( elt1->convRef().isNonnull() && elt2->convRef().isNonnull() ) {
      if(debug_ ) std::cout << " PFBlockAlgo.cc testLinkByVertex  Cconversion Refs are non null  " << std::endl;      
      if ( elt1->convRef() ==  elt2->convRef() ) {
	result=1.0;
	if(debug_ ) std::cout << " testLinkByVertex  Cconversion Refs are equal  " << std::endl;      
	return result;
      }
    } 
    
  }
  
  if (  elt1->trackType(reco::PFBlockElement::T_FROM_V0)  &&
             elt2->trackType(reco::PFBlockElement::T_FROM_V0)  ) {
    if(debug_ ) std::cout << " testLinkByVertex On V0 " << std::endl;
    if ( elt1->V0Ref().isNonnull() && elt2->V0Ref().isNonnull() ) {
      if(debug_ ) std::cout << " PFBlockAlgo.cc testLinkByVertex  V0 Refs are non null  " << std::endl;
      if ( elt1->V0Ref() ==  elt2->V0Ref() ) {
	result=1.0;
	if(debug_ ) std::cout << " testLinkByVertex  V0 Refs are equal  " << std::endl;
	return result;
      }
    }
  }

  return result;
}


void 
PFBlockAlgoNew::checkMaskSize( const reco::PFRecTrackCollection& tracks,
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
			    const Mask& scMask) const {

  if( !trackMask.empty() && 
      trackMask.size() != tracks.size() ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the track mask is different ";
    err += "from the size of the track vector.";
    throw std::length_error( err.c_str() );
  }

  if( !gsftrackMask.empty() && 
      gsftrackMask.size() != gsftracks.size() ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the gsf track mask is different ";
    err += "from the size of the gsftrack vector.";
    throw std::length_error( err.c_str() );
  }

  if( !ecalMask.empty() && 
      ecalMask.size() != ecals.size() ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the ecal mask is different ";
    err += "from the size of the ecal clusters vector.";
    throw std::length_error( err.c_str() );
  }
  
  if( !hcalMask.empty() && 
      hcalMask.size() != hcals.size() ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the hcal mask is different ";
    err += "from the size of the hcal clusters vector.";
    throw std::length_error( err.c_str() );
  }

  if( !hoMask.empty() && 
      hoMask.size() != hos.size() ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the ho mask is different ";
    err += "from the size of the ho clusters vector.";
    throw std::length_error( err.c_str() );
  }


  if( !hfemMask.empty() && 
      hfemMask.size() != hfems.size() ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the hfem mask is different ";
    err += "from the size of the hfem clusters vector.";
    throw std::length_error( err.c_str() );
  }

  if( !hfhadMask.empty() && 
      hfhadMask.size() != hfhads.size() ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the hfhad mask is different ";
    err += "from the size of the hfhad clusters vector.";
    throw std::length_error( err.c_str() );
  }

  if( !psMask.empty() && 
      psMask.size() != pss.size() ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the ps mask is different ";
    err += "from the size of the ps clusters vector.";
    throw std::length_error( err.c_str() );
  }
  
  if( !phMask.empty() && 
      phMask.size() != egphh.size() ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the photon mask is different ";
    err += "from the size of the photon vector.";
    throw std::length_error( err.c_str() );
  }
  
  if( !scMask.empty() && 
      scMask.size() != (sceb.size() + scee.size()) ) {
    string err = "PFBlockAlgoNew::setInput: ";
    err += "The size of the SC mask is different ";
    err += "from the size of the SC vectors.";
    throw std::length_error( err.c_str() );
  }  

}


std::ostream& operator<<(std::ostream& out, const PFBlockAlgoNew& a) {
  if(! out) return out;
  
  out<<"====== Particle Flow Block Algorithm ======= ";
  out<<endl;
  out<<"number of unassociated elements : "<<a.elements_.size()<<endl;
  out<<endl;
  
  for(PFBlockAlgoNew::IEC ie = a.elements_.begin(); 
      ie != a.elements_.end(); ++ie) {
    out<<"\t"<<**ie <<endl;
  }

  
  //   const PFBlockCollection& blocks = a.blocks();

  const std::auto_ptr< reco::PFBlockCollection >& blocks
    = a.blocks(); 
    
  if(!blocks.get() ) {
    out<<"blocks already transfered"<<endl;
  }
  else {
    out<<"number of blocks : "<<blocks->size()<<endl;
    out<<endl;
    
    for(PFBlockAlgoNew::IBC ib=blocks->begin(); 
	ib != blocks->end(); ++ib) {
      out<<(*ib)<<endl;
    }
  }

  return out;
}

bool 
PFBlockAlgoNew::goodPtResolution( const reco::TrackRef& trackref) {

  double P = trackref->p();
  double Pt = trackref->pt();
  double DPt = trackref->ptError();
  unsigned int NHit = trackref->hitPattern().trackerLayersWithMeasurement();
  unsigned int NLostHit = trackref->hitPattern().trackerLayersWithoutMeasurement();
  unsigned int LostHits = trackref->numberOfLostHits();
  double sigmaHad = sqrt(1.20*1.20/P+0.06*0.06) / (1.+LostHits);

  // iteration 1,2,3,4,5 correspond to algo = 1/4,5,6,7,8,9
  unsigned int Algo = 0; 
  switch (trackref->algo()) {
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
    Algo = useIterTracking_ ? 5 : 0;
    break;
  }

  // Protection against 0 momentum tracks
  if ( P < 0.05 ) return false;

  // Temporary : Reject all tracking iteration beyond 5th step. 
  if ( Algo > 4 ) return false;
 
  if (debug_) cout << " PFBlockAlgoNew: PFrecTrack->Track Pt= "
		   << Pt << " DPt = " << DPt << endl;
  if ( ( DPtovPtCut_[Algo] > 0. && 
	 DPt/Pt > DPtovPtCut_[Algo]*sigmaHad ) || 
       NHit < NHitCut_[Algo] ) { 
    // (Algo >= 3 && LostHits != 0) ) {
    if (debug_) cout << " PFBlockAlgoNew: skip badly measured track"
		     << ", P = " << P 
		     << ", Pt = " << Pt 
		     << " DPt = " << DPt 
		     << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
		     << ", Algo = " << Algo
		     << endl;
    if (debug_) cout << " cut is DPt/Pt < " << DPtovPtCut_[Algo] * sigmaHad << endl;
    if (debug_) cout << " cut is NHit >= " << NHitCut_[Algo] << endl;
    /*
    std::cout << "Track REJECTED : ";
    std::cout << ", P = " << P 
	      << ", Pt = " << Pt 
	      << " DPt = " << DPt 
	      << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
	      << ", Algo = " << Algo
	      << std::endl;
    */
    return false;
  }

  /*
  std::cout << "Track Accepted : ";
  std::cout << ", P = " << P 
       << ", Pt = " << Pt 
       << " DPt = " << DPt 
       << ", N(hits) = " << NHit << " (Lost : " << LostHits << "/" << NLostHit << ")"
       << ", Algo = " << Algo
       << std::endl;
  */
  return true;
}

int
PFBlockAlgoNew::muAssocToTrack( const reco::TrackRef& trackref,
			     const edm::Handle<reco::MuonCollection>& muonh) const {
  if(muonh.isValid() ) {
    for(unsigned j=0;j<muonh->size(); ++j) {
      reco::MuonRef muonref( muonh, j );
      if (muonref->track().isNonnull()) 
	if( muonref->track() == trackref ) return j;
    }
  }
  return -1; // not found
}

int 
PFBlockAlgoNew::muAssocToTrack( const reco::TrackRef& trackref,
			     const edm::OrphanHandle<reco::MuonCollection>& muonh) const {
  if(muonh.isValid() ) {
    for(unsigned j=0;j<muonh->size(); ++j) {
      reco::MuonRef muonref( muonh, j );
      if (muonref->track().isNonnull())
	if( muonref->track() == trackref ) return j;
    }
  }
  return -1; // not found
}


// This prefilter avoid to call associate when not necessary.
// ACHTUNG!!!! If you introduce new links check that they are not desables here
inline bool
PFBlockAlgoNew::linkPrefilter(const reco::PFBlockElement* last, 
			      const reco::PFBlockElement* next) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  const PFBlockElement::Type& type1 = (last)->type();
  const PFBlockElement::Type& type2 = (next)->type();
  const unsigned index = rowsize*std::max(type1,type2) + std::min(type1,type2);
  bool result = false;
  if( _linkTests[index] ) {
    result = _linkTests[index]->linkPrefilter(last,next);
  }
  return result;  
}

// a little history, ideas we may want to keep around for later
   /*
  // Links between the two preshower layers are not used for now - disable
  case PFBlockLink::PS1andPS2:
    {
      PFClusterRef  ps1ref = lowEl->clusterRef();
      PFClusterRef  ps2ref = highEl->clusterRef();
      assert( !ps1ref.isNull() );
      assert( !ps2ref.isNull() );
      // PJ - 14-May-09 : A link by rechit is needed here !
      // dist = testPS1AndPS2( *ps1ref, *ps2ref );
      dist = -1.;
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::TRACKandPS1:
  case PFBlockLink::TRACKandPS2:
    {
      //cout<<"TRACKandPS"<<endl;
      PFRecTrackRef trackref = lowEl->trackRefPF();
      PFClusterRef  clusterref = highEl->clusterRef();
      assert( !trackref.isNull() );
      assert( !clusterref.isNull() );
      // PJ - 14-May-09 : A link by rechit is needed here !
      dist = testTrackAndPS( *trackref, *clusterref );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
    // GSF Track/Brem Track and preshower cluster links are not used for now - disable
  case PFBlockLink::PS1andGSF:
  case PFBlockLink::PS2andGSF:
    {
      PFClusterRef  psref = lowEl->clusterRef();
      assert( !psref.isNull() );
      const reco::PFBlockElementGsfTrack *  GsfEl =  dynamic_cast<const reco::PFBlockElementGsfTrack*>(highEl);
      const PFRecTrack * myTrack =  &(GsfEl->GsftrackPF());
      // PJ - 14-May-09 : A link by rechit is needed here !
      dist = testTrackAndPS( *myTrack, *psref );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
  case PFBlockLink::PS1andBREM:
  case PFBlockLink::PS2andBREM:
    {
      PFClusterRef  psref = lowEl->clusterRef();
      assert( !psref.isNull() );
      const reco::PFBlockElementBrem * BremEl =  dynamic_cast<const reco::PFBlockElementBrem*>(highEl);
      const PFRecTrack * myTrack = &(BremEl->trackPF());
      // PJ - 14-May-09 : A link by rechit is needed here !
      dist = testTrackAndPS( *myTrack, *psref );
      linktest = PFBlock::LINKTEST_RECHIT;
      break;
    }
    */
