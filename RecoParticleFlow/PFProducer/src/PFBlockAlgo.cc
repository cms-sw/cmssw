#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
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

PFBlockAlgo::PFBlockAlgo() : 
  blocks_( new reco::PFBlockCollection ),  
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

void PFBlockAlgo::setLinkers(const std::vector<edm::ParameterSet>& confs) {
   constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  _linkTests.resize(rowsize*rowsize);
  const std::string prefix("PFBlockElement::");
  const std::string pfx_kdtree("KDTree");
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
    BlockElementLinkerBase * linker =
      BlockElementLinkerFactory::get()->create(linkerName,conf);
    _linkTests[index].reset(linker);
    // setup KDtree if requested
    const bool useKDTree = conf.getParameter<bool>("useKDTree");
    if( useKDTree ) {
      _kdtrees.emplace_back( KDTreeLinkerFactory::get()->create(pfx_kdtree+
								linkerName) );
      _kdtrees.back()->setTargetType(std::min(type1,type2));
      _kdtrees.back()->setFieldType(std::max(type1,type2));
    }
  }
}

void PFBlockAlgo::setImporters(const std::vector<edm::ParameterSet>& confs,
				  edm::ConsumesCollector& sumes) {
   _importers.reserve(confs.size());  
  for( const auto& conf : confs ) {
    const std::string& importerName = 
      conf.getParameter<std::string>("importerName");    
    BlockElementImporterBase * importer =
      BlockElementImporterFactory::get()->create(importerName,conf,sumes);
    _importers.emplace_back(importer);
  }
}

PFBlockAlgo::~PFBlockAlgo() {

#ifdef PFLOW_DEBUG
  if(debug_)
    cout<<"~PFBlockAlgo - number of remaining elements: "
	<<elements_.size()<<endl;
#endif  
}

void 
PFBlockAlgo::findBlocks() {
  // Glowinski & Gouzevitch
  for( const auto& kdtree : _kdtrees ) {
    kdtree->process();
  }  
  // !Glowinski & Gouzevitch
  // the blocks have not been passed to the event, and need to be cleared
  if( blocks_.get() ) blocks_->clear();
  else                blocks_.reset( new reco::PFBlockCollection );
  blocks_->reserve(elements_.size());
  for(IE ie = elements_.begin(); 
      ie != elements_.end();) {
    
#ifdef PFLOW_DEBUG
    if(debug_) {
      cout<<" PFBlockAlgo::findBlocks() ----------------------"<<endl;
      cout<<" element "<<**ie<<endl;
      cout<<" creating new block"<<endl;
    }
#endif

    blocks_->push_back( reco::PFBlock() );
    
    std::unordered_map<std::pair<size_t,size_t>, PFBlockLink > links;
    links.reserve(elements_.size());
    
    ie = associate(elements_, links, blocks_->back());    
    
    packLinks( blocks_->back(), links );
  }
  //std::cout << "(new) Found " << blocks_->size() << " PFBlocks!" << std::endl;
}

// start from first element in elements_
// partition elements until block grows no further
// return the start of the new block
PFBlockAlgo::IE
PFBlockAlgo::associate( PFBlockAlgo::ElementList& elems,
			   std::unordered_map<std::pair<size_t,size_t>,PFBlockLink>& links,
			   reco::PFBlock& block) {
  if( elems.size() == 0 ) return elems.begin();
  ElementList::iterator scan_upper(elems.begin()), search_lower(elems.begin()), 
    scan_lower(elems.begin());
  ++scan_upper; ++search_lower;
  double dist = -1;
  PFBlockLink::Type linktype = PFBlockLink::NONE;
  PFBlock::LinkTest linktest = PFBlock::LINKTEST_RECHIT;
  block.addElement(scan_lower->get()); // seed the block
  // the terminating condition of this loop is when the next range 
  // to scan has zero length (i.e. you have found no more nearest neighbours)
  do {     
    scan_upper = search_lower;
    // for each element added in the previous iteration we check to see what
    // elements are linked to it
    for( auto comp = scan_lower; comp != scan_upper; ++comp ) {
      // group everything that's linked to the current element:
      // std::partition moves all elements that return true for the 
      // function defined below (a.k.a. the linking function) to the
      // front of the range provided
      search_lower = 
	std::partition(search_lower,elems.end(),
		       [&](ElementList::value_type& a){	
			 dist = -1.0;			 
			 // compute linking info if it is possible
			 if( linkPrefilter(comp->get(), a.get()) ) {
			   link( comp->get(), a.get(), 
				 linktype, linktest, dist ); 
			 }
			 if( dist >= -0.5 ) {
			   const unsigned lidx = ((*comp)->type() < a->type() ? 
						  (*comp)->index() :
						  a->index() );
			   const unsigned uidx = ((*comp)->type() >= a->type() ?
						  (*comp)->index() :
						  a->index() );
			   block.addElement( a.get() ); 
			   links.emplace( std::make_pair(lidx,uidx),
					  PFBlockLink(linktype, linktest, dist,
						      lidx, uidx ) );
			   return true;
			 } else {
			   return false;
			 }
		       });
    }
    // we then update the scan range lower boundary to point to the
    // first element that we added in this round of association
    scan_lower = scan_upper;      
  } while( search_lower != scan_upper ); 
  // return the pointer to the first element not in the PFBlock we just made
  return elems.erase(elems.begin(),scan_upper);
}

void 
PFBlockAlgo::packLinks( reco::PFBlock& block, 
			   const std::unordered_map<std::pair<size_t,size_t>,PFBlockLink>& links ) const {
  
  
  const edm::OwnVector< reco::PFBlockElement >& els = block.elements();
  
  block.bookLinkData();
  unsigned elsize = els.size();
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
      const auto link_itr = links.find(std::make_pair(i2,i1));
      if( link_itr != links.end() ) {
	dist = link_itr->second.dist();
	linktest = link_itr->second.test();
	linked = true;
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

// see plugins/linkers for the functions that calculate distances
// for each available link type
inline bool
PFBlockAlgo::linkPrefilter(const reco::PFBlockElement* last, 
			      const reco::PFBlockElement* next) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  const PFBlockElement::Type& type1 = (last)->type();
  const PFBlockElement::Type& type2 = (next)->type();
  const unsigned index = rowsize*std::max(type1,type2) + std::min(type1,type2);
  bool result = false;
  if( index < _linkTests.size() && _linkTests[index] ) {
    result = _linkTests[index]->linkPrefilter(last,next);
  }
  return result;  
}

inline void 
PFBlockAlgo::link( const reco::PFBlockElement* el1, 
		      const reco::PFBlockElement* el2, 
		      PFBlockLink::Type& linktype, 
		      reco::PFBlock::LinkTest& linktest,
		      double& dist) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  dist=-1.0;
  linktest = PFBlock::LINKTEST_RECHIT; //rechit by default 
  PFBlockElement::Type type1 = el1->type();
  PFBlockElement::Type type2 = el2->type();
  linktype = static_cast<PFBlockLink::Type>(1<<(type1-1)|1<<(type2-1));
  const unsigned index = rowsize*std::max(type1,type2) + std::min(type1,type2);
  if(debug_ ) { 
    std::cout << " PFBlockAlgo links type1 " << type1 
	      << " type2 " << type2 << std::endl;
  }
  
  // index is always checked in the preFilter above, no need to check here
  dist = _linkTests[index]->testLink(el1,el2);
}

void PFBlockAlgo::updateEventSetup(const edm::EventSetup& es) {
  for( auto& importer : _importers ) {
    importer->updateEventSetup(es);
  }
}

// see plugins/importers and plugins/kdtrees
// for the definitions of available block element importers
// and kdtree preprocessors
void PFBlockAlgo::buildElements(const edm::Event& evt) {
  // import block elements as defined in python configuration
  for( const auto& importer : _importers ) {

    importer->importToBlock(evt,elements_);
  }
  
  // -------------- Loop over block elements ---------------------

  // Here we provide to all KDTree linkers the collections to link.
  // Glowinski & Gouzevitch
  
  for (ElementList::iterator it = elements_.begin();
       it != elements_.end(); ++it) {
    for( const auto& kdtree : _kdtrees ) {
      if( (*it)->type() == kdtree->targetType() ) {
	kdtree->insertTargetElt(it->get());
      }
      if( (*it)->type() == kdtree->fieldType() ) {
	kdtree->insertFieldClusterElt(it->get());
      }
    }    
  }
  //std::cout << "(new) imported: " << elements_.size() << " elements!" << std::endl;
}

std::ostream& operator<<(std::ostream& out, const PFBlockAlgo& a) {
  if(! out) return out;
  
  out<<"====== Particle Flow Block Algorithm ======= ";
  out<<endl;
  out<<"number of unassociated elements : "<<a.elements_.size()<<endl;
  out<<endl;
  
  for(PFBlockAlgo::IEC ie = a.elements_.begin(); 
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
    
    for(PFBlockAlgo::IBC ib=blocks->begin(); 
	ib != blocks->end(); ++ib) {
      out<<(*ib)<<endl;
    }
  }

  return out;
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
