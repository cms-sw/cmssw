#include "RecoParticleFlow/PFProducer/interface/PFBlockAlgo.h"
#include "RecoParticleFlow/PFProducer/interface/Utils.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFDisplacedVertex.h" // gouzevitch

#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"

#include <stdexcept>
#include <algorithm>
#include "TMath.h"

using namespace std;
using namespace reco;

#define INIT_ENTRY(name) {#name,name}

namespace {
  class QuickUnion{
  std::vector<unsigned> id_;
  std::vector<unsigned> size_;
  int count_;

  public:
    QuickUnion(const unsigned NBranches) {
      count_ = NBranches;
      id_.resize(NBranches);
      size_.resize(NBranches);
      for( unsigned i = 0; i < NBranches; ++i ) {
	id_[i] = i;
	size_[i] = 1;
      }
    }
    
    int count() const { return count_; }
    
    unsigned find(unsigned p) {
      while( p != id_[p] ) {
	id_[p] = id_[id_[p]];
	p = id_[p];
      }
      return p;
    }
    
    bool connected(unsigned p, unsigned q) { return find(p) == find(q); }
    
    void unite(unsigned p, unsigned q) {
      unsigned rootP = find(p);
      unsigned rootQ = find(q);
      id_[p] = q;
      
      if(size_[rootP] < size_[rootQ] ) { 
	id_[rootP] = rootQ; size_[rootQ] += size_[rootP]; 
      } else { 
	id_[rootQ] = rootP; size_[rootP] += size_[rootQ]; 
      }
      --count_;
    }
  };
}


//for debug only 
//#define PFLOW_DEBUG

PFBlockAlgo::PFBlockAlgo() : 
  blocks_( new reco::PFBlockCollection ),  
  debug_(false),
  elementTypes_( {
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
	INIT_ENTRY(PFBlockElement::HO),
	INIT_ENTRY(PFBlockElement::HGCAL)  
	  } ) {}

void PFBlockAlgo::setLinkers(const std::vector<edm::ParameterSet>& confs) {
   constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
   for( unsigned i = 0; i < rowsize; ++i ) {
     for( unsigned j = 0; j < rowsize; ++j ) {

       linkTestSquare_[i][j] = 0;
     }
   }
  linkTests_.resize(rowsize*rowsize);
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
    if( !(elementTypes_.count(link1) && elementTypes_.count(link2) ) ) {
      throw cms::Exception("InvalidBlockElementType")
	<< "One of \"" << link1 << "\" or \"" << link2 
	<< "\" are invalid block element types!";
    }
    const PFBlockElement::Type type1 = elementTypes_.at(link1);
    const PFBlockElement::Type type2 = elementTypes_.at(link2);    
    const unsigned index  = rowsize*std::max(type1,type2)+std::min(type1,type2);
    BlockElementLinkerBase * linker =
      BlockElementLinkerFactory::get()->create(linkerName,conf);
    linkTests_[index].reset(linker);
    linkTestSquare_[type1][type2] = index;
    linkTestSquare_[type2][type1] = index;
    // setup KDtree if requested
    const bool useKDTree = conf.getParameter<bool>("useKDTree");
    if( useKDTree ) {
      kdtrees_.emplace_back( KDTreeLinkerFactory::get()->create(pfx_kdtree+
								linkerName) );
      kdtrees_.back()->setTargetType(std::min(type1,type2));
      kdtrees_.back()->setFieldType(std::max(type1,type2));
    }
  }
}

void PFBlockAlgo::setImporters(const std::vector<edm::ParameterSet>& confs,
				  edm::ConsumesCollector& sumes) {
   importers_.reserve(confs.size());  
  for( const auto& conf : confs ) {
    const std::string& importerName = 
      conf.getParameter<std::string>("importerName");    
    BlockElementImporterBase * importer =
      BlockElementImporterFactory::get()->create(importerName,conf,sumes);
    importers_.emplace_back(importer);
  }
}

PFBlockAlgo::~PFBlockAlgo() {

#ifdef PFLOW_DEBUG
  if(debug_)
    cout<<"~PFBlockAlgo - number of remaining elements: "
	<<elements_.size()<<endl;
#endif  
}

void PFBlockAlgo::findBlocks() {
  // Glowinski & Gouzevitch
  for( const auto& kdtree : kdtrees_ ) {
    kdtree->process();
  }  
  // !Glowinski & Gouzevitch
  // the blocks have not been passed to the event, and need to be cleared
  if( blocks_.get() ) blocks_->clear();
  else                blocks_.reset( new reco::PFBlockCollection );
  blocks_->reserve(elements_.size());

  QuickUnion qu(bare_elements_.size());
  const auto elem_size = bare_elements_.size();
  for( unsigned i = 0; i < elem_size; ++i ) {
    for( unsigned j = 0; j < elem_size; ++j ) {
      if( qu.connected(i,j) || j == i ) continue;
      if( !linkTests_[linkTestSquare_[bare_elements_[i]->type()][bare_elements_[j]->type()]] ) {
        j = ranges_[bare_elements_[j]->type()].second;
        continue;
      }
      auto p1(bare_elements_[i]), p2(bare_elements_[j]);
      const PFBlockElement::Type type1 = p1->type();
      const PFBlockElement::Type type2 = p2->type();
      const unsigned index = linkTestSquare_[type1][type2];
      if( linkTests_[index]->linkPrefilter(p1,p2) ) {
        const double dist = linkTests_[index]->testLink(p1,p2);
        // compute linking info if it is possible
        if( dist > -0.5 ) {
          qu.unite(i,j);
        }
      }
    }
  }
  
  std::unordered_multimap<unsigned,unsigned> blocksmap(elements_.size());
  std::vector<unsigned> keys;
  keys.reserve(elements_.size());
  for( unsigned i = 0; i < elements_.size(); ++i ) {
    unsigned key = i; 
    while( key != qu.find(key) ) key = qu.find(key); // make sure we always find the root node...
    auto pos  = std::lower_bound(keys.begin(),keys.end(),key);
    if( pos == keys.end() || *pos != key ) {
      keys.insert(pos,key);      
    }
    blocksmap.emplace(key,i);
  }

  PFBlockLink::Type linktype = PFBlockLink::NONE;
  PFBlock::LinkTest linktest = PFBlock::LINKTEST_RECHIT;
  for( auto key : keys ) {
    blocks_->push_back( reco::PFBlock() );
    auto range = blocksmap.equal_range(key);
    auto& the_block = blocks_->back();
    ElementList::value_type::pointer p1(bare_elements_[range.first->second]);
    the_block.addElement(p1);
    const unsigned block_size = blocksmap.count(key) + 1;
    //reserve up to 1M or 8MB; pay rehash cost for more
    std::unordered_map<std::pair<unsigned int,unsigned int>, PFBlockLink > links(min(1000000u,block_size*block_size));
    auto itr = range.first;
    ++itr;
    for( ; itr != range.second; ++itr ) {
      ElementList::value_type::pointer p2(bare_elements_[itr->second]);
      const PFBlockElement::Type type1 = p1->type();
      const PFBlockElement::Type type2 = p2->type();        
      the_block.addElement(p2);
      linktest = PFBlock::LINKTEST_RECHIT; //rechit by default 
      linktype = static_cast<PFBlockLink::Type>(1<<(type1-1)|1<<(type2-1));
      const unsigned index = linkTestSquare_[type1][type2];
      if( nullptr != linkTests_[index] ) {
        const double dist = linkTests_[index]->testLink(p1,p2);
        links.emplace( std::make_pair(p1->index(), p2->index()) ,
                       PFBlockLink( linktype, linktest, dist,
                                    p1->index(), p2->index() ) );
      }
    }
    packLinks( the_block, links );    
  }
  
  bare_elements_.clear();
  elements_.clear();
}

void 
PFBlockAlgo::packLinks( reco::PFBlock& block, 
			   const std::unordered_map<std::pair<unsigned int,unsigned int>,PFBlockLink>& links ) const {
  constexpr unsigned rowsize = reco::PFBlockElement::kNBETypes;
  
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
        const PFBlockElement::Type type1 = els[i1].type();
        const PFBlockElement::Type type2 = els[i2].type();
        const auto minmax = std::minmax(type1,type2);
        const unsigned index = rowsize*minmax.second + minmax.first;
	PFBlockLink::Type linktype = PFBlockLink::NONE;
	bool bTestLink = ( nullptr == linkTests_[index] ? false : linkTests_[index]->linkPrefilter(&(els[i1]),&(els[i2])) );
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
  const PFBlockElement::Type type1 = (last)->type();
  const PFBlockElement::Type type2 = (next)->type();
  const unsigned index = rowsize*std::max(type1,type2) + std::min(type1,type2);
  bool result =  linkTests_[index]->linkPrefilter(last,next);
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
  const PFBlockElement::Type type1 = el1->type();
  const PFBlockElement::Type type2 = el2->type();
  linktype = static_cast<PFBlockLink::Type>(1<<(type1-1)|1<<(type2-1));
  const unsigned index = rowsize*std::max(type1,type2) + std::min(type1,type2);
  if(debug_ ) { 
    std::cout << " PFBlockAlgo links type1 " << type1 
	      << " type2 " << type2 << std::endl;
  }
  
  // index is always checked in the preFilter above, no need to check here
  dist = linkTests_[index]->testLink(el1,el2);
}

void PFBlockAlgo::updateEventSetup(const edm::EventSetup& es) {
  for( auto& importer : importers_ ) {
    importer->updateEventSetup(es);
  }
}

// see plugins/importers and plugins/kdtrees
// for the definitions of available block element importers
// and kdtree preprocessors
void PFBlockAlgo::buildElements(const edm::Event& evt) {
  // import block elements as defined in python configuration
  ranges_.fill(std::make_pair(0,0));
  elements_.clear();
  for( const auto& importer : importers_ ) {
    importer->importToBlock(evt,elements_);
  }

  std::sort(elements_.begin(),elements_.end(),
            [](const auto& a, const auto& b) { return a->type() < b->type(); } );
  
  bare_elements_.resize(elements_.size());
  for( unsigned i = 0; i < elements_.size(); ++i ) {
    bare_elements_[i] = elements_[i].get();
  }

  // list is now partitioned, so mark the boundaries so we can efficiently skip chunks  
  unsigned current_type = ( !elements_.empty() ? elements_[0]->type() : 0 );
  unsigned last_type = ( !elements_.empty() ? elements_.back()->type() : 0 );
  ranges_[current_type].first  = 0;
  ranges_[last_type].second = elements_.size()-1;
  for( size_t i = 0; i < elements_.size(); ++i ) {
    const auto the_type = elements_[i]->type();
    if( the_type != current_type ) {
      ranges_[the_type].first = i; 
      ranges_[current_type].second = i-1;      
      current_type = the_type;
    }
  }  
  // -------------- Loop over block elements ---------------------

  // Here we provide to all KDTree linkers the collections to link.
  // Glowinski & Gouzevitch
  
  for (ElementList::iterator it = elements_.begin();
       it != elements_.end(); ++it) {
    for( const auto& kdtree : kdtrees_ ) {
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

  const std::unique_ptr< reco::PFBlockCollection >& blocks
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
