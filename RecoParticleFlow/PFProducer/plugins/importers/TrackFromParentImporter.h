#ifndef __TrackFromParentImporter_H__
#define __TrackFromParentImporter_H__

#include "RecoParticleFlow/PFProducer/interface/BlockElementImporterBase.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

namespace pflow {
  namespace noop {
    // this adaptor class gets redefined later to match the 
    // needs of the collection and importing cuts we are using
    template<class Collection>
      class ParentCollectionAdaptor {
    public:  
      static bool check_importable(const typename Collection::value_type&) {
	return true;
      }
      static const std::vector<reco::PFRecTrackRef>& 
	get_track_refs(const typename Collection::value_type&) {
	return _empty;
      }
      static void set_element_info(reco::PFBlockElement*,
				   const typename edm::Ref<Collection>&) {    
      }
      static const std::vector<reco::PFRecTrackRef> _empty;
    };
  }
  namespace importers {    
    template<class Collection,class Adaptor=noop::ParentCollectionAdaptor<Collection> >
      class TrackFromParentImporter : public BlockElementImporterBase {
    public:
    TrackFromParentImporter(const edm::ParameterSet& conf,
			    edm::ConsumesCollector& sumes) :
      BlockElementImporterBase(conf,sumes),
	_src(sumes.consumes<Collection>(conf.getParameter<edm::InputTag>("source"))) {}
      
      void importToBlock( const edm::Event& ,
			  ElementList& ) const override;
      
    private:
      edm::EDGetTokenT<Collection> _src;
    };
    
    
    template<class Collection, class Adaptor>
      void TrackFromParentImporter<Collection,Adaptor>::
      importToBlock( const edm::Event& e, 
		     BlockElementImporterBase::ElementList& elems ) const {
      typedef BlockElementImporterBase::ElementList::value_type ElementType;  
      edm::Handle<Collection> pfparents;
      e.getByToken(_src,pfparents);
      elems.reserve(elems.size() + 2*pfparents->size());
      // setup our elements so that all the SCs are grouped together
      auto TKs_end = std::partition(elems.begin(),elems.end(),
				    [](const ElementType& a){
				      return a->type() == reco::PFBlockElement::TRACK;
				    });  
      // insert tracks into the element list, updating tracks that exist already
      auto bpar = pfparents->cbegin();
      auto epar = pfparents->cend();
      edm::Ref<Collection> parentRef;
      reco::PFBlockElement* trkElem = NULL;
      for( auto pfparent =  bpar; pfparent != epar; ++pfparent ) {
	if( Adaptor::check_importable(*pfparent) ) {
	  parentRef = edm::Ref<Collection>(pfparents,std::distance(bpar,pfparent));
	  const auto& pftracks = Adaptor::get_track_refs(*pfparent);
	  for( const auto& pftrack : pftracks ) {
	    auto tk_elem = std::find_if(elems.begin(),TKs_end,
					[&](const ElementType& a){
					  return ( a->trackRef() == 
						   pftrack->trackRef() );
					});
	    if( tk_elem != TKs_end ) { // if found flag the track, otherwise import
	      Adaptor::set_element_info(tk_elem->get(),parentRef);
	    } else {
	      trkElem = new reco::PFBlockElementTrack(pftrack);
	      Adaptor::set_element_info(trkElem,parentRef);
	      TKs_end = elems.insert(TKs_end,ElementType(trkElem));
	      ++TKs_end;
	    }
	  }
	}
      }// loop on tracking coming from common parent
      elems.shrink_to_fit();
    }
  } // importers
} // pflow
#endif
