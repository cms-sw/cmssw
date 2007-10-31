#ifndef RecoAlgos_CandidateProducer_h
#define RecoAlgos_CandidateProducer_h
/** \class CandidateProducer
 *
 * Framework module that produces a collection
 * of candidates from generic compoment
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.5 $
 *
 * $Id: CandidateProducer.h,v 1.5 2007/10/22 09:49:27 llista Exp $
 *
 */
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace converter {
  namespace helper {
    template<typename TColl>
    struct CandConverter { };
    
    struct ConcreteCreator { 
      template<typename CColl, typename Comp, typename Conv>
      static void create( size_t idx, CColl & cands, const Comp & components, Conv & converter ) {
	typename Conv::Candidate c;
	converter.convert( idx, components, c );
	cands.push_back( c );
      }
    };
    
    struct PolymorphicCreator { 
      template<typename CColl, typename Comp, typename Conv>
      static void create( size_t idx, CColl & cands, const Comp & components, Conv & converter ) {
	typename Conv::Candidate * c = new typename Conv::Candidate;
	converter.convert( idx, components, * c );
	cands.push_back( c );
      }
    };

    template<typename CColl>
    struct CandCreator { 
      typedef ConcreteCreator type;
    };

    template<>
    struct CandCreator<reco::CandidateCollection> { 
      typedef PolymorphicCreator type;
    };
  }
}

template<typename TColl, typename CColl, 
	 typename Conv = typename converter::helper::CandConverter<TColl>::type,
	 typename Creator = typename converter::helper::CandCreator<CColl>::type >
class CandidateProducer : public edm::EDProducer {
public:
  /// constructor from parameter set
  CandidateProducer( const edm::ParameterSet & cfg ) :
  src_( cfg.template getParameter<edm::InputTag>( "src" ) ), 
    converter_( cfg ) {
    produces<CColl>();
  }
  /// destructor
  ~CandidateProducer() { }
  
private:
  /// begin job
  void beginJob( const edm::EventSetup& es ) { converter_.beginJob( es ); }
  /// process one event
  void produce( edm::Event& e, const edm::EventSetup& );
  /// label of source collection and tag
  edm::InputTag src_;
  /// converter helper
  Conv converter_;
};

template<typename TColl, typename CColl, typename Conv, typename Creator>
void CandidateProducer<TColl, CColl, Conv, Creator>::produce( edm::Event& evt, const edm::EventSetup& ) {
  edm::Handle<TColl> components;
  evt.getByLabel( src_, components );
  std::auto_ptr<CColl> cands( new CColl );
  size_t size = components->size();
  cands->reserve( size );
  for( size_t idx = 0; idx != size; ++ idx )
    Creator::create( idx, * cands, components, converter_ );
  evt.put( cands );
}

#endif
