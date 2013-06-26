#ifndef UtilAlgos_NtpProducer_h
#define UtilAlgos_NtpProducer_h
/** \class NtpProducer
 *
 * Creates histograms defined in config file 
 *
 * \author: Luca Lista, INFN
 * 
 * Template parameters:
 * - C : Concrete candidate collection type
 *
 */
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

template<typename C>
class NtpProducer : public edm::EDProducer {
public:
  /// constructor from parameter set
  NtpProducer( const edm::ParameterSet& );
  /// destructor
  ~NtpProducer();
  
protected:
  /// process an event
  virtual void produce( edm::Event&, const edm::EventSetup& );

private:
  /// label of the collection to be read in
  edm::InputTag src_;
  /// variable tags
  std::vector<std::pair<std::string, StringObjectFunction<typename C::value_type> > > tags_;
};

template<typename C>
NtpProducer<C>::NtpProducer( const edm::ParameterSet& par ) : 
  src_( par.template getParameter<edm::InputTag>( "src" ) ) {
   std::vector<edm::ParameterSet> variables = 
                                   par.template getParameter<std::vector<edm::ParameterSet> >("variables");
   std::vector<edm::ParameterSet>::const_iterator 
     q = variables.begin(), end = variables.end();
   for (; q!=end; ++q) {
     std::string tag = q->getUntrackedParameter<std::string>("tag");
     StringObjectFunction<typename C::value_type> quantity(q->getUntrackedParameter<std::string>("quantity"));
     tags_.push_back(std::make_pair(tag, quantity));
     produces<std::vector<float> >(tag).setBranchAlias(tag);
   }   
}

template<typename C>
NtpProducer<C>::~NtpProducer() {
}

template<typename C>
void NtpProducer<C>::produce( edm::Event& iEvent, const edm::EventSetup& ) {
   edm::Handle<C> coll;
   iEvent.getByLabel(src_, coll);

   typename std::vector<std::pair<std::string, StringObjectFunction<typename C::value_type> > >::const_iterator 
     q = tags_.begin(), end = tags_.end();
   for(;q!=end; ++q) {
     std::auto_ptr<std::vector<float> > x(new std::vector<float>);
     x->reserve(coll->size());
     for (typename C::const_iterator elem=coll->begin(); elem!=coll->end(); ++elem ) {
       x->push_back(q->second(*elem));
     }
     iEvent.put(x, q->first);
   }
}

#endif
