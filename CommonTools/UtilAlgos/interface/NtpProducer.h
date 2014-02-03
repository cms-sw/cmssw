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
  virtual void produce( edm::Event&, const edm::EventSetup&) override;

private:
  /// label of the collection to be read in
  edm::InputTag src_;
  /// variable tags
  std::vector<std::pair<std::string, StringObjectFunction<typename C::value_type> > > tags_;
  bool lazyParser_;
  std::string prefix_;
  bool eventInfo_;
};

template<typename C>
NtpProducer<C>::NtpProducer( const edm::ParameterSet& par ) : 
  src_( par.template getParameter<edm::InputTag>( "src" ) ),
  lazyParser_( par.template getUntrackedParameter<bool>( "lazyParser", false ) ),
  prefix_( par.template getUntrackedParameter<std::string>( "prefix","" ) ),
  eventInfo_( par.template getUntrackedParameter<bool>( "eventInfo",true ) ){
  std::vector<edm::ParameterSet> variables = 
                                   par.template getParameter<std::vector<edm::ParameterSet> >("variables");
   std::vector<edm::ParameterSet>::const_iterator 
     q = variables.begin(), end = variables.end();
   if(eventInfo_){
     produces<unsigned int>( prefix_+"EventNumber" ).setBranchAlias( prefix_ + "EventNumber" );
     produces<unsigned int>( prefix_ + "RunNumber" ).setBranchAlias( prefix_ + "RunNumber" );
     produces<unsigned int>( prefix_ + "LumiBlock" ).setBranchAlias( prefix_ + "Lumiblock" );
   }
   for (; q!=end; ++q) {
     std::string tag = prefix_ + q->getUntrackedParameter<std::string>("tag");
     StringObjectFunction<typename C::value_type> quantity(q->getUntrackedParameter<std::string>("quantity"), lazyParser_);
     tags_.push_back(std::make_pair(tag, quantity));
     produces<std::vector<float> >(tag).setBranchAlias(tag);
     
   }   
}

template<typename C>
NtpProducer<C>::~NtpProducer() {
}

template<typename C>
void NtpProducer<C>::produce( edm::Event& iEvent, const edm::EventSetup&) {
   edm::Handle<C> coll;
   iEvent.getByLabel(src_, coll);
   if(eventInfo_){   
     std::auto_ptr<unsigned int> event( new unsigned int );
     std::auto_ptr<unsigned int> run( new unsigned int );
     std::auto_ptr<unsigned int> lumi( new unsigned int );
     *event = iEvent.id().event();
     *run = iEvent.id().run();
     *lumi = iEvent.luminosityBlock();
     iEvent.put( event, prefix_ + "EventNumber" );
     iEvent.put( run, prefix_ + "RunNumber" );
     iEvent.put( lumi, prefix_ + "LumiBlock" );
   }
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
