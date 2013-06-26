#ifndef CommonTools_UtilAlgos_ProductFromFwdPtrProducer_h
#define CommonTools_UtilAlgos_ProductFromFwdPtrProducer_h


/**
  \class    edm::ProductFromFwdPtrProducer ProductFromFwdPtrProducer.h "CommonTools/UtilAlgos/interface/ProductFromFwdPtrProducer.h"
  \brief    Produces a list of objects "by value" that correspond to the FwdPtr's from an input collection. 


  \author   Salvatore Rappoccio
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "CommonTools/UtilAlgos/interface/FwdPtrConversionFactory.h"
#include <vector>

namespace edm {



  template < class T, class H = ProductFromFwdPtrFactory<T> > 
  class ProductFromFwdPtrProducer : public edm::EDProducer {
  public :
    explicit ProductFromFwdPtrProducer( edm::ParameterSet const & params ) :
      src_   ( params.getParameter<edm::InputTag>("src") )
    {
      produces< std::vector<T> > ();
    }
    
    ~ProductFromFwdPtrProducer() {}

    virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override {

      edm::Handle< std::vector<edm::FwdPtr<T> > > hSrc;
      iEvent.getByLabel( src_, hSrc );
      
      std::auto_ptr< std::vector<T> > pOutput ( new std::vector<T> );
      
      for ( typename std::vector< edm::FwdPtr<T> >::const_iterator ibegin = hSrc->begin(),
	      iend = hSrc->end(),
	      i = ibegin; i!= iend; ++i ) {
	H factory;
	T t = factory(*i);
	pOutput->push_back( t );
      }
      
      
      iEvent.put( pOutput );      
    }
    
  protected :
    edm::InputTag src_;
  };
}

#endif
