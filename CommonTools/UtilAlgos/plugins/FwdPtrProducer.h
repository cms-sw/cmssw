#ifndef CommonTools_UtilAlgos_FwdPtrProducer_h
#define CommonTools_UtilAlgos_FwdPtrProducer_h


/**
  \class    edm::FwdPtrProducer FwdPtrProducer.h "CommonTools/UtilAlgos/interface/FwdPtrProducer.h"
  \brief    Produces a list of FwdPtr's to an input collection. 


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


  template < class T, class H = FwdPtrFromProductFactory<T> > 
  class FwdPtrProducer : public edm::EDProducer {
  public :
    explicit FwdPtrProducer( edm::ParameterSet const & params ) :
       src_( params.getParameter<edm::InputTag>("src") )
    {
      produces< std::vector< edm::FwdPtr<T> > > ();
    }
    
    ~FwdPtrProducer() {}

    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup){

      edm::Handle< edm::View<T> > hSrc;
      iEvent.getByLabel( src_, hSrc );
      
      std::auto_ptr< std::vector< edm::FwdPtr<T> > > pOutputFwdPtr ( new std::vector<edm::FwdPtr<T> > );
      
      for ( typename edm::View<T>::const_iterator ibegin = hSrc->begin(),
	      iend = hSrc->end(),
	      i = ibegin; i!= iend; ++i ) {
	H factory;
	FwdPtr<T> ptr = factory( *hSrc, i - ibegin );
	pOutputFwdPtr->push_back( ptr );
      }
      
      
      iEvent.put( pOutputFwdPtr );      
    }
    
  protected :
    edm::InputTag src_;
  };
}

#endif
