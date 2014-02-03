#ifndef CommonTools_UtilAlgos_FwdPtrCollectionFilter_h
#define CommonTools_UtilAlgos_FwdPtrCollectionFilter_h


/**
  \class    edm::FwdPtrCollectionFilter FwdPtrCollectionFilter.h "CommonTools/UtilAlgos/interface/FwdPtrCollectionFilter.h"
  \brief    Selects a list of FwdPtr's to a product T (templated) that satisfy a method S(T) (templated). Can also handle input as View<T>. 
            Can also have a factory class to create new instances of clones if desired. 


  \author   Salvatore Rappoccio
*/


#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "CommonTools/UtilAlgos/interface/FwdPtrConversionFactory.h"
#include <vector>

namespace edm {

  template < class T, class S, class H = ProductFromFwdPtrFactory<T> > 
  class FwdPtrCollectionFilter : public edm::EDFilter {
  public :
    explicit FwdPtrCollectionFilter() {}

    explicit FwdPtrCollectionFilter( edm::ParameterSet const & params ) :
    src_( params.getParameter<edm::InputTag>("src") ), filter_(false), makeClones_(false),
      selector_( params )
    {
      if ( params.exists("filter") ) {
	filter_ = params.getParameter<bool>("filter");
      }
      if ( params.exists("makeClones") ) {
	makeClones_ = params.getParameter<bool>("makeClones");
      }

      produces< std::vector< edm::FwdPtr<T> > > ();
      if ( makeClones_ ) {
	produces< std::vector<T> > ();
      }
    }
    
    ~FwdPtrCollectionFilter() {}

    virtual bool filter(edm::Event & iEvent, const edm::EventSetup& iSetup){

      std::auto_ptr< std::vector< edm::FwdPtr<T> > > pOutput ( new std::vector<edm::FwdPtr<T> > );
      
      std::auto_ptr< std::vector<T> > pClones ( new std::vector<T> );


      edm::Handle< std::vector< edm::FwdPtr<T> > > hSrcAsFwdPtr;
      edm::Handle< edm::View<T> > hSrcAsView;
      bool foundAsFwdPtr = iEvent.getByLabel( src_, hSrcAsFwdPtr );
      if ( !foundAsFwdPtr ) {
	iEvent.getByLabel( src_, hSrcAsView );	
      }

      // First try to access as a View<T>. If not a View<T>, look as a vector<FwdPtr<T> >
      if ( !foundAsFwdPtr ) {
	for ( typename edm::View<T>::const_iterator ibegin = hSrcAsView->begin(),
		iend = hSrcAsView->end(),
		i = ibegin; i!= iend; ++i ) {
	  if ( selector_( *i ) ) {
	    pOutput->push_back( edm::FwdPtr<T>( hSrcAsView->ptrAt( i - ibegin ), hSrcAsView->ptrAt( i - ibegin ) ) );
	    if ( makeClones_ ) {
	      H factory;	      
	      T outclone = factory( pOutput->back() );
	      pClones->push_back( outclone );
	    }
	  }
	}
      } else {
	for ( typename std::vector<edm::FwdPtr<T> >::const_iterator ibegin = hSrcAsFwdPtr->begin(),
		iend = hSrcAsFwdPtr->end(),
		i = ibegin; i!= iend; ++i ) {
	  if ( selector_( **i ) ) {
	    pOutput->push_back( *i );
	    if ( makeClones_ ) {
	      H factory;
	      T outclone = factory( pOutput->back() );
	      pClones->push_back( outclone );
	    }
	  }
	}

      }
      
      bool pass = pOutput->size() > 0;
      iEvent.put( pOutput );
      if ( makeClones_ )
	iEvent.put( pClones );
      if ( filter_ )
	return pass;
      else 
	return true;

    }
    
  protected :
    edm::InputTag src_;
    bool          filter_;
    bool          makeClones_;
    S             selector_; 
  };
}

#endif
