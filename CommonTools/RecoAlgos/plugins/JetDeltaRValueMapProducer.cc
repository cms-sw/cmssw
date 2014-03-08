/* \class PFJetSelector
 *
 * Selects jets with a configurable string-based cut,
 * and also writes out the constituents of the jet
 * into a separate collection.
 *
 * \author: Sal Rappoccio
 *
 *
 * for more details about the cut syntax, see the documentation
 * page below:
 *
 *   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePhysicsCutParser
 *
 *
 */


#include "FWCore/Framework/interface/EDProducer.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaR.h"

template < class T >
class JetDeltaRValueMapProducer : public edm::EDProducer {

public:

  typedef std::vector<T> JetsInput;
  typedef edm::ValueMap<float> JetValueMap; 


  JetDeltaRValueMapProducer ( edm::ParameterSet const & params ) :
      srcToken_( consumes< typename edm::View<T> >( params.getParameter<edm::InputTag>("src") ) ),
      matchedToken_( consumes< typename edm::View<T> >( params.getParameter<edm::InputTag>( "matched" ) ) ),
      distMin_( params.getParameter<double>( "distMin" ) ),
      value_( params.getParameter<std::string>("value") ),
      evaluation_( value_ )
  {
        produces< JetValueMap >(value_);
  }

  virtual ~JetDeltaRValueMapProducer() {}

    virtual void beginJob() override {}
    virtual void endJob() override {}

    virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {



      std::auto_ptr< JetValueMap > jetValueMap ( new JetValueMap() );
      edm::ValueMap<float>::Filler filler(*jetValueMap);


      edm::Handle< typename edm::View<T> > h_jets1;
      iEvent.getByToken( srcToken_, h_jets1 );
      edm::Handle< typename edm::View<T> > h_jets2;
      iEvent.getByToken( matchedToken_, h_jets2 );

      std::vector<float> values; values.reserve( h_jets1->size() );

      // Now set the Ptrs with the orphan handles.
      for ( typename edm::View<T>::const_iterator ibegin = h_jets1->begin(),
	      iend = h_jets1->end(), ijet = ibegin;
	    ijet != iend; ++ijet ) {
	bool found = false;
	for ( typename edm::View<T>::const_iterator jbegin = h_jets2->begin(),
		jend = h_jets2->end(), jjet = jbegin;
	      jjet != jend && !found; ++jjet ) {

	  if ( reco::deltaR( *ijet, *jjet) < distMin_ ) {
	    // Check the selection
	    float value = evaluation_(*jjet);
	    // Fill to the vector
	    values.push_back( value );
	    found = true;	    
	  }
	}// end loop over matched jets
      }// end loop over src jets
      
      filler.insert(h_jets1, values.begin(), values.end());
      filler.fill();

      // put  in Event
      iEvent.put(jetValueMap,value_);


    }

  protected:
    edm::EDGetTokenT< typename edm::View<T> >                  srcToken_;
    edm::EDGetTokenT< typename edm::View<T> >                  matchedToken_;
    double                         distMin_;
    std::string                    value_;
    StringObjectFunction<T>        evaluation_;

};

typedef JetDeltaRValueMapProducer<reco::Jet> RecoJetDeltaRValueMapProducer;

DEFINE_FWK_MODULE( RecoJetDeltaRValueMapProducer );
