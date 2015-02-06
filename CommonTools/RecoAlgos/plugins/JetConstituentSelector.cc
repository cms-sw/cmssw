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


#include "FWCore/Framework/interface/EDFilter.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "FWCore/Framework/interface/Event.h"

template < class T, typename C = std::vector<typename T::ConstituentTypeFwdPtr> >
class JetConstituentSelector : public edm::EDFilter {

public:

  typedef std::vector<T> JetsOutput;
  typedef C ConstituentsOutput;

  JetConstituentSelector ( edm::ParameterSet const & params ) :
      srcToken_( consumes< typename edm::View<T> >( params.getParameter<edm::InputTag>("src") ) ),
      cut_( params.getParameter<std::string>("cut") ),
      filter_(false),
      selector_( cut_ )
  {
	produces< JetsOutput >();
	produces< ConstituentsOutput > ("constituents");
  }

  virtual ~JetConstituentSelector() {}

    virtual void beginJob() override {}
    virtual void endJob() override {}

    virtual bool filter(edm::Event& iEvent, const edm::EventSetup& iSetup) override {

      std::auto_ptr< JetsOutput > jets ( new std::vector<T>() );
      std::auto_ptr< ConstituentsOutput > candsOut( new ConstituentsOutput  );

      edm::Handle< typename edm::View<T> > h_jets;
      iEvent.getByToken( srcToken_, h_jets );

      // Now set the Ptrs with the orphan handles.
      for ( typename edm::View<T>::const_iterator ibegin = h_jets->begin(),
	      iend = h_jets->end(), ijet = ibegin;
	    ijet != iend; ++ijet ) {

	// Check the selection
	if ( selector_(*ijet) ) {
	  // Add the jets that pass to the output collection
	  jets->push_back( *ijet );
	  for ( unsigned int ida = 0; ida < ijet->numberOfDaughters(); ++ida ) {
	    candsOut->push_back( typename ConstituentsOutput::value_type( ijet->daughterPtr(ida), ijet->daughterPtr(ida) ) );
	  }
	}
      }

      // put  in Event
      bool pass = jets->size() > 0;
      iEvent.put(jets);
      iEvent.put(candsOut, "constituents");

      if ( filter_ )
	return pass;
      else
	return true;

    }

  protected:
    edm::EDGetTokenT< typename edm::View<T> >                  srcToken_;
    std::string                    cut_;
    bool                           filter_;
    StringCutObjectSelector<T>   selector_;

};

typedef JetConstituentSelector<reco::PFJet> PFJetConstituentSelector;
typedef JetConstituentSelector<pat::Jet, std::vector< edm::FwdPtr<pat::PackedCandidate> > > PatJetConstituentSelector;
typedef JetConstituentSelector<reco::PFJet, std::vector< edm::FwdPtr<pat::PackedCandidate> > > MiniAODJetConstituentSelector;

DEFINE_FWK_MODULE( PFJetConstituentSelector );
DEFINE_FWK_MODULE( PatJetConstituentSelector );
DEFINE_FWK_MODULE( MiniAODJetConstituentSelector );
