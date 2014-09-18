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
#include "DataFormats/JetReco/interface/CATopJetTagInfo.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaR.h"

template < class T, class I >
class JetDeltaRTagInfoValueMapProducer : public edm::EDProducer {

public:

  typedef std::vector<T> JetsInput;
  typedef std::vector<I> TagInfosCollection;

  JetDeltaRTagInfoValueMapProducer ( edm::ParameterSet const & params ) :
      srcToken_( consumes< typename edm::View<T> >( params.getParameter<edm::InputTag>("src") ) ),
      matchedToken_( consumes< typename edm::View<T> >( params.getParameter<edm::InputTag>( "matched" ) ) ),
      matchedTagInfosToken_( consumes< typename edm::View<I> >( params.getParameter<edm::InputTag>( "matchedTagInfos" ) ) ),
      distMax_( params.getParameter<double>( "distMax" ) )
  {
        produces< TagInfosCollection >();
  }

  virtual ~JetDeltaRTagInfoValueMapProducer() {}

private:

  virtual void beginJob() override {}
  virtual void endJob() override {}

  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override {

    std::auto_ptr< TagInfosCollection > mappedTagInfos ( new TagInfosCollection() );


    edm::Handle< typename edm::View<T> > h_jets1;
    iEvent.getByToken( srcToken_, h_jets1 );
    edm::Handle< typename edm::View<T> > h_jets2;
    iEvent.getByToken( matchedToken_, h_jets2 );
    edm::Handle< typename edm::View<I> > h_tagInfos;
    iEvent.getByToken( matchedTagInfosToken_, h_tagInfos );


    std::vector<float> values( h_jets1->size(), -99999 );
    std::vector<bool> jets1_locks( h_jets1->size(), false );

    for ( typename edm::View<T>::const_iterator ibegin = h_jets2->begin(),
          iend = h_jets2->end(), ijet = ibegin;
          ijet != iend; ++ijet )
    {
      float matched_dR2 = 1e9;
      int matched_index = -1;
     
      for ( typename edm::View<T>::const_iterator jbegin = h_jets1->begin(),
            jend = h_jets1->end(), jjet = jbegin;
            jjet != jend; ++jjet )
      {
        int index=jjet - jbegin;

        if( jets1_locks.at(index) ) continue; // skip jets that have already been matched

        float temp_dR2 = reco::deltaR2(ijet->eta(),ijet->phi(),jjet->eta(),jjet->phi());
        if ( temp_dR2 < matched_dR2 )
        {
          matched_dR2 = temp_dR2;
          matched_index = index;
        }
      }// end loop over src jets

      I mappedTagInfo; 

      if( matched_index>=0 )
      {
        if ( matched_dR2 > distMax_*distMax_ )
          edm::LogWarning("MatchedJetsFarApart") << "Matched jets separated by dR greater than distMax=" << distMax_;
        else
        {
          jets1_locks.at(matched_index) = true;


          mappedTagInfo = h_tagInfos->at( matched_index );
        }
      }

      mappedTagInfos->push_back( mappedTagInfo );

    }// end loop over matched jets
    

    // put  in Event
    iEvent.put(mappedTagInfos);
  }

  edm::EDGetTokenT< typename edm::View<T> >  srcToken_;
  edm::EDGetTokenT< typename edm::View<T> >  matchedToken_;
  edm::EDGetTokenT< typename edm::View<I> >  matchedTagInfosToken_;
  double                                     distMax_;

};

typedef JetDeltaRTagInfoValueMapProducer<reco::Jet, reco::CATopJetTagInfo> RecoJetDeltaRTagInfoValueMapProducer;

DEFINE_FWK_MODULE( RecoJetDeltaRTagInfoValueMapProducer );
