/* \class JetDeltaRTagInfoValueMapProducer<T,I>
 *
 * Inputs two jet collections ("src" and "matched", type T), with
 * the second having tag infos run on them ("matchedTagInfos", type I). 
 * The jet collections are matched using delta-R matching. The
 * tag infos from the second collection are then rewritten into a
 * new TagInfoCollection, keyed to the first jet collection. 
 * This can be used in the miniAOD to associate the previously-run
 * CA8 "Top-Tagged" jets with their CATopTagInfos to the AK8 jets
 * that are stored in the miniAOD. 
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


#include "FWCore/Framework/interface/global/EDProducer.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/CATopJetTagInfo.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Math/interface/deltaR.h"

template < class T, class I >
class JetDeltaRTagInfoValueMapProducer : public edm::global::EDProducer<> {

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

  virtual void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override {

    std::auto_ptr< TagInfosCollection > mappedTagInfos ( new TagInfosCollection() );


    edm::Handle< typename edm::View<T> > h_jets1;
    iEvent.getByToken( srcToken_, h_jets1 );
    edm::Handle< typename edm::View<T> > h_jets2;
    iEvent.getByToken( matchedToken_, h_jets2 );
    edm::Handle< typename edm::View<I> > h_tagInfos;
    iEvent.getByToken( matchedTagInfosToken_, h_tagInfos );

    const double distMax2 = distMax_*distMax_;

    std::vector<bool> jets2_locks( h_jets2->size(), false );

    for ( typename edm::View<T>::const_iterator ibegin = h_jets1->begin(),
          iend = h_jets1->end(), ijet = ibegin;
          ijet != iend; ++ijet )
    {
      float matched_dR2 = 1e9;
      int matched_index = -1;

      //std::cout << "Looking at jet " << ijet - ibegin << ", mass = " << ijet->mass() << std::endl;
     
      for ( typename edm::View<T>::const_iterator jbegin = h_jets2->begin(),
            jend = h_jets2->end(), jjet = jbegin;
            jjet != jend; ++jjet )
      {
        int index=jjet - jbegin;

	//std::cout << "Checking jet " << index << ", mass = " << jjet->mass() << std::endl;

        if( jets2_locks.at(index) ) continue; // skip jets that have already been matched

        float temp_dR2 = reco::deltaR2(ijet->eta(),ijet->phi(),jjet->eta(),jjet->phi());
        if ( temp_dR2 < matched_dR2 )
        {
          matched_dR2 = temp_dR2;
          matched_index = index;
        }
      }// end loop over src jets

      I mappedTagInfo; 

      if( matched_index>=0 && static_cast<size_t>(matched_index) < h_tagInfos->size() )
      {
        if ( matched_dR2 > distMax2 )
          LogDebug("MatchedJetsFarApart") << "Matched jets separated by dR greater than distMax=" << distMax_;
        else
        {
          jets2_locks.at(matched_index) = true;
	  
          auto otherTagInfo = h_tagInfos->at( matched_index );
	  otherTagInfo.setJetRef( h_jets1->refAt( ijet - ibegin) );
	  mappedTagInfo = otherTagInfo;
	  //std::cout << "Matched! : " << matched_index << ", mass = " << mappedTagInfo.properties().topMass << std::endl;
        }
      }

      mappedTagInfos->push_back( mappedTagInfo );

    }// end loop over matched jets
    

    // put  in Event
    iEvent.put(mappedTagInfos);
  }

  const edm::EDGetTokenT< typename edm::View<T> >  srcToken_;
  const edm::EDGetTokenT< typename edm::View<T> >  matchedToken_;
  const edm::EDGetTokenT< typename edm::View<I> >  matchedTagInfosToken_;
  const double                                     distMax_;

};

typedef JetDeltaRTagInfoValueMapProducer<reco::Jet, reco::CATopJetTagInfo> RecoJetDeltaRTagInfoValueMapProducer;

DEFINE_FWK_MODULE( RecoJetDeltaRTagInfoValueMapProducer );
