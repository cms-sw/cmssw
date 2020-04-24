//
//

#ifndef PhysicsTools_PatAlgos_PATUserDataMerger_h
#define PhysicsTools_PatAlgos_PATUserDataMerger_h

/**
  \class    pat::PATUserDataMerger PATUserDataMerger.h "PhysicsTools/PatAlgos/interface/PATUserDataMerger.h"
  \brief    Assimilates pat::UserData into pat objects

            This expects one input:
		src:               The data to add to the objects that get passed to this
		                   object, which are ValueMaps to some type (like UserData or double).

		This will be called from PATUserDataHelper to handle the templated cases
		like UserData or double. PATUserDataHelper will then add all the instantiated
		cases.

  \author   Salvatore Rappoccio
  \version  $Id: PATUserDataMerger.h,v 1.10 2011/10/26 17:01:25 vadler Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <iostream>


namespace pat {
  namespace helper {
    struct AddUserInt {
        typedef int                       value_type;
        typedef edm::ValueMap<value_type> product_type;
        template<typename ObjectType>
        void addData(ObjectType &obj, const std::string & key, const value_type &val) { obj.addUserInt(key, val); }
    };
    struct AddUserFloat {
        typedef float                     value_type;
        typedef edm::ValueMap<value_type> product_type;
        template<typename ObjectType>
        void addData(ObjectType &obj, const std::string & key, const value_type &val) { obj.addUserFloat(key, val); }
    };
    struct AddUserPtr {
        typedef edm::Ptr<UserData>        value_type;
        typedef edm::ValueMap<value_type> product_type;
        template<typename ObjectType>
        void addData(ObjectType &obj, const std::string & key, const value_type &val) {
              obj.addUserDataFromPtr(key, val);
        }
    };
    struct AddUserCand {
        typedef reco::CandidatePtr        value_type;
        typedef edm::ValueMap<value_type> product_type;
        template<typename ObjectType>
        void addData(ObjectType &obj, const std::string & key, const value_type &val) { obj.addUserCand(key, val); }
    };

  }

  template<typename ObjectType, typename Operation>
  class PATUserDataMerger {

  public:

    PATUserDataMerger() {}
    PATUserDataMerger(const edm::ParameterSet & iConfig, edm::ConsumesCollector & iC);
    ~PATUserDataMerger() {}

    static void fillDescription(edm::ParameterSetDescription & iDesc);

    // Method to call from PATUserDataHelper to add information to the PATObject in question.
    void add(ObjectType & patObject,
	     edm::Event const & iEvent, edm::EventSetup const & iSetup);

  private:

    // configurables
    std::vector<edm::InputTag> userDataSrc_;
    std::vector<edm::EDGetTokenT<typename Operation::product_type> > userDataSrcTokens_;
    std::vector<std::string> labelPostfixesToStrip_, labels_;
    Operation                   loader_;

  };

}

// Constructor: Initilize user data src
template<typename ObjectType, typename Operation>
pat::PATUserDataMerger<ObjectType, Operation>::PATUserDataMerger(const edm::ParameterSet & iConfig, edm::ConsumesCollector & iC) :
  userDataSrc_(iConfig.getParameter<std::vector<edm::InputTag> >("src")),
  labelPostfixesToStrip_(iConfig.existsAs<std::vector<std::string>>("labelPostfixesToStrip") ? iConfig.getParameter<std::vector<std::string>>("labelPostfixesToStrip") : std::vector<std::string>())
{
  for ( std::vector<edm::InputTag>::const_iterator input_it = userDataSrc_.begin(); input_it !=  userDataSrc_.end(); ++input_it ) {
    userDataSrcTokens_.push_back( iC.consumes< typename Operation::product_type >( *input_it ) );
  }
  for (edm::InputTag tag : userDataSrc_) { // copy by value
      for (const std::string & stripme : labelPostfixesToStrip_) {
          auto match = tag.label().rfind(stripme);
          if (match == (tag.label().length() - stripme.length())) {
              tag = edm::InputTag(tag.label().substr(0, match), tag.instance(), tag.process());
          } 
      }
      labels_.push_back(tag.encode());
  }
}


/* ==================================================================================
     PATUserDataMerger::add
            This expects four inputs:
	        patObject:         ObjectType to add to

		from Event:
		userDataSrc:       The data to add, which is a ValueMap keyed by recoObject

		from Setup:
		none currently

		This will simply add the UserData *'s from the value map that are
		indexed by the reco objects, to the pat object's user data vector.
   ==================================================================================
*/

template<class ObjectType, typename Operation>
void
pat::PATUserDataMerger<ObjectType, Operation>::add(ObjectType & patObject,
						   edm::Event const & iEvent,
						   const edm::EventSetup& iSetup)
{

  typename std::vector<edm::EDGetTokenT<typename Operation::product_type> >::const_iterator token_begin = userDataSrcTokens_.begin(), token_it = userDataSrcTokens_.begin(), token_end = userDataSrcTokens_.end();

  for ( ; token_it != token_end; ++token_it ) {
    const std::string & encoded = (labels_.at(token_it - token_begin));

    // Declare the object handles:
    // ValueMap containing the values, or edm::Ptr's to the UserData that
    //   is associated to those PAT Objects
    edm::Handle<typename Operation::product_type> userData;

    // Get the objects by label
    if ( encoded.size() == 0 ) continue;
    iEvent.getByToken( *token_it, userData );

    edm::Ptr<reco::Candidate> recoObject = patObject.originalObjectRef();
    loader_.addData( patObject, encoded, (*userData)[recoObject]);

  }

}

template<class ObjectType, typename Operation>
void
pat::PATUserDataMerger<ObjectType, Operation>::fillDescription(edm::ParameterSetDescription & iDesc)
{
  iDesc.add<std::vector<edm::InputTag> >("src");
  iDesc.addOptional<std::vector<std::string>>("labelPostfixesToStrip", std::vector<std::string>());
}

#endif
