//
// $Id: PATUserDataMerger.h,v 1.11 2013/02/27 23:26:56 wmtan Exp $
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
  \version  $Id: PATUserDataMerger.h,v 1.11 2013/02/27 23:26:56 wmtan Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
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
    PATUserDataMerger(const edm::ParameterSet & iConfig);
    ~PATUserDataMerger() {}

    static void fillDescription(edm::ParameterSetDescription & iDesc);

    // Method to call from PATUserDataHelper to add information to the PATObject in question.
    void add(ObjectType & patObject,
	     edm::Event const & iEvent, edm::EventSetup const & iSetup);

  private:

    // configurables
    std::vector<edm::InputTag>  userDataSrc_;   // ValueMap containing the user data
    Operation                   loader_;

  };

}

// Constructor: Initilize user data src
template<typename ObjectType, typename Operation>
pat::PATUserDataMerger<ObjectType, Operation>::PATUserDataMerger(const edm::ParameterSet & iConfig) :
  userDataSrc_(iConfig.getParameter<std::vector<edm::InputTag> >("src") )
{
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

  std::vector<edm::InputTag>::const_iterator input_it = userDataSrc_.begin(),
//     input_begin = userDataSrc_.begin(), // warning from gcc461: variable 'input_begin' set but not used [-Wunused-but-set-variable]
    input_end = userDataSrc_.end();

  for ( ; input_it != input_end; ++input_it ) {

    // Declare the object handles:
    // ValueMap containing the values, or edm::Ptr's to the UserData that
    //   is associated to those PAT Objects
    edm::Handle<typename Operation::product_type> userData;

    // Get the objects by label
    if ( input_it->encode().size() == 0 ) continue;
    iEvent.getByLabel( *input_it, userData );

    edm::Ptr<reco::Candidate> recoObject = patObject.originalObjectRef();
    if ( userData->contains( recoObject.id() ) ) {
      loader_.addData( patObject, input_it->encode(), (*userData)[recoObject]);
    }

  }

}

template<class ObjectType, typename Operation>
void
pat::PATUserDataMerger<ObjectType, Operation>::fillDescription(edm::ParameterSetDescription & iDesc)
{
  iDesc.add<std::vector<edm::InputTag> >("src");
}

#endif
