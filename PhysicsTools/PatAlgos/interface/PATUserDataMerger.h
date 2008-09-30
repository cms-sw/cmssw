//
// $Id: PATUserDataMerger.h,$
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
  \version  $Id: PATUserDataMerger.h,$
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"

#include <iostream>


namespace pat {


  template<class ObjectType, class ValueType>
  class PATUserDataMerger {
    
  public:

    PATUserDataMerger() {}
    PATUserDataMerger(const edm::ParameterSet & iConfig);
    ~PATUserDataMerger() {}
    
    // Method to call from PATUserDataHelper to add information to the PATObject in question. 
    void add(PATObject<ObjectType> & patObject,
	     edm::Ptr<ObjectType> const & recoObject,
	     edm::Event const & iEvent, edm::EventSetup const & iSetup);

  private:

    // configurables
    std::vector<edm::InputTag>        userDataSrc_;   // ValueMap containing the user data

  };

// Constructor: Initilize user data src
template<class ObjectType, class ValueType>
PATUserDataMerger<ObjectType, ValueType>::PATUserDataMerger(const edm::ParameterSet & iConfig) :
  userDataSrc_(iConfig.getParameter<std::vector<edm::InputTag> >("src") )
{
}


/* ==================================================================================
     PATUserDataMerger::add 
            This expects four inputs:
	        patObject:         PATObject<ObjectType> to add to
		recoObject:        The base for the value map

		from Event:
		userDataSrc:       The data to add, which is a ValueMap keyed by recoObject
		
		from Setup:
		none currently

		This will simply add the UserData *'s from the value map that are 
		indexed by the reco objects, to the pat object's user data vector.
   ==================================================================================
*/

template<class ObjectType, class ValueType>
void PATUserDataMerger<ObjectType, ValueType>::add(PATObject<ObjectType> & patObject,
						   edm::Ptr<ObjectType> const & recoObject,
						   edm::Event const & iEvent, 
						   const edm::EventSetup & iSetup ) 
{

  std::vector<edm::InputTag>::const_iterator input_it = userDataSrc_.begin(),
    input_begin = userDataSrc_.begin(),
    input_end = userDataSrc_.end();

  for ( ; input_it != input_end; ++input_it ) {

    // Declare the object handles:
    // ValueMap containing Ptr's to the UserData that
    //   is associated to those PAT Objects
    edm::Handle<edm::ValueMap<edm::Ptr<ValueType> > > userData;

    // Get the objects by label
    iEvent.getByLabel( *input_it, userData );

    if ( userData->contains( recoObject.id() ) ) {
      patObject.addUserData( input_it->label(), *( (*userData)[recoObject] ) );
    }

  }
  
}


}
#endif
