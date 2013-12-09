//
//

#ifndef PhysicsTools_PatAlgos_PATUserDataHelper_h
#define PhysicsTools_PatAlgos_PATUserDataHelper_h

/**
  \class    pat::PATUserDataHelper PATUserDataHelper.h "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"
  \brief    Assists in assimilating all pat::UserData into pat objects.


            This will pull the following from the event stream (if they exist) and put them into the
	    object in question, all indexed by the reco objects that make up the pat objects in question:

	    * ValueMap<double>
	    * ValueMap<int>
	    * ValueMap<Ptr<UserData> >
	    * ValueMap<CandidatePtr>

	    This is accomplished by using PATUserDataMergers.

	    This also can add "in situ" string-parser-based methods directly.

  \author   Salvatore Rappoccio
  \version  $Id: PATUserDataHelper.h,v 1.8 2010/02/20 21:00:13 wmtan Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataMerger.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"


#include <iostream>


namespace pat {


  template<class ObjectType>
  class PATUserDataHelper {

  public:

    typedef StringObjectFunction<ObjectType>                      function_type;

    PATUserDataHelper() {}
    PATUserDataHelper(const edm::ParameterSet & iConfig, edm::ConsumesCollector && iC);
    ~PATUserDataHelper() {}

    static void fillDescription(edm::ParameterSetDescription & iDesc);

    // Adds information from user data to patObject,
    // using recoObject as the key
    void add(ObjectType & patObject,
	     edm::Event const & iEvent, edm::EventSetup const & iSetup);

  private:

    // Custom user data
    pat::PATUserDataMerger<ObjectType, pat::helper::AddUserPtr>      userDataMerger_;
    // User doubles
    pat::PATUserDataMerger<ObjectType, pat::helper::AddUserFloat>    userFloatMerger_;
    // User ints
    pat::PATUserDataMerger<ObjectType, pat::helper::AddUserInt>      userIntMerger_;
    // User candidate ptrs
    pat::PATUserDataMerger<ObjectType, pat::helper::AddUserCand>     userCandMerger_;

    // Inline functions that operate on ObjectType
    std::vector<std::string>                                          functionNames_;
    std::vector<std::string>                                          functionLabels_;
    std::vector<function_type >                                       functions_;

  };

// Constructor: Initilize user data src
template<class ObjectType>
PATUserDataHelper<ObjectType>::PATUserDataHelper(const edm::ParameterSet & iConfig, edm::ConsumesCollector && iC) :
  userDataMerger_   (iConfig.getParameter<edm::ParameterSet>("userClasses"), iC),
  userFloatMerger_  (iConfig.getParameter<edm::ParameterSet>("userFloats"), iC),
  userIntMerger_    (iConfig.getParameter<edm::ParameterSet>("userInts"), iC),
  userCandMerger_   (iConfig.getParameter<edm::ParameterSet>("userCands"), iC),
  functionNames_    (iConfig.getParameter<std::vector<std::string> >("userFunctions")),
  functionLabels_   (iConfig.getParameter<std::vector<std::string> >("userFunctionLabels"))
{

  // Make sure the sizes match
  if ( functionNames_.size() != functionLabels_.size() ) {
    throw cms::Exception("Size mismatch") << "userFunctions and userFunctionLabels do not have the same size, they must be the same\n";
  }
  // Loop over the function names, create a new string-parser function object
  // with all of them. This operates on ObjectType
  std::vector<std::string>::const_iterator funcBegin = functionNames_.begin(),
    funcEnd = functionNames_.end(),
    funcIt = funcBegin;
  for ( ; funcIt != funcEnd; ++funcIt) {
    functions_.push_back(  StringObjectFunction<ObjectType>( *funcIt ) );
  }
}


/* ==================================================================================
     PATUserDataHelper::add
            This expects four inputs:
	        patObject:         PATObject<ObjectType> to add to
		recoObject:        The base for the value maps
		iEvent:            Passed to the various data mergers
		iSetup:            "                                "

   ==================================================================================
*/

template<class ObjectType>
void PATUserDataHelper<ObjectType>::add(ObjectType & patObject,
					edm::Event const & iEvent,
					const edm::EventSetup& iSetup)
{

  // Add "complex" user data to the PAT object
  userDataMerger_.add(   patObject, iEvent, iSetup );
  userFloatMerger_.add(  patObject, iEvent, iSetup );
  userIntMerger_.add(    patObject, iEvent, iSetup );
  userCandMerger_.add(   patObject, iEvent, iSetup );

  // Add "inline" user-selected functions to the PAT object
  typename std::vector<function_type>::const_iterator funcBegin = functions_.begin(),
    funcEnd = functions_.end(),
    funcIt = funcBegin;
  if ( functionLabels_.size() == functions_.size() ) {
    for ( ; funcIt != funcEnd; ++funcIt) {
      double d = (*funcIt)( patObject );
      patObject.addUserFloat( functionLabels_[funcIt - funcBegin], d );
    }
  }


}


template<class ObjectType>
void PATUserDataHelper<ObjectType>::fillDescription(edm::ParameterSetDescription & iDesc)
{
  edm::ParameterSetDescription dataMergerPSet;
  pat::PATUserDataMerger<ObjectType, pat::helper::AddUserPtr>::fillDescription(dataMergerPSet);
  iDesc.add("userClasses", dataMergerPSet);
  iDesc.add("userFloats", dataMergerPSet);
  iDesc.add("userInts", dataMergerPSet);
  iDesc.add("userCands", dataMergerPSet);
  std::vector<std::string> emptyVectorOfStrings;
  iDesc.add<std::vector<std::string> >("userFunctions",emptyVectorOfStrings);
  iDesc.add<std::vector<std::string> >("userFunctionLabels",emptyVectorOfStrings);
}

}
#endif
