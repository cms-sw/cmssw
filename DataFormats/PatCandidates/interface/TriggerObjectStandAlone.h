#ifndef DataFormats_PatCandidates_TriggerObjectStandAlone_h
#define DataFormats_PatCandidates_TriggerObjectStandAlone_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerObjectStandAlone
//
// $Id: TriggerObjectStandAlone.h,v 1.1.2.1 2009/03/27 21:34:45 vadler Exp $
//
/**
  \class    pat::TriggerObjectStandAlone TriggerObjectStandAlone.h "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
  \brief    Analysis-level trigger object class

   TriggerObjectStandAlone implements a container for trigger objects' information within the 'pat' namespace.
   These Trigger objects keep also information on filters and paths ot be saved independently or embedded into PAT objects.
   It inherits from pat::TriggerObject and adds the following data members:
   - [to be filled]

  \author   Volker Adler
  \version  $Id: TriggerObjectStandAlone.h,v 1.1.2.1 2009/03/27 21:34:45 vadler Exp $
*/


#include "DataFormats/PatCandidates/interface/TriggerObject.h"

#include <map>
#include <string>
#include <vector>

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/Association.h"


namespace pat {
 
  class TriggerObjectStandAlone : public TriggerObject {
                                           
      /// data members
      std::vector< std::string > filterLabels_; // used for trigger match definition
      std::vector< std::string > pathNames_;    // used for trigger match definition

    public:

      /// constructors and desctructor
      TriggerObjectStandAlone()                                                             : TriggerObject()          {};
      TriggerObjectStandAlone( const TriggerObject & trigObj )                              : TriggerObject( trigObj ) {};
      TriggerObjectStandAlone( const reco::Particle::LorentzVector & vec, int id = 0 )      : TriggerObject( vec, id ) {};
      TriggerObjectStandAlone( const reco::Particle::PolarLorentzVector & vec, int id = 0 ) : TriggerObject( vec, id ) {};
      TriggerObjectStandAlone( const trigger::TriggerObject & trigObj )                     : TriggerObject( trigObj ) {};
      virtual ~TriggerObjectStandAlone() {};

      /// methods
      void addFilterLabel( const std::string & filterLabel ) { if ( ! hasFilterLabel( filterLabel ) ) filterLabels_.push_back( filterLabel ); };
      void addPathName( const std::string & pathName )       { if ( ! hasPathName( pathName ) )       pathNames_.push_back( pathName ); };
      std::vector< std::string > filterLabels() const { return filterLabels_; };
      std::vector< std::string > pathNames() const    { return pathNames_; };
      bool                       hasFilterLabel( const std::string & filterLabel ) const;
      bool                       hasPathName( const std::string & pathName ) const;
      TriggerObject              triggerObject(); // returns "pure" pat::TriggerObject w/o add-on

  };
  

  /// collection of TriggerObjectStandAlone
  typedef std::vector< TriggerObjectStandAlone >                       TriggerObjectStandAloneCollection;
  /// persistent reference to an item in a TriggerObjectStandAloneCollection
  typedef edm::Ref< TriggerObjectStandAloneCollection >                TriggerObjectStandAloneRef;
  /// persistent reference to a TriggerObjectStandAloneCollection product
  typedef edm::RefProd< TriggerObjectStandAloneCollection >            TriggerObjectStandAloneRefProd;
  /// vector of persistent references to items in the same TriggerObjectStandAloneCollection
  typedef edm::RefVector< TriggerObjectStandAloneCollection >          TriggerObjectStandAloneRefVector;
  /// association of TriggerObjectStandAlones to store matches to Candidates
  typedef edm::Association< TriggerObjectStandAloneCollection >        TriggerObjectStandAloneMatch;

}


#endif
