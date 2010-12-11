#ifndef DataFormats_PatCandidates_TriggerObjectStandAlone_h
#define DataFormats_PatCandidates_TriggerObjectStandAlone_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerObjectStandAlone
//
// $Id: TriggerObjectStandAlone.h,v 1.6 2010/06/16 15:40:52 vadler Exp $
//
/**
  \class    pat::TriggerObjectStandAlone TriggerObjectStandAlone.h "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
  \brief    Analysis-level trigger object class (stand-alone)

   TriggerObjectStandAlone implements a container for trigger objects' information within the 'pat' namespace.
   These Trigger objects keep also information on filters and paths ot be saved independently or embedded into PAT objects.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerObjectStandAlone

  \author   Volker Adler
  \version  $Id: TriggerObjectStandAlone.h,v 1.6 2010/06/16 15:40:52 vadler Exp $
*/


#include "DataFormats/PatCandidates/interface/TriggerObject.h"


namespace pat {

  class TriggerObjectStandAlone : public TriggerObject {

      /// data members
      std::vector< std::string > filterLabels_;           // used for trigger match definition
      std::vector< std::string > pathNames_;              // used for trigger match definition
      std::vector< bool >        pathLastFilterAccepted_; // vector alligned with 'pathNames_'
                                                          // true, if corresponding path accepted and trigger object was used in last filter

    public:

      /// constructors and desctructor
      TriggerObjectStandAlone()                                                             : TriggerObject()           {};
      TriggerObjectStandAlone( const TriggerObject & trigObj )                              : TriggerObject( trigObj )  {};
      TriggerObjectStandAlone( const reco::Particle::LorentzVector & vec, int id = 0 )      : TriggerObject( vec, id )  {};
      TriggerObjectStandAlone( const reco::Particle::PolarLorentzVector & vec, int id = 0 ) : TriggerObject( vec, id )  {};
      TriggerObjectStandAlone( const trigger::TriggerObject & trigObj )                     : TriggerObject( trigObj )  {};
      TriggerObjectStandAlone( const reco::LeafCandidate & leafCand )                       : TriggerObject( leafCand ) {};
      virtual ~TriggerObjectStandAlone() {};

      /// methods
      void addFilterLabel( const std::string & filterLabel ) { if ( ! hasFilterLabel( filterLabel ) ) filterLabels_.push_back( filterLabel ); };
      void addPathName( const std::string & pathName, bool pathLastFilterAccepted = true );
      std::vector< std::string > filterLabels() const                                                                  { return filterLabels_; };
      std::vector< std::string > pathNames( bool pathLastFilterAccepted = true ) const;
      bool                       hasFilterLabel( const std::string & filterLabel ) const;
      bool                       filter( const std::string & filterLabel ) const                                       { return hasFilterLabel( filterLabel ); };
      bool                       hasPathName( const std::string & pathName, bool pathLastFilterAccepted = true ) const;
      bool                       path( const std::string & pathName, bool pathLastFilterAccepted = true ) const        { return hasPathName( pathName, pathLastFilterAccepted ); };
      bool                       hasPathLastFilterAccepted() const                                                     { return ( pathLastFilterAccepted_.size() > 0 && pathLastFilterAccepted_.size() == pathNames_.size() );  };
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
  /// const iterator over vector of persistent references to items in the same TriggerObjectStandAloneCollection
  typedef edm::RefVectorIterator< TriggerObjectStandAloneCollection >  TriggerObjectStandAloneRefVectorIterator;
  /// association of TriggerObjectStandAlones to store matches to Candidates
  typedef edm::Association< TriggerObjectStandAloneCollection >        TriggerObjectStandAloneMatch;

}


#endif
