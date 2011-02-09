#ifndef DataFormats_PatCandidates_TriggerObjectStandAlone_h
#define DataFormats_PatCandidates_TriggerObjectStandAlone_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerObjectStandAlone
//
// $Id: TriggerObjectStandAlone.h,v 1.7 2010/12/11 21:25:44 vadler Exp $
//
/**
  \class    pat::TriggerObjectStandAlone TriggerObjectStandAlone.h "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
  \brief    Analysis-level trigger object class (stand-alone)

   TriggerObjectStandAlone implements a container for trigger objects' information within the 'pat' namespace.
   These Trigger objects keep also information on filters and paths ot be saved independently or embedded into PAT objects.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerObjectStandAlone

  \author   Volker Adler
  \version  $Id: TriggerObjectStandAlone.h,v 1.7 2010/12/11 21:25:44 vadler Exp $
*/


#include "DataFormats/PatCandidates/interface/TriggerObject.h"

#include <algorithm>


namespace pat {

  class TriggerObjectStandAlone : public TriggerObject {

      /// Data Members

      /// Vector of labels of all filters the trigger objects has been used in
      std::vector< std::string > filterLabels_;
      /// Vector of names of all paths the trigger objects has been used in
      std::vector< std::string > pathNames_;
      /// Vector alligned with 'pathNames_' of boolean indicating the usage of the trigger object
      /// An element is true, if the corresponding path succeeded and the trigger object was used in the last filter.
      /// The vector is empty for data (size 0), if the according information is not available in data.
      std::vector< bool > pathLastFilterAccepted_;

    public:

      /// Constructors and Destructor

      /// Default constructor
      TriggerObjectStandAlone() : TriggerObject() {};
      /// Constructor from pat::TriggerObject
      TriggerObjectStandAlone( const TriggerObject & trigObj ) : TriggerObject( trigObj ) {};
      /// Constructor from trigger::TriggerObject
      TriggerObjectStandAlone( const trigger::TriggerObject & trigObj ) : TriggerObject( trigObj ) {};
      /// Constructor from reco::Candidate
      TriggerObjectStandAlone( const reco::LeafCandidate & leafCand ) : TriggerObject( leafCand ) {};
      /// Constructors from Lorentz-vectors and (optional) PDG ID
      TriggerObjectStandAlone( const reco::Particle::LorentzVector & vec, int id = 0 ) : TriggerObject( vec, id ) {};
      TriggerObjectStandAlone( const reco::Particle::PolarLorentzVector & vec, int id = 0 ) : TriggerObject( vec, id ) {};

      /// Destructor
      virtual ~TriggerObjectStandAlone() {};

      /// Methods

      /// Adds a new filter label
      void addFilterLabel( const std::string & filterLabel ) { if ( ! hasFilterLabel( filterLabel ) ) filterLabels_.push_back( filterLabel ); };
      /// Adds a new path name
      void addPathName( const std::string & pathName, bool pathLastFilterAccepted = true );
      /// Gets all filter labels
      std::vector< std::string > filterLabels() const { return filterLabels_; };
      /// Gets all path names
      std::vector< std::string > pathNames( bool pathLastFilterAccepted = true ) const;
      /// Gets the pat::TriggerObject (parent class)
      TriggerObject triggerObject();
      /// Checks, if a certain filter label is assigned
      bool hasFilterLabel( const std::string & filterLabel ) const { return ( std::find( filterLabels_.begin(), filterLabels_.end(), filterLabel ) != filterLabels_.end()); };
      /// Checks, if a certain path name is assigned
      bool hasPathName( const std::string & pathName, bool pathLastFilterAccepted = true ) const;
      /// Checks, if the usage indicator vector has been filled
      bool hasPathLastFilterAccepted() const                                                     { return ( pathLastFilterAccepted_.size() > 0 && pathLastFilterAccepted_.size() == pathNames_.size() ); };

      /// Special methods for the cut string parser
      /// - argument types usable in the cut string parser
      /// - short names for readable configuration files

      /// Calls 'hasFilterLabel(...)'
      bool filter( const std::string & filterLabel ) const { return hasFilterLabel( filterLabel ); };
      /// Calls 'hasPathName(...)'
      bool path( const std::string & pathName, unsigned pathLastFilterAccepted = true ) const { return hasPathName( pathName, pathLastFilterAccepted ); };

  };


  /// Collection of TriggerObjectStandAlone
  typedef std::vector< TriggerObjectStandAlone >                       TriggerObjectStandAloneCollection;
  /// Persistent reference to an item in a TriggerObjectStandAloneCollection
  typedef edm::Ref< TriggerObjectStandAloneCollection >                TriggerObjectStandAloneRef;
  /// Persistent reference to a TriggerObjectStandAloneCollection product
  typedef edm::RefProd< TriggerObjectStandAloneCollection >            TriggerObjectStandAloneRefProd;
  /// Vector of persistent references to items in the same TriggerObjectStandAloneCollection
  typedef edm::RefVector< TriggerObjectStandAloneCollection >          TriggerObjectStandAloneRefVector;
  /// Const iterator over vector of persistent references to items in the same TriggerObjectStandAloneCollection
  typedef edm::RefVectorIterator< TriggerObjectStandAloneCollection >  TriggerObjectStandAloneRefVectorIterator;
  /// Association of TriggerObjectStandAlones to store matches to Candidates
  typedef edm::Association< TriggerObjectStandAloneCollection >        TriggerObjectStandAloneMatch;

}


#endif
