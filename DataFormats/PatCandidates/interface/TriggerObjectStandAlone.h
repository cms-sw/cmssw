#ifndef DataFormats_PatCandidates_TriggerObjectStandAlone_h
#define DataFormats_PatCandidates_TriggerObjectStandAlone_h


// -*- C++ -*-
//
// Package:    PatCandidates
// Class:      pat::TriggerObjectStandAlone
//
//
/**
  \class    pat::TriggerObjectStandAlone TriggerObjectStandAlone.h "DataFormats/PatCandidates/interface/TriggerObjectStandAlone.h"
  \brief    Analysis-level trigger object class (stand-alone)

   TriggerObjectStandAlone implements a container for trigger objects' information within the 'pat' namespace.
   These Trigger objects keep also information on filters and paths to be saved independently or embedded into PAT objects.
   The TriggerObjectStandAlone is also the data format used in the PAT trigger object matching.
   For detailed information, consult
   https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#TriggerObjectStandAlone

  \author   Volker Adler
*/


#include "DataFormats/PatCandidates/interface/TriggerObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/EventBase.h"
namespace edm { class TriggerNames; }

namespace pat {

  class TriggerObjectStandAlone : public TriggerObject {

      /// Data Members
      /// Keeping the old names of the data members for backward compatibility,
      /// although they refer only to HLT objects.

      /// Vector of labels of all HLT filters or names of L1 conditions the trigger objects has been used in
      std::vector< std::string > filterLabels_;
      std::vector< uint16_t >    filterLabelIndices_; 

      /// Vector of names of all HLT paths or L1 algorithms the trigger objects has been used in
      std::vector< std::string > pathNames_;  
      std::vector< uint16_t >    pathIndices_; 

      /// Vector alligned with 'pathNames_' of boolean indicating the usage of the trigger object
      /// An element is true, if the corresponding path succeeded and the trigger object was used in the last filter (HLT)
      /// or the corresponding algorithm succeeded as well as the corresponding condition (L1).
      /// The vector is empty for data (size 0), if the according information is not available.
      std::vector< bool > pathLastFilterAccepted_;
      /// Vector alligned with 'pathNames_' of boolean indicating the usage of the trigger object
      /// An element is true, if the corresponding path succeeded and the trigger object was used in an L3 filter (HLT only)
      /// The vector is empty for data (size 0), if the according information is not available.
      std::vector< bool > pathL3FilterAccepted_;

      edm::ParameterSetID  psetId_;

      /// Constants

      /// Constant defining the wild-card used in 'hasAnyName()'
      static const char wildcard_ = '*';

      /// Private methods

      /// Checks a string vector for occurence of a certain string, incl. wild-card mechanism
      bool hasAnyName( const std::string & name, const std::vector< std::string > & nameVec ) const;
      /// Adds a new HLT filter label or L1 condition name
      void addFilterOrCondition( const std::string & name ) { if ( ! hasFilterOrCondition( name ) ) filterLabels_.push_back( name ); };
      /// Adds a new HLT path or L1 algorithm name
      void addPathOrAlgorithm( const std::string & name, bool pathLastFilterAccepted, bool pathL3FilterAccepted );
      /// Gets all HLT filter labels or L1 condition names
      const std::vector< std::string > & filtersOrConditions() const { checkIfFiltersAreUnpacked(); return filterLabels_; };
      /// Gets all HLT path or L1 algorithm names
      std::vector< std::string > pathsOrAlgorithms( bool pathLastFilterAccepted, bool pathL3FilterAccepted ) const;
      /// Checks, if a certain HLT filter label or L1 condition name is assigned
      bool hasFilterOrCondition( const std::string & name ) const;
      /// Checks, if a certain HLT path or L1 algorithm name is assigned
      bool hasPathOrAlgorithm( const std::string & name, bool pathLastFilterAccepted, bool pathL3FilterAccepted ) const;
      /// Check, if the usage indicator vectors have been filled
      bool hasLastFilter() const { return ( !pathLastFilterAccepted_.empty() && pathLastFilterAccepted_.size() == pathNames_.size() ); };
      bool hasL3Filter() const { return ( !pathL3FilterAccepted_.empty() && pathL3FilterAccepted_.size() == pathNames_.size() ); };

      /// Check if trigger names have been packed by calling packPathNames() and not yet unpacked
      bool checkIfPathsAreUnpacked(bool throwIfPacked=true) const ;
      /// Check if trigger names have been packed by calling packFilterLabels() and not yet unpacked
      bool checkIfFiltersAreUnpacked(bool  throwIfPacked=true) const ;

    public:

      /// Constructors and Destructor

      /// Default constructor
      TriggerObjectStandAlone();
      /// Constructor from pat::TriggerObject
      TriggerObjectStandAlone( const TriggerObject & trigObj );
      /// Constructor from trigger::TriggerObject
      TriggerObjectStandAlone( const trigger::TriggerObject & trigObj );
      /// Constructor from reco::Candidate
      TriggerObjectStandAlone( const reco::LeafCandidate & leafCand );
      /// Constructors from Lorentz-vectors and (optional) PDG ID
      TriggerObjectStandAlone( const reco::Particle::LorentzVector & vec, int id = 0 );
      TriggerObjectStandAlone( const reco::Particle::PolarLorentzVector & vec, int id = 0 );
      TriggerObjectStandAlone( const TriggerObjectStandAlone& ) = default;

      /// Destructor
      ~TriggerObjectStandAlone() override {};

      /// Methods

      TriggerObjectStandAlone copy() const {
        return *this;
      }

      /// Adds a new HLT filter label
      void addFilterLabel( const std::string & filterLabel ) { addFilterOrCondition( filterLabel ); };
      /// Adds a new L1 condition name
      void addConditionName( const std::string & conditionName ) { addFilterOrCondition( conditionName ); };
      /// Adds a new HLT path name
      void addPathName( const std::string & pathName, bool pathLastFilterAccepted = true, bool pathL3FilterAccepted = true ) { addPathOrAlgorithm( pathName, pathLastFilterAccepted, pathL3FilterAccepted ); };
      /// Adds a new L1 algorithm name
      void addAlgorithmName( const std::string & algorithmName, bool algoCondAccepted = true ) { addPathOrAlgorithm( algorithmName, algoCondAccepted, false ); };
      /// Gets all HLT filter labels
      const std::vector< std::string > & filterLabels() const { return filtersOrConditions(); };
      /// Gets all L1 condition names
      const std::vector< std::string > & conditionNames() const { return filtersOrConditions(); };
      /// Gets all HLT path names
      std::vector< std::string > pathNames( bool pathLastFilterAccepted = false, bool pathL3FilterAccepted = true ) const { return pathsOrAlgorithms( pathLastFilterAccepted, pathL3FilterAccepted ); };
      /// Gets all L1 algorithm names
      std::vector< std::string > algorithmNames( bool algoCondAccepted = true ) const { return pathsOrAlgorithms( algoCondAccepted, false ); };
      /// Gets the pat::TriggerObject (parent class)
      TriggerObject triggerObject();
      /// Checks, if a certain HLT filter label is assigned
      bool hasFilterLabel( const std::string & filterLabel ) const { return hasFilterOrCondition( filterLabel ); };
      /// Checks, if a certain L1 condition name is assigned
      bool hasConditionName( const std::string & conditionName ) const { return hasFilterOrCondition( conditionName ); };
      /// Checks, if a certain HLT path name is assigned
      bool hasPathName( const std::string & pathName, bool pathLastFilterAccepted = false, bool pathL3FilterAccepted = true ) const { return hasPathOrAlgorithm( pathName, pathLastFilterAccepted, pathL3FilterAccepted ); };
      /// Checks, if a certain L1 algorithm name is assigned
      bool hasAlgorithmName( const std::string & algorithmName, bool algoCondAccepted = true ) const { return hasPathOrAlgorithm( algorithmName, algoCondAccepted, false ); };
      /// Checks, if a certain label of original collection is assigned (method overrides)
      bool hasCollection( const std::string & collName ) const override;
      bool hasCollection( const edm::InputTag & collName ) const override { return hasCollection( collName.encode() ); };
      /// Checks, if the usage indicator vector has been filled
      bool hasPathLastFilterAccepted() const { return hasLastFilter(); };
      bool hasAlgoCondAccepted() const { return hasLastFilter(); };
      bool hasPathL3FilterAccepted() const { return hasL3Filter(); };

      /// Special methods for the cut string parser
      /// - argument types usable in the cut string parser
      /// - short names for readable configuration files

      /// Calls 'hasFilterLabel(...)'
      bool filter( const std::string & filterLabel ) const { return hasFilterLabel( filterLabel ); };
      /// Calls 'hasConditionName(...)'
      bool cond( const std::string & conditionName ) const { return hasConditionName( conditionName ); };
      /// Calls 'hasPathName(...)'
      bool path( const std::string & pathName, unsigned pathLastFilterAccepted = 0, unsigned pathL3FilterAccepted = 1 ) const { return hasPathName( pathName, bool( pathLastFilterAccepted ), bool( pathL3FilterAccepted ) ); };
      /// Calls 'hasAlgorithmName(...)'
      bool algo( const std::string & algorithmName, unsigned algoCondAccepted = 1 ) const { return hasAlgorithmName( algorithmName, bool( algoCondAccepted ) ); };
      /// Calls 'hasCollection(...)' (method override)
      bool coll( const std::string & collName ) const override { return hasCollection( collName ); };

      ///set the psetid of the trigger process
      void setPSetID(const edm::ParameterSetID &psetId){psetId_=psetId;}
      
      const edm::ParameterSetID &psetID() const {return psetId_;}


      void packFilterLabels(const edm::EventBase &event,const edm::TriggerResults &res);
      ///  pack trigger names into indices 
      void packPathNames(const edm::TriggerNames &names) ;
      ///  unpack trigger names into indices 
      void unpackPathNames(const edm::TriggerNames &names) ;
      ///  pack filter labels into indices; note that the labels must be sorted!
      void packFilterLabels(const std::vector<std::string> &labels)  ;
      ///  unpack filter labels from indices
      void unpackFilterLabels(const std::vector<std::string> &labels)  ;
      ///  unpack filter labels from indices
      void unpackFilterLabels(const edm::EventBase & event,const edm::TriggerResults &res)  ;
      /// unpack both filter labels and trigger names
      void unpackNamesAndLabels(const edm::EventBase & event,const edm::TriggerResults &res);

      /// reduce the precision on the 4-vector
      void packP4();

      std::vector<std::string> const* allLabels(edm::ParameterSetID const& psetid,const edm::EventBase & event,const edm::TriggerResults &res) const ;

  };


  /// Collection of TriggerObjectStandAlone
  typedef std::vector< TriggerObjectStandAlone >                      TriggerObjectStandAloneCollection;
  /// Persistent reference to an item in a TriggerObjectStandAloneCollection
  typedef edm::Ref< TriggerObjectStandAloneCollection >               TriggerObjectStandAloneRef;
  /// Persistent reference to a TriggerObjectStandAloneCollection product
  typedef edm::RefProd< TriggerObjectStandAloneCollection >           TriggerObjectStandAloneRefProd;
  /// Vector of persistent references to items in the same TriggerObjectStandAloneCollection
  typedef edm::RefVector< TriggerObjectStandAloneCollection >         TriggerObjectStandAloneRefVector;
  /// Const iterator over vector of persistent references to items in the same TriggerObjectStandAloneCollection
  typedef edm::RefVectorIterator< TriggerObjectStandAloneCollection > TriggerObjectStandAloneRefVectorIterator;
  /// Association of TriggerObjectStandAlones to store matches to Candidates
  typedef edm::Association< TriggerObjectStandAloneCollection >       TriggerObjectStandAloneMatch;

}


#endif
