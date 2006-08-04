// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1ParticleMap
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Werner Sun
//         Created:  Wed Jul 26 14:42:56 EDT 2006
// $Id: L1ParticleMap.cc,v 1.3 2006/08/02 20:48:56 wsun Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"

using namespace l1extra ;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

std::string
L1ParticleMap::triggerNames_[ kNumOfL1TriggerTypes ] = {
   "SingleEM",
      "DoubleEM",
      "SingleMuon",
      "DoubleMuon",
      "SingleTau",
      "DoubleTau",
      "SingleJet",
      "DoubleJet",
      "TripleJet",
      "QuadJet",
      "HT",
      "MET",
      "HT+MET",
      "Jet+MET",
      "Tau+MET",
      "Muon+MET",
      "EM+MET",
      "Muon+Jet",
      "EM+Jet",
      "Muon+Tau",
      "EM+Tau",
      "EM+Muon",
      "SingleJet140",
      "SingleJet60",
      "SingleJet20"
      } ;

//
// constructors and destructor
//
L1ParticleMap::L1ParticleMap()
{
}

L1ParticleMap::L1ParticleMap(
   L1TriggerType triggerType,
   bool triggerDecision,
   const L1ObjectTypeVector& objectTypes,
   const L1EmParticleRefVector& emParticles,
   const L1JetParticleRefVector& jetParticles,
   const L1JetParticleRefVector& tauParticles,
   const L1MuonParticleRefVector& muonParticles,
   const L1EtMissParticleRefProd& etMissParticle,
   const L1IndexComboVector& indexCombos )
   : triggerType_( triggerType ),
     triggerDecision_( triggerDecision ),
     objectTypes_( objectTypes ),
     emParticles_( emParticles ),
     jetParticles_( jetParticles ),
     tauParticles_( tauParticles ),
     muonParticles_( muonParticles ),
     etMissParticle_( etMissParticle ),
     indexCombos_( indexCombos )
{
}

// L1ParticleMap::L1ParticleMap(const L1ParticleMap& rhs)
// {
//    // do actual copying here;
// }

L1ParticleMap::~L1ParticleMap()
{
}

//
// assignment operators
//
// const L1ParticleMap& L1ParticleMap::operator=(const L1ParticleMap& rhs)
// {
//   //An exception safe implementation is
//   L1ParticleMap temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

const L1ParticleMap::L1IndexComboVector&
L1ParticleMap::indexCombos() const
{
   if( indexCombos_.size() == 0 )
   {
      // Determine the number of non-global objects.  There should be 0 or 1.
      int numNonGlobal = 0 ;
      L1ObjectType nonGlobalType = kNumOfL1ObjectTypes ;
      int nonGlobalIndex = -1 ;
      for( int i = 0 ; i < numOfObjects() ; ++i )
      {
	 if( !objectTypeIsGlobal( objectTypes_[ i ] ) )
	 {
	    ++numNonGlobal ;
	    nonGlobalType = objectTypes_[ i ] ;
	    nonGlobalIndex = i ;
	 }
      }

      if( numNonGlobal == 0 )
      {
	 // Dummy entry for each object type.
	 L1IndexCombo tmpCombo ;

	 for( int i = 0 ; i < numOfObjects() ; ++i )
	 {
	    tmpCombo.push_back( 0 ) ;
	 }

	 indexCombos_.push_back( tmpCombo ) ;
      }
      else if( numNonGlobal == 1 )
      {
	 int nParticles = 0 ;

	 if( nonGlobalType == kEM )
	 {
	    nParticles = emParticles_.size() ;
	 }
	 else if( nonGlobalType == kJet )
	 {
	    nParticles = jetParticles_.size() ;
	 }
	 else if( nonGlobalType == kTau )
	 {
	    nParticles = tauParticles_.size() ;
	 }
	 else if( nonGlobalType == kMuon )
	 {
	    nParticles = muonParticles_.size() ;
	 }

	 for( int i = 0 ; i < nParticles ; ++i )
	 {
	    L1IndexCombo tmpCombo ;

	    for( int j = 0 ; j < numOfObjects() ; ++j )
	    {
	       if( j == nonGlobalIndex )
	       {		  
		  tmpCombo.push_back( i ) ;
	       }
	       else
	       {
		  tmpCombo.push_back( 0 ) ;
	       }
	    }

	    indexCombos_.push_back( tmpCombo ) ;
	 }
      }
   }

   return indexCombos_ ;
}

// const reco::ParticleKinematics*
// L1ParticleMap::particleInCombo( int aIndexInCombo,
// 				const L1IndexCombo& aCombo ) const
// {
//    L1ObjectType type = objectTypes_[ aIndexInCombo ] ;
//    int particleInList = aCombo[ aIndexInCombo ] ;

//    if( type == L1PhysObjectBase::kEM )
//    {
//       return dynamic_cast< const reco::ParticleKinematics* >(
// 	 emParticles_[ particleInList ].get() ) ;
//    }
//    else if( type == L1PhysObjectBase::kJet )
//    {
//       return dynamic_cast< const reco::ParticleKinematics* >(
// 	 jetParticles_[ particleInList ].get() ) ;
//    }
//    else if( type == L1PhysObjectBase::kTau )
//    {
//       return dynamic_cast< const reco::ParticleKinematics* >(
// 	 tauParticles_[ particleInList ].get() ) ;
//    }
//    else if( type == L1PhysObjectBase::kMuon )
//    {
//       return dynamic_cast< const reco::ParticleKinematics* >(
// 	 muonParticles_[ particleInList ].get() ) ;
//    }
//    else
//    {
//       return 0 ;
//    }
// }

const L1PhysObjectBase*
L1ParticleMap::physObjectInCombo( int aIndexInCombo,
				  const L1IndexCombo& aCombo ) const
{
   L1ObjectType type = objectTypes_[ aIndexInCombo ] ;
   int particleInList = aCombo[ aIndexInCombo ] ;

   if( type == kEM )
   {
      return dynamic_cast< const L1PhysObjectBase* >(
	 emParticles_[ particleInList ].get() ) ;
   }
   else if( type == kJet )
   {
      return dynamic_cast< const L1PhysObjectBase* >(
	 jetParticles_[ particleInList ].get() ) ;
   }
   else if( type == kTau )
   {
      return dynamic_cast< const L1PhysObjectBase* >(
	 tauParticles_[ particleInList ].get() ) ;
   }
   else if( type == kMuon )
   {
      return dynamic_cast< const L1PhysObjectBase* >(
	 muonParticles_[ particleInList ].get() ) ;
   }
   else if( type == kEtMiss || type == kEtTotal || type == kEtHad )
   {
      return dynamic_cast< const L1PhysObjectBase* >(
	 etMissParticle_.get() ) ;
   }
   else
   {
      return 0 ;
   }
}

const L1EmParticle*
L1ParticleMap::emParticleInCombo( int aIndexInCombo,
				  const L1IndexCombo& aCombo ) const
{
   L1ObjectType type = objectTypes_[ aIndexInCombo ] ;
   int particleInList = aCombo[ aIndexInCombo ] ;

   if( type == kEM )
   {
      return emParticles_[ particleInList ].get() ;
   }
   else
   {
      return 0 ;
   }
}

const L1JetParticle*
L1ParticleMap::jetParticleInCombo( int aIndexInCombo,
				   const L1IndexCombo& aCombo ) const
{
   L1ObjectType type = objectTypes_[ aIndexInCombo ] ;
   int particleInList = aCombo[ aIndexInCombo ] ;

   if( type == kJet )
   {
      return jetParticles_[ particleInList ].get() ;
   }
   else
   {
      return 0 ;
   }
}

const L1JetParticle*
L1ParticleMap::tauParticleInCombo( int aIndexInCombo,
				   const L1IndexCombo& aCombo ) const
{
   L1ObjectType type = objectTypes_[ aIndexInCombo ] ;
   int particleInList = aCombo[ aIndexInCombo ] ;

   if( type == kTau )
   {
      return tauParticles_[ particleInList ].get() ;
   }
   else
   {
      return 0 ;
   }
}

const L1MuonParticle*
L1ParticleMap::muonParticleInCombo( int aIndexInCombo,
				    const L1IndexCombo& aCombo ) const
{
   L1ObjectType type = objectTypes_[ aIndexInCombo ] ;
   int particleInList = aCombo[ aIndexInCombo ] ;

   if( type == kMuon )
   {
      return muonParticles_[ particleInList ].get() ;
   }
   else
   {
      return 0 ;
   }
}

const L1EtMissParticle*
L1ParticleMap::etMissParticleInCombo( int aIndexInCombo,
				      const L1IndexCombo& aCombo ) const
{
   L1ObjectType type = objectTypes_[ aIndexInCombo ] ;

   if( type == kEtMiss || type == kEtTotal || type == kEtHad )
   {
      return etMissParticle_.get() ;
   }
   else
   {
      return 0 ;
   }
}

// std::vector< const reco::ParticleKinematics* >
// L1ParticleMap::particleCombo( const L1IndexCombo& aCombo ) const
// {
//    std::vector< const reco::ParticleKinematics* > tmp ;

//    for( int i = 0 ; i < numOfObjects() ; ++i )
//    {
//       tmp.push_back( particleInCombo( i, aCombo ) ) ;
//    }

//    return tmp ;
// }

std::vector< const L1PhysObjectBase* >
L1ParticleMap::physObjectCombo( const L1IndexCombo& aCombo ) const
{
   std::vector< const L1PhysObjectBase* > tmp ;

   for( int i = 0 ; i < numOfObjects() ; ++i )
   {
      tmp.push_back( physObjectInCombo( i, aCombo ) ) ;
   }

   return tmp ;
}

//
// static member functions
//
const std::string&
L1ParticleMap::triggerName( L1TriggerType type )
{
   return triggerNames_[ type ] ;
}

L1ParticleMap::L1TriggerType
L1ParticleMap::triggerType( const std::string& name )
{
   for( int i = 0 ; i < kNumOfL1TriggerTypes ; ++i )
   {
      if( triggerNames_[ i ] == name )
      {
	 return ( L1TriggerType ) i ;
      }
   }

   return kNumOfL1TriggerTypes ;
}

bool
L1ParticleMap::objectTypeIsGlobal( L1ObjectType type )
{
   return type == kEtMiss || type == kEtTotal || type == kEtHad ;
}
