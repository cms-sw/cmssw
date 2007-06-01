// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1ParticleMap
// 
/**\class L1ParticleMap \file L1ParticleMap.cc DataFormats/L1Trigger/src/L1ParticleMap.cc \author Werner Sun
*/
//
// Original Author:  Werner Sun
//         Created:  Wed Jul 26 14:42:56 EDT 2006
// $Id: L1ParticleMap.cc,v 1.17 2007/05/23 05:09:11 wsun Exp $
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

// EG = isolated OR non-isolated
// Jet = central OR forward OR tau
std::string
L1ParticleMap::triggerNames_[ kNumOfL1TriggerTypes ] = {
   "A_SingleMu3",
   "A_SingleMu5",
   "A_SingleMu7",
   "A_SingleMu10",
   "A_SingleMu14",
   "A_SingleMu20",
   "A_SingleMu25",
   "A_SingleIsoEG5",
   "A_SingleIsoEG8",
   "A_SingleIsoEG10",
   "A_SingleIsoEG12",
   "A_SingleIsoEG15",
   "A_SingleIsoEG20",
   "A_SingleIsoEG25",
   "A_SingleEG5",
   "A_SingleEG10",
   "A_SingleEG15",
   "A_SingleEG20",
   "A_SingleEG25",
   "A_SingleJet15",
   "A_SingleJet20",
   "A_SingleJet30",
   "A_SingleJet50",
   "A_SingleJet70",
   "A_SingleJet100",
   "A_SingleJet150",
   "A_SingleJet200",
   "A_SingleTauJet10",
   "A_SingleTauJet20",
   "A_SingleTauJet30",
   "A_SingleTauJet40",
   "A_SingleTauJet80",
   "A_SingleTauJet100",
   "A_HTT100",
   "A_HTT200",
   "A_HTT250",
   "A_HTT300",
   "A_HTT400",
   "A_HTT500",
   "A_ETM20",
   "A_ETM30",
   "A_ETM40",
   "A_ETM50",
   "A_ETM60",
   "A_DoubleMu3",
   "A_DoubleIsoEG8",
   "A_DoubleIsoEG10",
   "A_DoubleEG5",
   "A_DoubleEG10",
   "A_DoubleEG15",
   "A_DoubleJet70",
   "A_DoubleJet100",
   "A_DoubleTauJet20",
   "A_DoubleTauJet30",
   "A_DoubleTauJet40",
   "A_Mu3_IsoEG5",
   "A_Mu7_IsoEG10",
   "A_Mu3_EG20",
   "A_Mu3_Jet15",
   "A_Mu7_Jet15",
   "A_Mu3_Jet100",
   "A_Mu10_Jet20",
   "A_Mu7_TauJet20",
   "A_Mu7_TauJet30",
   "A_IsoEG10_EG10",
   "A_IsoEG10_Jet15",
   "A_IsoEG10_Jet30",
   "A_IsoEG15_Jet20",
   "A_IsoEG15_Jet100",
   "A_IsoEG10_TauJet20",
   "A_IsoEG10_TauJet30",
   "A_EG10_Jet15",
   "A_EG20_Jet20",
   "A_EG20_Jet100",
   "A_EG20_TauJet40",
   "A_Jet100_TauJet40",
   "A_Mu3_HTT300",
   "A_IsoEG15_HTT300",
   "A_EG20_HTT300",
   "A_Jet100_HTT300",
   "A_TauJet40_HTT300",
   "A_Mu3_ETM30",
   "A_IsoEG15_ETM30",
   "A_EG20_ETM30",
   "A_Jet100_ETM40",
   "A_TauJet20_ETM20",
   "A_TauJet30_ETM30",
   "A_TauJet30_ETM40",
   "A_HTT200_ETM40",
   "A_TripleMu3",
   "A_TripleIsoEG5",
   "A_TripleEG10",
   "A_TripleJet50",
   "A_TripleTauJet40",
   "A_DoubleMu3_IsoEG5",
   "A_DoubleMu3_EG10",
   "A_DoubleIsoEG5_Mu3",
   "A_DoubleEG10_Mu3",
   "A_DoubleMu3_HTT200",
   "A_DoubleIsoEG5_HTT200",
   "A_DoubleEG10_HTT200",
   "A_DoubleJet50_HTT200",
   "A_DoubleTauJet40_HTT200",
   "A_DoubleMu3_ETM20",
   "A_DoubleIsoEG5_ETM20",
   "A_DoubleEG10_ETM20",
   "A_DoubleJet50_ETM20",
   "A_DoubleTauJet40_ETM20",
   "A_QuadJet30",
   "A_MinBias_HTT10",
   "A_ZeroBias"
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
   const L1EmParticleVectorRef& emParticles,
   const L1JetParticleVectorRef& jetParticles,
   const L1MuonParticleVectorRef& muonParticles,
   const L1EtMissParticleRefProd& etMissParticle,
   const L1IndexComboVector& indexCombos )
   : triggerType_( triggerType ),
     triggerDecision_( triggerDecision ),
     objectTypes_( objectTypes ),
     emParticles_( emParticles ),
     jetParticles_( jetParticles ),
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

const reco::LeafCandidate*
L1ParticleMap::candidateInCombo( int aIndexInCombo,
				 const L1IndexCombo& aCombo ) const
{
   L1ObjectType type = objectTypes_[ aIndexInCombo ] ;
   int particleInList = aCombo[ aIndexInCombo ] ;

   if( type == kEM )
   {
      return dynamic_cast< const reco::LeafCandidate* >(
	 emParticles_[ particleInList ].get() ) ;
   }
   else if( type == kJet )
   {
      return dynamic_cast< const reco::LeafCandidate* >(
	 jetParticles_[ particleInList ].get() ) ;
   }
   else if( type == kMuon )
   {
      return dynamic_cast< const reco::LeafCandidate* >(
	 muonParticles_[ particleInList ].get() ) ;
   }
   else if( type == kEtMiss || type == kEtTotal || type == kEtHad )
   {
      return dynamic_cast< const reco::LeafCandidate* >(
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

std::vector< const reco::LeafCandidate* >
L1ParticleMap::candidateCombo( const L1IndexCombo& aCombo ) const
{
   std::vector< const reco::LeafCandidate* > tmp ;

   for( int i = 0 ; i < numOfObjects() ; ++i )
   {
      tmp.push_back( candidateInCombo( i, aCombo ) ) ;
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
