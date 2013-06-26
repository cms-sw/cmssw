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
// $Id: L1ParticleMap.cc,v 1.27 2007/09/27 22:31:21 ratnik Exp $
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"  
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"  
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"  

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
   "L1_SingleMu3",
   "L1_SingleMu5",
   "L1_SingleMu7",
   "L1_SingleMu10",
   "L1_SingleMu14",
   "L1_SingleMu20",
   "L1_SingleMu25",
   "L1_SingleIsoEG5",
   "L1_SingleIsoEG8",
   "L1_SingleIsoEG10",
   "L1_SingleIsoEG12",
   "L1_SingleIsoEG15",
   "L1_SingleIsoEG20",
   "L1_SingleIsoEG25",
   "L1_SingleEG5",
   "L1_SingleEG8",
   "L1_SingleEG10",
   "L1_SingleEG12",
   "L1_SingleEG15",
   "L1_SingleEG20",
   "L1_SingleEG25",
   "L1_SingleJet15",
   "L1_SingleJet20",
   "L1_SingleJet30",
   "L1_SingleJet50",
   "L1_SingleJet70",
   "L1_SingleJet100",
   "L1_SingleJet150",
   "L1_SingleJet200",
   "L1_SingleTauJet10",
   "L1_SingleTauJet20",
   "L1_SingleTauJet30",
   "L1_SingleTauJet35",
   "L1_SingleTauJet40",
   "L1_SingleTauJet60",
   "L1_SingleTauJet80",
   "L1_SingleTauJet100",
   "L1_HTT100",
   "L1_HTT200",
   "L1_HTT250",
   "L1_HTT300",
   "L1_HTT400",
   "L1_HTT500",
   "L1_ETM10",
   "L1_ETM15",
   "L1_ETM20",
   "L1_ETM30",
   "L1_ETM40",
   "L1_ETM50",
   "L1_ETM60",
   "L1_ETT60",
   "L1_DoubleMu3",
   "L1_DoubleIsoEG8",
   "L1_DoubleIsoEG10",
   "L1_DoubleEG5",
   "L1_DoubleEG10",
   "L1_DoubleEG15",
   "L1_DoubleJet70",
   "L1_DoubleJet100",
   "L1_DoubleTauJet20",
   "L1_DoubleTauJet30",
   "L1_DoubleTauJet35",
   "L1_DoubleTauJet40",
   "L1_Mu3_IsoEG5",
   "L1_Mu5_IsoEG10",
   "L1_Mu3_EG12",
   "L1_Mu3_Jet15",
   "L1_Mu5_Jet15",
   "L1_Mu3_Jet70",
   "L1_Mu5_Jet20",
   "L1_Mu5_TauJet20",
   "L1_Mu5_TauJet30",
   "L1_IsoEG10_EG10",
   "L1_IsoEG10_Jet15",
   "L1_IsoEG10_Jet20",
   "L1_IsoEG10_Jet30",
   "L1_IsoEG10_Jet70",
   "L1_IsoEG10_TauJet20",
   "L1_IsoEG10_TauJet30",
   "L1_EG10_Jet15",
   "L1_EG12_Jet20",
   "L1_EG12_Jet70",
   "L1_EG12_TauJet40",
   "L1_Jet70_TauJet40",
   "L1_Mu3_HTT200",
   "L1_IsoEG10_HTT200",
   "L1_EG12_HTT200",
   "L1_Jet70_HTT200",
   "L1_TauJet40_HTT200",
   "L1_Mu3_ETM30",
   "L1_IsoEG10_ETM30",
   "L1_EG12_ETM30",
   "L1_Jet70_ETM40",
   "L1_TauJet20_ETM20",
   "L1_TauJet30_ETM30",
   "L1_TauJet30_ETM40",
   "L1_HTT100_ETM30",
   "L1_TripleMu3",
   "L1_TripleIsoEG5",
   "L1_TripleEG10",
   "L1_TripleJet50",
   "L1_TripleTauJet40",
   "L1_DoubleMu3_IsoEG5",
   "L1_DoubleMu3_EG10",
   "L1_DoubleIsoEG5_Mu3",
   "L1_DoubleEG10_Mu3",
   "L1_DoubleMu3_HTT200",
   "L1_DoubleIsoEG5_HTT200",
   "L1_DoubleEG10_HTT200",
   "L1_DoubleJet50_HTT200",
   "L1_DoubleTauJet40_HTT200",
   "L1_DoubleMu3_ETM20",
   "L1_DoubleIsoEG5_ETM20",
   "L1_DoubleEG10_ETM20",
   "L1_DoubleJet50_ETM20",
   "L1_DoubleTauJet40_ETM20",
   "L1_QuadJet30",
   "L1_ExclusiveDoubleIsoEG4", 
   "L1_ExclusiveDoubleJet60", 
   "L1_ExclusiveJet25_Gap_Jet25", 
   "L1_IsoEG10_Jet20_ForJet10",
   "L1_MinBias_HTT10",
   "L1_ZeroBias"
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
