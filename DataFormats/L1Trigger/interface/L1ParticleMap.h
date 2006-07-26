#ifndef L1Trigger_L1ParticleMap_h
#define L1Trigger_L1ParticleMap_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1ParticleMap
// 
/**\class L1ParticleMap L1ParticleMap.h DataFormats/L1Trigger/interface/L1ParticleMap.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Werner Sun
//         Created:  Fri Jul 14 19:46:30 EDT 2006
// $Id: L1ParticleMap.h,v 1.1 2006/07/17 20:35:19 wsun Exp $
// $Log: L1ParticleMap.h,v $
// Revision 1.1  2006/07/17 20:35:19  wsun
// First draft.
//
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtTotalPhys.h"
#include "DataFormats/L1Trigger/interface/L1EtHadPhys.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"

// forward declarations

namespace level1 {

   class L1ParticleMap
   {

      public:
	 typedef std::vector< const ParticleKinematics* >
	 L1ParticleCombination ;
	 typedef std::vector< L1ParticleCombination >
	 L1ParticleCombinationVector ;

	 L1ParticleMap();
	 virtual ~L1ParticleMap();

	 // ---------- const member functions ---------------------
	 int triggerIndex() const
	 { return m_triggerIndex ; }

	 // Number of particles, excluding global quantities, that
	 // participated in this trigger.
	 int numberOfTriggerParticles() const
	 { return m_particleTypes.size() ; }

	 const L1EmParticleRefVector& emParticles() const
	 { return m_emParticles ; }

	 const L1JetParticleRefVector& jetParticles() const
	 { return m_jetParticles ; }

	 const L1MuonParticleRefVector& muonParticles() const
	 { return m_muonParticles ; }

	 const edm::RefProd< L1EtTotalPhys >& etTotalPhys() const
	 { return m_etTotalPhys ; }

	 const edm::RefProd< L1EtHadPhys >& etHadPhys() const
	 { return m_etHadPhys ; }

	 const edm::RefProd< L1EtMissParticle >& etMissParticle() const
	 { return m_etMissParticle ; }

	 const std::vector< L1ParticleType >& particleTypes() const
	 { return m_particleTypes ; }

	 // If numberOfTriggerParticles() is 1, then there is no need to
	 // store the particle combinations.  In this case, the stored
	 // vector m_particleCombinations will be empty, and it will be
	 // filled upon request at analysis time.
	 const L1ParticleCombinationVector& particleCombinations() const
	 {
	    if( m_particleCombinations.size() == 0 &&
		numberOfTriggerParticles() == 1 )
	    {
	       if( *( particleTypes().begin() ) == kEM )
	       {
		  L1EmParticleRefVector::const_iterator itr =
		     m_emParticles.begin() ;
		  L1EmParticleRefVector::const_iterator end =
		     m_emParticles.end() ;

		  for( ; itr != end ; itr )
		  {
		     L1ParticleCombination tmpCombo ;
		     tmpCombo.push_back(
			dynamic_cast< const ParticleKinematics* >(
			   &( *itr ) ) ) ;
		     m_particleCombinations.push_back( tmpCombo ) ;
		  }
	       }
	       else if( *( particleTypes().begin() ) == kJet )
	       {
		  L1JetParticleRefVector::const_iterator itr =
		     m_jetParticles.begin() ;
		  L1JetParticleRefVector::const_iterator end =
		     m_jetParticles.end() ;

		  for( ; itr != end ; itr )
		  {
		     L1ParticleCombination tmpCombo ;
		     tmpCombo.push_back(
			dynamic_cast< const ParticleKinematics* >(
			   &( *itr ) ) ) ;
		     m_particleCombinations.push_back( tmpCombo ) ;
		  }
	       }
	       else if( *( particleTypes().begin() ) == kMuon )
	       {
		  L1MuonParticleRefVector::const_iterator itr =
		     m_muonParticles.begin() ;
		  L1MuonParticleRefVector::const_iterator end =
		     m_muonParticles.end() ;

		  for( ; itr != end ; itr )
		  {
		     L1ParticleCombination tmpCombo ;
		     tmpCombo.push_back(
			dynamic_cast< const ParticleKinematics* >(
			   &( *itr ) ) ) ;
		     m_particleCombinations.push_back( tmpCombo ) ;
		  }
	       }
	    }
	    else
	    {
	       return m_particleCombinations ;
	    }
	 }

	 // ---------- static member functions --------------------

	 // These static functions convert a pointer of type
	 // ParticleKinematics into a subclass pointer (using a
	 // dynamic_cast).  If the dynamic_cast fails (i.e. the object types
	 // do not match), a null pointer will be returned.
	 static const L1EmParticle* emParticle(
	    const ParticleKinematics* ptr )
	 {
	    return dynamic_cast< const L1EmParticle* >( ptr ) ;
	 }

	 static const L1JetParticle* jetParticle(
	    const ParticleKinematics* ptr )
	 {
	    return dynamic_cast< const L1JetParticle* >( ptr ) ;
	 }

	 static const L1MuonParticle* muonParticle(
	    const ParticleKinematics* ptr )
	 {
	    return dynamic_cast< const L1MuonParticle* >( ptr ) ;
	 }

	 // ---------- member functions ---------------------------

      private:
	 L1ParticleMap(const L1ParticleMap&); // stop default

	 const L1ParticleMap& operator=(const L1ParticleMap&); // stop default

	 // ---------- member data --------------------------------

	 // Index into trigger menu.
	 int m_triggerIndex ;

	 // Lists of particles that fired this trigger, perhaps in combination
	 // with another particle.
	 L1EmParticleRefVector m_emParticles ;
	 L1JetParticleRefVector m_jetParticles ;
	 L1MuonParticleRefVector m_muonParticles ;

	 // Global (event-wide) objects.  The Ref is null if the object
	 // was not used in this trigger.
	 edm::RefProd< L1EtTotalPhys > m_etTotalPhys ;
	 edm::RefProd< L1EtHadPhys > m_etHadPhys ;
	 edm::RefProd< L1EtMissParticle > m_etMissParticle ;

	 // Vector of length numberOfTriggerParticles that gives the
	 // type of each particle.
	 std::vector< L1ParticleType > m_particleTypes ;

	 // Particle combinations that fired this trigger.  The inner
	 // vector< int > has length numberOfTriggerParticles and contains
	 // references to the elements in m_emParticles, m_jetParticles, and
	 // m_muonParticles for a successful combination.  The particle type
	 // of each entry is given by m_particleTypes.
	 L1ParticleCombinationVector m_particleCombinations ;
   };

}

#endif
