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
// $Id: L1ParticleMap.h,v 1.2 2006/07/26 00:05:39 wsun Exp $
// $Log: L1ParticleMap.h,v $
// Revision 1.2  2006/07/26 00:05:39  wsun
// Structural mods for HLT use.
//
// Revision 1.1  2006/07/17 20:35:19  wsun
// First draft.
//
//

// system include files

// user include files
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtTotalPhys.h"
#include "DataFormats/L1Trigger/interface/L1EtHadPhys.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMapFwd.h"

// forward declarations

namespace l1extra {

   class L1ParticleMap
   {

      public:
	 enum L1TriggerType
	 {
	    kSingleElectron,
	    kSingleJet,
	    kSingleTau,
	    kSingleMuon,
	    kNumOfL1TriggerTypes
	 } ;

	 typedef std::vector< unsigned int > L1IndexCombo ;
	 typedef std::vector< L1IndexCombo > L1IndexComboVector ;
	 typedef L1PhysObjectBase::L1PhysObjectType L1ParticleType ;

	 L1ParticleMap();
	 virtual ~L1ParticleMap();

	 // ---------- const member functions ---------------------
	 L1TriggerType triggerType() const
	 { return triggerType_ ; }

	 // Indices of particle types (e/gamma, jets, and muons), excluding
	 // global quantities, that participated in this trigger.  The order
	 // of these type indices corresponds to the particles listed in each
	 // L1IndexCombo.
	 const std::vector< L1ParticleType >& nonGlobalParticleTypes() const
	 { return particleTypes_ ; }

	 // Number of particles (e/gamma, jets, and muons), excluding global
	 // quantities, that participated in this trigger.
	 int numOfNonGlobalParticles() const
	 { return particleTypes_.size() ; }

	 const L1EmParticleRefVector& emParticles() const
	 { return emParticles_ ; }

	 const L1JetParticleRefVector& jetParticles() const
	 { return jetParticles_ ; }

	 const L1MuonParticleRefVector& muonParticles() const
	 { return muonParticles_ ; }

	 const edm::RefProd< L1EtTotalPhys >& etTotalPhys() const
	 { return etTotalPhys_ ; }

	 const edm::RefProd< L1EtHadPhys >& etHadPhys() const
	 { return etHadPhys_ ; }

	 const edm::RefProd< L1EtMissParticle >& etMissParticle() const
	 { return etMissParticle_ ; }

	 // If numberOfTriggerParticles() is 1, then there is no need to
	 // store the particle combinations.  In this case, the stored
	 // vector m_particleCombinations will be empty, and it will be
	 // filled upon request at analysis time.
	 const L1IndexComboVector& indexCombos() const ;

	 // These functions retrieve the particle corresponding to a
	 // particular entry in a given combination.  The pointer is null
	 // if an error occurs (e.g. the particle requested does not match
	 // the type of the function).
	 const reco::ParticleKinematics* particleInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 const L1PhysObjectBase* physObjectInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 const L1EmParticle* emParticleInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 const L1JetParticle* jetParticleInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 const L1MuonParticle* muonParticleInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 // For a given particle combination, convert all the particles to
	 // ParticleKinematics pointers.
	 std::vector< const reco::ParticleKinematics* > particleCombo(
	    const L1IndexCombo& aCombo ) const ;

	 // For a given particle combination, convert all the particles to
	 // L1PhysObjectBase pointers.
	 std::vector< const L1PhysObjectBase* > physObjectCombo(
	    const L1IndexCombo& aCombo ) const ;

	 // ---------- static member functions --------------------

	 // ---------- member functions ---------------------------

      private:
	 // L1ParticleMap(const L1ParticleMap&); // stop default

	 // const L1ParticleMap& operator=(const L1ParticleMap&); // stop default

	 // ---------- member data --------------------------------

	 // Index into trigger menu.
	 L1TriggerType triggerType_ ;

	 // Lists of particles that fired this trigger, perhaps in combination
	 // with another particle.
	 L1EmParticleRefVector emParticles_ ;
	 L1JetParticleRefVector jetParticles_ ;
	 L1MuonParticleRefVector muonParticles_ ;

	 // Global (event-wide) objects.  The Ref is null if the object
	 // was not used in this trigger.
	 edm::RefProd< L1EtTotalPhys > etTotalPhys_ ;
	 edm::RefProd< L1EtHadPhys > etHadPhys_ ;
	 edm::RefProd< L1EtMissParticle > etMissParticle_ ;

	 // Vector of length numberOfTriggerParticles that gives the
	 // type of each particle.
	 std::vector< L1ParticleType > particleTypes_ ;

	 // Particle combinations that fired this trigger.  The inner
	 // vector< int > has length numberOfTriggerParticles and contains
	 // references to the elements in emParticles_, jetParticles_, and
	 // muonParticles_ for a successful combination.  The particle type
	 // of each entry is given by particleTypes_.
	 //
	 // This data member is mutable because if #particles = 1, then this
	 // vector is empty and is filled on request.
	 mutable L1IndexComboVector indexCombos_ ;
   };

}

#endif
