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
// $Id: L1ParticleMap.h,v 1.10 2006/08/28 03:10:40 wsun Exp $
// $Log: L1ParticleMap.h,v $
// Revision 1.10  2006/08/28 03:10:40  wsun
// Revamped L1ParticleMap to handle OR triggers.
//
// Revision 1.9  2006/08/23 23:09:04  wsun
// Separated iso/non-iso EM triggers and RefVectors.
//
// Revision 1.8  2006/08/10 18:47:41  wsun
// Removed L1PhysObjectBase; particle classes now derived from LeafCandidate.
//
// Revision 1.7  2006/08/06 15:32:26  wsun
// Added comment.
//
// Revision 1.6  2006/08/04 03:30:47  wsun
// Separated tau/jet bookkeeping, added static function objectTypeIsGlobal().
//
// Revision 1.5  2006/08/02 20:48:55  wsun
// Added more trigger lines, added mapping for global objects.
//
// Revision 1.4  2006/08/02 14:21:33  wsun
// Added trigger name dictionary, moved particle type enum to L1ParticleMap.
//
// Revision 1.3  2006/07/26 20:41:30  wsun
// Added implementation of L1ParticleMap.
//
// Revision 1.2  2006/07/26 00:05:39  wsun
// Structural mods for HLT use.
//
// Revision 1.1  2006/07/17 20:35:19  wsun
// First draft.
//
//

// system include files
#include <string>

// user include files
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMapFwd.h"

// forward declarations

namespace l1extra {

   class L1ParticleMap
   {

      public:
         enum L1ObjectType
	 {
            kEM,   // = isolated or non-isolated
            kJet,  // = central, forward, or tau
            kMuon,
	    kEtMiss,
	    kEtTotal,
	    kEtHad,
            kNumOfL1ObjectTypes
	 } ;

	 // For now, use trigger menu from PTDR:
	 // http://monicava.web.cern.ch/monicava/hlt_rates.htm#l1bits

	 // RelaxedEM = isolated OR non-isolated
	 // Jet = central OR forward OR tau

	 enum L1TriggerType
	 {
	    kSingleIsoEM,
	    kDoubleIsoEM,
	    kSingleRelaxedEM,
	    kDoubleRelaxedEM,
	    kSingleMuon,
	    kDoubleMuon,
	    kSingleTau,
	    kDoubleTau,
	    kSingleJet,
	    kDoubleJet,
	    kTripleJet,
	    kQuadJet,
	    kHT,
	    kMET,
	    kHTMET,
	    kJetMET,
	    kTauMET,
	    kMuonMET,
	    kIsoEMMET,
	    kMuonJet,
	    kIsoEMJet,
	    kMuonTau,
	    kIsoEMTau,
	    kIsoEMMuon,
	    kSingleJet140,
	    kSingleJet60,
	    kSingleJet20,
	    kMinBias,
	    kNumOfL1TriggerTypes
	 } ;

	 typedef std::vector< unsigned int > L1IndexCombo ;
	 typedef std::vector< L1IndexCombo > L1IndexComboVector ;
	 typedef std::vector< L1ObjectType > L1ObjectTypeVector ;

	 L1ParticleMap();
	 L1ParticleMap(
	    L1TriggerType triggerType,
	    bool triggerDecision,
	    const L1ObjectTypeVector& objectTypes,
	    const L1EmParticleVectorRef& emParticles =
	       L1EmParticleVectorRef(),
	    const L1JetParticleVectorRef& jetParticles =
	       L1JetParticleVectorRef(),
	    const L1MuonParticleVectorRef& muonParticles =
	       L1MuonParticleVectorRef(),
	    const L1EtMissParticleRefProd& etMissParticle =
	       L1EtMissParticleRefProd(),
	    const L1IndexComboVector& indexCombos =
	       L1IndexComboVector()
	    ) ;

	 virtual ~L1ParticleMap();

	 // ---------- const member functions ---------------------
	 L1TriggerType triggerType() const
	 { return triggerType_ ; }

	 const std::string& triggerName() const
	 { return triggerName( triggerType_ ) ; }

	 bool triggerDecision() const
	 { return triggerDecision_ ; }

	 // Indices of object types (see the above enum), that participated
	 // in this trigger.  The order of these type indices corresponds to
	 // the particles listed in each L1IndexCombo.
	 const L1ObjectTypeVector& objectTypes() const
	 { return objectTypes_ ; }

	 // Number of objects that participated in this trigger.
	 int numOfObjects() const
	 { return objectTypes_.size() ; }

	 const L1EmParticleVectorRef& emParticles() const
	 { return emParticles_ ; }

	 const L1JetParticleVectorRef& jetParticles() const
	 { return jetParticles_ ; }

	 const L1MuonParticleVectorRef& muonParticles() const
	 { return muonParticles_ ; }

	 const L1EtMissParticleRefProd& etMissParticle() const
	 { return etMissParticle_ ; }

	 // If there are zero or one non-global objects, then there is no need
	 // to store the object combinations.  In this case, the stored
	 // vector m_objectCombinations will be empty, and it will be
	 // filled upon request at analysis time.
	 const L1IndexComboVector& indexCombos() const ;

	 // These functions retrieve the object corresponding to a
	 // particular entry in a given combination.  The pointer is null
	 // if an error occurs (e.g. the particle requested does not match
	 // the type of the function).
	 const reco::LeafCandidate* candidateInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 const L1EmParticle* emParticleInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 const L1JetParticle* jetParticleInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 const L1MuonParticle* muonParticleInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 // This function just returns the single global object.
	 const L1EtMissParticle* etMissParticleInCombo(
	    int aIndexInCombo, const L1IndexCombo& aCombo ) const ;

	 // For a given particle combination, convert all the particles to
	 // reco::LeafCandidate pointers.
	 std::vector< const reco::LeafCandidate* > candidateCombo(
	    const L1IndexCombo& aCombo ) const ;

	 // ---------- static member functions --------------------
	 static const std::string& triggerName( L1TriggerType type ) ;
	 static L1TriggerType triggerType( const std::string& name ) ;
	 static bool objectTypeIsGlobal( L1ObjectType type ) ;

	 // ---------- member functions ---------------------------

      private:
	 // L1ParticleMap(const L1ParticleMap&); // stop default

	 // const L1ParticleMap& operator=(const L1ParticleMap&); // stop default

	 // ---------- member data --------------------------------

	 // Index into trigger menu.
	 L1TriggerType triggerType_ ;

	 bool triggerDecision_ ;

	 // Vector of length numOfObjects() that gives the
	 // type of each trigger object.
	 L1ObjectTypeVector objectTypes_ ;

	 // Lists of particles that fired this trigger, perhaps in combination
	 // with another particle.
	 L1EmParticleVectorRef emParticles_ ;
	 L1JetParticleVectorRef jetParticles_ ;
	 L1MuonParticleVectorRef muonParticles_ ;

	 // Global (event-wide) objects.  The Ref is null if the object
	 // was not used in this trigger.
	 L1EtMissParticleRefProd etMissParticle_ ;

	 // Object combinations that fired this trigger.  The inner
	 // vector< int > has length numOfObjects() and contains
	 // references to the elements in emParticles_, jetParticles_, and
	 // muonParticles_ for a successful combination.  A dummy index is
	 // entered for each global object in the trigger.  The object type
	 // of each entry is given by objectTypes_.
	 //
	 // This data member is mutable because if #particles = 1, then this
	 // vector is empty and is filled on request.
	 mutable L1IndexComboVector indexCombos_ ;

	 // Static array of trigger names.
	 static std::string triggerNames_[ kNumOfL1TriggerTypes ] ;
   };

}

#endif
