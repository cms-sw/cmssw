#ifndef L1Trigger_L1ParticleMap_h
#define L1Trigger_L1ParticleMap_h
// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1ParticleMap
// 
/**\class L1ParticleMap \file L1ParticleMap.h DataFormats/L1Trigger/interface/L1ParticleMap.h \author Werner Sun

 Description: L1Extra class for map between triggers and L1Extra particles.
*/
//
// Original Author:  Werner Sun
//         Created:  Fri Jul 14 19:46:30 EDT 2006
// $Id: L1ParticleMap.h,v 1.29 2007/09/27 22:31:18 ratnik Exp $
// $Log: L1ParticleMap.h,v $
// Revision 1.29  2007/09/27 22:31:18  ratnik
// QA campaign: merge includechecker changes back into the head, corresponding fixes being done in dependent packages
//
// Revision 1.28  2007/08/08 03:49:03  wsun
// Diffractive trigger threshold update from X. Rouby.
//
// Revision 1.27  2007/08/07 01:18:15  wsun
// Added JetMET calibration triggers from Len.
//
// Revision 1.26  2007/07/31 15:20:14  ratnik
// QA campaign: include cleanup based on CMSSW_1_7_X_2007-07-30-1600 includechecker results.
//
// Revision 1.25  2007/07/14 19:03:25  wsun
// Added diffractive triggers from X. Rouby and S. Ovyn.
//
// Revision 1.24  2007/06/16 16:50:03  wsun
// Added SingleTauJet35 and DoubleTauJet35 for 131HLT6.
//
// Revision 1.23  2007/06/15 19:27:31  wsun
// New L1 trigger table for 131HLT6.
//
// Revision 1.22  2007/06/03 00:06:30  wsun
// Revision of L1 trigger table for 131HLT5.
//
// Revision 1.21  2007/06/01 02:57:11  wsun
// New L1 trigger table for 131HLT5.
//
// Revision 1.20  2007/05/23 05:09:09  wsun
// L1 trigger table for 131HLT4.
//
// Revision 1.19  2007/05/15 14:52:40  wsun
// A_Mu3_IsoEG15 -> A_Mu7_IsoEG10
//
// Revision 1.18  2007/05/11 04:59:32  wsun
// Retweaked trigger table.
//
// Revision 1.17  2007/04/30 21:00:39  wsun
// QuadJet50 -> QuadJet20
//
// Revision 1.16  2007/04/23 18:33:31  wsun
// Another iteration of the L1 trigger table.
//
// Revision 1.15  2007/04/22 22:35:47  wsun
// Updated L1 trigger table yet again.
//
// Revision 1.14  2007/04/16 21:15:46  wsun
// Tweaks to trigger table for 131HLT.
//
// Revision 1.13  2007/04/13 17:50:46  wsun
// New trigger table for HLT exercise.
//
// Revision 1.12  2007/04/02 08:03:13  wsun
// Updated Doxygen documentation.
//
// Revision 1.11  2006/08/31 10:23:32  wsun
// Added MinBias trigger.
//
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
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h" 
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h" 
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h" 
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h" 

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
	    kSingleMu3,
	    kSingleMu5,
	    kSingleMu7,
	    kSingleMu10,
	    kSingleMu14,
	    kSingleMu20,
	    kSingleMu25,
	    kSingleIsoEG5,
	    kSingleIsoEG8,
	    kSingleIsoEG10,
	    kSingleIsoEG12,
	    kSingleIsoEG15,
	    kSingleIsoEG20,
	    kSingleIsoEG25,
	    kSingleEG5,
	    kSingleEG8,
	    kSingleEG10,
	    kSingleEG12,
	    kSingleEG15,
	    kSingleEG20,
	    kSingleEG25,
	    kSingleJet15,
	    kSingleJet20,
	    kSingleJet30,
	    kSingleJet50,
	    kSingleJet70,
	    kSingleJet100,
	    kSingleJet150,
	    kSingleJet200,
	    kSingleTauJet10,
	    kSingleTauJet20,
	    kSingleTauJet30,
	    kSingleTauJet35,
	    kSingleTauJet40,
	    kSingleTauJet60,
	    kSingleTauJet80,
	    kSingleTauJet100,
	    kHTT100,
	    kHTT200,
	    kHTT250,
	    kHTT300,
	    kHTT400,
	    kHTT500,
	    kETM10,
	    kETM15,
	    kETM20,
	    kETM30,
	    kETM40,
	    kETM50,
	    kETM60,
	    kETT60,
	    kDoubleMu3,
	    kDoubleIsoEG8,
	    kDoubleIsoEG10,
	    kDoubleEG5,
	    kDoubleEG10,
	    kDoubleEG15,
	    kDoubleJet70,
	    kDoubleJet100,
	    kDoubleTauJet20,
	    kDoubleTauJet30,
	    kDoubleTauJet35,
	    kDoubleTauJet40,
	    kMu3_IsoEG5,
	    kMu5_IsoEG10,
	    kMu3_EG12,
	    kMu3_Jet15,
	    kMu5_Jet15,
	    kMu3_Jet70,
	    kMu5_Jet20,
	    kMu5_TauJet20,
	    kMu5_TauJet30,
	    kIsoEG10_EG10,
	    kIsoEG10_Jet15,
	    kIsoEG10_Jet20,
	    kIsoEG10_Jet30,
	    kIsoEG10_Jet70,
	    kIsoEG10_TauJet20,
	    kIsoEG10_TauJet30,
	    kEG10_Jet15,
	    kEG12_Jet20,
	    kEG12_Jet70,
	    kEG12_TauJet40,
	    kJet70_TauJet40,
	    kMu3_HTT200,
	    kIsoEG10_HTT200,
	    kEG12_HTT200,
	    kJet70_HTT200,
	    kTauJet40_HTT200,
	    kMu3_ETM30,
	    kIsoEG10_ETM30,
	    kEG12_ETM30,
	    kJet70_ETM40,
	    kTauJet20_ETM20,
	    kTauJet30_ETM30,
	    kTauJet30_ETM40,
	    kHTT100_ETM30,
	    kTripleMu3,
	    kTripleIsoEG5,
	    kTripleEG10,
	    kTripleJet50,
	    kTripleTauJet40,
	    kDoubleMu3_IsoEG5,
	    kDoubleMu3_EG10,
	    kDoubleIsoEG5_Mu3,
	    kDoubleEG10_Mu3,
	    kDoubleMu3_HTT200,
	    kDoubleIsoEG5_HTT200,
	    kDoubleEG10_HTT200,
	    kDoubleJet50_HTT200,
	    kDoubleTauJet40_HTT200,
	    kDoubleMu3_ETM20,
	    kDoubleIsoEG5_ETM20,
	    kDoubleEG10_ETM20,
	    kDoubleJet50_ETM20,
	    kDoubleTauJet40_ETM20,
	    kQuadJet30,
            kExclusiveDoubleIsoEG4,
            kExclusiveDoubleJet60,
            kExclusiveJet25_Gap_Jet25,
            kIsoEG10_Jet20_ForJet10,
	    kMinBias_HTT10,
	    kZeroBias,
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
