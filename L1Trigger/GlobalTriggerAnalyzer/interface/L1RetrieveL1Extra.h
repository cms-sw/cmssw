#ifndef L1Trigger_GlobalTriggerAnalyzer_L1RetrieveL1Extra_h
#define L1Trigger_GlobalTriggerAnalyzer_L1RetrieveL1Extra_h

/**
 * \class L1RetrieveL1Extra
 *
 *
 * Description: retrieve L1Extra collection, return validity flag and pointer to collection.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Vasile Mihai Ghete   - HEPHY Vienna
 *
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <iosfwd>
#include <memory>
#include <vector>
#include <string>

// user include files
//   base classes

//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// L1Extra objects
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1HFRings.h"
#include "DataFormats/L1Trigger/interface/L1HFRingsFwd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// forward declarations


// class declaration
class L1RetrieveL1Extra {

public:

    // constructor(s)
    explicit L1RetrieveL1Extra(const edm::ParameterSet&);

    // destructor
    virtual ~L1RetrieveL1Extra();

public:

    /// validity for retrieval of L1Extra products (false: product not found)

    inline const bool validL1ExtraMuon() const {
        return m_validL1ExtraMuon;
    }

    //
    inline const bool validL1ExtraIsoEG() const {
        return m_validL1ExtraIsoEG;
    }

    inline const bool validL1ExtraNoIsoEG() const {
        return m_validL1ExtraNoIsoEG;
    }

    //
    inline const bool validL1ExtraCenJet() const {
        return m_validL1ExtraCenJet;
    }

    inline const bool validL1ExtraForJet() const {
        return m_validL1ExtraForJet;
    }

    inline const bool validL1ExtraTauJet() const {
        return m_validL1ExtraTauJet;
    }

    //
    inline const bool validL1ExtraETT() const {
        return m_validL1ExtraETT;
    }

    inline const bool validL1ExtraETM() const {
        return m_validL1ExtraETM;
    }

    inline const bool validL1ExtraHTT() const {
        return m_validL1ExtraHTT;
    }

    inline const bool validL1ExtraHTM() const {
        return m_validL1ExtraHTM;
    }

    //
    inline const bool validL1ExtraHfBitCounts() const {
        return m_validL1ExtraHfBitCounts;
    }

    inline const bool validL1ExtraHfRingEtSums() const {
        return m_validL1ExtraHfRingEtSums;
    }

    const bool validL1ExtraColl(const L1GtObject&) const;

    /// input tag for a given collection
    const edm::InputTag inputTagL1ExtraColl(const L1GtObject&) const;

    /// return retrieved L1Extra collections

    inline const l1extra::L1MuonParticleCollection* l1ExtraMuon() const {
        return m_l1ExtraMuon;
    }

    inline const l1extra::L1EmParticleCollection* l1ExtraIsoEG() const {
        return m_l1ExtraIsoEG;
    }
    inline const l1extra::L1EmParticleCollection* l1ExtraNoIsoEG() const {
        return m_l1ExtraNoIsoEG;
    }

    inline const l1extra::L1JetParticleCollection* l1ExtraCenJet() const {
        return m_l1ExtraCenJet;
    }
    inline const l1extra::L1JetParticleCollection* l1ExtraForJet() const {
        return m_l1ExtraForJet;
    }
    inline const l1extra::L1JetParticleCollection* l1ExtraTauJet() const {
        return m_l1ExtraTauJet;
    }

    inline const l1extra::L1EtMissParticleCollection* l1ExtraETT() const {
        return m_l1ExtraETT;
    }
    inline const l1extra::L1EtMissParticleCollection* l1ExtraETM() const {
        return m_l1ExtraETM;
    }
    inline const l1extra::L1EtMissParticleCollection* l1ExtraHTT() const {
        return m_l1ExtraHTT;
    }
    inline const l1extra::L1EtMissParticleCollection* l1ExtraHTM() const {
        return m_l1ExtraHTM;
    }

    inline const l1extra::L1HFRingsCollection* l1ExtraHfBitCounts() const {
        return m_l1ExtraHfBitCounts;
    }
    inline const l1extra::L1HFRingsCollection* l1ExtraHfRingEtSums() const {
        return m_l1ExtraHfRingEtSums;
    }

    /// retrieve L1Extra objects
    /// if a collection is not found, the corresponding m_valid(Object) is set to "false"
    void retrieveL1ExtraObjects(const edm::Event&, const edm::EventSetup&);

    /// user-friendly print of L1Extra
    /// TODO should have been defined in DataFormats for L1Extra collections...

    /// print L1GtObject object from bxInEvent, if checkBxInEvent is true,
    /// having the objIndexInColl order index in collection, if checkObjIndexInColl is true
    /// if checkBxInEvent and /or checkObjIndexInColl are false, print the objects without
    /// the bxInEvent and / or objIndexInColl check
    /// the combination checkBxInEvent = false, checkObjIndexInColl = true not supported
    void printL1Extra(std::ostream& oStr, const L1GtObject& gtObject,
            const bool checkBxInEvent, const int bxInEvent,
            const bool checkObjIndexInColl, const int objIndexInColl) const;

    /// print all L1GtObject objects from bxInEvent
    void printL1Extra(std::ostream&, const L1GtObject&, const int bxInEvent) const;

    /// print all L1GtObject objects from all bxInEvent
    void printL1Extra(std::ostream&, const L1GtObject&) const;

    /// print all L1Extra collections from a given BxInEvent
    void printL1Extra(std::ostream&, const int bxInEvent) const;

    /// print all L1Extra collections from all BxInEvent
    void printL1Extra(std::ostream&) const;

private:

    /// input parameters

    /// input tags for L1Extra objects

    edm::InputTag m_tagL1ExtraMuon;

    edm::InputTag m_tagL1ExtraIsoEG;
    edm::InputTag m_tagL1ExtraNoIsoEG;

    edm::InputTag m_tagL1ExtraCenJet;
    edm::InputTag m_tagL1ExtraForJet;
    edm::InputTag m_tagL1ExtraTauJet;

    edm::InputTag m_tagL1ExtraEtMissMET;
    edm::InputTag m_tagL1ExtraEtMissHTM;

    edm::InputTag m_tagL1ExtraHFRings;

    int m_nrBxInEventGmt;
    int m_nrBxInEventGct;

    /// validity for retrieval of L1Extra products (false: product not found)

    bool m_validL1ExtraMuon;

    bool m_validL1ExtraIsoEG;
    bool m_validL1ExtraNoIsoEG;

    bool m_validL1ExtraCenJet;
    bool m_validL1ExtraForJet;
    bool m_validL1ExtraTauJet;

    bool m_validL1ExtraETT;
    bool m_validL1ExtraETM;
    bool m_validL1ExtraHTT;
    bool m_validL1ExtraHTM;

    bool m_validL1ExtraHfBitCounts;
    bool m_validL1ExtraHfRingEtSums;

    /// retrieved L1Extra collections

    const l1extra::L1MuonParticleCollection* m_l1ExtraMuon;

    const l1extra::L1EmParticleCollection* m_l1ExtraIsoEG;
    const l1extra::L1EmParticleCollection* m_l1ExtraNoIsoEG;

    const l1extra::L1JetParticleCollection* m_l1ExtraCenJet;
    const l1extra::L1JetParticleCollection* m_l1ExtraForJet;
    const l1extra::L1JetParticleCollection* m_l1ExtraTauJet;

    const l1extra::L1EtMissParticleCollection* m_l1ExtraETT;
    const l1extra::L1EtMissParticleCollection* m_l1ExtraETM;
    const l1extra::L1EtMissParticleCollection* m_l1ExtraHTT;
    const l1extra::L1EtMissParticleCollection* m_l1ExtraHTM;

    const l1extra::L1HFRingsCollection* m_l1ExtraHfBitCounts;
    const l1extra::L1HFRingsCollection* m_l1ExtraHfRingEtSums;

};

#endif
