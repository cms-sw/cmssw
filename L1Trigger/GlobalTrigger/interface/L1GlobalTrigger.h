#ifndef GlobalTrigger_L1GlobalTrigger_h
#define GlobalTrigger_L1GlobalTrigger_h

/**
 * \class L1GlobalTrigger
 * 
 * 
 * Description: L1 Global Trigger producer.
 *  
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 * The CMSSW implementation of the L1 Global Trigger emulator
 * uses concepts and code from the ORCA L1 Global Trigger simulation,
 * authors: N. Neumeister, M. Fierro, M. Eder  - HEPHY Vienna.
 *  
 */

// system include files
#include <string>

#include <boost/cstdint.hpp>

// user include files

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// forward classes
class L1GlobalTriggerSetup;
class L1GlobalTriggerPSB;
class L1GlobalTriggerGTL;
class L1GlobalTriggerFDL;


// class declaration
class L1GlobalTrigger : public edm::EDProducer
{

public:

    explicit L1GlobalTrigger(const edm::ParameterSet&);
    ~L1GlobalTrigger();

    virtual void produce(edm::Event&, const edm::EventSetup&);

    // return pointer to setup
    inline const L1GlobalTriggerSetup* gtSetup() const
    {
        return m_gtSetup;
    }

    // return pointer to PSB
    inline const L1GlobalTriggerPSB* gtPSB() const
    {
        return m_gtPSB;
    }

    // return pointer to GTL
    inline const L1GlobalTriggerGTL* gtGTL() const
    {
        return m_gtGTL;
    }

    // return pointer to FDL
    inline const L1GlobalTriggerFDL* gtFDL() const
    {
        return m_gtFDL;
    }

private:

    static L1GlobalTriggerSetup* m_gtSetup;

    L1GlobalTriggerPSB* m_gtPSB;
    L1GlobalTriggerGTL* m_gtGTL;
    L1GlobalTriggerFDL* m_gtFDL;
    
    edm::InputTag m_muGmtInputTag;

    edm::InputTag m_caloGctInputTag;
    

};

#endif /*GlobalTrigger_L1GlobalTrigger_h*/
