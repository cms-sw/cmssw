#ifndef EventFilter_L1GlobalTriggerRawToDigi_L1GtTriggerMenuLiteProducer_h
#define EventFilter_L1GlobalTriggerRawToDigi_L1GtTriggerMenuLiteProducer_h

/**
 * \class L1GtTriggerMenuLiteProducer
 * 
 * 
 * Description: L1GtTriggerMenuLite producer.
 *
 * Implementation:
 *    Read the L1 trigger menu, the trigger masks and the prescale factor sets
 *    from event setup and save a lite version (top level menu, trigger masks
 *    for physics partition and prescale factor set) in Run Data.
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna 
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

// forward declarations
class L1GtStableParameters;
class L1GtTriggerMenu;
class L1GtTriggerMask;
class L1GtPrescaleFactors;

// class declaration
class L1GtTriggerMenuLiteProducer : public edm::one::EDProducer<edm::BeginRunProducer>
{

public:

    /// constructor(s)
    explicit L1GtTriggerMenuLiteProducer(const edm::ParameterSet&);

    /// destructor
    virtual ~L1GtTriggerMenuLiteProducer();

private:

    /// retrieve all the relevant L1 trigger event setup records
    /// and cache them to improve the speed
    void retrieveL1EventSetup(const edm::EventSetup&);

    virtual void beginJob() override final;
    void beginRunProduce(edm::Run&, const edm::EventSetup&) override final;

    virtual void produce(edm::Event&, const edm::EventSetup&) override final;

    virtual void endJob();

private:
    
    /// cached stuff

    /// stable parameters
    const L1GtStableParameters* m_l1GtStablePar;
    unsigned long long m_l1GtStableParCacheID;

    /// number of physics triggers
    unsigned int m_numberPhysTriggers;

    /// number of technical triggers
    unsigned int m_numberTechnicalTriggers;

    // trigger menu
    const L1GtTriggerMenu* m_l1GtMenu;
    unsigned long long m_l1GtMenuCacheID;

    const AlgorithmMap* m_algorithmMap;
    const AlgorithmMap* m_algorithmAliasMap;
    const AlgorithmMap* m_technicalTriggerMap;

    /// trigger masks
    const L1GtTriggerMask* m_l1GtTmAlgo;
    unsigned long long m_l1GtTmAlgoCacheID;
 
    const L1GtTriggerMask* m_l1GtTmTech;
    unsigned long long m_l1GtTmTechCacheID;
    
    const std::vector<unsigned int>* m_triggerMaskAlgoTrig;
    const std::vector<unsigned int>* m_triggerMaskTechTrig;

    /// prescale factors
    const L1GtPrescaleFactors* m_l1GtPfAlgo;
    unsigned long long m_l1GtPfAlgoCacheID;

    const L1GtPrescaleFactors* m_l1GtPfTech;
    unsigned long long m_l1GtPfTechCacheID;

    const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;
    const std::vector<std::vector<int> >* m_prescaleFactorsTechTrig;

private:

    /// index of physics DAQ partition
    unsigned int m_physicsDaqPartition;

};

#endif // EventFilter_L1GlobalTriggerRawToDigi_L1GtTriggerMenuLiteProducer_h
