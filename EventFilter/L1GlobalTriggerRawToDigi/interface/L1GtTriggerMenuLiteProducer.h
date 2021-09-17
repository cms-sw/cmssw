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
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"
#include "CondFormats/DataRecord/interface/L1GtStableParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// forward declarations
class L1GtStableParameters;
class L1GtTriggerMenu;
class L1GtTriggerMask;
class L1GtPrescaleFactors;

// class declaration
class L1GtTriggerMenuLiteProducer : public edm::one::EDProducer<edm::BeginRunProducer> {
public:
  /// constructor(s)
  explicit L1GtTriggerMenuLiteProducer(const edm::ParameterSet&);

  /// destructor
  ~L1GtTriggerMenuLiteProducer() override;

private:
  /// retrieve all the relevant L1 trigger event setup records
  /// and cache them to improve the speed
  void retrieveL1EventSetup(const edm::EventSetup&);

  void beginJob() final;
  void beginRunProduce(edm::Run&, const edm::EventSetup&) final;

  void produce(edm::Event&, const edm::EventSetup&) final;

  void endJob() override;

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

  /// EventSetup Tokens
  const edm::ESGetToken<L1GtStableParameters, L1GtStableParametersRcd> m_l1GtStableParamToken;
  const edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsAlgoTrigRcd> m_l1GtPfAlgoToken;
  const edm::ESGetToken<L1GtPrescaleFactors, L1GtPrescaleFactorsTechTrigRcd> m_l1GtPfTechToken;
  const edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd> m_l1GtTmAlgoToken;
  const edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskTechTrigRcd> m_l1GtTmTechToken;
  const edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> m_l1GtMenuToken;

private:
  /// index of physics DAQ partition
  unsigned int m_physicsDaqPartition;
};

#endif  // EventFilter_L1GlobalTriggerRawToDigi_L1GtTriggerMenuLiteProducer_h
