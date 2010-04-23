#ifndef EventFilter_CSCDigiToRawModule_h
#define EventFilter_CSCDigiToRawModule_h

/** \class CSCDigiToRawModule
 *
 *  $Date: 2008/06/26 18:43:21 $
 *  $Revision: 1.7 $
 *  \author A. Tumanov - Rice
 */

#include <FWCore/Framework/interface/EDProducer.h>

class CSCDigiToRaw;

class CSCDigiToRawModule : public edm::EDProducer {
 public:
  /// Constructor
  CSCDigiToRawModule(const edm::ParameterSet & pset);

  /// Destructor
  virtual ~CSCDigiToRawModule();

  // Operations
  virtual void produce( edm::Event&, const edm::EventSetup& );

 private:
  CSCDigiToRaw * packer;
  edm::InputTag theStripDigiTag;
  edm::InputTag theWireDigiTag;
  edm::InputTag theComparatorDigiTag;
  edm::InputTag theALCTDigiTag;
  edm::InputTag theCLCTDigiTag;
  edm::InputTag thePreTriggerTag;
  edm::InputTag theCorrelatedLCTDigiTag;
};
#endif


