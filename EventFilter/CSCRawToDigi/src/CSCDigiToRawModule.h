#ifndef EventFilter_CSCDigiToRawModule_h
#define EventFilter_CSCDigiToRawModule_h

/** \class CSCDigiToRawModule
 *
 *  $Date: 2007/10/08 22:01:06 $
 *  $Revision: 1.5 $
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
};
#endif


