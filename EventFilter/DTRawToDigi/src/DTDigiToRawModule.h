#ifndef EventFilter_DTDigiToRawModule_h
#define EventFilter_DTDigiToRawModule_h

/** \class DTDigiToRawModule
 *
 *  $Date: $
 *  $Revision: $
 *  \author N. Amapane - CERN
 */

#include <FWCore/Framework/interface/EDProducer.h>

class DTDigiToRaw;

class DTDigiToRawModule : public edm::EDProducer {
public:
  /// Constructor
  DTDigiToRawModule();

  /// Destructor
  virtual ~DTDigiToRawModule();

  // Operations
  virtual void produce( edm::Event&, const edm::EventSetup& );

private:
  DTDigiToRaw * packer;

};
#endif

