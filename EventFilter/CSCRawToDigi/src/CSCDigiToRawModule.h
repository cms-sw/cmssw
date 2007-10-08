#ifndef EventFilter_CSCDigiToRawModule_h
#define EventFilter_CSCDigiToRawModule_h

/** \class CSCDigiToRawModule
 *
 *  $Date: 2007/07/02 21:17:06 $
 *  $Revision: 1.4 $
 *  \author A. Tumanov - Rice
 */

#include <FWCore/Framework/interface/EDProducer.h>

#include <string.h>

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
  std::string digiCreator;
};
#endif


