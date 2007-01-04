#ifndef RPCRawToDigi_RPCPackingModule_H
#define RPCRawToDigi_RPCPackingModule_H

/** \class RPCPackingModule
 *  Driver class for digi to raw data conversions 
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm {class ParameterSet;}
namespace edm {class EventSetup; }
namespace edm {class Event; }

class RPCPackingModule : public edm::EDProducer {
public:

  /// ctor
  explicit RPCPackingModule( const edm::ParameterSet& );

    /// dtor
  virtual ~RPCPackingModule();
  /// initialisation. Retrieves cabling map from EventSetup.
  //virtual void beginJob( const edm::EventSetup& );

  /// dummy end of job
  //virtual void endJob() {}

  /// get data, convert to raw event, attach again to Event
  virtual void produce( edm::Event&, const edm::EventSetup& );

private:
// pack header  
// pack data
// pack trailer

private:

//  edm::InputTag digiLabel_;
  unsigned long eventCounter_;
};
#endif
