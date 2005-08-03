#ifndef DAQEcalTBInputService_H
#define DAQEcalTBInputService_H

/** \class EcalTBInputService
 *  An input service for ASCII data files. 
 *
 *  For the time being, reuses the services of DaqFileReader from DaqPrototype.
 *
 *  $Date: $
 *  $Revision:$
 *  \author N. Marinelli
 */

#include <PluginManager/ModuleDef.h>
#include <FWCore/Framework/interface/InputServiceMacros.h>
#include <FWCore/Framework/interface/InputService.h>
#include <FWCore/Framework/interface/InputServiceDescription.h>
#include <string>

class EcalTBDaqFileReader;

namespace raw{ class FEDRawData; }

namespace edm {

  class Retriever;
  class ParameterSet;
  class InputServiceDescription;
} 

  class DAQEcalTBInputService : public edm::InputService {

  public:
    DAQEcalTBInputService(const edm::ParameterSet& pset, 
			const edm::InputServiceDescription& desc);

    virtual ~DAQEcalTBInputService();

  private:
    virtual std::auto_ptr<edm::EventPrincipal> read();

    //void clear();

    edm::InputServiceDescription description_;
    
    edm::Retriever*  retriever_;
    EcalTBDaqFileReader * reader_;
    std::string filename_;

    int remainingEvents_;

  
  }; 
 
  DEFINE_FWK_INPUT_SERVICE(DAQEcalTBInputService)


#endif
