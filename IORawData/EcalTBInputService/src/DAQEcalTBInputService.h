#ifndef DAQEcalTBInputService_H
#define DAQEcalTBInputService_H

/** \class EcalTBInputService
 *  An input service for ASCII data files. 
 *
 *  For the time being, reuses the services of DaqFileReader from DaqPrototype.
 *
 *  $Date: 2005/08/05 14:31:54 $
 *  $Revision: 1.2 $
 *  \author N. Marinelli
 */

//#include <PluginManager/ModuleDef.h>
//#include "FWCore/Framework/interface/InputSourceMacros.h"
#include <FWCore/Framework/interface/InputSource.h>
#include <FWCore/Framework/interface/InputSourceDescription.h>
#include <FWCore/Framework/interface/ProductDescription.h>
#include <string>

class EcalTBDaqFileReader;
class FEDRawData;

namespace raw{ class FEDRawData; }

namespace edm {

  class Retriever;
  class ParameterSet;
  class InputSourceDescription;
} 

  class DAQEcalTBInputService : public edm::InputSource {

  public:
    DAQEcalTBInputService(const edm::ParameterSet& pset, 
			const edm::InputSourceDescription& desc);

    virtual ~DAQEcalTBInputService();

  private:
    virtual std::auto_ptr<edm::EventPrincipal> read();

    //void clear();

    edm::InputSourceDescription description_;
    edm::ProductDescription fedrawdataDescription_;
     
    edm::Retriever*  retriever_;
    EcalTBDaqFileReader * reader_;
    std::string filename_;

    int remainingEvents_;

  
  }; 
 
//  DEFINE_FWK_INPUT_SOURCE(DAQEcalTBInputService)


#endif
