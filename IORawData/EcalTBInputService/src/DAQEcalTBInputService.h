#ifndef DAQEcalTBInputService_H
#define DAQEcalTBInputService_H

/** \class EcalTBInputService
 *  An input service for ASCII data files. 
 *
 *  For the time being, reuses the services of DaqFileReader from DaqPrototype.
 *
 *  $Date: 2005/10/06 17:47:06 $
 *  $Revision: 1.3 $
 *  \author N. Marinelli
 */

//#include <PluginManager/ModuleDef.h>
//#include "FWCore/Framework/interface/InputSourceMacros.h"
#include <FWCore/Framework/interface/InputSource.h>
#include <FWCore/Framework/interface/InputSourceDescription.h>
#include <FWCore/Framework/interface/ProductDescription.h>
#include "FWCore/Framework/interface/ExternalInputSource.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include <string>

class EcalTBDaqFileReader;

namespace edm {
  class ParameterSet;
  class InputSourceDescription;
} 

class DAQEcalTBInputService : public edm::ExternalInputSource
{
  
  public:
    DAQEcalTBInputService(const edm::ParameterSet& pset, 
			const edm::InputSourceDescription& desc);

    virtual ~DAQEcalTBInputService();

  protected:
    virtual void setRunAndEventInfo();
    virtual bool produce(edm::Event & e);
  private:
/*     virtual std::auto_ptr<edm::EventPrincipal> read(); */

    //void clear();

    EcalTBDaqFileReader * reader_;
    bool isBinary_;    
    unsigned int fileCounter_;

  
  }; 
 
//  DEFINE_FWK_INPUT_SOURCE(DAQEcalTBInputService)


#endif
