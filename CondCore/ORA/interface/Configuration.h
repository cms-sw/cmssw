#ifndef INCLUDE_ORA_CONFIGURATION_H
#define INCLUDE_ORA_CONFIGURATION_H

#include "Properties.h"
//
#include <memory>
// externals
#include "CoralBase/MessageStream.h"

namespace ora {

  class IBlobStreamingService;
  class IReferenceHandler;
  
  class Configuration {
    public:

    static std::string automaticDatabaseCreation();
    static std::string automaticContainerCreation();
    static std::string automaticSchemaEvolution();
    
    public:

    Configuration();

    virtual ~Configuration();

    void setBlobStreamingService( IBlobStreamingService* service );
    
    IBlobStreamingService* blobStreamingService();

    void setReferenceHandler( IReferenceHandler* handler );
    
    IReferenceHandler* referenceHandler();

    Properties& properties();

    void setMessageVerbosity( coral::MsgLevel level );

    private:

    std::auto_ptr<IBlobStreamingService> m_blobStreamingService;
    
    std::auto_ptr<IReferenceHandler> m_referenceHandler;

    Properties m_properties;

  };
}

#endif
