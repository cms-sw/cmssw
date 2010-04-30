#ifndef INCLUDE_ORA_CONTAINERSCHEMA_H
#define INCLUDE_ORA_CONTAINERSCHEMA_H

#include "MappingTree.h"
#include "Sequences.h"
// externals
#include "Reflex/Type.h"

namespace coral {
  class ISchema;
}

namespace ora {

  class DatabaseSession;
  class IBlobStreamingService;
  class IReferenceHandler;

  class ContainerSchema {
    public:

    ContainerSchema( int containerId,
                     const std::string& containerName,
                     const Reflex::Type& containerType,
                     DatabaseSession& session );

    ContainerSchema( int containerId,
                     const std::string& containerName,
                     const std::string& className,
                     DatabaseSession& session );

    ~ContainerSchema();

    void create();

    void drop();

    void evolve();

    void create( const Reflex::Type& dependentClass );

    void evolve( const Reflex::Type& dependentClass, MappingTree& baseMapping );
    
    const Reflex::Type& type();

    MappingTree& mapping( bool writeEnabled=false);

    bool extendIfRequired( const Reflex::Type& dependentClass );
    
    MappingElement& mappingForDependentClass( const Reflex::Type& dependentClass, bool writeEnabled=false );

    bool mappingForDependentClasses( std::vector<MappingElement>& destination );

    Sequences& containerSequences();

    IBlobStreamingService* blobStreamingService();

    IReferenceHandler* referenceHandler();

    int containerId();

    const std::string& containerName();

    const std::string& className();

    const std::string& mappingVersion();

    coral::ISchema& storageSchema();

    private:
    void checkClassDict();
    bool loadMappingForDependentClass( const Reflex::Type& dependentClass );
    void extend( const Reflex::Type& dependentClass );
    private:

    int m_containerId;
    std::string m_containerName;
    std::string m_className;
    Reflex::Type m_classDict;
    DatabaseSession& m_session;
    bool m_loaded;
    Sequences m_containerSchemaSequences;
    MappingTree m_mapping;
    std::map<std::string,MappingTree*> m_dependentMappings;
    
  };
}
  
#endif
