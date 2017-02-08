#ifndef INCLUDE_ORA_CONTAINERSCHEMA_H
#define INCLUDE_ORA_CONTAINERSCHEMA_H

#include "MappingTree.h"
#include "Sequences.h"
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

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
                     const edm::TypeWithDict& containerType,
                     DatabaseSession& session );

    ContainerSchema( int containerId,
                     const std::string& containerName,
                     const std::string& className,
                     DatabaseSession& session );

    ~ContainerSchema();

    void create();

    void drop();

    void evolve();

    void create( const edm::TypeWithDict& dependentClass );

    void evolve( const edm::TypeWithDict& dependentClass, MappingTree& baseMapping );
    
    void setAccessPermission( const std::string& principal, bool forWrite );
    
    const edm::TypeWithDict& type();

    MappingTree& mapping( bool writeEnabled=false);

    bool extendIfRequired( const edm::TypeWithDict& dependentClass );
    
    MappingElement& mappingForDependentClass( const edm::TypeWithDict& dependentClass, bool writeEnabled=false );

    bool mappingForDependentClasses( std::vector<MappingElement>& destination );

    Sequences& containerSequences();

    IBlobStreamingService* blobStreamingService();

    IReferenceHandler* referenceHandler();

    int containerId();

    const std::string& containerName();

    const std::string& className();

    const std::string& mappingVersion();

    coral::ISchema& storageSchema();

    DatabaseSession& dbSession();

    private:
    void initClassDict();
    bool loadMappingForDependentClass( const edm::TypeWithDict& dependentClass );
    void extend( const edm::TypeWithDict& dependentClass );
    void getTableHierarchy( const std::set<std::string>& containerMappingVersions, std::vector<std::string>& destination );
    private:

    int m_containerId;
    std::string m_containerName;
    std::string m_className;
    edm::TypeWithDict m_classDict;
    DatabaseSession& m_session;
    bool m_loaded;
    Sequences m_containerSchemaSequences;
    MappingTree m_mapping;
    std::map<std::string,MappingTree*> m_dependentMappings;
    
  };
}
  
#endif
