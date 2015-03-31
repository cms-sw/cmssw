#ifndef INCLUDE_ORA_MAPPINGDATABASE_H
#define INCLUDE_ORA_MAPPINGDATABASE_H

#include "Sequences.h"
//
#include <string>
#include <map>
#include <vector>
#include <set>

namespace edm {
  class TypeWithDict;
}

namespace ora {

  class IDatabaseSchema;
  class MappingTree;
  class MappingElement;
  class MappingRawData;
  class MappingRawElement;

    /**
       @class MappingDatabase MappingDatabase.h
       Utility class to manage the object-relational mappings for the C++ classes.
    */
  class MappingDatabase {

    public:

    static std::string versionOfClass( const edm::TypeWithDict& dictionary );
    public:
    /// Constructor
    explicit MappingDatabase( IDatabaseSchema& schema );

    /// Destructor
    ~MappingDatabase();

    void setUp();

    std::string newMappingVersionForContainer( const std::string& className );
    
    std::string newMappingVersionForDependentClass( const std::string& containerName, const std::string& className );

    bool getMappingByVersion( const std::string& version, MappingTree& destination  );
    
    void removeMapping( const std::string& version );

    bool getMappingForContainer( const std::string& className, const std::string& classVersion, int containerId, MappingTree& destination  );

    bool getBaseMappingForContainer( const std::string& className, int containerId, MappingTree& destination  );

    bool getDependentMappingsForContainer( int containerId, std::vector<MappingElement>& destination  );

    bool getDependentClassesForContainer( int containerId, std::set<std::string>& list );

    bool getClassVersionListForMappingVersion( const std::string& mappingVersion, std::set<std::string>& destination );

    bool getClassVersionListForContainer( int containerId, std::map<std::string,std::string>& versionMap );

    void insertClassVersion( const edm::TypeWithDict& dictionaryEntry, int dependencyIndex, int containerId, const std::string& mappingVersion, bool asBase=false );

    void insertClassVersion( const std::string& className, const std::string& classVersion, int dependencyIndex, int containerId, const std::string& mappingVersion, bool asBase=false );

    void setMappingVersionForClass( const edm::TypeWithDict& dictionaryEntry, int containerId, const std::string& mappingVersion , bool dependency=false);

    void storeMapping( const MappingTree& mappingStructure );
    
    bool getMappingVersionsForContainer( int containerId, std::set<std::string>& versionList );

    const std::set<std::string>& versions();
    
    void clear();
    
    private:
    void buildElement( MappingElement& parentElement, const std::string& scopeName,
                       std::map<std::string,std::vector<MappingRawElement> >& innerElements );
    void unfoldElement( const MappingElement& element, MappingRawData& destination );

    private:
      
    /// The schema in use
    IDatabaseSchema& m_schema;
    NamedSequence m_mappingSequence;
    std::set<std::string> m_versions;
    bool m_isLoaded;

  };

}

#endif
