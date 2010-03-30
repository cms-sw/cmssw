#ifndef INCLUDE_ORA_MAPPINGGENERATOR_H
#define INCLUDE_ORA_MAPPINGGENERATOR_H

#include "TableRegister.h"
//
#include <string>
#include <vector>
#include <map>
#include <set>

namespace Reflex{
  class Type;
}

namespace coral {
  class ISchema;
}

namespace ora {
  
  class MappingTree;
  class MappingElement;
  class ArrayMember;

  class MappingGenerator {

    public:

    // Calculate the number of columns required for a given class
    static size_t sizeInColumns(const Reflex::Type& topLevelClassType );
    static std::pair<bool,size_t> sizeInColumnsForCArray( const Reflex::Type& arrayType );
    
    public:
    /// Constructor
    explicit MappingGenerator( coral::ISchema& schema );
    
    /// Destructor
    ~MappingGenerator();

    bool createNewMapping( const std::string& containerName, const Reflex::Type& classDictionary, MappingTree& destination );

    bool createNewMapping( const std::string& containerName, const Reflex::Type& classDictionary,
                           MappingTree& baseMapping, MappingTree& destination );

    bool createNewDependentMapping( const Reflex::Type& dependentClassDictionary, MappingTree& parentClassMapping,
                                    MappingTree& destination );

    bool createNewDependentMapping( const Reflex::Type& dependentClassDictionary, MappingTree& parentClassMapping,
                                    MappingTree& dependentClassBaseMapping, MappingTree& destination );

    TableRegister& tableRegister();

    private:

    static void _sizeInColumns(const Reflex::Type& typ, size_t& sz, bool& hasDependencies );
    static void _sizeInColumnsForCArray(const Reflex::Type& typ, size_t& sz, bool& hasDependencies );

    bool processClass( const std::string& containerName, const Reflex::Type& classDictionary, MappingTree& destination );

    bool processDependentClass( const Reflex::Type& classDictionary,MappingTree& parentClassMapping,
                                MappingTree& destination );
    
    bool buildCArrayElementTree( MappingElement& topElement );
    
    bool processBaseClasses( MappingElement& mappingElement,
                             const Reflex::Type& classDictionary,
                             std::vector<ArrayMember>& carrays);

    bool processObject( MappingElement& mappingElement,
                        const Reflex::Type& classDictionary );

    bool processItem( MappingElement& parentelement,
                      const std::string& attributeName,
                      const std::string& attributeNameForSchema,
                      const Reflex::Type& attributeType,
                      bool arraysInBlob );

    bool processPrimitive( MappingElement& parentelement,
                           const std::string& attributeName,
                           const std::string& attributeNameForSchema,
                           const Reflex::Type& attributeType );

    bool processBlob( MappingElement& parentelement,
                      const std::string& attributeName,
                      const std::string& attributeNameForSchema,
                      const Reflex::Type& attributeType );

    bool processLeafElement( const std::string& elementType,
                             MappingElement& parentelement,
                             const std::string& attributeName,
                             const std::string& attributeNameForSchema,
                             const std::string& typeName );

    bool processEmbeddedClass( MappingElement& parentelement,
                               const std::string& attributeName,
                               const std::string& attributeNameForSchema,
                               const Reflex::Type& attributeType );
    
    bool processOraReference( MappingElement& parentelement,
                              const std::string& attributeName,
                              const std::string& attributeNameForSchema,
                              const Reflex::Type& attributeType );

    bool processArray( MappingElement& parentelement,
                       const std::string& attributeName,
                       const std::string& attributeNameForSchema,
                       const Reflex::Type& attributeType );

    bool processCArray( MappingElement& parentelement,
                        const std::string& attributeName,
                        const std::string& attributeNameForSchema,
                        const Reflex::Type& attributeType );

    bool processCArrayItem( MappingElement& parentelement,
                            const std::string& attributeName,
                            const std::string& attributeNameForSchema,
                            const Reflex::Type& attributeType,
                            const Reflex::Type& arrayElementType );

    bool processInlineCArrayItem( MappingElement& parentelement,
                                  const std::string& attributeName,
                                  const std::string& attributeNameForSchema,
                                  const Reflex::Type& attributeType,
                                  const Reflex::Type& arrayElementType );

    bool processOraPtr( MappingElement& parentelement,
                        const std::string& attributeName,
                        const std::string& attributeNameForSchema,
                        const Reflex::Type& attributeType,
                        bool arraysInBlobs);

    bool processUniqueReference( MappingElement& parentelement,
                                 const std::string& attributeName,
                                 const std::string& attributeNameForSchema,
                                 const Reflex::Type& attributeType );

    private:

    coral::ISchema& m_schema;
    TableRegister m_tableRegister;

  };

}

#endif
