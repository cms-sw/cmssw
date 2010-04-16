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

  class MappingGenerator {

    public:
    /// Constructor
    explicit MappingGenerator( coral::ISchema& schema );
    
    /// Destructor
    ~MappingGenerator();

    void createNewMapping( const std::string& containerName, const Reflex::Type& classDictionary, MappingTree& destination );

    void createNewMapping( const std::string& containerName, const Reflex::Type& classDictionary,
                           const MappingTree& baseMapping, MappingTree& destination );

    void createNewDependentMapping( const Reflex::Type& dependentClassDictionary, const MappingTree& parentClassMapping,
                                    MappingTree& destination );

    void createNewDependentMapping( const Reflex::Type& dependentClassDictionary, const MappingTree& parentClassMapping,
                                    const MappingTree& dependentClassBaseMapping, MappingTree& destination );

    private:
    coral::ISchema& m_schema;
    TableRegister m_tableRegister;

  };

}

#endif
