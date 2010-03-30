#ifndef INCLUDE_ORA_MAPPINGRULES_H
#define INCLUDE_ORA_MAPPINGRULES_H

//
#include <string>
#include <locale>
#include <algorithm>

namespace Reflex{
  class Type;
}

namespace ora {

  class MappingRules {
    public:

      static const size_t MaxTableNameLength  = 20;
      static const size_t MaxColumnNameLength = 30;
      
      static const size_t ClassNameLengthForSchema  = 7;

      static const size_t MaxColumnsPerTable = 100;
      static const size_t MaxColumnsForInlineCArray = 13;
    
    public:

      static std::string containerIdSequenceName();

      static std::string mappingSequenceName();

      static std::string sequenceNameForContainer( const std::string& containerName );
      
      /// Returns the name of the index for the identity of an object mapped to a table
      static std::string indexNameForIdentity( const std::string& tableName );
      /// Returns the name of the foreign key to the identity of the parrent table
      static std::string fkNameForIdentity( const std::string& tableName );

      static std::string newMappingVersionForContainer( const std::string& containerName, int iteration );

      static std::string newMappingVersionForDependentClass( const std::string& containerName, const std::string& className, int iteration );

      static std::string classVersionPropertyNameInDictionary();

      static std::string mappingPropertyNameInDictionary();

      static bool isMappedToBlob(const std::string& mappingProperty);

      static std::string baseClassVersion();

      static std::string defaultClassVersion(const std::string& className);

      static std::string classId( const std::string& className, const std::string& classVersion );
      
      static std::string classVersionFromId( const std::string& classId );
      
      static std::string baseIdForClass( const std::string& className );

    public:
      /// Returns the default name for a table given a class name
      static std::string fullNameForSchema( const std::string& parentVariableName, const std::string& variableName );
      static std::string tableNameForClass( const std::string& className, size_t maxLength=MaxTableNameLength );
      static std::string tableNameForVariable( const std::string& variableName,const std::string& mainTableName );
      static std::string tableNameForDependency( const std::string& mainTableName, unsigned int index );
      static std::string columnNameForId();
      static std::string columnNameForRefColumn();
      static std::string columnNameForPosition();
      static std::string columnNameForVariable( const std::string& variableName, const std::string& scope, bool forData=true );
      static std::string columnNameForOID( const std::string& variableName, const std::string& scope, unsigned int index );
      static std::string columnNameForRefMetadata( const std::string& variableName, const std::string& scope );
      static std::string columnNameForRefId( const std::string& variableName, const std::string& scope );
      static std::string sequenceNameForDependentClass( const std::string& containerName, const std::string& className );

      static std::string formatForSchema( const std::string& variableName, size_t maxLength);
      static std::string newNameForSchemaObject( const std::string& initialName, unsigned int index, size_t maxLength);
      static std::string newNameForDepSchemaObject( const std::string& initialName, unsigned int index, size_t maxLength);
      /// Forms the scoped variable names for all of the fields
      static std::string scopedVariableName( const std::string& variableName, const std::string& scope );
      /// Forms the scoped variable names for the schema objects
      static std::string scopedVariableForSchemaObjects( const std::string& variableName, const std::string& scope );
      /// Forms the variable names for the c-array variable
      static std::string variableNameForArrayIndex( const std::string& arrayVariable, unsigned int index );
      /// Forms the variable names for the c-array columns
      static std::string variableNameForArrayColumn( unsigned int arrayIndex );
      static std::string variableNameForArrayColumn( const std::string& arrayVariable, unsigned int arrayIndex );

      static std::string variableNameForArrayColumn( const Reflex::Type& array );
      
      static std::string referenceColumnKey( const std::string& tableName, const std::string& columnName );
      
      static std::string variableNameForContainerValue();
      static std::string variableNameForContainerKey();
      
    private:

      static std::string shortName( const std::string& nameCut, size_t maxLenght);
      static std::string nameForSchema( const std::string& variableName );

      struct ToUpper {
          ToUpper(const std::locale& l):loc(l) {}
          char operator() (char c) const  { return std::toupper(c,loc); }
          std::locale const& loc;
      };
  };
}

#endif

