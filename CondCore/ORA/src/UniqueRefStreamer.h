#ifndef INCLUDE_ORA_UNIQUEREFSTREAMER_H
#define INCLUDE_ORA_UNIQUEREFSTREAMER_H

#include "IRelationalStreamer.h"
#include "DataElement.h"
#include "RelationalBuffer.h"
//
#include <memory>
#include <boost/shared_ptr.hpp>
// externals
#include "Reflex/Type.h"


namespace ora {

  class MappingElement;
  class ContainerSchema;
  class RelationalRefLoader;
  class DependentClassReader;

  std::string uniqueRefNullLabel();
  
  class UniqueRefWriter :  public IRelationalWriter{

    public:

      UniqueRefWriter( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~UniqueRefWriter();
      
      bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

      void setRecordId( const std::vector<int>& identity );
      
      /// Writes a data element
      void write( int oid, const void* data );

    private:

      Reflex::Type m_objectType;
      MappingElement& m_mappingElement;
      int m_columnIndexes[2];
      ContainerSchema& m_schema;
      DataElement* m_dataElement;
      IRelationalData* m_relationalData;
      RelationalBuffer* m_operationBuffer;
  };
  
  class UniqueRefUpdater :  public IRelationalUpdater {

    public:

      UniqueRefUpdater( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~UniqueRefUpdater();
      
      bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);
      
      void setRecordId( const std::vector<int>& identity );
      
      /// Updates a data element
      void update( int oid,
                   const void* data );

    private:

      UniqueRefWriter m_writer;

  };

  class UniqueRefReader : public IRelationalReader {

      public:

      UniqueRefReader( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~UniqueRefReader();
      
      bool build( DataElement& offset, IRelationalData& relationalData);

      void select( int oid );
      
      void setRecordId( const std::vector<int>& identity );

      /// Reads a data element
      void read( void* destination );

      void clear();

    private:

      Reflex::Type m_objectType;
      MappingElement& m_mappingElement;
      int m_columnIndexes[2];
      ContainerSchema& m_schema;
      DataElement* m_dataElement;
      IRelationalData* m_relationalData;
      std::vector<boost::shared_ptr<RelationalRefLoader> > m_loaders;
  };

  class UniqueRefStreamer : public IRelationalStreamer 
  {
    public:
    UniqueRefStreamer( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~UniqueRefStreamer();

    IRelationalWriter* newWriter();

    IRelationalUpdater* newUpdater();

    IRelationalReader* newReader();
    
    private:
    Reflex::Type m_objectType;
    MappingElement& m_mapping;
    ContainerSchema& m_schema;
  }; 

}
#endif

    
      
