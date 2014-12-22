#ifndef INCLUDE_ORA_UNIQUEREFSTREAMER_H
#define INCLUDE_ORA_UNIQUEREFSTREAMER_H

#include "IRelationalStreamer.h"
#include "DataElement.h"
#include "RelationalBuffer.h"
//
#include <memory>
#include <boost/shared_ptr.hpp>
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"


namespace ora {

  class MappingElement;
  class ContainerSchema;
  class RelationalRefLoader;
  class DependentClassReader;

  std::string uniqueRefNullLabel();
  
  class UniqueRefWriter :  public IRelationalWriter{

    public:

      UniqueRefWriter( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~UniqueRefWriter();
      
      bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

      void setRecordId( const std::vector<int>& identity );
      
      /// Writes a data element
      void write( int oid, const void* data );

    private:

      edm::TypeWithDict m_objectType;
      MappingElement& m_mappingElement;
      int m_columnIndexes[2];
      ContainerSchema& m_schema;
      DataElement* m_dataElement;
      IRelationalData* m_relationalData;
      RelationalBuffer* m_operationBuffer;
  };
  
  class UniqueRefUpdater :  public IRelationalUpdater {

    public:

      UniqueRefUpdater( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

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

      UniqueRefReader( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~UniqueRefReader();
      
      bool build( DataElement& offset, IRelationalData& relationalData);

      void select( int oid );
      
      void setRecordId( const std::vector<int>& identity );

      /// Reads a data element
      void read( void* destination );

      void clear();

    private:

      edm::TypeWithDict m_objectType;
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
    UniqueRefStreamer( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~UniqueRefStreamer();

    IRelationalWriter* newWriter();

    IRelationalUpdater* newUpdater();

    IRelationalReader* newReader();
    
    private:
    edm::TypeWithDict m_objectType;
    MappingElement& m_mapping;
    ContainerSchema& m_schema;
  }; 

}
#endif

    
      
