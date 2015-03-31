#ifndef INCLUDE_ORA_PVECTORSTREAMER_H
#define INCLUDE_ORA_PVECTORSTREAMER_H

#include "STLContainerStreamer.h"

namespace ora {

  class PVectorWriter: public IRelationalWriter {
    
    public:
      /// Constructor
      PVectorWriter( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );
      virtual ~PVectorWriter();

      bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer );
      void setRecordId( const std::vector<int>& identity );
      void write( int oid,const void* data );

    private:
      STLContainerWriter m_writer;
  };
  
  class PVectorUpdater : public IRelationalUpdater {

    public:

    /// Constructor
    PVectorUpdater(const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );
    virtual ~PVectorUpdater();

    bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer);
    void setRecordId( const std::vector<int>& identity );
    void update( int oid,const void* data );

    private:
    RelationalBuffer* m_buffer;
    STLContainerWriter m_writer;
  };

  class PVectorReader : public IRelationalReader {

    public:
    
    /// Constructor
    PVectorReader(const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~PVectorReader();
    
    bool build( DataElement& offset, IRelationalData& relationalData );
    void select( int oid );
    void setRecordId( const std::vector<int>& identity );
    void read( void* address );
    void clear();

    private:
    STLContainerReader m_reader;
  };

  class PVectorStreamer : public IRelationalStreamer 
  {
    public:
    PVectorStreamer( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~PVectorStreamer();

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

    
