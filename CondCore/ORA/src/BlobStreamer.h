#ifndef INCLUDE_ORA_BLOBSTREAMER_H
#define INCLUDE_ORA_BLOBSTREAMER_H

#include "IRelationalStreamer.h"
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace ora {

  class MappingElement;
  class ContainerSchema;
  class DataElement;
  class IBlobStreamingService;
  
  class BlobWriterBase  {

    public:
    BlobWriterBase( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~BlobWriterBase();

      bool buildDataElement( DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer );
      
      void bindData( const void* data );

    private:
      edm::TypeWithDict m_objectType;
      MappingElement& m_mapping;
      int m_columnIndex;
      ContainerSchema& m_schema;
      
    private:
      DataElement* m_dataElement;
      IRelationalData* m_relationalData;
      RelationalBuffer* m_relationalBuffer;
      IBlobStreamingService* m_blobWriter;
      bool m_useCompression;
  };

  class BlobWriter : public BlobWriterBase, public IRelationalWriter {
    
    public:
    BlobWriter( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~BlobWriter();

    bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

    void write( int oid, const void* data );
    
  };
  

  class BlobUpdater : public BlobWriterBase, public IRelationalUpdater {
    
    public:
    BlobUpdater( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~BlobUpdater();

    bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

    void update( int oid, const void* data );
    
  };

  class BlobReader : public IRelationalReader {
    
    public:
    BlobReader( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~BlobReader();

    bool build(DataElement& dataElement, IRelationalData& relationalData );

    void select( int oid );

    void setRecordId( const std::vector<int>& identity );

    void read( void* data );

    void clear();

    private:
      edm::TypeWithDict m_objectType;
      MappingElement& m_mapping;
      int m_columnIndex;
      ContainerSchema& m_schema;
      
    private:
      DataElement* m_dataElement;
      IRelationalData* m_relationalData;
      IBlobStreamingService* m_blobReader;

  };

  class BlobStreamer : public IRelationalStreamer 
  {
    public:
    BlobStreamer( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~BlobStreamer();

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

    
      
