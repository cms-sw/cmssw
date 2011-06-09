#ifndef INCLUDE_ORA_ORAREFERENCESTREAMER_H
#define INCLUDE_ORA_ORAREFERENCESTREAMER_H

#include "IRelationalStreamer.h"
// externals
#include "Reflex/Type.h"

namespace ora {

  class MappingElement;
  class ContainerSchema;
  
  class OraReferenceStreamerBase {

    public:

    explicit OraReferenceStreamerBase( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& schema );

    virtual ~OraReferenceStreamerBase();

      bool buildDataElement( DataElement& dataElement, IRelationalData& relationalData );
      
      void bindDataForUpdate( const void* data );

      void bindDataForRead( void* data );

    private:

      Reflex::Type m_objectType;
      MappingElement& m_mapping;
      int m_columnIndexes[2];
      ContainerSchema& m_schema;
      DataElement* m_dataElement;
      DataElement* m_dataElemOId0;
      DataElement* m_dataElemOId1;
      IRelationalData* m_relationalData;
  };

  class OraReferenceWriter : public OraReferenceStreamerBase, public IRelationalWriter {
    public:
    explicit OraReferenceWriter( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& schema );

    virtual ~OraReferenceWriter();

    bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

    void write( int oid, const void* data );
    
  };
  

  class OraReferenceUpdater : public OraReferenceStreamerBase, public IRelationalUpdater {
    public:
    explicit OraReferenceUpdater( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& schema );

    virtual ~OraReferenceUpdater();

    bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

    void update( int oid, const void* data );
    
  };

  class OraReferenceReader : public OraReferenceStreamerBase, public IRelationalReader {
    public:
    explicit OraReferenceReader( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& schema );

    virtual ~OraReferenceReader();

    bool build(DataElement& dataElement, IRelationalData& relationalData );

    void select( int oid );

    void setRecordId( const std::vector<int>& identity );

    void read( void* data );

    void clear();
    
  };

  class OraReferenceStreamer : public IRelationalStreamer 
  {
    public:
    explicit OraReferenceStreamer( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& schema );

    ~OraReferenceStreamer();

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

    
      
