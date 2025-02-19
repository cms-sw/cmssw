#ifndef INCLUDE_ORA_PRIMITIVESTREAMER_H
#define INCLUDE_ORA_PRIMITIVESTREAMER_H

#include "IRelationalStreamer.h"
// externals
#include "Reflex/Type.h"

namespace ora {

  class MappingElement;
  class DataElement;

  class PrimitiveStreamerBase  {

      public:

      PrimitiveStreamerBase( const Reflex::Type& objectType, MappingElement& mapping );

      virtual ~PrimitiveStreamerBase();

      bool buildDataElement( DataElement& dataElement, IRelationalData& relationalData );
      
      void bindDataForUpdate( const void* data );

      void bindDataForRead( void* data );

    private:

      Reflex::Type m_objectType;
      MappingElement& m_mapping;
      int m_columnIndex;
      DataElement* m_dataElement;
      IRelationalData* m_relationalData;
  };

  class PrimitiveWriter : public PrimitiveStreamerBase, public IRelationalWriter {
    public:
    PrimitiveWriter( const Reflex::Type& objectType, MappingElement& mapping );

    virtual ~PrimitiveWriter();

    bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

    void write( int oid, const void* data );
    
  };
  

  class PrimitiveUpdater : public PrimitiveStreamerBase, public IRelationalUpdater {
    public:
    PrimitiveUpdater( const Reflex::Type& objectType, MappingElement& mapping );

    virtual ~PrimitiveUpdater();

    bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

    void update( int oid, const void* data );
    
  };

  class PrimitiveReader : public PrimitiveStreamerBase, public IRelationalReader {
    public:
    PrimitiveReader( const Reflex::Type& objectType, MappingElement& mapping );

    virtual ~PrimitiveReader();

    bool build(DataElement& dataElement, IRelationalData& relationalData );

    void select( int oid );

    void setRecordId( const std::vector<int>& identity );

    void read( void* data );

    void clear();

  };

  class PrimitiveStreamer : public IRelationalStreamer 
  {
    public:
    PrimitiveStreamer( const Reflex::Type& objectType, MappingElement& mapping );

    ~PrimitiveStreamer();

    IRelationalWriter* newWriter();

    IRelationalUpdater* newUpdater();

    IRelationalReader* newReader();

    private:
    Reflex::Type m_objectType;
    MappingElement& m_mapping;
  }; 
  
}


#endif

    
      
