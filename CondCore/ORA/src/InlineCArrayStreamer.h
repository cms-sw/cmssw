#ifndef INCLUDE_ORA_INLINECARRAYSTREAMER_H
#define INCLUDE_ORA_INLINECARRAYSTREAMER_H

#include "IRelationalStreamer.h"
#include "RelationalStreamerFactory.h"
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace ora {

  class MappingElement;
  class ContainerSchema;
  class DataElement;

  class InlineCArrayStreamerBase  {

    public:

    InlineCArrayStreamerBase( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~InlineCArrayStreamerBase();

    bool buildDataElement( DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer* operationBuffer );
    
    virtual void processArrayElement( DataElement& arrayElementOffset, IRelationalData& relationalData, MappingElement& arrayElementMapping, RelationalBuffer* operationBuffer ) = 0;

    protected:

    edm::TypeWithDict m_objectType;
    edm::TypeWithDict m_arrayType;
    RelationalStreamerFactory m_streamerFactory;
    private:

    MappingElement& m_mapping;
  };

  class InlineCArrayWriter : public InlineCArrayStreamerBase, public IRelationalWriter {
    public:
    InlineCArrayWriter( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~InlineCArrayWriter();

    void processArrayElement( DataElement& arrayElementOffset, IRelationalData& relationalData, MappingElement& arrayElementMapping, RelationalBuffer* operationBuffer );

    bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

    void write( int oid, const void* data );

    private:
    std::vector<IRelationalWriter*> m_writers;
  };
  

  class InlineCArrayUpdater : public InlineCArrayStreamerBase, public IRelationalUpdater {
    public:
    InlineCArrayUpdater( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~InlineCArrayUpdater();

    void processArrayElement( DataElement& arrayElementOffset, IRelationalData& relationalData, MappingElement& arrayElementMapping, RelationalBuffer* operationBuffer );

    bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

    void update( int oid, const void* data );
    private:
    std::vector<IRelationalUpdater*> m_updaters;
    
  };

  class InlineCArrayReader : public InlineCArrayStreamerBase, public IRelationalReader {
    public:
    InlineCArrayReader( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~InlineCArrayReader();

    void processArrayElement( DataElement& arrayElementOffset, IRelationalData& relationalData, MappingElement& arrayElementMapping, RelationalBuffer* operationBuffer );

    bool build(DataElement& dataElement, IRelationalData& relationalData );

    void select( int oid );

    void setRecordId( const std::vector<int>& identity );

    void read( void* data );

    void clear();
    
    private:
    std::vector<IRelationalReader*> m_readers;
  };

  class InlineCArrayStreamer : public IRelationalStreamer 
  {
    public:
    InlineCArrayStreamer( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~InlineCArrayStreamer();

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

    
      
