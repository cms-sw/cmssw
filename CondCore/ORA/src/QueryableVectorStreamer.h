#ifndef INCLUDE_ORA_QUERYABLEVECTORSTREAMER_H
#define INCLUDE_ORA_QUERYABLEVECTORSTREAMER_H

#include "IRelationalStreamer.h"
#include "DataElement.h"
//
#include <memory>
#include <boost/shared_ptr.hpp>
// externals
#include "Reflex/Type.h"

namespace ora {

  class MappingElement;
  class ContainerSchema;
  class IRelationalData;
  class IArrayHandler;
  class BulkInsertOperation;
  class IVectorLoader;
  
  class QueryableVectorWriter: public IRelationalWriter {
    
    public:
      /// Constructor
      QueryableVectorWriter( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );
      virtual ~QueryableVectorWriter();

      bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer );
      void setRecordId( const std::vector<int>& identity );
      void write( int oid,const void* data );
    public:
      Reflex::Type& objectType();
      MappingElement& mapping();
      DataElement* dataElement();
      IArrayHandler* arrayHandler();

    private:
      Reflex::Type m_objectType;
      MappingElement& m_mappingElement;
      ContainerSchema& m_schema;
      std::vector<int> m_recordId;
      DataElement m_localElement;

    private:
      DataElement* m_offset;
      BulkInsertOperation* m_insertOperation;
      std::auto_ptr<IArrayHandler> m_arrayHandler;
      std::auto_ptr<IRelationalWriter> m_dataWriter;
  };
  
  class QueryableVectorUpdater : public IRelationalUpdater {

    public:

    /// Constructor
    QueryableVectorUpdater(const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );
    virtual ~QueryableVectorUpdater();

    bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer);
    void setRecordId( const std::vector<int>& identity );
    void update( int oid,const void* data );

    private:
    RelationalBuffer* m_buffer;
    QueryableVectorWriter m_writer;
  };

  class QueryableVectorReader : public IRelationalReader {

    public:
    
    /// Constructor
    QueryableVectorReader(const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~QueryableVectorReader();
    
    bool build( DataElement& offset, IRelationalData& relationalData );
    void select( int oid );
    void setRecordId( const std::vector<int>& identity );
    void read( void* address );
    void clear();

    private:
    Reflex::Type m_objectType;
    MappingElement& m_mapping;
    ContainerSchema& m_schema;
    DataElement* m_dataElement;
    std::vector<boost::shared_ptr<IVectorLoader> > m_loaders;
    std::vector<int> m_tmpIds;
  };

  class QueryableVectorStreamer : public IRelationalStreamer 
  {
    public:
    QueryableVectorStreamer( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~QueryableVectorStreamer();

    IRelationalWriter* newWriter();

    IRelationalUpdater* newUpdater();

    IRelationalReader* newReader();
    
    private:
    Reflex::Type m_objectType;
    MappingElement& m_mapping;
    ContainerSchema& m_schema;
  }; 

  
}

inline
Reflex::Type&
ora::QueryableVectorWriter::objectType(){
  return m_objectType;
}

inline
ora::MappingElement&
ora::QueryableVectorWriter::mapping(){
  return m_mappingElement;
}

inline
ora::DataElement*
ora::QueryableVectorWriter::dataElement(){
  return m_offset;
}

inline
ora::IArrayHandler*
ora::QueryableVectorWriter::arrayHandler(){
  return m_arrayHandler.get();
}

#endif

    
