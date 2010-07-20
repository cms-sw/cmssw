#ifndef INCLUDE_ORA_STLCONTAINERSTREAMER_H
#define INCLUDE_ORA_STLCONTAINERSTREAMER_H

#include "DataElement.h"
#include "IRelationalStreamer.h"
#include "RelationalDeleter.h"
//
#include <memory>
// externals
#include "Reflex/Type.h"

namespace ora {

  class MappingElement;
  class ContainerSchema;

  class IArrayHandler;
  class BulkInsertOperation;
  class MultiRecordSelectOperation;

  class STLContainerWriter : public IRelationalWriter {
    
    public:
      /// Constructor
      STLContainerWriter( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );
      virtual ~STLContainerWriter();

      MappingElement& mapping();
      DataElement* dataElement();
      IArrayHandler* arrayHandler();

    public:

      bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer );
      
      void setRecordId( const std::vector<int>& identity );

      /// Writes a data element
      void write( int oid,const void* data );

    private:
      Reflex::Type m_objectType;
      MappingElement& m_mappingElement;
      ContainerSchema& m_schema;
      std::vector<int> m_recordId;
      DataElement m_localElement;

    private:
      bool m_associative;
      DataElement* m_offset;
      BulkInsertOperation* m_insertOperation;
      std::auto_ptr<IArrayHandler> m_arrayHandler;
      std::auto_ptr<IRelationalWriter> m_keyWriter; 
      std::auto_ptr<IRelationalWriter> m_dataWriter;
  };
  
  class STLContainerUpdater : public IRelationalUpdater {

    public:

    /// Constructor
    STLContainerUpdater(const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~STLContainerUpdater();

    bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

      /// Updates a data element
    void update( int oid,const void* data );

   private:

    RelationalDeleter m_deleter;
    STLContainerWriter m_writer;
  };

  class STLContainerReader : public IRelationalReader {

    public:
    
    /// Constructor
    STLContainerReader(const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~STLContainerReader();
    
    bool build( DataElement& offset, IRelationalData& relationalData );

    void select( int oid );
    
    void setRecordId( const std::vector<int>& identity );

    /// Reads a data element
    void read( void* address );

    void clear();

    private:
      Reflex::Type m_objectType;
      MappingElement& m_mappingElement;
      ContainerSchema& m_schema;
      std::vector<int> m_recordId;
      DataElement m_localElement;
      
    private:
      bool m_associative;
      DataElement* m_offset;
      std::auto_ptr<MultiRecordSelectOperation> m_query;
      std::auto_ptr<IArrayHandler> m_arrayHandler;
      std::auto_ptr<IRelationalReader> m_keyReader;
      std::auto_ptr<IRelationalReader> m_dataReader;
  };

  class STLContainerStreamer : public IRelationalStreamer 
  {
    public:
    STLContainerStreamer( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~STLContainerStreamer();

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
ora::MappingElement&
ora::STLContainerWriter::mapping(){
  return m_mappingElement;
}

inline
ora::DataElement*
ora::STLContainerWriter::dataElement(){
  return m_offset;
}

inline
ora::IArrayHandler*
ora::STLContainerWriter::arrayHandler(){
  return m_arrayHandler.get();
}

#endif

    
