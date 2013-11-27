#ifndef INCLUDE_ORA_CARRAYSTREAMER_H
#define INCLUDE_ORA_CARRAYSTREAMER_H

#include "DataElement.h"
#include "IRelationalStreamer.h"
#include "RelationalDeleter.h"
//
#include <memory>
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace ora {

  class MappingElement;
  class ContainerSchema;

  class IArrayHandler;
  class MultiRecordInsertOperation;
  class MultiRecordSelectOperation;

  class CArrayWriter : public IRelationalWriter {
    
    public:
      /// Constructor
      CArrayWriter( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );
      virtual ~CArrayWriter();

      bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer );
      
      void setRecordId( const std::vector<int>& identity );

      /// Writes a data element
      void write( int oid,const void* data );

    private:
      edm::TypeWithDict m_objectType;
      MappingElement& m_mappingElement;
      ContainerSchema& m_schema;
      std::vector<int> m_recordId;
      DataElement m_localElement;

    private:
      DataElement* m_offset;
      MultiRecordInsertOperation* m_insertOperation;
      std::auto_ptr<IArrayHandler> m_arrayHandler;
      std::auto_ptr<IRelationalWriter> m_dataWriter;
      
  };
  
  class CArrayUpdater : public IRelationalUpdater {

    public:

    /// Constructor
    CArrayUpdater(const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~CArrayUpdater();

    bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

    void setRecordId( const std::vector<int>& identity );

      /// Updates a data element
    void update( int oid,const void* data );

    private:

    RelationalDeleter m_deleter;
    CArrayWriter m_writer;
  };

  class CArrayReader : public IRelationalReader {

    public:
    
    /// Constructor
    CArrayReader(const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~CArrayReader();
    
    bool build( DataElement& offset, IRelationalData& relationalData );

    void select( int oid );
    
    void setRecordId( const std::vector<int>& identity );

    /// Reads a data element
    void read( void* address );

    void clear();

    private:
      edm::TypeWithDict m_objectType;
      MappingElement& m_mappingElement;
      ContainerSchema& m_schema;
      std::vector<int> m_recordId;
      DataElement m_localElement;
      
    private:
      DataElement* m_offset;
      std::auto_ptr<MultiRecordSelectOperation> m_query;
      std::auto_ptr<IArrayHandler> m_arrayHandler;
      std::auto_ptr<IRelationalReader> m_dataReader;
  };

  class CArrayStreamer : public IRelationalStreamer 
  {
    public:
    CArrayStreamer( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~CArrayStreamer();

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

    
