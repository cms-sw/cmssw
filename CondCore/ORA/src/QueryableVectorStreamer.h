#ifndef INCLUDE_ORA_QUERYABLEVECTORSTREAMER_H
#define INCLUDE_ORA_QUERYABLEVECTORSTREAMER_H

#include "PVectorStreamer.h"
#include "DataElement.h"
//
#include <memory>
#include <boost/shared_ptr.hpp>
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

namespace ora {

  class MappingElement;
  class ContainerSchema;
  class IRelationalData;
  class IArrayHandler;
  class MultiRecordInsertOperation;
  class IVectorLoader;
  
  class QueryableVectorWriter: public IRelationalWriter {
    
    public:
      /// Constructor
      QueryableVectorWriter( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );
      virtual ~QueryableVectorWriter();

      bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer );
      void setRecordId( const std::vector<int>& identity );
      void write( int oid,const void* data );
    private:
      edm::TypeWithDict m_objectType;
      DataElement* m_offset;
      DataElement m_localElement;
    private:  
      PVectorWriter m_writer;
  };
  
  class QueryableVectorUpdater : public IRelationalUpdater {

    public:

    /// Constructor
    QueryableVectorUpdater(const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );
    virtual ~QueryableVectorUpdater();

    bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer);
    void setRecordId( const std::vector<int>& identity );
    void update( int oid,const void* data );

    private:
      edm::TypeWithDict m_objectType;
      DataElement* m_offset;
      DataElement m_localElement;
    private:
    PVectorUpdater m_updater;
  };

  class QueryableVectorReader : public IRelationalReader {

    public:
    
    /// Constructor
    QueryableVectorReader(const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    virtual ~QueryableVectorReader();
    
    bool build( DataElement& offset, IRelationalData& relationalData );
    void select( int oid );
    void setRecordId( const std::vector<int>& identity );
    void read( void* address );
    void clear();

    private:
    edm::TypeWithDict m_objectType;
    MappingElement& m_mapping;
    ContainerSchema& m_schema;
    DataElement* m_dataElement;
    std::vector<boost::shared_ptr<IVectorLoader> > m_loaders;
    std::vector<int> m_tmpIds;
  };

  class QueryableVectorStreamer : public IRelationalStreamer 
  {
    public:
    QueryableVectorStreamer( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~QueryableVectorStreamer();

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

    
