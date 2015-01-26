#ifndef INCLUDE_ORA_ORAPTRSTREAMER_H
#define INCLUDE_ORA_ORAPTRSTREAMER_H

#include "IRelationalStreamer.h"
#include "DataElement.h"
//
#include <memory>
#include <boost/shared_ptr.hpp>
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"


namespace ora {

  class MappingElement;
  class ContainerSchema;
  class IPtrLoader;
  class OraPtrReadBuffer;
  
  class OraPtrWriter :  public IRelationalWriter{

    public:

      OraPtrWriter( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~OraPtrWriter();
      
      bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

      void setRecordId( const std::vector<int>& identity );
      
      /// Writes a data element
      void write( int oid, const void* data );

    private:

      edm::TypeWithDict m_objectType;
      MappingElement& m_mappingElement;
      ContainerSchema& m_schema;
      DataElement m_localElement;
      DataElement* m_dataElement;
      std::auto_ptr<IRelationalWriter> m_writer;
  };
  
  class OraPtrUpdater :  public IRelationalUpdater {

    public:

      OraPtrUpdater( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~OraPtrUpdater();
      
      bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);
      
      void setRecordId( const std::vector<int>& identity );
      
      /// Updates a data element
      void update( int oid,
                   const void* data );

    private:

      edm::TypeWithDict m_objectType;
      MappingElement& m_mappingElement;
      ContainerSchema& m_schema;
      DataElement m_localElement;
      DataElement* m_dataElement;
      std::auto_ptr<IRelationalUpdater> m_updater;
  };

  class OraPtrReader : public IRelationalReader {

      public:

      OraPtrReader( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~OraPtrReader();
      
      bool build( DataElement& offset, IRelationalData& relationalData);

      void select( int oid );
      
      void setRecordId( const std::vector<int>& identity );

      /// Reads a data element
      void read( void* destination );

      void clear();

    private:

      edm::TypeWithDict m_objectType;
      DataElement* m_dataElement;
      std::auto_ptr<OraPtrReadBuffer> m_readBuffer;
      std::vector<boost::shared_ptr<IPtrLoader> > m_loaders;
      std::vector<int> m_tmpIds;
  };

  class OraPtrStreamer : public IRelationalStreamer 
  {
    public:
    OraPtrStreamer( const edm::TypeWithDict& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~OraPtrStreamer();

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

    
      
