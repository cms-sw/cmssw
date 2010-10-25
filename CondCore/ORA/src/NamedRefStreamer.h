#ifndef INCLUDE_ORA_NAMEDREFSTREAMER_H
#define INCLUDE_ORA_NAMEDREFSTREAMER_H

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

  std::string namedRefNullLabel();
  
  class NamedReferenceStreamerBase {

    public:

    explicit NamedReferenceStreamerBase( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& schema );

    virtual ~NamedReferenceStreamerBase();

      bool buildDataElement( DataElement& dataElement, IRelationalData& relationalData );
      
      void bindDataForUpdate( const void* data );

      void bindDataForRead( void* data );

    private:

      Reflex::Type m_objectType;
      const std::vector<std::string>& m_columns;
      ContainerSchema& m_schema;
      DataElement* m_dataElement;
      DataElement* m_refNameDataElement;
      DataElement* m_ptrDataElement;
      DataElement* m_flagDataElement;
      IRelationalData* m_relationalData;
  };

  class NamedRefWriter :  public NamedReferenceStreamerBase, public IRelationalWriter{

    public:

      NamedRefWriter( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~NamedRefWriter();
      
      bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);

      void setRecordId( const std::vector<int>& identity );
      
      /// Writes a data element
      void write( int oid, const void* data );

  };
  
  class NamedRefUpdater :  public NamedReferenceStreamerBase, public IRelationalUpdater {

    public:

      NamedRefUpdater( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~NamedRefUpdater();
      
      bool build(DataElement& dataElement, IRelationalData& relationalData, RelationalBuffer& operationBuffer);
      
      void setRecordId( const std::vector<int>& identity );
      
      /// Updates a data element
      void update( int oid,
                   const void* data );

  };

  class NamedRefReader : public NamedReferenceStreamerBase, public IRelationalReader {

      public:

      NamedRefReader( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

      virtual ~NamedRefReader();
      
      bool build( DataElement& offset, IRelationalData& relationalData);

      void select( int oid );
      
      void setRecordId( const std::vector<int>& identity );

      /// Reads a data element
      void read( void* destination );

      void clear();

  };

  class NamedRefStreamer : public IRelationalStreamer 
  {
    public:
    NamedRefStreamer( const Reflex::Type& objectType, MappingElement& mapping, ContainerSchema& contSchema );

    ~NamedRefStreamer();

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

    
      
