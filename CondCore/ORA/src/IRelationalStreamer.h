#ifndef INCLUDE_ORA_IRELATIONALSTREAMER_H
#define INCLUDE_ORA_IRELATIONALSTREAMER_H

#define MAXARRAYSIZE 65000

//
#include <vector>

namespace ora {

  class DataElement;
  class IRelationalData;
  class RelationalBuffer;
  
  class IRelationalWriter {
    
      public:

      /// Destructor
      virtual ~IRelationalWriter(){
      }

      virtual bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer ) = 0;

      virtual void setRecordId( const std::vector<int>& identity ) = 0;
      
      /// Writes a data element
      virtual void write( int oid, const void* data ) = 0;
      
  };
  
  
  class IRelationalUpdater {

    public:

    virtual ~IRelationalUpdater(){
    }

    virtual bool build( DataElement& offset, IRelationalData& relationalData, RelationalBuffer& operationBuffer) = 0;

    virtual void setRecordId( const std::vector<int>& identity ) = 0;

      /// Updates a data element
    virtual void update( int oid, const void* data ) = 0;
  };
  

  class IRelationalReader {

    public:
    
    virtual ~IRelationalReader(){
    }
    
    virtual bool build( DataElement& offset, IRelationalData& relationalData ) = 0;

    virtual void select( int oid ) = 0;

    virtual void setRecordId( const std::vector<int>& identity ) = 0;

    /// Reads a data element
    virtual void read( void* address ) = 0;

    virtual void clear() = 0;

  };

  class IRelationalStreamer {
    public:
      /// Destructor
      virtual ~IRelationalStreamer(){
      }

      virtual IRelationalWriter* newWriter() = 0;
      
      virtual IRelationalUpdater* newUpdater() = 0;

      virtual IRelationalReader* newReader() = 0;
  };
}

#endif

    
