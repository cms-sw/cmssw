#ifndef INCLUDE_ORA_IBLOBSTREAMINGSERVICE_H
#define INCLUDE_ORA_IBLOBSTREAMINGSERVICE_H

#include <boost/shared_ptr.hpp>

namespace coral {
  class Blob;
}

namespace edm {
    class TypeWithDict;
}

namespace ora {
  
  /// Interface for a Streaming Service. 
  class IBlobStreamingService
  {
  public:
    /// Empty destructor
    virtual ~IBlobStreamingService() {}

    virtual boost::shared_ptr<coral::Blob> write( const void* addressOfInputData, const edm::TypeWithDict & classDictionary, bool useCompression = true ) = 0;

    /// Reads an object from a Blob and fills-in the container
    virtual void read( const coral::Blob& blobData, void* addressOfContainer, const edm::TypeWithDict& classDictionary ) = 0;
  };

}

#endif
