#ifndef INCLUDE_ORA_IBLOBSTREAMINGSERVICE_H
#define INCLUDE_ORA_IBLOBSTREAMINGSERVICE_H

#include <boost/shared_ptr.hpp>

namespace coral {
  class Blob;
}

namespace Reflex {
  class Type;
}

namespace ora {
  
  /// Interface for a Streaming Service. 
  class IBlobStreamingService
  {
  public:
    /// Empty destructor
    virtual ~IBlobStreamingService() {}

    /// Streams an object an returns by constant reference the underlying blob
    virtual boost::shared_ptr<coral::Blob> write( const void* addressOfInputData, const Reflex::Type& classDictionary ) = 0;

    /// Reads an object from a Blob and fills-in the container
    virtual void read( const coral::Blob& blobData, void* addressOfContainer, const Reflex::Type& classDictionary ) = 0;
  };

}

#endif
