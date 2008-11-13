#ifndef CondCore_DBCommon_BlobStreamingService_h
#define CondCore_DBCommon_BlobStreamingService_h

#include "POOLCore/IBlobStreamingService.h"
#include "ObjectRelationalAccess/ObjectRelationalClassUtils.h"
#include "CoralKernel/Service.h"

namespace cond {

  /// default dictionary prerequisite template argument, always true

  struct NoDictPrereq {
    inline bool operator () ( const ROOT::Reflex::Type& classDictionary )
    { return true; }
  };

  /// implementation builder template for the IBlobStreamingService interface

  template<class Writer_t, class Reader_t, class DictPrereq_t = NoDictPrereq>
  class BlobStreamingService : public coral::Service,
    virtual public pool::IBlobStreamingService
  {
  protected:
    typedef BlobStreamingService<Writer_t, Reader_t, DictPrereq_t> parent;

  public:

  /// Standard Constructor with a component key
  explicit BlobStreamingService( const std::string& key ):coral::Service( key ) {}
  
  /// Standard Destructor
  virtual ~BlobStreamingService() {}
  
  /// Returns a new NON-PORTABLE streamer for writing into a BLOB
  pool::IBlobWriter* newWriter( const ROOT::Reflex::Type& classDictionary,
				const std::string& version ) const
  {
    if ( DictPrereq_t()( classDictionary ) )
    return new Writer_t( classDictionary );
    return 0;
  }

  /// Returns a new NON-PORTABLE streamer for reading from a BLOB
  pool::IBlobReader* newReader( const ROOT::Reflex::Type& classDictionary,
				const std::string& version ) const
  {
    if ( DictPrereq_t()( classDictionary ) )
    return new Reader_t( classDictionary );
    return 0;
  }
  };
}
#endif // COND_BLOBSTREAMINGSERVICE_H
