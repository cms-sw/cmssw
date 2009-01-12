#ifndef COND_PRIMITIVESCONTAINERSTREAMERS_H
#define COND_PRIMITIVESCONTAINERSTREAMERS_H

#include "Reflex/Type.h"
#include "CoralBase/Blob.h"
#include "POOLCore/IBlobStreamingService.h"
#include "ObjectRelationalAccess/ObjectRelationalClassUtils.h"
namespace cond {
  // prerequisite check for supported dictionary type
  
  struct PrimitiveContainerDictPrereq {
    bool operator () ( const Reflex::Type& classDictionary )
    {
      return pool::ObjectRelationalClassUtils::isTypeNonAssociativeContainer(classDictionary);
      // && classDictionary.TemplateArgumentAt(0).IsFundamental();
    }
  };

  class BlobWriter : virtual public pool::IBlobWriter
  {
  public:
    /// Constructor
    explicit BlobWriter( const Reflex::Type& type );

    /// Empty destructor
    virtual ~BlobWriter();

    /// Streams an object an returns by constant reference the underlying blob
    const coral::Blob& write( const void* addressOfInputData );

  private:
    /// The type
    Reflex::Type m_type;

    /// The blob data
    coral::Blob m_blob;
  };


  class BlobReader : virtual public pool::IBlobReader
  {
  public:
    /// Constructor
    BlobReader( const Reflex::Type& type );

    /// Empty destructor
    virtual ~BlobReader();

    /// Reads an object from a Blob and returns the starting address of the object
    void read( const coral::Blob& blobData,
               void* containerAddress ) const;
  private:
    /// The type
    Reflex::Type m_type;
  };

}

#endif // COND_PRIMITIVESCONTAINERSTREAMERS_H
