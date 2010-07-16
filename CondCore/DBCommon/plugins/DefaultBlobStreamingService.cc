#include "CondCore/DBCommon/interface/BlobStreamerPluginFactory.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "DefaultBlobStreamingService.h"
//
#include <cstring>
//
#include "Reflex/Member.h"
#include "Reflex/Object.h"
#include "RVersion.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(5,19,0)
using namespace ROOT;
#endif

namespace cond {

  bool isTypeNonAssociativeContainer(const Reflex::Type& typ){
    Reflex::TypeTemplate tt = typ.TemplateFamily();
    if (! tt) {
      return false;
    } else {
      std::string contName = tt.Name(Reflex::SCOPED|Reflex::FINAL);
      if(  contName == "std::vector"              ||
           contName == "std::list"                ||
           contName == "std::deque"               ||
           contName == "std::stack"               ||
           contName == "std::set"                 ||
           contName == "std::multiset"            ||
           contName == "__gnu_cxx::hash_set"      ||
           contName == "__gnu_cxx::hash_multiset" ||
           contName == "ora::PVector"            ||
           contName == "ora::QueryableVector"){
        return true;
      }
    }
    return false;
  }
  
}

DEFINE_EDM_PLUGIN(cond::BlobStreamerPluginFactory,cond::DefaultBlobStreamingService,"COND/Services/DefaultBlobStreamingService");

cond::DefaultBlobStreamingService::DefaultBlobStreamingService(){
}

cond::DefaultBlobStreamingService::~DefaultBlobStreamingService(){
}

boost::shared_ptr<coral::Blob>
cond::DefaultBlobStreamingService::write( const void* addressOfInputData,
                                          const TypeH& classDictionary ){
  boost::shared_ptr<coral::Blob> theBlob( new coral::Blob );
  if( !isTypeNonAssociativeContainer(classDictionary) ){
    throw cond::Exception( "DefaultBlobStreamingService::write Associative container are not supported." );    
  }  
  Reflex::Object theContainer( classDictionary, const_cast<void*>( addressOfInputData ) );
  //get the container size
  Reflex::Member sizeMethod = classDictionary.MemberByName( "size" );
  if ( ! sizeMethod )
    throw cond::Exception( "DefaultBlobStreamingService::write No size method is defined for the container" );
  size_t containerSize=0;
#if ROOT_VERSION_CODE < ROOT_VERSION(5,19,0)
  ROOT::Reflex::Object ret = sizeMethod.Invoke(theContainer);
  if( !ret.TypeOf() || !ret.Address() )
    throw cond::Exception( "DefaultBlobStreamingService::write Could not invoke the size method on the container" );
  containerSize = *(static_cast<size_t*>(ret.Address()));
#else
  Reflex::Object* ret=0;
  sizeMethod.Invoke(theContainer,ret);
  if( !ret->TypeOf() || !ret->Address() )
    throw cond::Exception( "DefaultBlobStreamingService::write Could not invoke the size method on the container" );
  containerSize = *(static_cast<size_t*>(ret->Address()));
#endif
 
  if(containerSize==0){
    return theBlob;
  }
  size_t elementSize=classDictionary.TemplateArgumentAt(0).SizeOf();
  Reflex::Member beginMethod = classDictionary.MemberByName( "begin" );
  if ( ! beginMethod )
    throw cond::Exception( "DefaultBlobStreamingService::write No begin method is defined for the container" );
  TypeH iteratorType = beginMethod.TypeOf().ReturnType();
  Reflex::Member dereferenceMethod = iteratorType.MemberByName( "operator*" );
  if ( ! dereferenceMethod )
    throw cond::Exception( "DefaultBlobStreamingService::write Could not retrieve the dereference method of the container's iterator" );
  // Create an iterator
  void* elementAddress=0;
  void* startingAddress=0;
#if ROOT_VERSION_CODE < ROOT_VERSION(5,19,0)
  ROOT::Reflex::Object iteratorObject = beginMethod.Invoke( ROOT::Reflex::Object( classDictionary, const_cast< void * > ( addressOfInputData ) ) );
  elementAddress = dereferenceMethod.Invoke( iteratorObject ).Address();
  theBlob->resize( containerSize * elementSize );
  startingAddress = theBlob->startingAddress();
  ::memcpy( startingAddress, elementAddress, containerSize*elementSize );
  iteratorObject.Destruct();
#else
  Reflex::Object* containercontents=0;
  Reflex::Object* iteratorObject=0;
  beginMethod.Invoke( ROOT::Reflex::Object( classDictionary, const_cast< void * > ( addressOfInputData ) ),iteratorObject );
  if( !iteratorObject ) throw cond::Exception( "BlobWriter::write Could not retrieve the iterator object" );
  dereferenceMethod.Invoke( *iteratorObject, containercontents);
  if ( ! containercontents )
    throw cond::Exception( "DefaultBlobStreamingService::write Could not retrieve the content of the container" );
  elementAddress=containercontents->Address();
  theBlob->resize( containerSize * elementSize );
  startingAddress = theBlob->startingAddress();
  ::memcpy( startingAddress, elementAddress, containerSize*elementSize );
  iteratorObject->Destruct();
#endif
  return theBlob;
}

void cond::DefaultBlobStreamingService::read( const coral::Blob& blobData,
                                             void* containerAddress,
                                             const TypeH& classDictionary ){
  if( !isTypeNonAssociativeContainer(classDictionary) ){
    throw cond::Exception( "DefaultBlobStreamingService::read Associative container are not supported." );    
  }  
  const void * srcstartingAddress=blobData.startingAddress();
  long bsize=blobData.size();
  if(bsize==0){
    return;
  }
  Reflex::Member clearMethod = classDictionary.MemberByName( "clear" );
  if ( ! clearMethod )
    throw cond::Exception( "DefaultBlobStreamingService::read Could not retrieve the clear method of the container" );
  Reflex::Object containerObject( classDictionary, containerAddress );
  // Clear the container
  clearMethod.Invoke( containerObject );
  Reflex::Member resizeMethod = classDictionary.MemberByName( "resize",TypeH::ByName("void (size_t)") );
  if ( ! resizeMethod )
    throw cond::Exception( "DefaultBlobStreamingService::read Could not retrieve the resize method of the container" );
  // resize the container
  size_t elementSize=classDictionary.TemplateArgumentAt(0).SizeOf();
  std::vector<void *> v(1);
  size_t containerSize = bsize / elementSize;
  v[0] = (void*)(&containerSize);
#if ROOT_VERSION_CODE < ROOT_VERSION(5,19,0)
  resizeMethod.Invoke( containerObject,v );
  ROOT::Reflex::Member beginMethod = classDictionary.MemberByName( "begin" );
  if ( ! beginMethod )
    throw cond::Exception( "DefaultBlobStreamingService::read No begin method is defined for the container" );
  ROOT::Reflex::Object iteratorObject = beginMethod.Invoke( ROOT::Reflex::Object( classDictionary, const_cast< void * > ( containerAddress ) ) );
  ROOT::Reflex::Type iteratorType = beginMethod.TypeOf().ReturnType();
  //get first element address
  ROOT::Reflex::Member dereferenceMethod = iteratorType.MemberByName( "operator*" );
  if ( ! dereferenceMethod )
    throw cond::Exception( "BlobReader::read Could not retrieve the dereference method of the container's iterator" );
  void* elementAddress = dereferenceMethod.Invoke( iteratorObject ).Address();
  ::memcpy( elementAddress, srcstartingAddress, (size_t)bsize);
  iteratorObject.Destruct();
#else
  Reflex::Object* m=0;
  resizeMethod.Invoke( containerObject,m,v );
  // Create an iterator
  Reflex::Member beginMethod = classDictionary.MemberByName( "begin" );
  if ( ! beginMethod )
    throw cond::Exception( "DefaultBlobStreamingService::read No begin method is defined for the container" );
  Reflex::Object* iteratorObject=0; 
  beginMethod.Invoke( Reflex::Object( classDictionary, const_cast< void * > ( containerAddress ) ),iteratorObject );
  if ( ! iteratorObject )
    throw cond::Exception( "DefaultBlobStreamingService::read Could not retrieve the iterator of the container" );
  Reflex::Type iteratorType = beginMethod.TypeOf().ReturnType();
  //get first element address
  Reflex::Member dereferenceMethod = iteratorType.MemberByName( "operator*" );
  if ( ! dereferenceMethod )
    throw cond::Exception( "DefaultBlobStreamingService::read Could not retrieve the dereference method of the container's iterator" );
  void* elementAddress = 0;
  Reflex::Object* ret=0;
  dereferenceMethod.Invoke( *iteratorObject, ret );
  if ( ! ret )
    throw cond::Exception( "DefaultBlobStreamingService::read Could not retrieve the content of the container" );
  ret->Address();
  ::memcpy( elementAddress, srcstartingAddress, (size_t)bsize);
  iteratorObject->Destruct();
#endif
}

