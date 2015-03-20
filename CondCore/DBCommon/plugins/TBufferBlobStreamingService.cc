#include "CondCore/DBCommon/interface/BlobStreamerPluginFactory.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "TBufferBlobStreamingService.h"
//
#include <algorithm>
#include <typeinfo>
#include <string>
#include <cstring>
//
#include "TBufferFile.h"

#include "FWCore/Utilities/interface/TypeWithDict.h"

typedef void (TBuffer::*WriteArrayFn_t)(const void *obj, Int_t n);
typedef void (TBuffer::*ReadArrayFn_t)(void *obj, Int_t n);

#define PRIMITIVE(x) { \
	typeid(x), \
	reinterpret_cast<WriteArrayFn_t>( \
		(void (TBuffer::*)(const x*, Int_t))&TBuffer::WriteFastArray), \
	reinterpret_cast<ReadArrayFn_t>( \
		(void (TBuffer::*)(x*, Int_t))&TBuffer::ReadFastArray) \
}

struct Primitive {
	const std::type_info	&type;
	WriteArrayFn_t		writeArrayFn;
	ReadArrayFn_t		readArrayFn;

	inline bool operator == (const std::type_info &other) const
	{ return type == other; }
} static const primitives[] = {
	PRIMITIVE(Bool_t),
	PRIMITIVE(Char_t),
	PRIMITIVE(UChar_t),
	PRIMITIVE(Short_t),
	PRIMITIVE(UShort_t),
	PRIMITIVE(Int_t),
	PRIMITIVE(UInt_t),
	PRIMITIVE(Long_t),
	PRIMITIVE(ULong_t),
	PRIMITIVE(Long64_t),
	PRIMITIVE(ULong64_t),
	PRIMITIVE(Float_t),
	PRIMITIVE(Double_t)
};

static const std::size_t nPrimitives =
				(sizeof primitives / sizeof primitives[0]);

#undef PRIMTIVE

cond::TBufferBlobTypeInfo::TBufferBlobTypeInfo( edm::TypeWithDict const & type_)
 : m_arraySize(0), m_class(0), m_primitive(0)
{
  edm::TypeWithDict type = type_;
  while(true) {
    type = type.finalType();

    if (!type.isArray())
      break;

    if (!m_arraySize)
      m_arraySize = 1;
    m_arraySize *= type.arrayLength();
    type = type.toType();
  }

  if (type.isClass()) {
    const std::type_info &typeInfo = type.typeInfo();
    m_class = TClass::GetClass(typeInfo);
    if (!m_class)
      throw cond::Exception("TBufferBlobTypeInfo::TBufferBlobTypeInfo "
                            "No ROOT class registered for " + type.name());
  } else if (type.isFundamental()) {
    if (!m_arraySize)
        throw cond::Exception("TBufferBlobTypeInfo::TBufferBlobTypeInfo "
                              "Only arrays of primitive types supported. "
	                      "Please to not use a Blob for this member.");

    m_primitive = std::find(primitives, primitives + nPrimitives,
                            type.typeInfo()) - primitives;
    if (m_primitive >= nPrimitives)
      throw cond::Exception("TBufferBlobTypeInfo::TBufferBlobTypeInfo "
                            "Cannot handle primitive type " + type.name());
  } else
    throw cond::Exception("TBufferBlobTypeInfo::TBufferBlobTypeInfo "
                          "Cannot handle C++ type " + type.name());
}


cond::TBufferBlobStreamingService::TBufferBlobStreamingService(){
}

cond::TBufferBlobStreamingService::~TBufferBlobStreamingService(){
}

#include <boost/bind.hpp>
namespace {
  inline char * reallocInBlob( boost::shared_ptr<coral::Blob> theBlob, char* p, size_t newsize, size_t oldsize) {
    // various checks missing....
    theBlob->resize(newsize);
    return (char*)theBlob->startingAddress();
    
  }
}

boost::shared_ptr<coral::Blob> cond::TBufferBlobStreamingService::write( const void* addr,
									 edm::TypeWithDict const & classDictionary,
                                                                         bool ){
  TBufferBlobTypeInfo theType( classDictionary );
  if (theType.m_class && theType.m_class->GetActualClass(addr) != theType.m_class)
    throw cond::Exception("TBufferBlobWriter::write object to stream is "
                          "not of actual class.");
  
  boost::shared_ptr<coral::Blob> theBlob( new coral::Blob );
  //theBlob->resize(1024);
  
  // with new root...
  // TBufferFile buffer(TBufferFile::kWrite, theBlob->size(), theBlob->startingAddress(), kFALSE, boost::bind(reallocInBlob, theBlob,_1,_2,_3));
  TBufferFile buffer(TBufferFile::kWrite);
  buffer.InitMap();
  
  if (theType.m_arraySize && !theType.m_class)
    (buffer.*primitives[theType.m_primitive].writeArrayFn)(addr, theType.m_arraySize);
  else if (theType.m_arraySize)
    buffer.WriteFastArray(const_cast<void*>(addr), theType.m_class, theType.m_arraySize);
  else
    buffer.StreamObject(const_cast<void*>(addr), theType.m_class);

  Int_t size = buffer.Length();

  theBlob->resize(size);
  void *startingAddress = theBlob->startingAddress();
  std::memcpy(startingAddress, buffer.Buffer(), size);

  return theBlob;
}

void cond::TBufferBlobStreamingService::read( const coral::Blob& blobData,
                                              void* addr,
                                               edm::TypeWithDict const & classDictionary ){
  TBufferBlobTypeInfo theType( classDictionary );
  const void *startingAddress = blobData.startingAddress();
  size_t size = blobData.size();
  if (!size)
    return;

  TBufferFile buffer(TBufferFile::kRead, size,
                 const_cast<void*>(startingAddress), kFALSE);

  buffer.InitMap();

  if (theType.m_arraySize && !theType.m_class)
    (buffer.*primitives[theType.m_primitive].readArrayFn)(addr, theType.m_arraySize);
  else if (theType.m_arraySize)
    buffer.ReadFastArray(addr, theType.m_class, theType.m_arraySize);
  else
    buffer.StreamObject(addr, theType.m_class);
}

