#include <algorithm>
#include <typeinfo>
#include <string>

#include "RVersion.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,15,0)
#include "TBufferFile.h"
typedef TBufferFile CONDRootBuffer;
#else
#include "TBuffer.h"
typedef TBuffer CONDRootBuffer;
#endif

#include "Reflex/Reflex.h"
#include "Cintex/Cintex.h"

#include "CondCore/DBCommon/interface/Exception.h"

#include "TBufferBlobStreamer.h"

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

cond::TBufferBlobTypeInfo::TBufferBlobTypeInfo(const TypeH& type_)
 : m_arraySize(0), m_class(0), m_primitive(0)
{
  static bool cintexInitialized = false;
  if (!cintexInitialized) {
    cintexInitialized = true;
    ROOT::Cintex::Cintex::Enable();
  }

  TypeH type = type_;
  while(true) {
    type = type.FinalType();

    if (!type.IsArray())
      break;

    if (!m_arraySize)
      m_arraySize = 1;
    m_arraySize *= type.ArrayLength();
    type = type.ToType();
  }

  if (type.IsClass()) {
    const std::type_info &typeInfo = type.TypeInfo();
    m_class = TClass::GetClass(typeInfo);
    if (!m_class)
      throw cond::Exception("TBufferBlobTypeInfo::TBufferBlobTypeInfo "
                            "No ROOT class registered for " + type.Name());
  } else if (type.IsFundamental()) {
    if (!m_arraySize)
        throw cond::Exception("TBufferBlobTypeInfo::TBufferBlobTypeInfo "
                              "Only arrays of primitive types supported. "
	                      "Please to not use a Blob for this member.");

    m_primitive = std::find(primitives, primitives + nPrimitives,
                            type.TypeInfo()) - primitives;
    if (m_primitive >= nPrimitives)
      throw cond::Exception("TBufferBlobTypeInfo::TBufferBlobTypeInfo "
                            "Cannot handle primitive type " + type.Name());
  } else
    throw cond::Exception("TBufferBlobTypeInfo::TBufferBlobTypeInfo "
                          "Cannot handle C++ type " + type.Name());
}

cond::TBufferBlobWriter::TBufferBlobWriter(const TypeH &type):
  m_type(type),
  m_blob()
{
}

cond::TBufferBlobWriter::~TBufferBlobWriter()
{
}

const coral::Blob &cond::TBufferBlobWriter::write(const void *addr)
{
  //std::cout<<"TBufferBlobWriter::write"<<std::endl;
  if (m_type.m_class && m_type.m_class->GetActualClass(addr) != m_type.m_class)
    throw cond::Exception("TBufferBlobWriter::write object to stream is "
                          "not of actual class.");

  CONDRootBuffer buffer(CONDRootBuffer::kWrite);
  buffer.InitMap();

  if (m_type.m_arraySize && !m_type.m_class)
    (buffer.*primitives[m_type.m_primitive].writeArrayFn)(addr, m_type.m_arraySize);
  else if (m_type.m_arraySize)
    buffer.WriteFastArray(const_cast<void*>(addr), m_type.m_class, m_type.m_arraySize);
  else
    buffer.StreamObject(const_cast<void*>(addr), m_type.m_class);

  Int_t size = buffer.Length();

  m_blob.resize(size);
  void *startingAddress = m_blob.startingAddress();

  std::memcpy(startingAddress, buffer.Buffer(), size);

  return m_blob;
}

cond::TBufferBlobReader::TBufferBlobReader(const TypeH& type):
  m_type(type)
{
}

cond::TBufferBlobReader::~TBufferBlobReader()
{
}

void cond::TBufferBlobReader::read(const coral::Blob &blobData,
                                   void *addr) const
{
  const void *startingAddress = blobData.startingAddress();
  size_t size = blobData.size();
  if (!size)
    return;

  CONDRootBuffer buffer(CONDRootBuffer::kRead, size,
                 const_cast<void*>(startingAddress), kFALSE);

  buffer.InitMap();

  if (m_type.m_arraySize && !m_type.m_class)
    (buffer.*primitives[m_type.m_primitive].readArrayFn)(addr, m_type.m_arraySize);
  else if (m_type.m_arraySize)
    buffer.ReadFastArray(addr, m_type.m_class, m_type.m_arraySize);
  else
    buffer.StreamObject(addr, m_type.m_class);
}
