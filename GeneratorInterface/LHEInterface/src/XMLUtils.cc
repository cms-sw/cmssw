#include <iostream>
#include <memory>
#include <string>
#include <cstring>

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>

#include "FWCore/Utilities/interface/Exception.h"

#include "Utilities/StorageFactory/interface/IOTypes.h"
#include "Utilities/StorageFactory/interface/Storage.h"

#include "XMLUtils.h"

XERCES_CPP_NAMESPACE_USE
#define BUF_SIZE 8192 

namespace lhef {

StorageWrap::StorageWrap(Storage *storage) :
	storage(storage)
{
}

StorageWrap::~StorageWrap()
{
	storage->close();
}

unsigned int XMLDocument::XercesPlatform::instances = 0;

XMLDocument::XercesPlatform::XercesPlatform()
{
	if (!instances++) {
		try {
			XMLPlatformUtils::Initialize();
		} catch(const XMLException &e) {
			throw cms::Exception("XMLDocument")
				<< "XMLPlatformUtils::Initialize failed "
				   "because of: "
				<< XMLSimpleStr(e.getMessage()) << std::endl;
		}
	}
}

XMLDocument::XercesPlatform::~XercesPlatform()
{
	if (!--instances)
		XMLPlatformUtils::Terminate();
}

XMLDocument::XMLDocument(std::auto_ptr<std::istream> &in, Handler &handler) :
	platform(new XercesPlatform()),
	source(new STLInputSource(in)),
	parser(XMLReaderFactory::createXMLReader()),
	done(false)
{
	init(handler);
}

XMLDocument::XMLDocument(std::auto_ptr<StorageWrap> &in, Handler &handler) :
	platform(new XercesPlatform()),
	source(new StorageInputSource(in)),
	parser(XMLReaderFactory::createXMLReader()),
	done(false)
{
	init(handler);
}

void XMLDocument::init(Handler &handler)
{
	try {
		parser->setFeature(XMLUni::fgSAX2CoreValidation, false);
		parser->setFeature(XMLUni::fgSAX2CoreNameSpaces, false);
		parser->setFeature(XMLUni::fgXercesSchema, false);
		parser->setFeature(XMLUni::fgXercesSchemaFullChecking, false);

		parser->setContentHandler(&handler);
		parser->setLexicalHandler(&handler);
		parser->setErrorHandler(&handler);

		if (!parser->parseFirst(*source, token))
			throw cms::Exception("XMLParseError")
				<< "SAXParser::parseFirst failed" << std::endl;
	} catch(const XMLException &e) {
		throw cms::Exception("XMLDocument")
			<< "XMLPlatformUtils::Initialize failed because of "
			<< XMLSimpleStr(e.getMessage())	<< std::endl;
	} catch(const SAXException &e) {
		throw cms::Exception("XMLDocument")
			<< "XML parser reported: "
			<< XMLSimpleStr(e.getMessage()) << "." << std::endl;  
        }
}

XMLDocument::~XMLDocument()
{
}

bool XMLDocument::parse()
{
	try {
		if (done || parser->getErrorCount())
			return false;

		done = !parser->parseNext(token);
	} catch(const XMLException &e) {
		throw cms::Exception("XMLDocument")
			<< "XMLPlatformUtils::Initialize failed because of "
			<< XMLSimpleStr(e.getMessage())	<< std::endl;
	} catch(const SAXException &e) {
		throw cms::Exception("XMLDocument")
			<< "XML parser reported: "
			<< XMLSimpleStr(e.getMessage()) << "." << std::endl;  
        }

	return !done;
}

CBInputStream::Reader::~Reader()
{
}

CBInputStream::CBInputStream(Reader &reader) :
	reader(reader)
{
}

CBInputStream::~CBInputStream()
{
}

unsigned int CBInputStream::readBytes(XMLByte* const buf,
                                      const unsigned int size)
{
	char *rawBuf = reinterpret_cast<char*>(buf);
	unsigned int bytes = size * sizeof(XMLByte);
	unsigned int read = 0;

	while(read < bytes) {
		if (buffer.empty()) {
			buffer = reader.data();
			if (buffer.empty())
				break;
		}

		unsigned int len = buffer.length();
		unsigned int rem = bytes - read;
		if (rem < len) {
			std::memcpy(rawBuf + read, buffer.c_str(), rem);
			buffer.erase(0, rem);
			read += rem;
			break;
		}

		std::memcpy(rawBuf + read, buffer.c_str(), len);
		buffer.clear();
		read += len;
	}

	read /= sizeof(XMLByte);
	pos += read;

	return read;
}

STLInputStream::STLInputStream(std::istream &in) :
	in(in)
{
	if (in.bad())
		throw cms::Exception("FileStreamError")
			<< "I/O stream bad in STLInputStream::STLInputStream()"
			<< std::endl;
}

STLInputStream::~STLInputStream()
{
}

unsigned int STLInputStream::readBytes(XMLByte* const buf,
                                       const unsigned int size)
{
	char *rawBuf = reinterpret_cast<char*>(buf);
	unsigned int bytes = size * sizeof(XMLByte);
	in.read(rawBuf, bytes);
	unsigned int readBytes = in.gcount();

	if (in.bad())
		throw cms::Exception("FileStreamError")
			<< "I/O stream bad in STLInputStream::readBytes()"
			<< std::endl;

	unsigned int read = (unsigned int)(readBytes / sizeof(XMLByte));
	unsigned int rest = (unsigned int)(readBytes % sizeof(XMLByte));
	for(unsigned int i = 1; i <= rest; i++)
		in.putback(rawBuf[readBytes - i]);

	pos += read;
	return read;
}

StorageInputStream::StorageInputStream(StorageWrap &in) :
	in(in),
        lstr(LZMA_STREAM_INIT),
        compression_(false)
{
  // Check the kind of file.
  char header[6];
  unsigned int s = in->read(header, 6);
  in->position(0, Storage::SET);
  // Let's use lzma to start with.
  if (header[1] == '7'
      && header[2] == 'z'
      && header[3] == 'X'
      && header[4] == 'Z')
  {
    compression_ = true;
    lstr = LZMA_STREAM_INIT;
    // We store the beginning of the outBuffer to make sure
    // we can always update previous results.

#if LZMA_VERSION <= UINT32_C(49990030)
    int ret = lzma_auto_decoder(&lstr, NULL, NULL);
#else
    int ret = lzma_auto_decoder(&lstr, -1, 0);
#endif

    if (ret != LZMA_OK)
    {
      lzma_end(&lstr);
      throw cms::Exception("IO") << "Error while reading compressed LHE file";
    }
  }
}

StorageInputStream::~StorageInputStream()
{
  lzma_end(&(lstr));
}

unsigned int StorageInputStream::readBytes(XMLByte* const buf,
                                           const unsigned int size)
{
  // Simple read-in write-out in case
  if (!compression_)
  {
    void *rawBuf = reinterpret_cast<void*>(buf);
    unsigned int bytes = size * sizeof(XMLByte);
    unsigned int readBytes = in->read(rawBuf, bytes);

    unsigned int read = (unsigned int)(readBytes / sizeof(XMLByte));
    unsigned int rest = (unsigned int)(readBytes % sizeof(XMLByte));
    if (rest)
      in->position(-(IOOffset)rest, Storage::CURRENT);

    pos += read;
    return read;
  }
  // Compressed case. 
  // We simply read as many bytes as we can and we 
  // uncompress them in the output buffer.
  // We never decompress more bytes we were asked by
  // xerces. 
  // In case we read from file more bytes than needed 
  // we simply rollback by the (hopefully) correct amount.
  // If we don't read enough bytes, we simply return
  // the amount of bytes read and we wait for being called
  // again by xerces.
  unsigned int bytes = size * sizeof(XMLByte);
  uint8_t inBuf[BUF_SIZE];
  unsigned int rd = in->read((void*)inBuf, BUF_SIZE);
  lstr.next_in = inBuf;
  lstr.avail_in = rd;
  lstr.next_out = buf;
  lstr.avail_out = bytes;

  int ret = lzma_code(&lstr, LZMA_RUN);
  if(ret != LZMA_OK && ret != LZMA_STREAM_END) {  /* decompression error */
    lzma_end(&lstr);
    throw cms::Exception("IO") << "Error while reading compressed LHE file";
  }

  // If we did not consume everything we put it back.
  if (lstr.avail_in)
    in->position(-(IOOffset)lstr.avail_in - lstr.total_in, Storage::CURRENT);    
  pos += lstr.total_out;
  return lstr.total_out;
}

} // namespace lhef
