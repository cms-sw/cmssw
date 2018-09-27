#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstring>

#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <xercesc/sax2/XMLReaderFactory.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "Utilities/StorageFactory/interface/IOTypes.h"
#include "Utilities/StorageFactory/interface/Storage.h"

#include "XMLUtils.h"

XERCES_CPP_NAMESPACE_USE

namespace lhef {

StorageWrap::StorageWrap(std::unique_ptr<Storage> storage) :
	storage(std::move(storage))
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
			cms::concurrency::xercesInitialize();
		} catch(const XMLException &e) {
			throw cms::Exception("XMLDocument")
				<< "cms::concurrency::xercesInitialize failed "
				   "because of: "
				<< XMLSimpleStr(e.getMessage()) << std::endl;
		}
	}
}

XMLDocument::XercesPlatform::~XercesPlatform()
{
	if (!--instances)
		cms::concurrency::xercesTerminate();
}

XMLDocument::XMLDocument(std::unique_ptr<std::istream> &in, Handler &handler) :
	platform(new XercesPlatform()),
	source(new STLInputSource(in)),
	parser(XMLReaderFactory::createXMLReader()),
	done(false)
{
	init(handler);
}

XMLDocument::XMLDocument(std::unique_ptr<StorageWrap> &in, Handler &handler) :
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
			<< "cms::concurrency::xercesInitialize failed because of "
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
			<< "cms::concurrency::xercesInitialize failed because of "
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

XMLSize_t CBInputStream::readBytes(XMLByte* const buf,
				   const XMLSize_t size)
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

XMLSize_t STLInputStream::readBytes(XMLByte* const buf,
				    const XMLSize_t size)
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
        compression_(false),
        lasttotal_(0)
{
   buffer_.reserve(bufferSize_);
  // Check the kind of file.
  char header[6];
  /*unsigned int s = */ in->read(header, 6);
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


XMLSize_t StorageInputStream::readBytes(XMLByte* const buf, const XMLSize_t size)
{
        // Compression code is not able to handle sizeof(XMLByte) > 1.
    assert(sizeof(XMLByte) == sizeof(unsigned char));

    if (! (buffLoc_ < buffTotal_) )
    {
        int rd = in->read((void*)&buffer_[0], buffer_.capacity());
             // Storage layer is supposed to throw exceptions instead of returning errors; just-in-case
        if (rd < 0)
        {
            edm::Exception ex(edm::errors::FileReadError);
            ex << "Error while reading buffered LHE file";
            throw ex;
        }
        buffLoc_=0;
        buffTotal_=rd;
        if (buffTotal_ == 0)
        {
            return 0;
        }
    }
    unsigned int dataRead;
    if (!compression_)
    {
        dataRead = std::min(buffTotal_-buffLoc_, static_cast<unsigned int>(size));
        memcpy(buf, &buffer_[buffLoc_], dataRead);
        buffLoc_ += dataRead;
    }
    else
    {
        dataRead = buffTotal_-buffLoc_;
        lstr.next_in = &buffer_[buffLoc_];
        lstr.avail_in = dataRead;
        lstr.next_out = buf;
        lstr.avail_out = size;
        int ret = lzma_code(&lstr, LZMA_RUN);
        if(ret != LZMA_OK && ret != LZMA_STREAM_END)
        {  /* decompression error */
            lzma_end(&lstr);
            throw cms::Exception("IO") << "Error while reading compressed LHE file (error code " << ret << ")";
        }
        dataRead -= lstr.avail_in;
        buffLoc_ += dataRead;
            // Decoder was unable to make progress; reset stream and try again.
            // If this becomes problematic, we can make the buffer circular.
        if (!dataRead)
        {
                // NOTE: lstr.avail_in == buffTotal-buffLoc_
            in->position(-(IOOffset)(lstr.avail_in), Storage::CURRENT);
            buffLoc_ = 0;
            buffTotal_ = 0;
            return readBytes(buf, size);
        }
        dataRead = (size - lstr.avail_out);
    }
    return dataRead;
}

} // namespace lhef
