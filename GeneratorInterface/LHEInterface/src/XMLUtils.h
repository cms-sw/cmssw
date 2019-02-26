#ifndef GeneratorInterface_LHEInterface_XMLUtils_h
#define GeneratorInterface_LHEInterface_XMLUtils_h

#include <iostream>
#include <string>
#include <memory>
#include <vector>

#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/XMLChar.hpp>
#include <xercesc/util/BinInputStream.hpp>
#include <xercesc/framework/XMLPScanToken.hpp>
#include <xercesc/sax/InputSource.hpp>
#include <xercesc/sax2/DefaultHandler.hpp>
#include <xercesc/sax2/SAX2XMLReader.hpp>
#include <lzma.h>


class Storage;

namespace lhef {

class StorageWrap {
    public:
	StorageWrap(std::unique_ptr<Storage> storage);
	~StorageWrap();

	Storage *operator -> () { return storage.get(); }
	const Storage *operator -> () const { return storage.get(); }

    private:
	std::unique_ptr<Storage>	storage;
};

class XMLDocument {
    public:
	class Handler : public XERCES_CPP_NAMESPACE_QUALIFIER DefaultHandler {};

	XMLDocument(std::unique_ptr<std::istream> &in, Handler &handler);
	XMLDocument(std::unique_ptr<StorageWrap> &in, Handler &handler);
	virtual ~XMLDocument();

	bool parse();

        static std::shared_ptr<void> platformHandle() { return std::make_shared<XercesPlatform>(); }
    private:
	class XercesPlatform {
	    public:
		XercesPlatform();
		~XercesPlatform();

	    private:
		// do not make any kind of copies
		XercesPlatform(const XercesPlatform &orig) = delete;
		XercesPlatform &operator = (const XercesPlatform &orig) = delete;

		static unsigned int instances;
	};

	void init(Handler &handler);

	std::unique_ptr<XercesPlatform>					platform;

	std::unique_ptr<XERCES_CPP_NAMESPACE_QUALIFIER InputSource>	source;
	std::unique_ptr<XERCES_CPP_NAMESPACE_QUALIFIER SAX2XMLReader>	parser;

	XERCES_CPP_NAMESPACE_QUALIFIER XMLPScanToken			token;

	bool								done;
};

class XMLSimpleStr {
    public:
	XMLSimpleStr(const XMLCh *str) :
		string(XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(str))
	{}

	~XMLSimpleStr()
	{ XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&string); }

	inline operator const char *() const
	{ return string; }

	inline static bool isAllSpaces(const XMLCh *str, unsigned int length)
	{ return XERCES_CPP_NAMESPACE_QUALIFIER
				XMLChar1_0::isAllSpaces(str, length); }

	inline static bool isSpace(XMLCh ch)
	{ return XERCES_CPP_NAMESPACE_QUALIFIER
				XMLChar1_0::isWhitespace(ch); }

    private:
	char	*string;
};

class XMLUniStr {
    public:
	XMLUniStr(const char *str) :
		unicode(XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(str))
	{}

	~XMLUniStr()
	{ XERCES_CPP_NAMESPACE_QUALIFIER XMLString::release(&unicode); }

	operator const XMLCh *() const
	{ return unicode; }

    private:
	XMLCh	*unicode;
};

template<typename T>
class XMLInputSourceWrapper :
			public XERCES_CPP_NAMESPACE_QUALIFIER InputSource {
    public:
	typedef typename T::Stream_t Stream_t;

	XMLInputSourceWrapper(std::unique_ptr<Stream_t> &obj) : obj(std::move(obj)) {}
	~XMLInputSourceWrapper() override {}

	XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream* makeStream() const override
	{ return new T(*obj); }

    private:
	std::unique_ptr<Stream_t>	obj;
};

class CBInputStream : public XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream {
    public:
	class Reader {
	    public:
		virtual ~Reader();
		virtual const std::string &data()= 0;
	};

	typedef Reader Stream_t;

	CBInputStream(Reader &in);
	~CBInputStream() override;

	XMLFilePos curPos() const override { return pos; }

	XMLSize_t readBytes(XMLByte *const buf,
				    const XMLSize_t size) override;

        const XMLCh* getContentType() const override { return nullptr; }

    private:
	Reader		&reader;
	std::string	buffer;
	unsigned int	pos;
};

class STLInputStream : public XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream {
    public:
	typedef std::istream Stream_t;

	STLInputStream(std::istream &in);
	~STLInputStream() override;

	XMLFilePos curPos() const override { return pos; }

	XMLSize_t readBytes(XMLByte *const buf,
				    const XMLSize_t size) override;

        const XMLCh* getContentType() const override { return nullptr; }

    private:
	std::istream	&in;
	unsigned int	pos;
};

class StorageInputStream :
		public XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream {
    public:
	typedef StorageWrap Stream_t;

	StorageInputStream(StorageWrap &in);
	~StorageInputStream() override;

	XMLFilePos curPos() const override { return pos; }

	XMLSize_t readBytes(XMLByte *const buf,
				    const XMLSize_t size) override;

        const XMLCh* getContentType() const override { return nullptr; }

    private:
	StorageWrap	&in;
	unsigned int	pos;
        lzma_stream     lstr;
        bool            compression_;
        unsigned int    lasttotal_;

        unsigned int buffLoc_ = 0,buffTotal_ = 0;
        std::vector<uint8_t> buffer_;
        static constexpr unsigned bufferSize_ = 16*1024*1024;
};

typedef XMLInputSourceWrapper<CBInputStream> CBInputSource;
typedef XMLInputSourceWrapper<STLInputStream> STLInputSource;
typedef XMLInputSourceWrapper<StorageInputStream> StorageInputSource;

} // namespace lhef

#endif // GeneratorInterface_LHEInterface_XMLUtils_h
