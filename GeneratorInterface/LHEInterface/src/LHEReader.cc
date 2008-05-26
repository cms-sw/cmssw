#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>

#include <boost/bind.hpp>

#include <xercesc/sax2/Attributes.hpp>
#include <xercesc/dom/DOM.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

#include "Utilities/StorageFactory/interface/IOTypes.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

#include "XMLUtils.h"

XERCES_CPP_NAMESPACE_USE

namespace lhef {

class LHEReader::Source {
    public:
	Source() {}
	virtual ~Source() {}
	virtual XMLDocument *createReader(XMLDocument::Handler &handler) = 0;
};

class LHEReader::FileSource : public LHEReader::Source {
    public:
	FileSource(const std::string &fileURL) :
		fileStream(StorageFactory::get()->open(fileURL,
		                                       IOFlags::OpenRead))
	{
		if (!fileStream.get())
			throw cms::Exception("FileOpenError")
				<< "Could not open LHE file \""
				<< fileURL << "\" for reading"
				<< std::endl;
	}

	~FileSource() {}

	XMLDocument *createReader(XMLDocument::Handler &handler)
	{ return new XMLDocument(fileStream, handler); }

    private:
	std::auto_ptr<Storage>		fileStream;
};

class LHEReader::XMLHandler : public XMLDocument::Handler {
    public:
	XMLHandler() :
		impl(0), gotObject(kNone), mode(kNone),
		xmlHeader(0), headerOk(false) {}
	~XMLHandler()
	{ if (xmlHeader) xmlHeader->release(); }

	enum Object {
		kNone = 0,
		kHeader,
		kInit,
		kComment,
		kEvent
	};

    protected:
	void startElement(const XMLCh *const uri,
	                  const XMLCh *const localname,
	                  const XMLCh *const qname,
	                  const Attributes &attributes);

	void endElement(const XMLCh *const uri,
	                const XMLCh *const localname,
	                const XMLCh *const qname);

	void characters(const XMLCh *const data, const unsigned int length);
	void comment(const XMLCh *const data, const unsigned int length);

    private:
	friend class LHEReader;

	DOMImplementation		*impl;
	std::string			buffer;
	Object				gotObject;
	Object				mode;
	DOMDocument			*xmlHeader;
	std::vector<DOMElement*>	xmlNodes;
	bool				headerOk;
	std::vector<LHERunInfo::Header>	headers;
};

static void attributesToDom(DOMElement *dom, const Attributes &attributes)
{
	for(unsigned int i = 0; i < attributes.getLength(); i++) {
		const XMLCh *name = attributes.getQName(i);
		const XMLCh *value = attributes.getValue(i);

		dom->setAttribute(name, value);
	}
}

static void fillHeader(LHERunInfo::Header &header, const char *data,
                       int len = -1)
{
	const char *end = len >= 0 ? (data + len) : 0;
	while(*data && (!end || data < end)) {
		std::size_t len = std::strcspn(data, "\r\n");
		if (end && data + len > end)
			len = end - data;
		if (data[len] == '\r' && data[len + 1] == '\n')
			len += 2;
		else if (data[len])
			len++;
		header.addLine(std::string(data, len));
		data += len;
	}
}

void LHEReader::XMLHandler::startElement(const XMLCh *const uri,
                                         const XMLCh *const localname,
                                         const XMLCh *const qname,
                                         const Attributes &attributes)
{
	std::string name((const char*)XMLSimpleStr(qname));

	if (!headerOk) {
		if (name != "LesHouchesEvents")
			throw cms::Exception("InvalidFormat")
				<< "LHE file has invalid header" << std::endl;
		headerOk = true;
		return;
	}

	if (mode == kHeader) {
		DOMElement *elem = xmlHeader->createElement(qname);
		attributesToDom(elem, attributes);
		xmlNodes.back()->appendChild(elem);
		xmlNodes.push_back(elem);
		return;
	} else if (mode != kNone)
		throw cms::Exception("InvalidFormat")
			<< "LHE file has invalid format" << std::endl;

	if (name == "header") {
		if (!impl)
			impl = DOMImplementationRegistry::getDOMImplementation(
							XMLUniStr("Core"));
		xmlHeader = impl->createDocument(0, qname, 0);
		xmlNodes.resize(1);
		xmlNodes[0] = xmlHeader->getDocumentElement();
		mode = kHeader;
	} if (name == "init")
		mode = kInit;
	else if (name == "event")
		mode = kEvent;

	if (mode == kNone)
		throw cms::Exception("InvalidFormat")
			<< "LHE file has invalid format" << std::endl;

	buffer.clear();
}

void LHEReader::XMLHandler::endElement(const XMLCh *const uri,
                                       const XMLCh *const localname,
                                       const XMLCh *const qname)
{
	if (mode) {
		if (mode == kHeader && xmlNodes.size() > 1) {
			xmlNodes.resize(xmlNodes.size() - 1);
			return;
		} else if (mode == kHeader) {
			std::auto_ptr<DOMWriter> writer(
				static_cast<DOMImplementationLS*>(
                                                impl)->createDOMWriter());
			writer->setEncoding(XMLUniStr("UTF-8"));
			XMLSimpleStr buffer(
					writer->writeToString(*xmlNodes[0]));
			LHERunInfo::Header header("header");
			fillHeader(header, (const char*)buffer + 8,
			           std::strlen(buffer) - 17);
			headers.push_back(header);

			xmlHeader->release();
			xmlHeader = 0;
		}

		if (gotObject != kNone)
			throw cms::Exception("InvalidState")
				<< "Unexpected pileup in"
				    " LHEReader::XMLHandler::endElement"
				<< std::endl;

		gotObject = mode;
		mode = kNone;
	}
}

void LHEReader::XMLHandler::characters(const XMLCh *const data_,
                                       const unsigned int length)
{
	if (mode == kHeader) {
		DOMText *text = xmlHeader->createTextNode(data_);
		xmlNodes.back()->appendChild(text);
		return;
	}

	if (XMLSimpleStr::isAllSpaces(data_, length))
		return;

	unsigned int offset = 0;
	while(offset < length && XMLSimpleStr::isSpace(data_[offset]))
		offset++;

	XMLSimpleStr data(data_ + offset);

	if (mode == kNone)
		throw cms::Exception("InvalidFormat")
			<< "LHE file has invalid format" << std::endl;

	buffer.append(data);
}

void LHEReader::XMLHandler::comment(const XMLCh *const data_,
                                    const unsigned int length)
{
	if (mode == kHeader) {
		DOMComment *comment = xmlHeader->createComment(data_);
		xmlNodes.back()->appendChild(comment);
		return;
	}

	XMLSimpleStr data(data_);

	LHERunInfo::Header header;
	fillHeader(header, data);
	headers.push_back(header);
}

LHEReader::LHEReader(const edm::ParameterSet &params) :
	fileURLs(params.getUntrackedParameter< std::vector<std::string> >("fileNames")),
	firstEvent(params.getUntrackedParameter<unsigned int>("seekEvent", 0)),
	maxEvents(params.getUntrackedParameter<int>("limitEvents", -1)),
	curIndex(0), handler(new XMLHandler())
{
}

LHEReader::LHEReader(const std::vector<std::string> &fileNames,
                     unsigned int firstEvent) :
	fileURLs(fileNames), firstEvent(firstEvent),
	curIndex(0), handler(new XMLHandler())
{
}

LHEReader::~LHEReader()
{
}

boost::shared_ptr<LHEEvent> LHEReader::next()
{
	while(curDoc.get() || curIndex < fileURLs.size()) {
		if (!curDoc.get()) {
			curSource.reset(new FileSource(fileURLs[curIndex++]));
			curDoc.reset(curSource->createReader(*handler));
			curRunInfo.reset();
		}

		XMLHandler::Object event = handler->gotObject;
		handler->gotObject = XMLHandler::kNone;

		std::istringstream data;
		if (event != XMLHandler::kNone) {
			data.str(handler->buffer);
			handler->buffer.clear();
		}

		switch(event) {
		    case XMLHandler::kNone:
			if (!curDoc->parse())
				curDoc.reset();
			break;

		    case XMLHandler::kHeader:
			break;

		    case XMLHandler::kInit:
			curRunInfo.reset(new LHERunInfo(data));
			std::for_each(handler->headers.begin(),
			              handler->headers.end(),
			              boost::bind(&LHERunInfo::addHeader,
			                          curRunInfo.get(), _1));
			handler->headers.clear();
			break;

		    case XMLHandler::kComment:
			break;

		    case XMLHandler::kEvent:
			if (!curRunInfo.get())
				throw cms::Exception("InvalidState")
					<< "Got LHE event without"
					   " initialization." << std::endl;

			if (firstEvent > 0) {
				firstEvent--;
				continue;
			}

			if (maxEvents == 0)
				return boost::shared_ptr<LHEEvent>();
			else if (maxEvents > 0)
				maxEvents--;

			return boost::shared_ptr<LHEEvent>(
					new LHEEvent(curRunInfo, data));
		}
	}

	return boost::shared_ptr<LHEEvent>();
}

} // namespace lhef
