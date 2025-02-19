#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <cctype>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>

#include <boost/bind.hpp>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

#include "XMLUtils.h"

XERCES_CPP_NAMESPACE_USE

static int skipWhitespace(std::istream &in)
{
	int ch;
	do {
		ch = in.get();
	} while(std::isspace(ch));
	if (ch != std::istream::traits_type::eof())
		in.putback(ch);
	return ch;
}

namespace lhef {

LHERunInfo::LHERunInfo(std::istream &in)
{
	in >> heprup.IDBMUP.first >> heprup.IDBMUP.second
	   >> heprup.EBMUP.first >> heprup.EBMUP.second
	   >> heprup.PDFGUP.first >> heprup.PDFGUP.second
	   >> heprup.PDFSUP.first >> heprup.PDFSUP.second
	   >> heprup.IDWTUP >> heprup.NPRUP;
	if (!in.good())
		throw cms::Exception("InvalidFormat")
			<< "Les Houches file contained invalid"
			   " header in init section." << std::endl;

	heprup.resize();

	for(int i = 0; i < heprup.NPRUP; i++) {
		in >> heprup.XSECUP[i] >> heprup.XERRUP[i]
		   >> heprup.XMAXUP[i] >> heprup.LPRUP[i];
		if (!in.good())
			throw cms::Exception("InvalidFormat")
				<< "Les Houches file contained invalid data"
				   " in header payload line " << (i + 1)
				<< "." << std::endl;
	}

	while(skipWhitespace(in) == '#') {
		std::string line;
		std::getline(in, line);
		comments.push_back(line + "\n");
	}

	if (!in.eof())
		edm::LogWarning("Generator|LHEInterface")
			<< "Les Houches file contained spurious"
			   " content after the regular data." << std::endl;

	init();
}

LHERunInfo::LHERunInfo(const HEPRUP &heprup) :
	heprup(heprup)
{
	init();
}

LHERunInfo::LHERunInfo(const HEPRUP &heprup,
                       const std::vector<LHERunInfoProduct::Header> &headers,  
                       const std::vector<std::string> &comments) :
	heprup(heprup)
{
	std::copy(headers.begin(), headers.end(),
	          std::back_inserter(this->headers));
	std::copy(comments.begin(), comments.end(),
	          std::back_inserter(this->comments));

	init();
}
                                                      
LHERunInfo::LHERunInfo(const LHERunInfoProduct &product) :
	heprup(product.heprup())
{
	std::copy(product.headers_begin(), product.headers_end(),
	          std::back_inserter(headers));
	std::copy(product.comments_begin(), product.comments_end(),
	          std::back_inserter(comments));

	init();
}
                                                      
LHERunInfo::~LHERunInfo()
{
}

void LHERunInfo::init()
{
	for(int i = 0; i < heprup.NPRUP; i++) {
		Process proc;

		proc.process = heprup.LPRUP[i];
		proc.heprupIndex = (unsigned int)i;

		processes.push_back(proc);
	}

	std::sort(processes.begin(), processes.end());
}

bool LHERunInfo::operator == (const LHERunInfo &other) const
{
	return heprup == other.heprup;
}

void LHERunInfo::count(int process, CountMode mode, double eventWeight,
                       double brWeight, double matchWeight)
{
	std::vector<Process>::iterator proc =
		std::lower_bound(processes.begin(), processes.end(), process);
	if (proc == processes.end() || proc->process != process)
		return;

	switch(mode) {
	    case kAccepted:
		proc->acceptedBr.add(eventWeight * brWeight * matchWeight);
		proc->accepted.add(eventWeight * matchWeight);
	    case kKilled:
		proc->killed.add(eventWeight * matchWeight);
	    case kSelected:
		proc->selected.add(eventWeight);
	    case kTried:
		proc->tried.add(eventWeight);
	}
}

LHERunInfo::XSec LHERunInfo::xsec() const
{
	double sigSelSum = 0.0;
	double sigSum = 0.0;
	double sigBrSum = 0.0;
	double err2Sum = 0.0;
	double errBr2Sum = 0.0;

	for(std::vector<Process>::const_iterator proc = processes.begin();
	    proc != processes.end(); ++proc) {
		unsigned int idx = proc->heprupIndex;

		double sigmaSum, sigma2Sum, sigma2Err, xsec;
		switch(std::abs(heprup.IDWTUP)) {
		    case 2:
			sigmaSum = proc->tried.sum * heprup.XSECUP[idx];
			sigma2Sum = proc->tried.sum2 * heprup.XSECUP[idx]
			                             * heprup.XSECUP[idx];
			sigma2Err = proc->tried.sum2 * heprup.XERRUP[idx]
			                             * heprup.XERRUP[idx];
			break;
		    case 3:
			sigmaSum = proc->tried.n * heprup.XSECUP[idx];
			sigma2Sum = sigmaSum * heprup.XSECUP[idx];
			sigma2Err = proc->tried.n * heprup.XERRUP[idx]
			                          * heprup.XERRUP[idx];
			break;
		    default:
			xsec = proc->tried.sum / proc->tried.n;
			sigmaSum = proc->tried.sum * xsec;
			sigma2Sum = proc->tried.sum2 * xsec * xsec;
			sigma2Err = 0.0;
		}

		if (!proc->killed.n)
			continue;

		double sigmaAvg = sigmaSum / proc->tried.sum;
		double fracAcc = proc->killed.sum / proc->selected.sum;
		double fracBr = proc->accepted.sum > 0.0 ?
		                proc->acceptedBr.sum / proc->accepted.sum : 1;
		double sigmaFin = sigmaAvg * fracAcc * fracBr;
		double sigmaFinBr = sigmaFin * fracBr;

		double relErr = 1.0;
		if (proc->killed.n > 1) {
			double sigmaAvg2 = sigmaAvg * sigmaAvg;
			double delta2Sig =
				(sigma2Sum / proc->tried.n - sigmaAvg2) /
				(proc->tried.n * sigmaAvg2);
			double delta2Veto =
				((double)proc->selected.n - proc->killed.n) /
				((double)proc->selected.n * proc->killed.n);
			double delta2Sum = delta2Sig + delta2Veto
			                   + sigma2Err / sigma2Sum;
			relErr = (delta2Sum > 0.0 ?
					std::sqrt(delta2Sum) : 0.0);
		}
		double deltaFin = sigmaFin * relErr;
		double deltaFinBr = sigmaFinBr * relErr;

		sigSelSum += sigmaAvg;
		sigSum += sigmaFin;
		sigBrSum += sigmaFinBr;
		err2Sum += deltaFin * deltaFin;
		errBr2Sum += deltaFinBr * deltaFinBr;
	}

	XSec result;
	result.value = sigBrSum;
	result.error = std::sqrt(errBr2Sum);

	return result;
}

void LHERunInfo::statistics() const
{
	double sigSelSum = 0.0;
	double sigSum = 0.0;
	double sigBrSum = 0.0;
	double err2Sum = 0.0;
	double errBr2Sum = 0.0;
	unsigned long nAccepted = 0;
	unsigned long nTried = 0;

	std::cout << std::endl;
	std::cout << "Process and cross-section statistics" << std::endl;
	std::cout << "------------------------------------" << std::endl;
	std::cout << "Process\tevents\ttried\txsec [pb]\t\taccepted [%]"
	          << std::endl;

	for(std::vector<Process>::const_iterator proc = processes.begin();
	    proc != processes.end(); ++proc) {
		unsigned int idx = proc->heprupIndex;

		double sigmaSum, sigma2Sum, sigma2Err, xsec;
		switch(std::abs(heprup.IDWTUP)) {
		    case 2:
			sigmaSum = proc->tried.sum * heprup.XSECUP[idx];
			sigma2Sum = proc->tried.sum2 * heprup.XSECUP[idx]
			                             * heprup.XSECUP[idx];
			sigma2Err = proc->tried.sum2 * heprup.XERRUP[idx]
			                             * heprup.XERRUP[idx];
			break;
		    case 3:
			sigmaSum = proc->tried.n * heprup.XSECUP[idx];
			sigma2Sum = sigmaSum * heprup.XSECUP[idx];
			sigma2Err = proc->tried.n * heprup.XERRUP[idx]
			                          * heprup.XERRUP[idx];
			break;
		    default:
			xsec = proc->tried.sum / proc->tried.n;
			sigmaSum = proc->tried.sum * xsec;
			sigma2Sum = proc->tried.sum2 * xsec * xsec;
			sigma2Err = 0.0;
		}

		if (!proc->selected.n) {
			std::cout << proc->process << "\t0\t0\tn/a\t\t\tn/a"
			          << std::endl;
			continue;
		}

		double sigmaAvg = sigmaSum / proc->tried.sum;
		double fracAcc = proc->killed.sum / proc->selected.sum;
		double fracBr = proc->accepted.sum > 0.0 ?
		                proc->acceptedBr.sum / proc->accepted.sum : 1;
		double sigmaFin = sigmaAvg * fracAcc;
		double sigmaFinBr = sigmaFin * fracBr;

		double relErr = 1.0;
		if (proc->killed.n > 1) {
			double sigmaAvg2 = sigmaAvg * sigmaAvg;
			double delta2Sig =
				(sigma2Sum / proc->tried.n - sigmaAvg2) /
				(proc->tried.n * sigmaAvg2);
			double delta2Veto =
				((double)proc->selected.n - proc->killed.n) /
				((double)proc->selected.n * proc->killed.n);
			double delta2Sum = delta2Sig + delta2Veto
			                   + sigma2Err / sigma2Sum;
			relErr = (delta2Sum > 0.0 ?
					std::sqrt(delta2Sum) : 0.0);
		}
		double deltaFin = sigmaFin * relErr;
		double deltaFinBr = sigmaFinBr * relErr;

		std::cout << proc->process << "\t"
		          << proc->accepted.n << "\t"
		          << proc->tried.n << "\t"
		          << std::scientific << std::setprecision(3)
		          << sigmaFinBr << " +/- "
		          << deltaFinBr << "\t"
		          << std::fixed << std::setprecision(1)
		          << (fracAcc * 100) << std::endl;

		nAccepted += proc->accepted.n;
		nTried += proc->tried.n;
		sigSelSum += sigmaAvg;
		sigSum += sigmaFin;
		sigBrSum += sigmaFinBr;
		err2Sum += deltaFin * deltaFin;
		errBr2Sum += deltaFinBr * deltaFinBr;
	}

	std::cout << "Total\t"
	          << nAccepted << "\t"
	          << nTried << "\t"
	          << std::scientific << std::setprecision(3)
	          << sigBrSum << " +/- "
	          << std::sqrt(errBr2Sum) << "\t"
	          << std::fixed << std::setprecision(1)
	          << (sigSum / sigSelSum * 100) << std::endl;
}

LHERunInfo::Header::Header() :
	xmlDoc(0)
{
}

LHERunInfo::Header::Header(const std::string &tag) :
	LHERunInfoProduct::Header(tag), xmlDoc(0)
{
}

LHERunInfo::Header::Header(const Header &orig) :
	LHERunInfoProduct::Header(orig), xmlDoc(0)
{
}

LHERunInfo::Header::Header(const LHERunInfoProduct::Header &orig) :
	LHERunInfoProduct::Header(orig), xmlDoc(0)
{
}

LHERunInfo::Header::~Header()
{
	if (xmlDoc)
		xmlDoc->release();
}

static void fillLines(std::vector<std::string> &lines, const char *data,
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
		lines.push_back(std::string(data, len));
		data += len;
	}
}

static std::vector<std::string> domToLines(const DOMNode *node)
{
	std::vector<std::string> result;
	DOMImplementation *impl =
		DOMImplementationRegistry::getDOMImplementation(
							XMLUniStr("Core"));
	std::auto_ptr<DOMWriter> writer(
		static_cast<DOMImplementationLS*>(impl)->createDOMWriter());

	writer->setEncoding(XMLUniStr("UTF-8"));
	XMLSimpleStr buffer(writer->writeToString(*node));

	const char *p = std::strchr((const char*)buffer, '>') + 1;
	const char *q = std::strrchr(p, '<');
	fillLines(result, p, q - p);

	return result;
}

std::vector<std::string> LHERunInfo::findHeader(const std::string &tag) const
{
	const LHERunInfo::Header *header = 0;
	for(std::vector<Header>::const_iterator iter = headers.begin();
	    iter != headers.end(); ++iter) {
		if (iter->tag() == tag)
			return std::vector<std::string>(iter->begin(),
			                                iter->end());
		if (iter->tag() == "header")
			header = &*iter;
	}

	if (!header)
		return std::vector<std::string>();

	const DOMNode *root = header->getXMLNode();
	if (!root)
		return std::vector<std::string>();

	for(const DOMNode *iter = root->getFirstChild();
	    iter; iter = iter->getNextSibling()) {
		if (iter->getNodeType() != DOMNode::ELEMENT_NODE)
			continue;
		if (tag == (const char*)XMLSimpleStr(iter->getNodeName()))
			return domToLines(iter);
	}

	return std::vector<std::string>();
}

namespace {
	class HeaderReader : public CBInputStream::Reader {
	    public:
		HeaderReader(const LHERunInfo::Header *header) :
			header(header), mode(kHeader),
			iter(header->begin())
		{
		}

		const std::string &data()
		{
			switch(mode) {
			    case kHeader:
				tmp = "<" + header->tag() + ">";
				mode = kBody;
				break;
			    case kBody:
				if (iter != header->end())
					return *iter++;
				tmp = "</" + header->tag() + ">";
				mode = kFooter;
				break;
			    case kFooter:
				tmp.clear();
			}

			return tmp;
		}

	    private:
		enum Mode {
			kHeader,
			kBody,
			kFooter
		};

		const LHERunInfo::Header		*header;
		Mode					mode;
		LHERunInfo::Header::const_iterator	iter;
		std::string				tmp;
	};
} // anonymous namespace

const DOMNode *LHERunInfo::Header::getXMLNode() const
{
	if (tag().empty())
		return 0;

	if (!xmlDoc) {
		XercesDOMParser parser;
		parser.setValidationScheme(XercesDOMParser::Val_Auto);
		parser.setDoNamespaces(false);
		parser.setDoSchema(false);
		parser.setValidationSchemaFullChecking(false);

		HandlerBase errHandler;
		parser.setErrorHandler(&errHandler);
		parser.setCreateEntityReferenceNodes(false);

		try {
			std::auto_ptr<CBInputStream::Reader> reader(
						new HeaderReader(this));
			CBInputSource source(reader);
			parser.parse(source);
			xmlDoc = parser.adoptDocument();
		} catch(const XMLException &e) {
			throw cms::Exception("Generator|LHEInterface")
				<< "XML parser reported DOM error no. "
				<< (unsigned long)e.getCode()
				<< ": " << XMLSimpleStr(e.getMessage()) << "."
				<< std::endl;
		} catch(const SAXException &e) {
			throw cms::Exception("Generator|LHEInterface")
				<< "XML parser reported: "
				<< XMLSimpleStr(e.getMessage()) << "."
				<< std::endl;
		}
	}

	return xmlDoc->getDocumentElement();
}

std::pair<int, int> LHERunInfo::pdfSetTranslation() const
{
	int pdfA = -1, pdfB = -1;

	if (heprup.PDFGUP.first >= 0) {
		pdfA = heprup.PDFSUP.first;
	}

	if (heprup.PDFGUP.second >= 0) {
		pdfB = heprup.PDFSUP.second;
	}

	return std::make_pair(pdfA, pdfB);
}

} // namespace lhef
