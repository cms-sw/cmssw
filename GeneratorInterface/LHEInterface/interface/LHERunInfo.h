#ifndef GeneratorInterface_LHEInterface_LHERunInfo_h
#define GeneratorInterface_LHEInterface_LHERunInfo_h

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

#ifndef XERCES_CPP_NAMESPACE_QUALIFIER
#	define UNDEF_XERCES_CPP_NAMESPACE_QUALIFIER
#	define XERCES_CPP_NAMESPACE_QUALIFIER dummy::
namespace dummy {
	class DOMNode;
	class DOMDocument;
}
#endif

namespace lhef {

class LHERunInfo {
    public:
	LHERunInfo(std::istream &in);
	LHERunInfo(const HEPRUP &heprup);
	LHERunInfo(const HEPRUP &heprup,
	           const std::vector<LHERunInfoProduct::Header> &headers,
	           const std::vector<std::string> &comments);
	LHERunInfo(const LHERunInfoProduct &product);
	~LHERunInfo();

	class Header : public LHERunInfoProduct::Header {
	    public:
		Header();
		Header(const std::string &tag);
		Header(const Header &orig);
		Header(const LHERunInfoProduct::Header &orig);
		~Header();

#ifndef UNDEF_XERCES_CPP_NAMESPACE_QUALIFIER
		const XERCES_CPP_NAMESPACE_QUALIFIER DOMNode
							*getXMLNode() const;
#endif

	    private:
		mutable XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *xmlDoc;
	};

	const HEPRUP *getHEPRUP() const { return &heprup; } 

	bool operator == (const LHERunInfo &other) const;
	inline bool operator != (const LHERunInfo &other) const
	{ return !(*this == other); }

	const std::vector<Header> &getHeaders() const { return headers; }
	const std::vector<std::string> &getComments() const { return comments; }

	std::vector<std::string> findHeader(const std::string &tag) const;

	void addHeader(const Header &header) { headers.push_back(header); }
	void addComment(const std::string &line) { comments.push_back(line); }

	enum CountMode {
		kTried = 0,
		kSelected,
		kKilled,
		kAccepted
	};

	struct XSec {
		XSec() : value(0.0), error(0.0) {}

		double	value;
		double	error;
	};

	void count(int process, CountMode count, double eventWeight = 1.0,
	           double brWeight = 1.0, double matchWeight = 1.0);
	XSec xsec() const;
	void statistics() const;

	std::pair<int, int> pdfSetTranslation() const;

    private:
	struct Counter {
		Counter() : n(0), sum(0.0), sum2(0.0) {}

		inline void add(double weight)
		{
			n++;
			sum += weight;
			sum2 += weight * weight;
		}

		unsigned int	n;
		double		sum;
		double		sum2;
	};

	struct Process {
		int		process;
		unsigned int	heprupIndex;
		Counter		tried;
		Counter		selected;
		Counter		killed;
		Counter		accepted;
		Counter		acceptedBr;

		inline bool operator < (const Process &other) const
		{ return process < other.process; }
		inline bool operator < (int process) const
		{ return this->process < process; }
		inline bool operator == (int process) const
		{ return this->process == process; }
	};

	void init();

	HEPRUP				heprup;
	std::vector<Process>		processes;
	std::vector<Header>		headers;
	std::vector<std::string>	comments;
};

} // namespace lhef

#ifdef UNDEF_XERCES_CPP_NAMESPACE_QUALIFIER
#	undef XERCES_CPP_NAMESPACE_QUALIFIER
#endif

#endif // GeneratorRunInfo_LHEInterface_LHERunInfo_h
