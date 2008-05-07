#ifndef GeneratorInterface_LHEInterface_LHECommon_h
#define GeneratorInterface_LHEInterface_LHECommon_h

#include <iostream>
#include <memory>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommonProduct.h"

namespace lhef {

class LHECommon {
    public:
	LHECommon(std::istream &in);
	LHECommon(const HEPRUP &heprup);
	~LHECommon();

	typedef LHECommonProduct::Header Header;

	const HEPRUP *getHEPRUP() const { return &heprup; } 

	bool operator == (const LHECommon &other) const;
	inline bool operator != (const LHECommon &other) const
	{ return !(*this == other); }

	const std::vector<Header> &getHeaders() const { return headers; }

	void addHeader(const Header &header) { headers.push_back(header); }

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
	           double matchWeight = 1.0);
	XSec xsec() const;

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

		inline bool operator < (const Process &other) const
		{ return process < other.process; }
		inline bool operator < (int process) const
		{ return this->process < process; }
		inline bool operator == (int process) const
		{ return this->process == process; }
	};

	void init();

	HEPRUP			heprup;
	std::vector<Process>	processes;
	std::vector<Header>	headers;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_LHECommon_h
