#ifndef GeneratorInterface_LHEInterface_BranchingRatios_h
#define GeneratorInterface_LHEInterface_BranchingRatios_h

#include <map>

#include <boost/shared_ptr.hpp>

namespace lhef {

class LHEEvent;
class Hadronisation;

class BranchingRatios {
    public:
	BranchingRatios() {}
	~BranchingRatios() {}

	void reset();

	void set(int pdgId, bool both = true, double value = 1.0);
	double getFactor(const Hadronisation *hadronisation,
	                 const boost::shared_ptr<LHEEvent> &event) const;

    private:
	double get(int pdgId, const Hadronisation *hadronisation) const;

	mutable std::map<int, double>	cache;
};

} // namespace lhef

#endif // GeneratorCommon_LHEInterface_BranchingRatios_h
