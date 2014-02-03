#ifndef GeneratorInterface_LHEInterface_LHEEvent_h
#define GeneratorInterface_LHEInterface_LHEEvent_h

#include <iostream>
#include <utility>
#include <memory>
#include <vector>
#include <string>

#include <boost/shared_ptr.hpp>

#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/PdfInfo.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHEEventProduct.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

namespace lhef {

class LHEEvent {
    public:
	LHEEvent(const boost::shared_ptr<LHERunInfo> &runInfo,
	         std::istream &in);
	LHEEvent(const boost::shared_ptr<LHERunInfo> &runInfo,
	         const HEPEUP &hepeup);
	LHEEvent(const boost::shared_ptr<LHERunInfo> &runInfo,
	         const HEPEUP &hepeup,
	         const LHEEventProduct::PDF *pdf,
	         const std::vector<std::string> &comments);
	LHEEvent(const boost::shared_ptr<LHERunInfo> &runInfo,
	         const LHEEventProduct &product);
	~LHEEvent();

	typedef LHEEventProduct::PDF PDF;
	typedef LHEEventProduct::WGT WGT;

	const boost::shared_ptr<LHERunInfo> &getRunInfo() const { return runInfo; }
	const HEPEUP *getHEPEUP() const { return &hepeup; }
	const HEPRUP *getHEPRUP() const { return runInfo->getHEPRUP(); }
	const PDF *getPDF() const { return pdf.get(); }
	const std::vector<std::string> &getComments() const { return comments; }
	const int getReadAttempts() { return readAttemptCounter; }

	void addWeight(const WGT& wgt) { weights_.push_back(wgt); }
	void setPDF(std::auto_ptr<PDF> pdf) { this->pdf = pdf; }

	double originalXWGTUP() const { return originalXWGTUP_; }
	const std::vector<WGT>& weights() const { return weights_; }

	void addComment(const std::string &line) { comments.push_back(line); }

	static void removeParticle(lhef::HEPEUP &hepeup, int index);
	void removeResonances(const std::vector<int> &ids);

	void count(LHERunInfo::CountMode count,
	           double weight = 1.0, double matchWeight = 1.0);

	void attempted() { readAttemptCounter++; return; }
	
	void fillPdfInfo(HepMC::PdfInfo *info) const;
	void fillEventInfo(HepMC::GenEvent *hepmc) const;

	std::auto_ptr<HepMC::GenEvent> asHepMCEvent() const;

	static const HepMC::GenVertex *findSignalVertex(
			const HepMC::GenEvent *event, bool status3 = true);

	static void fixHepMCEventTimeOrdering(HepMC::GenEvent *event);

    private:
	static bool checkHepMCTree(const HepMC::GenEvent *event);
	HepMC::GenParticle *makeHepMCParticle(unsigned int i) const;

	const boost::shared_ptr<LHERunInfo>	runInfo;

	HEPEUP					hepeup;
	std::auto_ptr<PDF>			pdf;
	std::vector<WGT>	          	weights_;
	std::vector<std::string>		comments;
	bool					counted;
	int                                     readAttemptCounter;
	double                                  originalXWGTUP_;
};

} // namespace lhef

#endif // GeneratorEvent_LHEInterface_LHEEvent_h
