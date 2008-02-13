#include <iostream>
#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommon.h"

#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

namespace lhef {

LHEEvent::LHEEvent(const boost::shared_ptr<LHECommon> &common,
                   std::istream &in) :
	common(common)
{
	hepeup.NUP = 0;
	hepeup.XPDWUP.first = hepeup.XPDWUP.second = 0.0;

	in >> hepeup.NUP >> hepeup.IDPRUP >> hepeup.XWGTUP
	   >> hepeup.SCALUP >> hepeup.AQEDUP >> hepeup.AQCDUP;
	if (!in.good())
		throw cms::Exception("InvalidFormat")
			<< "Les Houches file contained invalid"
			   " event header." << std::endl;

	hepeup.resize();

	for(int i = 0; i < hepeup.NUP; i++) {
		in >> hepeup.IDUP[i] >> hepeup.ISTUP[i]
		   >> hepeup.MOTHUP[i].first >> hepeup.MOTHUP[i].second
		   >> hepeup.ICOLUP[i].first >> hepeup.ICOLUP[i].second
		   >> hepeup.PUP[i][0] >> hepeup.PUP[i][1] >> hepeup.PUP[i][2]
		   >> hepeup.PUP[i][3] >> hepeup.PUP[i][4]
		   >> hepeup.VTIMUP[i] >> hepeup.SPINUP[i];
		if (!in.good())
			throw cms::Exception("InvalidFormat")
				<< "Les Houches file contained invalid event"
				   " in particle line " << (i + 1)
				<< "." << std::endl;
	}

	int ch;
	do {
		ch = in.get();
	} while(std::isspace(ch));

	if (!in.eof())
		edm::LogWarning("Generator|LHEInterface")
			<< "Les Houches file contained spurious"
			   " content after event data." << std::endl;
}

LHEEvent::~LHEEvent()
{
}

} // namespace lhef
