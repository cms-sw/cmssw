#include <iostream>
#include <string>
#include <cctype>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHECommon.h"

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

LHECommon::LHECommon(std::istream &in, const std::string &comment)
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

	skipWhitespace(in);
	if (!in.eof())
		edm::LogWarning("Generator|LHEInterface")
			<< "Les Houches file contained spurious"
			   " content after the regular data." << std::endl;
}

LHECommon::~LHECommon()
{
}

bool LHECommon::operator == (const LHECommon &other) const
{
	return heprup == other.heprup;
}

} // namespace lhef
