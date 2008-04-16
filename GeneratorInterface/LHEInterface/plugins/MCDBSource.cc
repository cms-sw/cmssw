#include <iostream>
#include <string>

#include "mcdb.hpp"

#include "FWCore/Framework/interface/GeneratedInputSource.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "GeneratorInterface/LHEInterface/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"

#include "LHESource.h"

using namespace lhef;

class MCDBSource : public LHESource {
    public:
	explicit MCDBSource(const edm::ParameterSet &params,
	                    const edm::InputSourceDescription &desc);
	virtual ~MCDBSource();

    private:
	mcdb::MCDB	mcdb;
};

MCDBSource::MCDBSource(const edm::ParameterSet &params,
                       const edm::InputSourceDescription &desc) :
        LHESource(params, desc, 0)
{
	unsigned int articleId = params.getParameter<unsigned int>("articleID");

	edm::LogInfo("Generator|LHEInterface")
		<< "Reading article id " << articleId << " from MCDB."
		<< std::endl;

	mcdb::Article article = mcdb.getArticle(articleId);

	edm::LogInfo("Generator|LHEInterface")
		<< "Title: " << article.title() << std::endl
		<< "First author: " << article.authors()[0].firstName() << " "
		<< article.authors()[0].lastName() << std::endl
		<< "Number of authors: " << article.authors().size() << std::endl
		<< "Abstract: " << article.abstract() << std::endl
		<< "Generator: " << article.generator().name()
		<< ", " << article.generator().version() << std::endl
		<< "Number of files: " << article.files().size() << std::endl
		<< "Files: " << std::endl;

	std::vector<std::string> supportedProtocols = 
			params.getParameter< std::vector<std::string> >(
							"supportedProtocols");

	unsigned int firstEvent =
		params.getUntrackedParameter<unsigned int>("seekEvent", 0);

	std::vector<std::string> fileURLs;
	for(std::vector<mcdb::File>::iterator file = article.files().begin();
	    file != article.files().end(); ++file) {
		int nEvents = file->eventsNumber();
		if ((int)firstEvent > nEvents) {
			firstEvent -= nEvents;
			continue;
		}

		bool found = false;
		for(std::vector<std::string>::const_iterator prot =
						supportedProtocols.begin();
		    prot != supportedProtocols.end(); ++prot) {
			for(std::vector<std::string>::const_iterator path =
							file->paths().begin();
			    path != file->paths().end(); ++path) {
				if (path->substr(0, prot->length() + 1) ==
				    *prot + ":") {
					fileURLs.push_back(*path);
					found = true;
					break;
				}
			}
			if (found)
				break;
		}

		if (!found)
			throw cms::Exception("Generator|LHEInterface")
				<< "MCDB did not contain any URLs with"
				   " supported protocols for at least one"
				   " file." << std::endl;
	}

	reader.reset(new LHEReader(fileURLs, firstEvent));
}

MCDBSource::~MCDBSource()
{
}

DEFINE_ANOTHER_FWK_INPUT_SOURCE(MCDBSource);
