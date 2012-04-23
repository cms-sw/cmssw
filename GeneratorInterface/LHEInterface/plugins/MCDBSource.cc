#include <iostream>
#include <utility>
#include <string>

#include <boost/regex.hpp>

#include "mcdb.hpp"

#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/LHEInterface/interface/LHEReader.h"

#include "LHESource.h"

using namespace lhef;

class MCDBSource : public LHESource {
    public:
	explicit MCDBSource(const edm::ParameterSet &params,
	                    const edm::InputSourceDescription &desc);
	virtual ~MCDBSource();
};

static std::pair<std::vector<std::string>, unsigned int>
			getFileURLs(const edm::ParameterSet &params)
{
	unsigned int articleId = params.getParameter<unsigned int>("articleID");

	edm::LogInfo("Generator|LHEInterface")
		<< "Reading article id " << articleId << " from MCDB."
		<< std::endl;

	mcdb::MCDB mcdb;

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
		params.getUntrackedParameter< std::vector<std::string> >(
							"supportedProtocols");

	boost::regex filter(params.getUntrackedParameter<std::string>(
							"filter", "\\.lhef?$"),
	                    boost::regex_constants::normal |
	                    boost::regex_constants::icase);

	unsigned int firstEvent =
		params.getUntrackedParameter<unsigned int>("skipEvents", 0);

    unsigned int fcount = 0;
	std::vector<std::string> fileURLs;
	for(std::vector<mcdb::File>::iterator file = article.files().begin();
	    file != article.files().end(); ++file) {
		std::string fileURL;
		for(std::vector<std::string>::const_iterator prot =
						supportedProtocols.begin();
		    prot != supportedProtocols.end(); ++prot) {
			for(std::vector<std::string>::const_iterator path =
							file->paths().begin();
			    path != file->paths().end(); ++path) {
				if (path->substr(0, prot->length() + 1) ==
				    *prot + ":") {
					fileURL = *path;
					break;
				}
			}
			if (!fileURL.empty())
				break;
		}

		if (fileURL.empty())
			throw cms::Exception("Generator|LHEInterface")
				<< "MCDB did not contain any URLs with"
				   " supported protocols for at least one"
				   " file." << std::endl;

		if (!boost::regex_search(fileURL, filter))
			continue;

		int nEvents = file->eventsNumber();
		if (nEvents > 0 && (int)firstEvent >= nEvents) {
			firstEvent -= nEvents;
			continue;
		}

		fileURLs.push_back(fileURL);
        fcount++;
        edm::LogInfo("Generator|LHEInterface") << "Adding file n. " << fcount << " " << fileURL;
	}

	return std::make_pair(fileURLs, firstEvent);
}

static edm::ParameterSet augmentPSetFromMCDB(const edm::ParameterSet &params)
{
	// note that this is inherently ugly, but the only way to
	// pass the file URLs to ExternalInputSource, as the MCDB client
	// is nothing but an MCDB to URL converter
	// all modified parameters are untracked, so the provenance is
	// unchanged

	std::pair<std::vector<std::string>, unsigned int> result =
							getFileURLs(params);

	edm::ParameterSet newParams = params;
	newParams.addUntrackedParameter<std::vector<std::string> >(
						"fileNames", result.first);
	newParams.addUntrackedParameter<unsigned int>(
						"skipEvents", result.second);

	return newParams;
}

MCDBSource::MCDBSource(const edm::ParameterSet &params,
                       const edm::InputSourceDescription &desc) :
        LHESource(augmentPSetFromMCDB(params), desc)
{
}

MCDBSource::~MCDBSource()
{
}

DEFINE_FWK_INPUT_SOURCE(MCDBSource);
