#include <string>
#include <vector>
#include <memory>
#include <ostream>

#include <boost/shared_ptr.hpp>

#include <TFile.h>
#include <TDirectory.h>
#include <TObject.h>

#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVATrainer/interface/TrainerMonitoring.h"

namespace { // anonymous

class ROOTContextSentinel {
    public:
	ROOTContextSentinel() : dir(gDirectory), file(gFile) {}
	~ROOTContextSentinel() { gDirectory = dir; gFile = file; }

    private:
	TDirectory	*dir;
	TFile		*file;
};

} // anonymous namespace

namespace PhysicsTools {

TrainerMonitoring::TrainerMonitoring(const std::string &fileName)
{
	ROOTContextSentinel ctx;

	rootFile.reset(TFile::Open(fileName.c_str(), "RECREATE"));
}

TrainerMonitoring::~TrainerMonitoring()
{
}

void TrainerMonitoring::write()
{
	ROOTContextSentinel ctx;

	typedef std::map<std::string, boost::shared_ptr<Module> > Map;
	for(Map::const_iterator iter = modules.begin();
	    iter != modules.end(); ++iter) {
		rootFile->cd();
		TDirectory *dir = rootFile->mkdir(iter->first.c_str());
		dir->cd();
		iter->second->write(dir);
	}
}

TrainerMonitoring::Module::Module()
{
}

TrainerMonitoring::Module::~Module()
{
}

void TrainerMonitoring::Module::write(TDirectory *dir)
{
	typedef std::map<std::string, boost::shared_ptr<Object> > Map;
	for(Map::const_iterator iter = data.begin();
	    iter != data.end(); ++iter)
		iter->second->write(dir);
}

void TrainerMonitoring::Module::add(Object *object)
{
	boost::shared_ptr<Object> ptr(object);
	if (!data.insert(std::make_pair(object->getName(), ptr)).second)
		throw cms::Exception("DuplicateNode")
			<< "Node \"" << object->getName() << "\" already"
			   " exists." << std::endl;
}

TrainerMonitoring::Module *TrainerMonitoring::book(const std::string &name)
{
	boost::shared_ptr<Module> module(new Module);
	if (!modules.insert(std::make_pair(name, module)).second)
		throw cms::Exception("DuplicateModule")
			<< "Module \"" << name << "\" already"
			   " exists." << std::endl;

	return module.get();
}

} // namespace PhysicsTools
