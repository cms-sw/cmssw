#include <algorithm>
#include <iterator>
#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <cstddef>
#include <cstring>

#include <TString.h>


#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/zstream.h"
#include "PhysicsTools/MVAComputer/interface/memstream.h"

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/BitSet.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

using namespace PhysicsTools;

static std::size_t getStreamSize(std::ifstream &in)
{
	in.seekg(0, std::ios::end);
	std::ifstream::pos_type end = in.tellg();
	in.seekg(0, std::ios::beg);
	std::ifstream::pos_type begin = in.tellg();

	return (std::size_t)(end - begin);
}

static Calibration::VarProcessor*
getCalibration(const std::string &file, const std::vector<std::string> &names)
{
	std::unique_ptr<Calibration::ProcExternal> calib(
					new Calibration::ProcExternal);

	std::ifstream in(file.c_str(), std::ios::binary | std::ios::in);
	if (!in.good())
		throw cms::Exception("mvaWeightsToCalibration")
			<< "Weights file \"" << file << "\" "
			   "cannot be opened for reading." << std::endl;

	char buf[512];

	while(in.good() && !TString(buf).BeginsWith("Method"))
		in.getline(buf, 512);
	if (!in.good())
		throw cms::Exception("mvaWeightsToCalibration")
			<< "Weights file \"" << file << "\" "
			   "is not a TMVA weights file." << std::endl;

	TString ls(buf);
	Int_t idx1 = ls.First(':') + 2;
	Int_t idx2 = ls.Index(' ', idx1) - idx1;
	if (idx2 < 0)
		idx2 = ls.Length();
	TString fullname = ls(idx1, idx2);
	idx1 = fullname.First(':');
	Int_t idxtit = (idx1 < 0 ? fullname.Length() : idx1);
	TString methodName = fullname(0, idxtit);

	std::size_t size = getStreamSize(in) + methodName.Length();
	for(std::vector<std::string>::const_iterator iter = names.begin();
	    iter != names.end(); ++iter)
		size += iter->size() + 1;
	size += (size / 32) + 128;

	char *buffer = nullptr;
	try {
		buffer = new char[size];
		ext::omemstream os(buffer, size);
		/* call dtor of ozs at end */ {
			ext::ozstream ozs(&os);
			ozs << methodName << "\n";
			ozs << names.size() << "\n";
			for(std::vector<std::string>::const_iterator iter =
								names.begin();
			    iter != names.end(); ++iter)
				ozs << *iter << "\n";
			ozs << in.rdbuf();
			ozs.flush();
		}
		size = os.end() - os.begin();
		calib->store.resize(size);
		std::memcpy(&calib->store.front(), os.begin(), size);
	} catch(...) {
		delete[] buffer;
		throw;
	}
	delete[] buffer;
	in.close();

	calib->method = "ProcTMVA";

	return calib.release();
}

int main(int argc, char **argv)
{
	if (argc < 4) {
		std::cerr << "Syntax: " << argv[0] << " <input> "
		          << "<output> <var1> [<var2>...]" << std::endl;
		return 1;
	}


	std::vector<std::string> names;
	for(int i = 3; i < argc; i++)
		names.push_back(argv[i]);

	try {
		std::unique_ptr<Calibration::VarProcessor> proc(
					getCalibration(argv[1], names));

		BitSet inputVars(names.size());
		for(std::size_t i = 0; i < names.size(); i++)
			inputVars[i] = true;
		proc->inputVars = Calibration::convert(inputVars);

		Calibration::MVAComputer mva;
		std::copy(names.begin(), names.end(),
		          std::back_inserter(mva.inputSet));
		mva.addProcessor(proc.get());
		mva.output = names.size();

		MVAComputer::writeCalibration(argv[2], &mva);
	} catch(cms::Exception const& e) {
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
