#include <iostream>
#include <fstream>
#include <utility>
#include <sstream>
#include <string>
#include <vector>


#include "FWCore/Utilities/interface/Exception.h"

#include "PhysicsTools/MVAComputer/interface/Calibration.h"
#include "PhysicsTools/MVAComputer/interface/BitSet.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputer.h"

using namespace PhysicsTools;

static std::string escape(const std::string &arg)
{
	std::string result;
	for(std::string::const_iterator c = arg.begin(); c != arg.end(); ++c)
		switch(*c) {
		    case '<':
			result += "&lt;";
			break;
		    case '>':
			result += "&gt;";
			break;
		    case '&':
			result += "&amp;";
			break;
		    case '"':
			result += "&quot;";
			break;
		    default:
			result += *c;
		}

	return result;
}

static std::string mkNumber(const char *prefix, unsigned int index)
{
	std::ostringstream ss;
	ss << prefix << index;
	return ss.str();
}

int main(int argc, char **argv)
{
	if (argc != 3) {
		std::cerr << "Syntax: " << argv[0] << " <input.mva> "
		             "<output.xml>\n";
		return 1;
	}


	try {
		Calibration::MVAComputer *calib =
				MVAComputer::readCalibration(argv[1]);
		if (!calib) {
			std::cerr << "MVA calibration could not be read."
		                  << std::endl;
			return 1;
		}

		std::ofstream out(argv[2]);
		if (!out.good()) {
			std::cerr << "XML description file could not be "
			             "opened for writing."
		                  << std::endl;
			return 1;
		}

		out << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>" << std::endl;
		out << "<MVATrainer>" << std::endl;
		out << "\t<!-- Warning: Auto-generated file from MVA calibration extractor. -->" << std::endl;
		out << "\t<!--          This trainer configuration is incomplete! -->" << std::endl;
		out << "\t<general>" << std::endl;
		out << "\t\t<option name=\"trainfiles\">train_%1$s%2$s.%3$s</option>" << std::endl;
		out << "\t</general>" << std::endl;

		std::vector< std::pair<std::string, std::string> > vars;

		out << "\t<input id=\"input\">" << std::endl;
		for(std::vector<Calibration::Variable>::const_iterator iter =
						calib->inputSet.begin();
		    iter != calib->inputSet.end(); ++iter) {
			
			out << "\t\t<var name=\"" << escape(iter->name)
			    << "\" multiple=\"true\" optional=\"true\"/>"
			    << std::endl;
			vars.push_back(std::make_pair("input", iter->name));
		}
		out << "\t</input>" << std::endl;

		unsigned int procId = 0;
		std::string proc = "input";
		std::vector<Calibration::VarProcessor*> procs =
						calib->getProcessors();
		for(std::vector<Calibration::VarProcessor*>::const_iterator
			iter = procs.begin(); iter != procs.end(); ++iter) {
			std::string name = (*iter)->getInstanceName();

			PhysicsTools::BitSet in =
				PhysicsTools::Calibration::convert(
							(*iter)->inputVars);

			int j = 0;
			for(unsigned int i = vars.size(); i < in.size(); i++) {
				std::string var = mkNumber("var", ++j);
				out << "\t\t\t<var name=\"" << var
				    << "\"/>" << std::endl;

				vars.push_back(std::make_pair(proc, var));
			}

			if (iter != procs.begin()) {
				out << "\t\t</output>" << std::endl;
				out << "\t</processor>" << std::endl;
			}

			proc = mkNumber("proc", ++procId);
			out << "\t<processor id=\"" << proc
			    << "\" name=\"" << name << "\">" << std::endl;
			out << "\t\t<input>" << std::endl;

			for(unsigned int i = 0; i < in.size(); i++) {
				if (in[i])
					out << "\t\t\t<var source=\""
					    << escape(vars.at(i).first)
					    << "\" name=\""
					    << escape(vars.at(i).second)
					    << "\"/>" << std::endl;
			}

			out << "\t\t</input>" << std::endl;
			out << "\t\t<config>" << std::endl;
			out << "\t\t\t<!-- FILL ME -->" << std::endl;
			out << "\t\t</config>" << std::endl;
			out << "\t\t<output>" << std::endl;
		}

		int j = 0;
		for(unsigned int i = vars.size(); i <= calib->output; i++) {
			std::string var = mkNumber("var", ++j);

			out << "\t\t\t<var name=\"" << var
			    << "\"/>" << std::endl;

			vars.push_back(std::make_pair(proc, var));
		}

		if (!procs.empty()) {
			out << "\t\t</output>" << std::endl;
			out << "\t</processor>" << std::endl;
		}

		out << "\t<output>" << std::endl;
		out << "\t\t<var source=\""
		    << escape(vars.at(calib->output).first)
		    << "\" name=\""
		    << escape(vars.at(calib->output).second)
		    << "\"/>" << std::endl;
		out << "\t</output>" << std::endl;

		out << "</MVATrainer>" << std::endl;
	} catch(cms::Exception e) {
		std::cerr << e.what() << std::endl;
	}

	return 0;
}
