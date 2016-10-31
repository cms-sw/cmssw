// Various standard headers
#include <iostream>
#include <string>
#include <fstream>
#include <cctype>
#include <cassert>
#include <algorithm>
#include <set>

// ROOT headers
#include "TROOT.h"
#include "TFile.h"

// Serializaton headers
#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"

// Local headers
#include "CondTools/Hcal/interface/CmdLine.h"
#include "CondTools/Hcal/interface/visualizeHFPhase1PMTParams.h"
#include "CondTools/Hcal/interface/parseHcalDetId.h"

using namespace cmdline;

static void print_usage(const char* progname,
                        const char* options,
                        const char* description)
{
    std::cout << "\nUsage: " << progname << ' ' << options
              << " [-r referenceFile] idFile cutFile outputFile\n\n"
              << "Program arguments are:\n\n"
              << "  idfile       is the name of the text file containing the list of PMTs\n"
              << "               for which the cuts should be visualized. This file should\n"
              << "               contain the list of detector ids, one per line, in the format\n\n"
              << "               ieta  iphi  depth  subdetector\n\n"
              << "               Entries in this file which do not correspond to HF will be\n"
              << "               ignored, and depth 3 and 4 will be converted into 1 and 2.\n"
              << "               Duplicate entries will be ignored as well. Character '#'\n"
              << "               at the beginning of a line can be used to add comments.\n\n"
              << "  cutfile      is the name of the binary file containing the cuts\n"
              << "               to visualize. This file should be created by the\n"
              << "               \"write_HFPhase1PMTParams\" program.\n\n"
              << "  outputFile   is the name of the root file to write\n\n"
              << "Program options are:\n\n"
              << description << '\n'
              << "  -r           (default is not to plot reference cuts) reference binary file\n"
              << std::endl;
}

static bool is_comment_or_blank(const std::string& s)
{
    if (s.empty())
        return true;
    if (s[0] == '#' || s[0] == '\0')
        return true;
    for (const char* pc = s.c_str(); *pc; ++pc)
	if (!isspace(*pc))
	    return false;
    return true;
}

static std::vector<HcalDetId> exctractHFPmtIDs(std::vector<HcalDetId>& input)
{
    std::set<HcalDetId> s;
    const unsigned n = input.size();
    for (unsigned i=0; i<n; ++i)
        if (input[i].subdet() == HcalForward)
            s.insert(input[i].baseDetId());
    return std::vector<HcalDetId>(s.begin(), s.end());
}

int main(int argc, char *argv[])
{
    using namespace std;

    // Parse the input arguments
    CmdLine cmdline(argc, argv);

    VisualizationOptions options;
    if (argc == 1)
    {
        print_usage(cmdline.progname(), options.options(), options.description());
        return 0;
    }

    // Arguments which may be specified on the command line
    std::string idFile, cutFile, outputFile, referenceFile;

    try {
        options.load(cmdline);
        cmdline.option("-r") >> referenceFile;

        cmdline.optend();
        if (cmdline.argc() != 3)
            throw CmdLineError("wrong number of command line arguments");
        cmdline >> idFile >> cutFile >> outputFile;
    }
    catch (const CmdLineError& e) {
        cerr << "Error in " << cmdline.progname() << ": " << e.str() << endl;
        return 1;
    }

    // Load the vector of ids
    vector<HcalDetId> idVec;
    {
        ifstream is(idFile.c_str());
        if (!is.is_open())
        {
            cerr << "Failed to open file \"" << idFile << "\". Exiting." << endl;
            return 1;
        }
        string line;
        for (unsigned lineNumber = 1; getline(is, line); ++lineNumber)
            if (!is_comment_or_blank(line))
            {
                HcalDetId id = parseHcalDetId(line);
                if (id.rawId())
                    idVec.push_back(id);
                else
                    cerr << "Invalid detector id entry on line " << lineNumber
                         << " : \"" << line << "\". Ignored." << endl;
            }
    }

    // We just need HF PMT ids, not all possible ids
    const vector<HcalDetId>& pmtIds = exctractHFPmtIDs(idVec);
    if (pmtIds.empty())
    {
        cerr << "No valid HF PMT ids found in file " << idFile << ". Exiting." << endl;
        return 1;
    }

    // Load the table of cuts
    HFPhase1PMTParams cuts;
    {
        std::ifstream is(cutFile.c_str(), std::ios::binary);
        if (!is.is_open())
        {
            cerr << "Failed to open file \"" << cutFile << "\". Exiting." << endl;
            return 1;
        }
        eos::portable_iarchive ar(is);
        ar & cuts;
    }

    // Load the reference table of cuts
    HFPhase1PMTParams refcuts;
    HFPhase1PMTParams* refptr = nullptr;
    if (!referenceFile.empty())
    {
        std::ifstream is(referenceFile.c_str(), std::ios::binary);
        if (!is.is_open())
        {
            cerr << "Failed to open file \"" << referenceFile << "\". Exiting." << endl;
            return 1;
        }
        eos::portable_iarchive ar(is);
        ar & refcuts;
        refptr = &refcuts;
    }

    // Initialize root
    TROOT root("showCuts", "Cut Illustrator");
    root.SetBatch(kTRUE);

    // Open the output root file. Note that, due to idiosyncratic root
    // object ownership policies, root objects (histograms, etc) allocated
    // on the heap will be owned by this file.
    TFile of(outputFile.c_str(), "RECREATE");
    if (!of.IsOpen())
    {
        cerr << "Failed to open file \"" << outputFile << "\". Exiting." << endl;
        return 1;
    }

    visualizeHFPhase1PMTParams(pmtIds, cuts, options, refptr);
    
    // Write the output file
    of.Write();
    of.Close();

    return 0;
}
