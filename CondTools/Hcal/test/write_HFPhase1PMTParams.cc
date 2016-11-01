#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>

#include "CondTools/Hcal/interface/CmdLine.h"

#include "CondFormats/Serialization/interface/eos/portable_iarchive.hpp"
#include "CondFormats/Serialization/interface/eos/portable_oarchive.hpp"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "CondTools/Hcal/interface/make_HFPhase1PMTParams.h"

using namespace std;
using namespace cmdline;

static void print_usage(const char* progname)
{
    cout << "\nUsage: " << progname << " code outputfile\n\n"
         << "Argument \"code\" must be 0, 1, 2, or 3:\n\n"
         << "  0 -- write parameters for data (calls make_HFPhase1PMTParams_data)\n"
         << "  1 -- write parameters for MC (calls make_HFPhase1PMTParams_mc)\n"
         << "  2 -- write dummy (all pass) parameters (calls make_HFPhase1PMTParams_dummy)\n"
         << "  3 -- write parameters for testing (calls make_HFPhase1PMTParams_test)\n"
         << endl;
}

int main(int argc, char *argv[])
{
    CmdLine cmdline(argc, argv);

    if (argc == 1)
    {
        print_usage(cmdline.progname());
        return 0;
    }

    unsigned code;
    string outputfile;

    try {
        cmdline.optend();
        if (cmdline.argc() != 2)
            throw CmdLineError("wrong number of command line arguments");
        cmdline >> code >> outputfile;
    }
    catch (const CmdLineError& e) {
        std::cerr << "Error in " << cmdline.progname() << ": "
                  << e.str() << std::endl;
        return 1;
    }

    // Make the object
    std::unique_ptr<HFPhase1PMTParams> p1;
    if (code == 0)
        p1 = make_HFPhase1PMTParams_data();
    else if (code == 1)
        p1 = make_HFPhase1PMTParams_mc();
    else if (code == 2)
        p1 = make_HFPhase1PMTParams_dummy();
    else if (code == 3)
        p1 = make_HFPhase1PMTParams_test();
    else
    {
        cerr << "Error: invalid code \"" << code << "\"." << endl;
        print_usage(cmdline.progname());
        return 1;
    }

    // Are we using a text file as output?
    bool usingTxt = false;
    std::ios_base::openmode mode = std::ios::binary;
    {
        const unsigned outlen = strlen(outputfile.c_str());
        if (outlen >= 4)
            usingTxt = strcmp(".txt", outputfile.c_str() + outlen - 4) == 0;
        if (usingTxt)
            mode = std::ios_base::openmode();
    }

    // Write the object out
    {
        std::ofstream of(outputfile, mode);
        if (!of.is_open())
        {
            cerr << "Failed to open file " << outputfile << endl;
            return 1;
        }
        if (usingTxt)
        {
            boost::archive::text_oarchive ar(of);
            ar & *p1;
        }
        else
        {
            eos::portable_oarchive ar(of);
            ar & *p1;
        }
    }

    // Read it back in
    HFPhase1PMTParams p2;
    {
        std::ifstream is(outputfile, mode);
        if (usingTxt)
        {
            boost::archive::text_iarchive ar(is);
            ar & p2;
        }
        else
        {
            eos::portable_iarchive ar(is);
            ar & p2;
        }
    }

    // Make sure that they are the same
    assert(*p1 == p2);

    return 0;
}
