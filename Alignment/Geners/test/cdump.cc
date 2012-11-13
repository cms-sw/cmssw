// The following program dumps contents of an object catalog stored
// in a "geners" binary metafile to the standard output

#include <map>
#include <fstream>
#include <iostream>
#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/ContiguousCatalog.hh"
#include "Alignment/Geners/interface/CatalogIO.hh"
#include "Alignment/Geners/interface/CStringStream.hh"

#include "CmdLine.hh"

using namespace gs;
using namespace std;

static void print_usage(const char* progname)
{
    cout << "\nUsage: " << progname << " [-a] [-c] [-f] [-i] [-s] filename\n\n"
         << "This program prints the contents of \"geners\" catalog files to the standard\n"
         << "output. These files can be usually recognized by their \".gsbmf\" extension\n"
         << "(Generic Serialization Binary MetaFile). Normally, the program prints class\n"
         << "names, item names in the archive, and archive categories for all items in\n"
         << "the catalog. This default behavior can be modified with option switches.\n"
         << "The meaning of the switches is as follows:\n\n"
         << " -a   Print catalog annotations, if any.\n\n"
         << " -c   Print default archive compression mode.\n\n"
         << " -f   Full dump. Print complete info for each catalog entry.\n\n"
         << " -i   Include the catalog item ids into the printout.\n\n"
         << " -s   Print only the summary statistics for item types. If option \"-f\" is given\n"
         << "      together with \"-s\", the summary will be printed after the full dump.\n"
         << endl;
}

int main(int argc, char const* argv[])
{
    CmdLine cmdline(argc, argv);

    if (argc == 1)
    {
        print_usage(cmdline.progname());
        return 0;
    }

    std::string inputfile;
    bool printAnnotations = false;
    bool printCompressionMode = false;
    bool fullDump = false;
    bool printIds = false;
    bool summaryMode = false;

    try {
        printAnnotations     = cmdline.has("-a");
        printCompressionMode = cmdline.has("-c");
        fullDump             = cmdline.has("-f");
        printIds             = cmdline.has("-i");
        summaryMode          = cmdline.has("-s");

        cmdline.optend();

        if (cmdline.argc() != 1)
            throw CmdLineError("wrong number of command line arguments");
        cmdline >> inputfile;
    }
    catch (CmdLineError& e) {
        cerr << "Error in " << cmdline.progname() << ": "
             << e.str() << endl;
        print_usage(cmdline.progname());
        return 1;
    }

    ifstream in(inputfile.c_str(), ios_base::binary);
    if (!in.is_open())
    {
        cerr << "Error: failed to open file \"" << inputfile << "\"" << endl;
        return 1;
    }

    unsigned compressionCode = 0, mergeLevel = 0;
    std::vector<std::string> annotations;
    CPP11_auto_ptr<ContiguousCatalog> cat;
    try 
    {
        cat = CPP11_auto_ptr<ContiguousCatalog>(readBinaryCatalog<ContiguousCatalog>(
                      in, &compressionCode, &mergeLevel, &annotations, true));
    }
    catch (std::exception& e)
    {
        cerr << "Failed to read catalog from file \""
             << inputfile << "\". " << e.what() << endl;
        return 1;
    }

    if (printCompressionMode)
    {
        CStringStream::CompressionMode mode = 
            static_cast<CStringStream::CompressionMode>(compressionCode);
        cout << "Default compression mode: "
             << CStringStream::compressionModeName(mode, false) << endl;
    }
    if (printAnnotations)
    {
        const unsigned nAnnotations = annotations.size();
        for (unsigned i=0; i<nAnnotations; ++i)
        {
            if (i) cout << '\n';
            cout << "Annotation " << i << ": " << annotations[i] << endl;
        }
        if (!nAnnotations)
            cout << "This catalog does not have any annotations" << endl;
    }

    std::map<std::string,unsigned> typecount;

    const unsigned long long first = cat->smallestId();
    const unsigned long long last = cat->largestId();    
    for (unsigned long long id = first; id <= last; ++id)
    {
        if (!cat->itemExists(id)) continue;

        CPP11_shared_ptr<const CatalogEntry> e = cat->retrieveEntry(id);
        if (fullDump)
        {
            if (id != first) cout << '\n';
            e->humanReadable(cout);
        }
        else if (!summaryMode)
        {
            if (printIds)
                cout << e->id() << "  ";
            cout << e->type().name() << "  " << '"' << e->name() << '"'
                 << "  " << '"' << e->category() << '"' << endl;
        }
        if (summaryMode)
            typecount[e->type().name()]++;
    }

    if (summaryMode)
    {
        if (fullDump)
            cout << '\n';
        for (std::map<std::string,unsigned>::const_iterator it = typecount.begin();
             it != typecount.end(); ++it)
            cout << it->second << ' ' << it->first << endl;
    }

    return 0;
}
