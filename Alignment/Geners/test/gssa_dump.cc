#include <map>
#include <cstring>
#include "Alignment/Geners/interface/IOException.hh"

#include "CmdLine.hh"

#include "Alignment/Geners/interface/stringArchiveIO.hh"

using namespace gs;
using namespace std;

static void print_usage(const char* progname)
{
    cout << "\nUsage: " << progname << " [-f] [-s] filename\n\n"
         << "This program prints the contents of \"geners\" string archives to the standard\n"
         << "output. Files which contain string archive can be usually recognised by their\n"
         << "extensions \".gssa\" (Generic Serialization String Archive) and \".gssaz\"\n"
         << "(string archive with compression). Normally, the program prints item ids,\n"
         << "class names, item names, and archive categories for all items in the archive.\n"
         << "This default behavior can be modified with option switches. The meaning of\n"
         << "the switches is as follows:\n\n"
         << " -f   Full dump. Print complete info for each catalog entry.\n\n"
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
    bool fullDump = false;
    bool summaryMode = false;

    try {
        fullDump    = cmdline.has("-f");
        summaryMode = cmdline.has("-s");

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

    StringArchive* ar = 0;
    try
    {
        ar = readCompressedStringArchiveExt(inputfile.c_str());
    }
    catch (std::exception& e)
    {
        cerr << "Failed to load string archive from file \""
             << inputfile << "\" (" << e.what() << ')' << endl;
        return 1;
    }

    std::map<std::string,unsigned> typecount;

    const unsigned long long first = ar->smallestId();
    const unsigned long long last = ar->largestId();    
    for (unsigned long long id = first; id <= last; ++id)
    {
        if (!ar->itemExists(id)) continue;

        CPP11_shared_ptr<const CatalogEntry> e = ar->catalogEntry(id);
        if (fullDump)
        {
            if (id != first) cout << '\n';
            e->humanReadable(cout);
        }
        else if (!summaryMode)
            cout << e->id() << "  "
                 << e->type().name() << "  " << '"' << e->name() << '"'
                 << "  " << '"' << e->category() << '"' << endl;
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
