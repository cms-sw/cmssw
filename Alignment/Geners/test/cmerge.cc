// The following program merges contents of several object catalogs
// stored in "geners" binary metafiles

#include <iostream>
#include <fstream>
#include <vector>

#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/ContiguousCatalog.hh"
#include "Alignment/Geners/interface/CatalogIO.hh"
#include "Alignment/Geners/interface/uriUtils.hh"

#include "CmdLine.hh"

using namespace gs;
using namespace std;


static void print_usage(const char* progname)
{
    cout << "\nUsage: " << progname << " -o output_file input_file0 input_file1 ...\n\n"
         << "This program merges the contents of several object catalogs stored in \"geners\"\n"
         << "binary metafiles. The output metafile can be used as a single \"geners\" archive\n"
         << "which includes all items from the merged catalogs.\n"
         << endl;
}


static const char* update_uri(const char* inputfile,
                              const std::string& originalUri)
{
    static std::string result, old_inputfile, old_URI;

    if (old_inputfile != inputfile || old_URI != originalUri)
    {
        old_inputfile = inputfile;
        old_URI = originalUri;
        result = joinDir1WithName2(inputfile, originalUri.c_str());
    }
    return result.c_str();
}


static ItemLocation update_location(const char* inputfile,
                                    const ItemLocation& original)
{
    ItemLocation loc(original);
    loc.setURI(update_uri(inputfile, original.URI()));
    return loc;
}


int main(int argc, char const* argv[])
{
    CmdLine cmdline(argc, argv);

    if (argc == 1)
    {
        print_usage(cmdline.progname());
        return 0;
    }

    std::string outputfile;
    std::vector<std::string> inputfiles;

    try {
        cmdline.require("-o") >> outputfile;
        cmdline.optend();
        
        while (cmdline)
        {
            std::string s;
            cmdline >> s;
            inputfiles.push_back(s);
        }

        if (inputfiles.empty())
            throw CmdLineError("must specify at least one input file");
    }
    catch (CmdLineError& e) {
        cerr << "Error in " << cmdline.progname() << ": "
             << e.str() << endl;
        print_usage(cmdline.progname());
        return 1;
    }

    // Before we perform any significant data processing,
    // make sure that we can open all input files
    const unsigned nfiles = inputfiles.size();
    for (unsigned ifile=0; ifile<nfiles; ++ifile)
    {
        const std::string& inputfile(inputfiles[ifile]);
        ifstream in(inputfile.c_str(), ios_base::binary);
        if (!in.is_open())
        {
            cerr << "Error: failed to open file \""
                 << inputfile << "\"" << endl;
            return 1;
        }
    }

    // Variables which summarize the combined catalog
    unsigned totalMergeLevel = 0, lastCompressionCode = 0;
    bool compressionCodeMixed = false;
    std::vector<std::string> allAnnotations;
    ContiguousCatalog merged;

    // Now, do the real cycle over the input files
    for (unsigned ifile=0; ifile<nfiles; ++ifile)
    {
        const std::string& inputfile(inputfiles[ifile]);
        const unsigned long long fileOffset = merged.size();

        ifstream in(inputfile.c_str(), ios_base::binary);
        if (!in.is_open())
        {
            cerr << "Error: failed to open file \""
                 << inputfile << "\"" << endl;
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

        // Update compression code info
        if (ifile)
            if (compressionCode != lastCompressionCode)
                compressionCodeMixed = true;
        lastCompressionCode = compressionCode;

        // Update merge level
        if (mergeLevel == 0) ++mergeLevel;
        totalMergeLevel += mergeLevel;

        // Update annotations
        std::copy(annotations.begin(), annotations.end(),
                  std::back_inserter(allAnnotations));

        const unsigned long long last = cat->largestId();
        for (unsigned long long id=cat->smallestId(); id<=last; ++id)
        {
            if (!cat->itemExists(id)) continue;
            CPP11_shared_ptr<const CatalogEntry> e=cat->retrieveEntry(id);
            const unsigned long long newid = merged.makeEntry(
                *e, e->compressionCode(), e->itemLength(),
                update_location(inputfile.c_str(),e->location()),
                e->offset() + fileOffset);
            assert(newid);
        }
    }

    ofstream of(outputfile.c_str(), ios_base::binary);
    if (!of.is_open())
    {
        cerr << "Error: failed to open file \""
             << outputfile << "\"" << endl;
        return 1;
    }
    const unsigned compress = compressionCodeMixed ? 1 : lastCompressionCode;
    if (!writeBinaryCatalog(of, compress, totalMergeLevel,
                            allAnnotations, merged))
    {
        cerr << "Error: failed to write merged catalog to file \""
             << outputfile << "\"" << endl;
        return 1;
    }
    of.close();
 
    return 0;
}
