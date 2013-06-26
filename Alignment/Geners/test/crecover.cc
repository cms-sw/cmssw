// This is the catalog recovery program. For it to be successful, the
// original archive must have been created using the "cat=s" option.

#include <iostream>
#include <fstream>

#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/BinaryArchiveBase.hh"
#include "Alignment/Geners/interface/ContiguousCatalog.hh"
#include "Alignment/Geners/interface/CatalogIO.hh"
#include "Alignment/Geners/interface/IOException.hh"
#include "Alignment/Geners/interface/streamposIO.hh"

#include "CmdLine.hh"

using namespace gs;
using namespace std;

static void print_usage(const char* progname)
{
    cout << "\nUsage: " << progname << " archive_name [annotation]\n\n"
         << "This program recovers the \"geners\" catalog data from binary archives.\n"
         << "The recovery is possible only if the archive mode string included the\n"
         << "\"cat=i\" section which instructs the binary archive to write a duplicate\n"
         << "copy of item metadata into the data stream. This mode of operation\n"
         << "is recommended in the development stage of your project, so that the\n"
         << "catalog can be recovered even after an unexpected program crash.\n\n"
         << "The first command line argument is the name of the \"geners\" archive\n"
         << "without extension or data file number. The second argument is optional.\n"
         << "If provided, it will become the annotation string added to the recovered\n"
         << "catalog.\n"
         << endl;
}

namespace {
    class CatalogRecovery : public BinaryArchiveBase
    {
        inline std::ostream& plainOutputStream() {return f_;}

        inline std::istream& plainInputStream(
            unsigned long long /* id */, unsigned* /* compressionCode */,
            unsigned long long* /* length */) {return f_;}

        inline void openFile(const unsigned long count)
        {
            std::ostringstream os;
            os << AbsArchive::name() << '.' << count << ".gsbd";
            const std::string& fname = os.str();
            openDataFile(f_, fname.c_str());
            cout << "Processing file " << fname << "\n";
            cout.flush();
        }

        inline unsigned long long addToCatalog(
            const AbsRecord& /* record */, unsigned /* compressCode */,
            unsigned long long /* itemLength */)
        {
            return 0ULL;
        }

        fstream f_;
        ContiguousCatalog cat_;

    public:
        inline CatalogRecovery(const char* name)
            : BinaryArchiveBase(name, "r") {}

        inline ~CatalogRecovery() {if (f_.is_open()) f_.close();}

        inline std::string catalogFileName() const
        {
            return AbsArchive::name() + ".gsbmf";
        }

        inline unsigned long long processFile(const unsigned long fileNumber)
        {
            openFile(fileNumber);
            if (!injectMetadata())
                throw IOInvalidData("Item metadata was not written into the data stream. "
                                    "Catalog can not be recovered.");
            assert(catalogEntryClassId());
            assert(itemLocationClassId());

            const std::streampos zeropos(0);
            unsigned long long count = 0;
            for (f_.peek(); !f_.eof(); f_.peek())
            {
                unsigned long long id = 0;
                std::streampos jump(0);
                read_pod(f_, &jump);
                if (f_.fail() || jump == zeropos) return count;
                f_.seekp(jump);
                if (f_.fail()) return count;
                const CatalogEntry* entry = CatalogEntry::read(
                    *catalogEntryClassId(), *itemLocationClassId(), f_);
                if (entry)
                {
                    id = cat_.makeEntry(*entry, entry->compressionCode(),
                                        entry->itemLength(), entry->location(),
                                        entry->offset());
                    delete entry;
                }
                if (!id) return count;
                ++count;
            }
            return count;
        }

        inline void writeCatalog(const std::vector<std::string>& annotations)
        {
            const std::string& catFile = catalogFileName();
            ofstream cats(catFile.c_str());
            if (!cats)
            {
                std::ostringstream os;
                os << "In CatalogRecovery::writeCatalog: "
                   << "failed to open catalog data file \""
                   << catFile << '"';
                throw IOOpeningFailure(os.str());
            }

            const unsigned compress = static_cast<unsigned>(compressionMode());
            if (!writeBinaryCatalog(cats, compress, 1, annotations, cat_))
            {
                std::ostringstream os;
                os << "In CatalogRecovery::writeCatalog: "
                   << "failed to write catalog data to file "
                   << catFile;
                throw IOWriteFailure(os.str());
            }
        }

        inline void flush() {}
    };
}

int main(int argc, char const* argv[])
{
    CmdLine cmdline(argc, argv);

    if (argc == 1)
    {
        print_usage(cmdline.progname());
        return 0;
    }

    std::string archiveName;
    std::vector<std::string> annotations;
    try {
        cmdline.optend();

        const unsigned cmdargc = cmdline.argc();
        if (cmdargc != 1 && cmdargc != 2)
            throw CmdLineError("wrong number of command line arguments");
        cmdline >> archiveName;
        if (cmdargc > 1)
        {
            annotations.resize(1);
            cmdline >> annotations[0];
        }
    }
    catch (CmdLineError& e) {
        cerr << "Error in " << cmdline.progname() << ": "
             << e.str() << endl;
        print_usage(cmdline.progname());
        return 1;
    }

    CatalogRecovery recovery(archiveName.c_str());

    // Check if the catalog file already exists. Do not overwrite.
    const std::string& catFile = recovery.catalogFileName();
    {
        ifstream cats(catFile.c_str());
        if (cats)
        {
            cout << "Catalog file \"" << catFile << "\" already exists" << endl;
            return 1;
        }
    }

    unsigned long long itemCount = 0;
    unsigned long fileCount = 0;
    try {
        for (; ; ++fileCount)
            itemCount += recovery.processFile(fileCount);
    }
    catch (IOOpeningFailure& e) {
        if (!fileCount)
        {
            cout << "Archive \"" << archiveName << "\" could not be open. "
                 << e.what() << endl;
            return 1;
        }
    }
    catch (exception& e) {
        cout << "Error in data processing cycle. "
             << e.what() << endl;
        return 1;
    }

    try {
        recovery.writeCatalog(annotations);
    }
    catch (exception& e) {
        cout << e.what() << endl;
        return 1;
    }

    cout << "Wrote catalog file \"" << catFile << "\"\n"
         << "The catalog contains metadata for " << itemCount
         << " items found in " << fileCount << " data file"
         << (fileCount == 1UL ? "" : "s") << endl;
    return 0;
}
