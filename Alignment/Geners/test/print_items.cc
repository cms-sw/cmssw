//
// It is expected that the code of this program will be extended by the
// users of the "geners" package as they develop their storable classes.
// To add printing capability for archived objects of a new class,
// make sure that "operator<<" is defined between std::ostream and this
// class. Then add the relevant header file below and insert the
// "printable_type" statement for your class together with other similar
// statements already present in this program.
//

#include <set>
#include <map>
#include <iostream>

#include "Alignment/Geners/interface/ClassId.hh"
#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/MultiFileArchive.hh"
#include "Alignment/Geners/interface/complexIO.hh"
#include "Alignment/Geners/interface/Reference.hh"

#include "CmdLine.hh"

using namespace gs;
using namespace std;

static void print_usage(const char* progname)
{
    cout << "\nUsage: " << progname << " [-n] [-c] archive_name item_name item_category\n\n"
         << "Optional switches \"-n\" and \"-c\" turn on regular expression search for item\n"
         << "name and category, respectively. For each matching item, the program prints\n"
         << "its class and its text representation, as defined by \"operator <<\".\n"
         << endl;
}

namespace {
    typedef unsigned long (*PrinterFunction)(
        MultiFileArchive&, const SearchSpecifier&, const SearchSpecifier&);

    template <class T>
    struct Printer
    {
        static unsigned long print(MultiFileArchive& ar,
                                   const SearchSpecifier& name,
                                   const SearchSpecifier& category)
        {
            ClassId id(ClassId::makeId<T>());
            Reference<T> ref(ar, name, category);
            const unsigned long nItems = ref.size();
            for (unsigned long i=0; i<nItems; ++i)
                cout << id.name() << "  " << *ref.get(i) << endl;
            return nItems;
        }
    };
}

#define printable_type(sometype) do {                 \
    ClassId id(ClassId::makeId< sometype >());        \
    typemap[id.name()] = &Printer< sometype >::print; \
} while(0);

int main(int argc, char const* argv[])
{
    typedef std::map<std::string,PrinterFunction> Typemap;

    CmdLine cmdline(argc, argv);
    if (argc == 1)
    {
        print_usage(cmdline.progname());
        return 0;
    }

    bool useRegexForName = false, useRegexForCategory = false;
    std::string archiveName, nameIn, categoryIn;

    try {
        useRegexForName = cmdline.has("-n");
        useRegexForCategory = cmdline.has("-c");

        cmdline.optend();

        const unsigned cmdargc = cmdline.argc();
        if (cmdargc != 3)
            throw CmdLineError("wrong number of command line arguments");
        cmdline >> archiveName >> nameIn >> categoryIn;
    }
    catch (CmdLineError& e) {
        cerr << "Error in " << cmdline.progname() << ": "
             << e.str() << endl;
        print_usage(cmdline.progname());
        return 1;
    }

    MultiFileArchive mar(archiveName.c_str(), "r");
    if (!mar.isOpen())
    {
        cerr << mar.error() << endl;
        return 1;
    }

    SearchSpecifier name(nameIn, useRegexForName);
    SearchSpecifier category(categoryIn, useRegexForCategory);
    std::vector<unsigned long long> found;
    mar.itemSearch(name, category, &found);
    const unsigned long nfound = found.size();
    if (!nfound)
    {
        cout << "No items found" << endl;
        return 0;
    }

    Typemap typemap;

    // You can add more printable types to the following collection
    // as long as the type supports "<<" operator with ostream
    printable_type(bool);
    printable_type(char);
    printable_type(unsigned char);
    printable_type(signed char);
    printable_type(short);
    printable_type(unsigned short);
    printable_type(int);
    printable_type(long);
    printable_type(long long);
    printable_type(unsigned);
    printable_type(unsigned long);
    printable_type(unsigned long long);
    printable_type(float);
    printable_type(double);
    printable_type(long double);
    printable_type(std::complex<float>);
    printable_type(std::complex<double>);
    printable_type(std::complex<long double>);
    printable_type(std::string);


    // We need to call the generic printer exactly once for each distinct
    // type of the items satisfying the search criterion
    std::set<std::string> typenames;
    unsigned long nprinted = 0;
    for (unsigned long i=0; i<nfound; ++i)
    {
        CPP11_shared_ptr<const CatalogEntry> entry = 
            mar.catalogEntry(found[i]);
        const ClassId& type = entry->type();
        const std::string& tname = type.name();
        if (typenames.insert(tname).second)
        {
            Typemap::iterator it = typemap.find(tname);
            if (it != typemap.end())
                nprinted += (*it->second)(mar, name, category);
        }
    }
    if (nprinted != nfound)
    {
        const unsigned long notprinted = nfound - nprinted;
        cout << "Found " << notprinted << " non-printable item" 
             << (notprinted == 1UL ? "" : "s") << endl;
    }

    return 0;
}
