#include <iostream>
#include <sstream>

#include "Alignment/Geners/interface/Reference.hh"

#include "JetMETCorrections/FFTJetObjects/interface/loadFFTJetInterpolationTable.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"


static void dumpArchiveMetadata(gs::StringArchive& ar, std::ostream& os)
{
    const unsigned long long idSmall = ar.smallestId();
    if (!idSmall)
        os << "!!! No records in the archive !!!" << std::endl;
    else
    {
        const unsigned long long idLarge = ar.largestId();
        unsigned long long count = 0;
        for (unsigned long long id = idSmall; id <= idLarge; ++id)
            if (ar.itemExists(id))
            {
                CPP11_shared_ptr<const gs::CatalogEntry> e = 
                    ar.catalogEntry(id);
                os << '\n';
                e->humanReadable(os);
                ++count;
            }
        os << '\n' << count << " records in the archive" << std::endl;
    }
}


CPP11_auto_ptr<npstat::StorableMultivariateFunctor>
loadFFTJetInterpolationTable(const edm::ParameterSet& ps,
                             gs::StringArchive& ar, const bool verbose)
{
    gs::SearchSpecifier nameSearch(ps.getParameter<std::string>("name"),
                                   ps.getParameter<bool>("nameIsRegex"));
    gs::SearchSpecifier categorySearch(ps.getParameter<std::string>("category"),
                                       ps.getParameter<bool>("categoryIsRegex"));
    gs::Reference<npstat::StorableMultivariateFunctor> ref(
        ar, nameSearch, categorySearch);

    // Require that we get a unique item for this search
    if (!ref.unique())
    {
        std::ostringstream os;
        os << "Error in loadFFTJetInterpolationTable: table with name \""
           << nameSearch.pattern() << "\" ";
        if (nameSearch.useRegex())
            os << "(regex) ";
        os << "and category \""
           << categorySearch.pattern() << "\" ";
        if (categorySearch.useRegex())
            os << "(regex) ";
        os << "is not ";
        if (ref.empty())
            os << "found";
        else
            os << "unique";
        os << " in the archive. Archive contents are:\n";
        dumpArchiveMetadata(ar, os);
        throw cms::Exception("FFTJetBadConfig", os.str());
    }

    CPP11_auto_ptr<npstat::StorableMultivariateFunctor> p = ref.get(0);
    if (verbose)
    {
        std::cout << "In loadFFTJetInterpolationTable: loaded table with metadata"
                  << std::endl;
        CPP11_shared_ptr<const gs::CatalogEntry> e = ref.indexedCatalogEntry(0);
        e->humanReadable(std::cout);
        std::cout << std::endl;
        std::cout << "Actual table class name is \""
                  << p->classId().name() << '"' << std::endl;
    }
    return p;
}
