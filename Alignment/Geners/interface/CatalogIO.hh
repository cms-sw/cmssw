//=========================================================================
// CatalogIO.hh
//
// The following functions provide an operational definition of
// the "standard" binary catalog file format
//
// I. Volobouev
// November 2010
//=========================================================================

#ifndef GENERS_CATALOGIO_HH_
#define GENERS_CATALOGIO_HH_

#include <iostream>

#include "Alignment/Geners/interface/AbsCatalog.hh"

namespace gs {
    struct CatalogFormat1
    {
        enum {ID = 713265489};
    };

    // In the following, it is assumed that the catalog is stored
    // in memory and that it has a stream dedicated to it. The
    // function returns "true" on success.
    //
    // Programs which make new catalogs should set "mergeLevel" to 1.
    // Programs which combine existing catalogs should add up the
    // merge levels of those catalogs to come up with the "mergeLevel".
    //
    // "annotations" are arbitrary strings (which should be just combined
    // when catalogs are merged).
    bool writeBinaryCatalog(std::ostream& os, unsigned compressionCode,
                            unsigned mergeLevel,
                            const std::vector<std::string>& annotations,
                            const AbsCatalog& catalog,
                            unsigned formatId = CatalogFormat1::ID);

    // In the following, it is assumed that the Catalog class
    // has a "read" function which builds the catalog on the heap.
    // The "allowReadingByDifferentClass" parameter specifies
    // whether the catalog class is allowed to read something
    // written by another catalog class. This function returns NULL
    // pointer on failure. Note that the user must call "delete"
    // on the returned pointer at some point in the future.
    template <class Catalog>
    Catalog* readBinaryCatalog(std::istream& is, unsigned* compressionCode,
                               unsigned* mergeLevel,
                               std::vector<std::string>* annotations,
                               bool allowReadingByDifferentClass,
                               unsigned formatId = CatalogFormat1::ID);
}

#include <sstream>
#include <cassert>

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

namespace gs {
    template <class Catalog>
    Catalog* readBinaryCatalog(std::istream& is, unsigned* compressionCode,
                               unsigned* mergeLevel,
                               std::vector<std::string>* annotations,
                               const bool allowReadingByDifferentClass,
                               const unsigned expectedformatId)
    {
        assert(compressionCode);
        assert(mergeLevel);
        assert(annotations);

        is.seekg(0, std::ios_base::beg);

        unsigned formatId = 0, endianness = 0;
        unsigned char sizelong = 0;
        read_pod(is, &formatId);
        read_pod(is, &endianness);
        read_pod(is, &sizelong);

        if (is.fail()) throw IOReadFailure(
            "In gs::readBinaryCatalog: input stream failure");

        if (endianness != 0x01020304 ||
              formatId != expectedformatId ||
              sizelong != sizeof(long))
            throw IOInvalidData("In gs::readBinaryCatalog: not \"geners\" "
                                "binary metafile or incompatible system "
                                "architecture");

        read_pod(is, compressionCode);
        read_pod(is, mergeLevel);
        read_pod_vector(is, annotations);
        ClassId id(is, 1);
        Catalog* readback = 0;

        ClassId catId(ClassId::makeId<Catalog>());
        if (id.name() == catId.name())
            // The reading is done by the same class as the writing.
            // Make sure the "read" function gets the correct class version.
            readback = Catalog::read(id, is);
        else
        {
            if (!allowReadingByDifferentClass)
            {
                std::ostringstream os;
                os << "In gs::readBinarCatalog: incompatible "
                   << "catalog class: written by \"" << id.name()
                   << "\", but reading is attempted by \""
                   << catId.name() << '"';
                throw IOInvalidData(os.str());
            }

            // The reading is not done by the same class as the writing.
            // All bets are off, and it is up to the user to decide whether
            // this makes sense. However, to maintain compatibility with
            // version 1 archives, we need to pass this version to the
            // catalog read function.
            if (id.version() == 1)
                catId.setVersion(1);

            readback = Catalog::read(catId, is);
        }

        // Catalogs do not necessarily know their size in advance,
        // so that they might read until the end of file is encountered.
        // However, the eof flag on the stream can result in various
        // problems later (for example, in writing to the stream).
        // Remove this flag.
        if (is.eof() && !is.fail() && !is.bad())
        {
            is.clear();
            is.seekg(0, std::ios_base::end);
        }

        return readback;
    }
}


#endif // GENERS_CATALOGIO_HH_

