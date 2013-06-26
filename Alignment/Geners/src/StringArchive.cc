#include <sstream>
#include <cassert>
#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/StringArchive.hh"
#include "Alignment/Geners/interface/streamposIO.hh"

namespace gs {
    void StringArchive::search(AbsReference& reference)
    {
        std::vector<unsigned long long> idlist;
        catalog_.search(reference.namePattern(),
                        reference.categoryPattern(),
                        &idlist);
        const unsigned long nfound = idlist.size();
        for (unsigned long i=0; i<nfound; ++i)
        {
            CPP11_shared_ptr<const CatalogEntry> pentry = 
                catalog_.retrieveEntry(idlist[i]);
            if (reference.isIOCompatible(*pentry))
                addItemToReference(reference, idlist[i]);
        }
    }

    std::istream& StringArchive::inputStream(const unsigned long long id)
    {
        if (!id) throw gs::IOInvalidArgument(
            "In gs::StringArchive::inputStream: invalid item id");
        unsigned cCode;
        std::streampos pos;
        unsigned long long itemLen;
        if (!catalog_.retrieveStreampos(id, &cCode, &itemLen, &pos))
        {
            std::ostringstream os;
            os << "In gs::StringArchive::inputStream: "
               << "failed to locate item with id " << id;
            throw gs::IOInvalidArgument(os.str());
        }
        stream_.seekg(pos);
        return stream_;
    }

    bool StringArchive::isEqual(const AbsArchive& cata) const
    {
        const StringArchive& r = static_cast<const StringArchive&>(cata);
        return lastpos_ == r.lastpos_ && 
               name() == r.name() && 
               stream_ == r.stream_ &&
               catalog_ == r.catalog_;        
    }

    bool StringArchive::write(std::ostream& of) const
    {
        write_pod(of, lastpos_);
        write_pod(of, name());
        return !of.fail() &&
               stream_.classId().write(of) &&
               stream_.write(of) &&
               catalog_.classId().write(of) &&
               catalog_.write(of);
    }

    StringArchive* StringArchive::read(const ClassId& id, std::istream& in)
    {
        static const ClassId current(ClassId::makeId<StringArchive>());
        current.ensureSameId(id);

        std::streampos lastpos;
        read_pod(in, &lastpos);
        std::string nam;
        read_pod(in, &nam);
        if (in.fail()) throw IOReadFailure(
            "In gs::StringArchive::read: input stream failure");
        CPP11_auto_ptr<StringArchive> archive(new StringArchive(nam.c_str()));
        archive->lastpos_ = lastpos;
        ClassId streamId(in, 1);
        CharBuffer::restore(streamId, in, &archive->stream_);
        ClassId catId(in, 1);
        CPP11_auto_ptr<ContiguousCatalog> p(ContiguousCatalog::read(catId, in));
        assert(p.get());
        archive->catalog_ = *p;
        return archive.release();
    }
}
