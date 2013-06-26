#include "Alignment/Geners/interface/ItemLocation.hh"
#include "Alignment/Geners/interface/streamposIO.hh"
#include "Alignment/Geners/interface/IOException.hh"

namespace gs {
    bool ItemLocation::operator==(const ItemLocation& r) const
    {
        if (pos_ != r.pos_)
            return false;
        if (URI_ != r.URI_)
            return false;
        // If global URI does not exist, local URI must coincide exactly.
        // If global URI exists, local URI may or may not be defined.
        if (URI_.size())
        {
            if (cachedItemURI_.size() && r.cachedItemURI_.size())
                if (cachedItemURI_ != r.cachedItemURI_)
                    return false;
        }
        else
            if (cachedItemURI_ != r.cachedItemURI_)
                return false;
        return true;
    }

    bool ItemLocation::write(std::ostream& of) const
    {
        // The code for storing the stream offset is necessarily
        // going to be implementation-dependent. The C++ standard
        // does not specify std::streampos with enough detail.
        write_pod(of, pos_);
        write_pod(of, URI_);
        write_pod(of, cachedItemURI_);
        return !of.fail();
    }

    ItemLocation* ItemLocation::read(const ClassId& id, std::istream& in)
    {
        static const ClassId current(ClassId::makeId<ItemLocation>());
        current.ensureSameId(id);

        std::streampos pos;
        read_pod(in, &pos);

        std::string globURI;
        read_pod(in, &globURI);

        std::string cachedItemURI;
        read_pod(in, &cachedItemURI);

        if (in.fail())
            throw IOReadFailure("In ItemLocation::read: input stream failure");

        return new ItemLocation(pos, globURI.c_str(), cachedItemURI.c_str());
    }
}
