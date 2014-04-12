#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/ItemDescriptor.hh"

namespace gs {
    ItemDescriptor::ItemDescriptor()
        : classId_(ClassId::invalidId())
    {
    }

    ItemDescriptor::ItemDescriptor(
        const ClassId& classId, const char* ioPrototype,
        const char* name, const char* categ)
        : classId_(classId),
          ioProto_(ioPrototype ? ioPrototype : ""),
          nameCat_(name ? std::string(name) : std::string(""),
                   categ ? std::string(categ) : std::string(""))
    {
        if (classId_.name().empty()) throw gs::IOInvalidArgument(
            "In ItemDescriptor constructor: invalid class id");
    }

    bool ItemDescriptor::isSameClassIdandIO(const ItemDescriptor& r) const
    {
        return !classId_.name().empty() && 
               classId_.name() == r.classId_.name() &&
               ioProto_ == r.ioProto_;
    }

    bool ItemDescriptor::isEqual(const ItemDescriptor& r) const
    {
        return !classId_.name().empty() && 
               classId_ == r.classId_ && 
               ioProto_ == r.ioProto_ &&
               nameCat_ == r.nameCat_;
    }
}
