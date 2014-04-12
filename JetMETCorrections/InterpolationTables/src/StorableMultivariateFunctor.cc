#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "JetMETCorrections/InterpolationTables/interface/StorableMultivariateFunctorReader.h"

namespace npstat {
    void StorableMultivariateFunctor::validateDescription(
        const std::string& description) const
    {
        if (description_ != description) 
        {
            std::string mesage = 
                "In StorableMultivariateFunctor::validateDescription: "
                "argument description string \"";
            mesage += description;
            mesage += "\" is different from the object description string \"";
            mesage += description_;
            mesage += "\"";
            throw npstat::NpstatRuntimeError(mesage.c_str());
        }
    }

    StorableMultivariateFunctor* StorableMultivariateFunctor::read(
        const gs::ClassId& id, std::istream& in)
    {
        return StaticStorableMultivariateFunctorReader::instance().read(id, in);
    }
}
