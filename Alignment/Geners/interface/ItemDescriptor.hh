// Minimal item descriptor for archive I/O

#ifndef GENERS_ITEMDESCRIPTOR_HH_
#define GENERS_ITEMDESCRIPTOR_HH_

#include <string>
#include <utility>
#include <typeinfo>

#include "Alignment/Geners/interface/ClassId.hh"

namespace gs {
    class ItemDescriptor
    {
    public:
        ItemDescriptor();
        ItemDescriptor(const ClassId& classId, const char* ioPrototype,
                       const char* name, const char* category);
        inline virtual ~ItemDescriptor() {}

        inline const ClassId& type() const {return classId_;}
        inline const std::string& ioPrototype() const {return ioProto_;}
        inline const std::string& name() const {return nameCat_.first;}
        inline const std::string& category() const {return nameCat_.second;}
        inline const std::pair<std::string,std::string>& nameAndCategory()
            const {return nameCat_;}

        inline bool operator==(const ItemDescriptor& r) const
            {return (typeid(*this) == typeid(r)) && this->isEqual(r);}
        inline bool operator!=(const ItemDescriptor& r) const
            {return !(*this == r);}

        // The following returns "true" if the class id and
        // I/O prototype of this item coincide with those of
        // the argument
        bool isSameClassIdandIO(const ItemDescriptor& r) const;

        // The following method checks I/O prototype only
        // allowing for class id mismatch
        inline bool isSameIOPrototype(const ItemDescriptor& r) const
            {return ioProto_ == r.ioProto_;}

    protected:
        virtual bool isEqual(const ItemDescriptor&) const;

    private:
        ClassId classId_;
        std::string ioProto_;
        std::pair<std::string,std::string> nameCat_;
    };
}

#endif // GENERS_ITEMDESCRIPTOR_HH_

