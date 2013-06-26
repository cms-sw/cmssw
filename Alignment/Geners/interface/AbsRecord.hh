#ifndef GENERS_ABSRECORD_HH_
#define GENERS_ABSRECORD_HH_

#include "Alignment/Geners/interface/ItemDescriptor.hh"

namespace gs {
    class AbsArchive;
    class AbsRecord;
}

gs::AbsArchive& operator<<(gs::AbsArchive& ar, const gs::AbsRecord& record);

namespace gs {
    class AbsRecord : public ItemDescriptor
    {
    public:
        inline virtual ~AbsRecord() {}

        // Item id will be set to non-0 value upon writing the item
        // into the archive. When the id is not 0, the record can no
        // longer be written out (if you really want to write out the
        // same item again, make another record).
        inline unsigned long long id() const {return itemId_;}

        // Item length will be set to non-0 value upon writing the item
        // into the archive.
        inline unsigned long long itemLength() const {return itemLength_;}

    protected:
        inline AbsRecord() : ItemDescriptor(), itemId_(0), itemLength_(0) {}
        inline AbsRecord(const ClassId& classId, const char* ioPrototype,
                         const char* name, const char* category)
            : ItemDescriptor(classId, ioPrototype, name, category),
              itemId_(0), itemLength_(0) {}

    private:
        friend gs::AbsArchive& ::operator<<(gs::AbsArchive& ar,
                                            const gs::AbsRecord& record);

        // The following functions must be overriden by derived classes.
        // "writeData" should return "true" upon success.
        virtual bool writeData(std::ostream& os) const = 0;
        
        mutable unsigned long long itemId_;
        mutable unsigned long long itemLength_;
    };
}

#endif // GENERS_ABSRECORD_HH_

