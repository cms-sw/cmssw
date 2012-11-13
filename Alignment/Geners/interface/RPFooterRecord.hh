#ifndef GENERS_RPFOOTERRECORD_HH_
#define GENERS_RPFOOTERRECORD_HH_

#include "Alignment/Geners/interface/AbsRecord.hh"
#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    namespace Private {
        template<class Ntuple>
        class RPFooterRecord : public AbsRecord
        {
        public:
            inline RPFooterRecord(const Ntuple& obj)
                : AbsRecord(obj.classId(), "gs::RPFooter",
                            obj.name_.c_str(), obj.category_.c_str()),
                  obj_(obj) {}

            inline bool writeData(std::ostream& os) const
            {
                write_pod(os, obj_.fillCount_);
                write_pod(os, obj_.headerSaved_);
                return write_item(os, obj_.idlist_, false);
            }

        private:
            RPFooterRecord();
            const Ntuple& obj_;
        };
    }
}

#endif // GENERS_RPFOOTERRECORD_HH_

