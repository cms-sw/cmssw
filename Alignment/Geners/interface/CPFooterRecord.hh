#ifndef GENERS_CPFOOTERRECORD_HH_
#define GENERS_CPFOOTERRECORD_HH_

#include "Alignment/Geners/interface/AbsRecord.hh"
#include "Alignment/Geners/interface/GenericIO.hh"

namespace gs {
    namespace Private {
        template<class Ntuple>
        class CPFooterRecord : public AbsRecord
        {
        public:
            inline CPFooterRecord(const Ntuple& obj)
                : AbsRecord(obj.classId(), "gs::CPFooter",
                            obj.name_.c_str(), obj.category_.c_str()),
                  obj_(obj) {}

            inline bool writeData(std::ostream& os) const
            {
                write_pod(os, obj_.fillCount_);
                write_pod(os, obj_.headerSaved_);
                return !os.fail() && write_item(os, obj_.bufIds_, false);
            }

        private:
            CPFooterRecord();
            const Ntuple& obj_;
        };
    }
}

#endif // GENERS_CPFOOTERRECORD_HH_

