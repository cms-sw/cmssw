#ifndef GENERS_RPHEADERRECORD_HH_
#define GENERS_RPHEADERRECORD_HH_

#include "Alignment/Geners/interface/AbsRecord.hh"
#include "Alignment/Geners/interface/binaryIO.hh"

namespace gs {
    namespace Private {
        template<class Ntuple>
        class RPHeaderRecord : public AbsRecord
        {
        public:
            inline RPHeaderRecord(const Ntuple& obj)
                : AbsRecord(obj.thisClass_, "gs::RPHeader",
                            obj.name_.c_str(), obj.category_.c_str()),
                  obj_(obj) {}

            inline bool writeData(std::ostream& os) const
            {
                obj_.thisClass_.write(os);
                obj_.bufferClass_.write(os);
                write_pod_vector(os, obj_.colNames_);
                write_pod(os, obj_.title_);
                write_pod(os, obj_.bufferSize_);
                return !os.fail();
            }

        private:
            RPHeaderRecord();
            const Ntuple& obj_;
        };
    }
}

#endif // GENERS_RPHEADERRECORD_HH_

