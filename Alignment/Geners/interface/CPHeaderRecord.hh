#ifndef GENERS_CPHEADERRECORD_HH_
#define GENERS_CPHEADERRECORD_HH_

#include "Alignment/Geners/interface/AbsRecord.hh"
#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/CharBuffer.hh"

namespace gs {
    namespace Private {
        template<class Ntuple>
        class CPHeaderRecord : public AbsRecord
        {
        public:
            inline CPHeaderRecord(const Ntuple& obj)
                : AbsRecord(obj.thisClass_, "gs::CPHeader",
                            obj.name_.c_str(), obj.category_.c_str()),
                  obj_(obj) {}

            inline bool writeData(std::ostream& os) const
            {
                obj_.thisClass_.write(os);
                obj_.bufferClass_.write(os);
                obj_.cbClass_.write(os);
                write_pod_vector(os, obj_.colNames_);
                write_pod(os, obj_.title_);
                write_pod(os, obj_.bufferSize_);
                return !os.fail() && obj_.dumpColumnClassIds(os);
            }

        private:
            CPHeaderRecord();
            const Ntuple& obj_;
        };
    }
}

#endif // GENERS_CPHEADERRECORD_HH_

