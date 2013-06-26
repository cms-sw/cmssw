// Record for row packer buffer

#ifndef GENERS_RPBUFFERRECORD_HH_
#define GENERS_RPBUFFERRECORD_HH_

#include "Alignment/Geners/interface/AbsRecord.hh"
#include "Alignment/Geners/interface/streamposIO.hh"

namespace gs {
    namespace Private {
        template<class Ntuple>
        class RPBufferRecord : public AbsRecord
        {
        public:
            inline RPBufferRecord(const Ntuple& obj)
                : AbsRecord(obj.fillBuffer_.classId(), "gs::RPBuffer",
                            obj.name_.c_str(), obj.category_.c_str()),
                  obj_(obj) {}

            inline bool writeData(std::ostream& os) const
            {
                write_pod(os, obj_.firstFillBufferRow_);
                write_pod_vector(os, obj_.fillBufferOffsets_);
                return obj_.fillBuffer_.write(os);
            }

        private:
            RPBufferRecord();
            const Ntuple& obj_;
        };
    }
}

#endif // GENERS_RPBUFFERRECORD_HH_

