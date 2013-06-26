// Record for column buffer used by column packer

#ifndef GENERS_CPBUFFERRECORD_HH_
#define GENERS_CPBUFFERRECORD_HH_

#include "Alignment/Geners/interface/AbsRecord.hh"
#include "Alignment/Geners/interface/ColumnBuffer.hh"
#include "Alignment/Geners/interface/binaryIO.hh"

namespace gs {
    namespace Private {
        class CPBufferRecord : public AbsRecord
        {
        public:
            inline CPBufferRecord(const ColumnBuffer& obj, const char* name,
                                  const char* category, unsigned long col)
                : AbsRecord(obj.classId(), "gs::CPBuffer", name, category),
                  obj_(obj), column_(col) {}

            inline bool writeData(std::ostream& os) const
            {
                write_pod(os, column_);
                return !os.fail() && obj_.write(os);
            }

        private:
            CPBufferRecord();
            const ColumnBuffer& obj_;
            const unsigned long column_;
        };
    }
}

#endif // GENERS_CPBUFFERRECORD_HH_

