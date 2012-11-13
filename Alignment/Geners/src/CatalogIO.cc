#include "Alignment/Geners/interface/CatalogIO.hh"

namespace gs {
    bool writeBinaryCatalog(std::ostream& os, const unsigned compressionCode,
                            const unsigned mergeLevel,
                            const std::vector<std::string>& annotations,
                            const AbsCatalog& catalog, const unsigned formatId)
    {
        os.seekp(0, std::ios_base::beg);

        const unsigned endianness = 0x01020304;
        const unsigned char sizelong = sizeof(long);

        write_pod(os, formatId);
        write_pod(os, endianness);
        write_pod(os, sizelong);
        write_pod(os, compressionCode);
        write_pod(os, mergeLevel);
        write_pod_vector(os, annotations);

        return !os.fail() && 
               catalog.classId().write(os) && catalog.write(os);
    }
}
