#ifndef GENERS_URIUTILS_HH_
#define GENERS_URIUTILS_HH_

#include <string>

namespace gs {
    std::string localFileURI(const char* filename);

    std::string fileTail(const char* filename);

    std::string fileDirname(const char* filename);

    std::string joinDir1WithName2(const char* fname1, const char* fname2);
}

#endif // GENERS_URIUTILS_HH_

