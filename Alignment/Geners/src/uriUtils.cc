#include <cstring>
#include <cassert>
#include "Alignment/Geners/interface/IOException.hh"

#include "Alignment/Geners/interface/uriUtils.hh"

namespace gs {
    std::string localFileURI(const char* filename)
    {
        assert(filename);
        if (strlen(filename) == 0) throw gs::IOInvalidArgument(
            "In gs::localFileURI: empty file name");
        std::string uri("file://");
        if (filename[0] != '/')
            uri += "./";
        uri += filename;
        return uri;
    }

    std::string fileTail(const char* filename)
    {
        assert(filename);
        const char *progname = std::strrchr(filename, '/');
        if (progname)
            ++progname;
        else
            progname = filename;
        return std::string(progname);
    }

    std::string fileDirname(const char* filename)
    {
        assert(filename);
        const char *progname = std::strrchr(filename, '/');
        if (progname)
            return std::string(filename, progname - filename);
        else
            return std::string(".");
    }

    std::string joinDir1WithName2(const char* fname1, const char* fname2)
    {
        std::string res(fileDirname(fname1));
        res += '/';
        res += fileTail(fname2);
        return res;
    }
}
