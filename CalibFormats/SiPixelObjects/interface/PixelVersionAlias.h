#ifndef PixelVersionAlias_h
#define PixelVersionAlias_h

namespace pos{
  class PixelVersionAlias {

  public:
    PixelVersionAlias(std::string path, unsigned int version, std::string alias) {
      path_=path;
      version_=version;
      alias_=alias;
    }

    std::string path() {return path_;}
    unsigned int version() {return version_;}
    std::string alias() {return alias_;}

  private:

    std::string path_;
    unsigned int version_;
    std::string alias_;
  };
}
#endif
