#ifndef EVF_RATESTAT
#define EVF_RATESTAT

#include <string>
#include <vector>

namespace evf{

  class CurlPoster;

  class RateStat{
  public:
    RateStat(std::string iDieUrl);
    ~RateStat();
    void sendStat(const unsigned char *, size_t, unsigned int);
    void sendLegenda(const std::string &);
  private:
    std::string iDieUrl_;
    CurlPoster *poster_;

  };
}
#endif
