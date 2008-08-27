#ifndef EVF_SQUIDNET_H
#define EVF_SQUIDNET_H

#include <string>

namespace evf
{
  class SquidNet 
    {
    public:
      SquidNet(unsigned int, std::string const &);
      virtual ~SquidNet(){}
      bool check();
    private:
      unsigned int port_;
      std::string proxy_;
      std::string urlToGet_;
    };
}
#endif
