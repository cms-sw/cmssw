#ifndef EventFilter_L1TRawToDigi_Packer_h
#define EventFilter_L1TRawToDigi_Packer_h

#include "EventFilter/L1TRawToDigi/interface/Block.h"
#include "EventFilter/L1TRawToDigi/interface/PackerTokens.h"

namespace edm {
  class Event;
}

namespace l1t {
  class L1TDigiToRaw;

  class Packer {
  public:
    virtual Blocks pack(const edm::Event&, const PackerTokens*) = 0;
    void setBoard(unsigned board) { board_ = board; };
    unsigned board() { return board_; };
    virtual ~Packer() = default;

  private:
    unsigned board_{0};
  };

  typedef std::vector<std::shared_ptr<Packer>> Packers;
}  // namespace l1t

#endif
