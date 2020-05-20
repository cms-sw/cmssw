#ifndef L1Trigger_TrackFindingTracklet_interface_FPGAWord_h
#define L1Trigger_TrackFindingTracklet_interface_FPGAWord_h

#include <string>
#include <cassert>

namespace trklet {

  class FPGAWord {
  public:
    FPGAWord();

    FPGAWord(int value, int nbits, bool positive = true, int line = -1, const char* file = nullptr);

    ~FPGAWord() = default;

    void set(int value, int nbits, bool positive = true, int line = -1, const char* file = nullptr);

    std::string str() const;

    //return the nbits starting with the lsb. lsb=0 means the least significant bit
    unsigned int bits(unsigned int lsb, unsigned int nbit) const;

    int value() const { return value_; }
    int nbits() const { return nbits_; }

    bool atExtreme() const;

    bool operator==(const FPGAWord& other) const;

  private:
    int value_{-1};
    int nbits_{-1};
    bool positive_{true};
  };
};  // namespace trklet
#endif
