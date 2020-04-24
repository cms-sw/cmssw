#ifndef CordicXilinx_H
#define CordicXilinx_H

#include <vector>
#include <array>
#include <stdint.h>

class CordicXilinx
{
  public:
    CordicXilinx(int inputBits, int outputBits, bool debug=false);

    // Fills aPhi and aMagnitude with result scaled to 2^outputBits
    // input must be signed and in the range [ -2^(inputBits-1), 2^(inputBits-1) )
    void operator() ( int32_t xInput , int32_t yInput , int32_t& aPhi , uint32_t& aMagnitude ) const;

    // Returns angle in fixed point according to internalBits_ precision
    // The integer part is 3 bits (i.e. -4 to 3)
    // Valid input is -pi < angle < pi
    int encodeAngle(const double angleFloat) const;

  private:
    const int inputBits_;
    const int outputBits_;
    const bool debug_;

    std::vector<int> rotations_;
    // For (probably unnecessary) optimization
    std::array<int, 3> encodedAngles_;
    enum { Pi, HalfPi, NHalfPi };
    int iterations_;
    int internalBits_;
    int scaleFactor_;
};

#endif
