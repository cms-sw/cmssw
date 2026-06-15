#ifndef DataFormats_HGCalReco_throwOutOfBoundsException_h
#define DataFormats_HGCalReco_throwOutOfBoundsException_h

namespace ticllayer {
  void throwOutOfBoundsException(float eta, float phi, int bin, int size) noexcept(false);
}

#endif
