#ifndef RECOLOCALCALO_HCALLASERRECO_HCALLASERUNPACKER_H 
#define RECOLOCALCALO_HCALLASERRECO_HCALLASERUNPACKER_H 1

class FEDRawData;
class HcalLaserDigi;

class HcalLaserUnpacker {
public:
  HcalLaserUnpacker();
  void unpack(const FEDRawData& raw, HcalLaserDigi& digi) const;
};

#endif // RECOLOCALCALO_HCALLASERRECO_HCALLASERUNPACKER_H 
