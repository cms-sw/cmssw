#ifndef DATAFORMATS_SISTRIPCLUSTERINFO_H
#define DATAFORMATS_SISTRIPCLUSTERINFO_H

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include <vector>

class SiStripClusterInfo {
public:

  SiStripClusterInfo() : detId_(0) {};

  SiStripClusterInfo(const SiStripCluster& cluster):
    detId_(cluster.geographicalId()),
    FirstStrip(cluster.firstStrip()),
    StripAmplitudes(cluster.amplitudes()),    
    Position(cluster.barycenter()),
    Width(cluster.amplitudes().size()),
    MaxCharge(0),
    MaxPosition(0)
    {};
  

  short firstStrip() const {return FirstStrip;}
  unsigned int geographicalId() const {return detId_;}
  const std::vector<short>&  stripAmplitudes() const {return StripAmplitudes;}
  const std::vector<float>&  stripNoises()     const {return StripNoises;}
  
  float charge()    const {return Charge;}
  float noise()     const {return Noise;}
  float position()  const {return Position;}
  float width()     const {return Width;}
  float maxCharge() const {return MaxCharge;}
  float maxPos()    const {return MaxPosition;}
  float chargeL()   const {return ChargeL;}
  float chargeR()   const {return ChargeR;}
  
  void setCharge(const float& value) {Charge=value;}  
  void setNoise(const float& value) {Noise=value;}
  void setStripNoises(std::vector<float>& value) {StripNoises=value;}
  void setMaxCharge(const float& value) {MaxCharge=value;}
  void setMaxPos(const float& value) {MaxPosition=value;}
  void setChargeL(const float& value) {ChargeL=value;}
  void setChargeR(const float& value) {ChargeR=value;}

private:

  unsigned int         detId_;
  short                FirstStrip;
  std::vector<short>   StripAmplitudes;
  std::vector<float>   StripNoises;

  float Charge;
  float Noise;
  float Position;
  int   Width;
  float MaxCharge;
  float MaxPosition;
  float ChargeL;
  float ChargeR;
};

// Comparison operators
inline bool operator<( const SiStripClusterInfo& one, const SiStripClusterInfo& other) {
  if(one.geographicalId() == other.geographicalId()) {
    float StoN_one = (one.noise()!= 0) ? one.charge()/one.noise() : .0;
    float StoN_other = (other.noise()!= 0) ? other.charge()/other.noise() : .0;
 
    return StoN_one < StoN_other;
  }
  return one.geographicalId() < other.geographicalId();
}
#endif // DATAFORMATS_SISTRIPCLUSTER_H
