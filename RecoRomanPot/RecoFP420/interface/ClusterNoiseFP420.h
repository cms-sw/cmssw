#ifndef ClusterNoiseFP420_h
#define ClusterNoiseFP420_h

#include <vector>
#include <map>
#include <iostream>
#include <cstdint>
//#define mynsdebug0

typedef float ElectrodNoise;
typedef bool ElectrodDisable;

class ClusterNoiseFP420 {
public:
  ClusterNoiseFP420();
  ~ClusterNoiseFP420();

  class ElectrodData {
  public:
    ElectrodNoise getNoise() const { return static_cast<ElectrodNoise>(std::abs(Data) / 10.0); }
    ElectrodDisable getDisable() const { return ((Data > 0) ? false : true); }  // if Data <=0 then electrode is disable
    void setData(short data) { Data = data; }
    void setData(float noise_, bool disable_) {
      short noise = static_cast<short>(noise_ * 10.0 + 0.5) & 0x01FF;
      Data =
          (disable_ ? -1 : 1) * noise;  // Data = sign(+/-1) * Noise(Adc count). if Data <=0 then electrode is disable

#ifdef mynsdebug0
      std::cout << std::fixed << "ClusterNoiseFP420.h:: ElectrodData: noise= " << noise_ << " \t"
                << ": disable= " << disable_ << " \t"
                << "sign Data(=noise*10.0 + 0.5)= " << Data << " \t"
                << "in getNoise we do: abs(Data)/10.0, so it is OK"
                << " \t" << std::endl;
#endif
    };

  private:
    //the short type is assured to be 16 bit in CMSSW???
    short Data;  // Data = sign(+/-1) * Noise(Adc count). if Data <=0 then electrode is disable
  };

  std::map<uint32_t, std::vector<ElectrodData> > m_noises;
};

typedef std::vector<ClusterNoiseFP420::ElectrodData> ElectrodNoiseVector;
typedef std::vector<ClusterNoiseFP420::ElectrodData>::const_iterator ElectrodNoiseVectorIterator;
typedef std::map<uint32_t, ElectrodNoiseVector> ElectrodNoiseMap;
typedef std::map<uint32_t, ElectrodNoiseVector>::const_iterator ElectrodNoiseMapIterator;

#endif
