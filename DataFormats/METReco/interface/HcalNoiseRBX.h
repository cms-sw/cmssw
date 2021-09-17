#ifndef _DATAFORMATS_METRECO_HCALNOISERBX_H_
#define _DATAFORMATS_METRECO_HCALNOISERBX_H_

//
// HcalNoiseRBX.h
//
//   description: Container class of RBX information to study anomalous noise in the HCAL.
//                The information for HcalNoiseHPD's are filled in RecoMET/METProducers/HcalNoiseInfoProducer,
//                but the idnumber is managed by DataFormats/METReco/HcalNoiseRBXArray.
//                Essentially contains 4 HcalNoiseHPDs.
//
//   author: J.P. Chou, Brown
//
//

#include "DataFormats/METReco/interface/HcalNoiseHPD.h"

namespace reco {

  //
  // forward declarations
  //

  class HcalNoiseRBX;

  //
  // typedefs
  //

  typedef std::vector<HcalNoiseRBX> HcalNoiseRBXCollection;

  class HcalNoiseRBX {
    friend class HcalNoiseInfoProducer;  // allows this class the fill the HPDs with info
    friend class HcalNoiseRBXArray;      // allows this class to manage the idnumber

  public:
    // constructors
    HcalNoiseRBX();

    // destructor
    ~HcalNoiseRBX();

    //
    // Detector ID accessors
    //

    // accessors
    int idnumber(void) const;

    //
    // other accessors
    //

    // returns a vector of HcalNoiseHPDs
    // this is expensive and deprecated.  One should use the iterator accessor method instead (provided below)
    const std::vector<HcalNoiseHPD> HPDs(void) const;
    inline std::vector<HcalNoiseHPD>::const_iterator HPDsBegin(void) const { return hpds_.begin(); }
    inline std::vector<HcalNoiseHPD>::const_iterator HPDsEnd(void) const { return hpds_.end(); }

    // return HPD with the highest rechit energy in the RBX
    // individual rechits only contribute if they have E>threshold
    std::vector<HcalNoiseHPD>::const_iterator maxHPD(double threshold = 1.5) const;

    // pedestal subtracted fC information for all of the pixels in the RBX
    const std::vector<float> allCharge(void) const;
    float allChargeTotal(void) const;
    float allChargeHighest2TS(unsigned int firstts = 4) const;
    float allChargeHighest3TS(unsigned int firstts = 4) const;

    // total number of adc zeros in the RBX
    int totalZeros(void) const;

    // largest number of adc zeros from a single channel in the RBX
    int maxZeros(void) const;

    // sum of the energy of rechits in the RBX with E>threshold
    double recHitEnergy(double theshold = 1.5) const;
    double recHitEnergyFailR45(double threshold = 1.5) const;

    // minimum and maximum time for rechits in the RBX with E>threshold
    double minRecHitTime(double threshold = 20.0) const;
    double maxRecHitTime(double threshold = 20.0) const;

    // total number of rechits above some threshold in the RBX
    int numRecHits(double threshold = 1.5) const;
    int numRecHitsFailR45(double threshold = 1.5) const;

    // calotower properties integrated over the entire RBX
    double caloTowerHadE(void) const;
    double caloTowerEmE(void) const;
    double caloTowerTotalE(void) const;
    double caloTowerEmFraction(void) const;

    // helper function to get the unique calotowers
    struct twrcomp {
      inline bool operator()(const CaloTower& t1, const CaloTower& t2) const { return t1.id() < t2.id(); }
    };
    typedef std::set<CaloTower, twrcomp> towerset_t;

  private:
    // members
    int idnumber_;

    // the hpds
    std::vector<HcalNoiseHPD> hpds_;

    // the charge
    std::vector<float> allCharge_;

    void uniqueTowers(towerset_t& twrs_) const;
  };

}  // namespace reco

#endif
