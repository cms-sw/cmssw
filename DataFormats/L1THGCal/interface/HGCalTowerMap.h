#ifndef DataFormats_L1TCalorimeter_HGCalTowerMap_h
#define DataFormats_L1TCalorimeter_HGCalTowerMap_h

#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include <unordered_map>

namespace l1t {
  
  class HGCalTowerMap;
  typedef BXVector<HGCalTowerMap> HGCalTowerMapBxCollection;

  class HGCalTowerMap {

  public:

    HGCalTowerMap(): nEtaBins_(0), nPhiBins_(0), layer_(0) {}
    
    HGCalTowerMap( int nEtaBins, int nPhiBins );

    HGCalTowerMap( const std::vector<double>& etaBins, const std::vector<double>& phiBins );

    ~HGCalTowerMap();

    void setLayer( const unsigned layer ) { layer_ = layer; }


    int nEtaBins() const { return nEtaBins_; }
    int nPhiBins() const { return nPhiBins_; }
    const vector<double>& etaBins() const { return etaBins_; }
    const vector<double>& phiBins() const { return phiBins_; }
    const l1t::HGCalTower& tower(int iEta, int iPhi) { return towerMap_[bin_id(iEta,iPhi)]; }

    int iEta(const double eta) const;
    int iPhi(const double phi) const;
    int layer() const { return layer_;}

    HGCalTowerMap& operator+=(const HGCalTowerMap& map);
    void addTower(int iEta, int iPhi, const l1t::HGCalTower& tower) { towerMap_[bin_id(iEta,iPhi)] += tower; }

  private:

    static constexpr double kEtaMin_ = 1.479;
    static constexpr double kEtaMax_ = 3.;
    static constexpr double kEtaMinLoose_ = 1.401; //BH has some TC below 1.479
    static constexpr double kEtaMaxLoose_ = 3.085; //FH has some TC above 3.0
    static constexpr double kPhiMin_ = -M_PI;
    static constexpr double kPhiMax_ = +M_PI;  
    
    int nEtaBins_;
    int nPhiBins_;
    vector<double> etaBins_;
    vector<double> phiBins_;
    std::unordered_map<int,l1t::HGCalTower>  towerMap_;
    unsigned layer_;

    int bin_id(int iEta,int iPhi) const;

  };

}

#endif


