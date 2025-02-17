#ifndef DataFormats_ME0StubPrimitive_H
#define DataFormats_ME0StubPrimitive_H

#include <vector>
#include <cstdint>
#include <string>
#include <iostream>

#include "DataFormats/MuonDetId/interface/GEMDetId.h"

class ME0StubPrimitive final {
public:
    // Constructors
    ME0StubPrimitive();
    ME0StubPrimitive(int lc_,
                     int hc_,
                     int id_,
                     int strip_,
                     int partition_);
    ME0StubPrimitive(int lc_,
                     int hc_,
                     int id_,
                     int strip_,
                     int partition_,
                     double bx_);
    ME0StubPrimitive(int lc_,
                     int hc_,
                     int id_,
                     int strip_,
                     int partition_,
                     double bx_,
                     std::vector<double>& centroid_);

    // clone
    ME0StubPrimitive* clone() const { return new ME0StubPrimitive(*this); }

    // Get private variable
    int LayerCount() const {return lc;}
    int HitCount() const {return hc;}
    int PatternId() const {return id;}
    int Strip() const {return strip;}
    int EtaPartition() const {return partition;}
    int BX() const {return bx;}
    double SubStrip() const {return substrip;}
    double BendingAngle() const {return bend_ang;}
    double MSE() const {return mse;}
    std::vector<double> Centroids() const {return centroid;}
    int Quality() const {return quality;}
    int MaxClusterSize() const {return max_cluster_size;}
    int MaxNoise() const {return max_noise;}

    // Set private variable
    void SetLayerCount(int lc_) {lc=lc_;}
    void SetHitCount(int hc_) {hc=hc_;}
    void SetPatternId(int id_) {id=id_;}
    void SetStrip(int strip_) {strip=strip_;}
    void SetEtaPartition(int partition_) {partition=partition_;}
    void SetBX(double bx_) {bx=bx_;}
    void SetCentroids(std::vector<double> centroid_) {centroid=centroid_;}
    void SetMaxClusterSize(int max_cluster_size_) {max_cluster_size=max_cluster_size_;}
    void SetMaxNoise(int max_noise_) {max_noise=max_noise_;}

    void reset();
    void update_quality();
    void fit(int max_span=37);

    // operators
    bool operator==(const ME0StubPrimitive& other) {
        if (lc == 0 && other.lc == 0) {return true;}
        return (quality == other.quality);
    }
    bool operator>(const ME0StubPrimitive& other) {return (quality > other.quality);}
    bool operator<(const ME0StubPrimitive& other) {return (quality < other.quality);}
    bool operator>=(const ME0StubPrimitive& other) {return (quality >= other.quality);}
    bool operator<=(const ME0StubPrimitive& other) {return (quality <= other.quality);}
    // ostream
    friend std::ostream& operator<<(std::ostream& os, const ME0StubPrimitive& stub) {
        os << "id=" << stub.PatternId() << ", lc=" << stub.LayerCount() 
           << ", strip=" << stub.Strip() << ", prt=" << stub.EtaPartition()
           << ", quality=" << stub.Quality();
        return os;
    }
private:
    int lc, hc, id, strip, partition;
    double bx = -9999;
    std::vector<double> centroid;
    double substrip = 0.0;
    double bend_ang = 0.0;
    double mse = 9999;
    int quality = 0;
    int max_cluster_size=0;
    int max_noise=0;
    bool ignore_bend = false;
    std::vector<double> llse_fit(const std::vector<double>& x, const std::vector<double>& y);
};

#endif