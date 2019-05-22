#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"

using namespace l1t;

HGCalMulticluster::HGCalMulticluster(const LorentzVector p4, int pt, int eta, int phi)
    : HGCalClusterT<l1t::HGCalCluster>(p4, pt, eta, phi), hOverE_(-99), hOverEValid_(false) {}

HGCalMulticluster::HGCalMulticluster(const edm::Ptr<l1t::HGCalCluster> &clusterSeed, float fraction)
    : HGCalClusterT<l1t::HGCalCluster>(clusterSeed, fraction), hOverE_(-99), hOverEValid_(false) {}

HGCalMulticluster::~HGCalMulticluster() {}
