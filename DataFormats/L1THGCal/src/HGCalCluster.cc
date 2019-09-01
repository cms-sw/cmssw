#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace l1t;

HGCalCluster::HGCalCluster(const LorentzVector p4, int pt, int eta, int phi)
    : HGCalClusterT<l1t::HGCalTriggerCell>(p4, pt, eta, phi), module_(0) {}

HGCalCluster::HGCalCluster(const edm::Ptr<l1t::HGCalTriggerCell> &tcSeed)
    : HGCalClusterT<l1t::HGCalTriggerCell>(tcSeed), module_(0) {}

HGCalCluster::~HGCalCluster() {}
