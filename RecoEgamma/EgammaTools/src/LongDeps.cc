#include "RecoEgamma/EgammaTools/interface/LongDeps.h"


LongDeps::LongDeps(float radius, const std::vector<float>& energyPerLayer, float energyEE,float energyFH,float energyBH,const std::set<int> layers):
energyPerLayer_(energyPerLayer),radius_(radius),energyEE_(energyEE),energyFH_(energyFH),
energyBH_(energyBH),layers_(layers) {
    lay_Efrac10_ = 0;
    lay_Efrac90_ = 0;
    float lay_energy = 0;
    for (unsigned lay = 1; lay < 52; ++lay) {
        lay_energy += energyPerLayer_[lay];
        if (lay_Efrac10_ == 0 && lay_energy > 0.1 * energyEE_){
            lay_Efrac10_ = lay;
        }
        if (lay_Efrac90_ == 0 && lay_energy > 0.9 * energyEE_){
            lay_Efrac90_ = lay;
        }
    }
    float e4 = energyPerLayer_[1]+energyPerLayer_[2]+energyPerLayer_[3]+energyPerLayer_[4];
    float etot = energyEE_ + energyFH_ + energyBH_;
    e4oEtot_ =  (etot > 0.) ? e4/etot : -1. ;
}
