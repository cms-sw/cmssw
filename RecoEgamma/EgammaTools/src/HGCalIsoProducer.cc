/*
 * HGCalIsoProducer.cc
 *
 *  Created on: 13 Oct 2017
 *      Author: jkiesele
 */

#include "DataFormats/Math/interface/deltaR.h"
#include <stdexcept>
#include "../interface/HGCalIsoProducer.h"

HGCalIsoProducer::HGCalIsoProducer():dr2_(0.15*0.15),mindr2_(0),rechittools_(nullptr),debug_(false),nlayers_(30){

    allHitMap_ = new std::map<DetId, const HGCRecHit *>();
    mapassigned_=false;
    setNRings(5);
}

HGCalIsoProducer::~HGCalIsoProducer(){
    if(!mapassigned_ &&allHitMap_)
        delete allHitMap_;
}

void HGCalIsoProducer::produceHGCalIso(const reco::CaloClusterPtr & seed){

    if(!rechittools_)
        throw std::runtime_error("HGCalIsoProducer::produceHGCalIso: rechittools not set");

    for(auto& r:isoringdeposits_)
        r=0;


    //this could be replaced by the hit map created by storeRecHits in PCAhelpers
    std::vector<DetId>seedhits;
    const std::vector<std::pair<DetId,float > > & seedhitmap= seed->hitsAndFractions();
    for(const auto& h:seedhitmap)
        seedhits.push_back(h.first);

    for(const auto& hit: *allHitMap_) {

        const GlobalPoint position = rechittools_->getPosition(hit.first);
        float eta=rechittools_->getEta(position, 0);//assume vertex at z=0
        float phi=rechittools_->getPhi(position);

        float deltar2=reco::deltaR2(eta,phi,seed->eta(),seed->phi());

        if(deltar2>dr2_ || deltar2<mindr2_) continue;

        size_t layer=rechittools_->getLayerWithOffset(hit.first);
        if(layer>=nlayers_) continue;

        const size_t& ring=ringasso_.at(layer);

        //do not consider hits associated to the photon cluster
        if(std::find(seedhits.begin(),seedhits.end(),hit.first)==seedhits.end()){
            isoringdeposits_.at(ring)+=hit.second->energy();
        }
    }
}

void HGCalIsoProducer::setNRings(const size_t nrings){
    if(nrings>nlayers_)
        throw std::logic_error("PhotonHGCalIsoProducer::setNRings: max number of rings reached");

    ringasso_.clear();
    isoringdeposits_.clear();
    size_t separator=nlayers_/nrings;
    size_t counter=0;
    for(size_t i=0;i<nlayers_+1;i++){
        ringasso_.push_back(counter);
        //the last ring might be larger.
        if(i && !(i%separator) && (int)counter<(int)nrings-1){
            counter++;
        }
    }
    isoringdeposits_.resize(nrings,0);
}

//copied from electronIDProducer - can be merged
void HGCalIsoProducer::fillHitMap(const HGCRecHitCollection & rechitsEE,
                                 const HGCRecHitCollection & rechitsFH,
                                 const HGCRecHitCollection & rechitsBH) {
    allHitMap_->clear();
    unsigned hitsize = rechitsEE.size();
    for ( unsigned i=0; i< hitsize ; ++i) {
        (*allHitMap_)[rechitsEE[i].detid()] = & rechitsEE[i];
    }

    if (debug_)
        std::cout << " EE " << hitsize << " RecHits " << std::endl;
    hitsize = rechitsFH.size();
    for ( unsigned i=0; i< hitsize ; ++i) {
        (*allHitMap_)[rechitsFH[i].detid()] = & rechitsFH[i];
    }
    if (debug_)
        std::cout << " FH " << hitsize << " RecHits " << std::endl;
    hitsize = rechitsBH.size();
    for ( unsigned i=0; i< hitsize ; ++i) {
        (*allHitMap_)[rechitsBH[i].detid()] = & rechitsBH[i];
    }
    if (debug_)
        std::cout << " BH " << hitsize << " RecHits " << std::endl;
    if( debug_)
        std::cout << " Stored " << allHitMap_->size() << " rechits " << std::endl;

}



void HGCalIsoProducer::setHitMap(std::map<DetId, const HGCRecHit *> * hitmap){
    if(!mapassigned_ && allHitMap_)
        delete allHitMap_;
    mapassigned_=true;
    allHitMap_=hitmap;
}

const float& HGCalIsoProducer::getIso(const size_t& ring)const{
    if(ring>=isoringdeposits_.size())
        throw std::out_of_range("HGCalIsoProducer::getIso: ring index out of range");
    return isoringdeposits_.at(ring);
}
