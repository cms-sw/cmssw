#include "GeneratorInterface/GenFilters/plugins/MCFilterZboostHelper.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"


#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <vector>
#include <iostream>


class MCMultiParticleMassFilter : public edm::global::EDFilter<> {
    public:
        explicit MCMultiParticleMassFilter(const edm::ParameterSet&);
        ~MCMultiParticleMassFilter() override;

    private:
        bool filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const override;
        bool recurseLoop(std::vector<HepMC::GenParticle*>& particlesThatPassCuts, std::vector<int>& indices, int depth) const;
    
    /* Member data */
    const edm::EDGetTokenT<edm::HepMCProduct> token_;
    const std::vector<int> particleID;
    double ptMin;
    double etaMax;
    int status;
    const double minTotalMassSq;
    const double maxTotalMassSq;
    int nParticles;

    int particleSumTo;
    int particleProdTo;
};



using namespace edm;
using namespace std;

MCMultiParticleMassFilter::MCMultiParticleMassFilter(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
        iConfig.getUntrackedParameter<edm::InputTag>("src", edm::InputTag("generator", "unsmeared")))),
    particleID(iConfig.getParameter<std::vector<int>>("ParticleID")),
    ptMin(iConfig.getParameter<double>("PtMin")),
    etaMax(iConfig.getParameter<double>("EtaMax")),
    status(iConfig.getParameter<int>("Status")),
    minTotalMassSq(iConfig.getParameter<double>("minTotalMass")*iConfig.getParameter<double>("minTotalMass")),
    maxTotalMassSq(iConfig.getParameter<double>("maxTotalMass")*iConfig.getParameter<double>("maxTotalMass")){

    nParticles = particleID.size();

    particleSumTo = 0;
    particleProdTo = 1;
    for(const int i : particleID){        
        particleSumTo += i;
        particleProdTo *= i;
    }
}

MCMultiParticleMassFilter::~MCMultiParticleMassFilter(){

}


bool MCMultiParticleMassFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
    edm::Handle<edm::HepMCProduct> evt;
    iEvent.getByToken(token_, evt);
    const HepMC::GenEvent* myGenEvent = evt->GetEvent();

    std::vector<HepMC::GenParticle*> particlesThatPassCuts = std::vector<HepMC::GenParticle*>();
    for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p){
        if(
            ((*p)->status() == status) &&
            ((*p)->momentum().perp() > ptMin) &&
            (std::fabs((*p)->momentum().eta()) < etaMax)
        ){
            for(const int i : particleID){
                if(i == (*p)->pdg_id()){
                    //if the particle ID is one of the ones you specified, then add it 
                    particlesThatPassCuts.push_back(*p);
                    break;
                }
            }
        }
    }
    int nIterables = particlesThatPassCuts.size();
    // cout << "The number of iterables for this event is=" << nIterables << endl;
    if(nIterables < nParticles){
        // cout << "too few particles!" << endl;
        return false;
    } else{
        // cout << "Running Loop" << endl;
        int i = 0;
        while((nIterables - i) >= nParticles){
        // for(int i = 0; i < nIterables; i++){
            // if((nIterables - i) < nParticles){
            //     continue;
            // }
            vector<int> indices;
            indices.push_back(i);
            // cout << "running recursion starting from index " << i << endl;
            bool success = recurseLoop(particlesThatPassCuts, indices, 1);
            if(success){
                return true;
            }
            i++;
        }
    }
    return false;
}

bool MCMultiParticleMassFilter::recurseLoop(std::vector<HepMC::GenParticle*>& particlesThatPassCuts, std::vector<int>& indices, int depth) const{
    int lastIndex = indices.back();
    if(lastIndex >= (int)(particlesThatPassCuts.size()) ){
        return false;
    } else if(depth == nParticles){
        int tempSum = 0;
        int tempProd = 1;
        
        double px,py,pz,e;
        px = py = pz = e = 0;
        for(const int i : indices){
            int id = particlesThatPassCuts[i]->pdg_id();
            tempSum += id;
            tempProd *= id;
            HepMC::FourVector tempVec = particlesThatPassCuts[i]->momentum();
            px += tempVec.px();
            py += tempVec.py();
            pz += tempVec.pz();
            e += tempVec.e();
        }
        if(
            (tempSum != particleSumTo) || 
            (tempProd != particleProdTo)
        ){
            return false;//check if the ids are what you want
        }
        double invMassSq = e*e - px*px - py*py - pz*pz;
        if(
            (invMassSq >= minTotalMassSq) && //taking the root is computationally expensive!
            (invMassSq <= maxTotalMassSq)
        ){
            // cout << "passed: m=" << sqrt(invMassSq) << endl;
            // cout << "summed vector is " << px << ", " << py << ", " << pz << ", " << e << endl;
            return true;
        }
        // cout << "filtered out: m=" << invMassSq << endl;
        return false;
    } else{
        indices.push_back(lastIndex + 1);
        return recurseLoop(particlesThatPassCuts, indices, depth + 1);
    }
}
DEFINE_FWK_MODULE(MCMultiParticleMassFilter);
