#ifndef TrkAnalyzerUtils_h_
#define TrkAnalyzerUtils_h_

#include <vector>
#include <cmath>
 
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace std;
using namespace reco;
using namespace edm;

//rounds to first few nonzero sig figs
float rndSF(float value, int nSignificantDigits) 
{
  if(value==0) return 0; 
  
  float dSign = (value > 0.0) ? 1 : -1; 
  value *= dSign; 

  int nOffset = static_cast<int>(log10(value)); 
  if(nOffset>=0) ++nOffset;  

  float dScale = pow(10.0,nSignificantDigits-nOffset); 

  return dSign * static_cast<float>(round(value*dScale))/dScale;    
}

//keeps first n digits after the decimal place
inline float rndDP(float value, int nPlaces)
{
  return float(int(value*pow(10,nPlaces))/pow(10,nPlaces));
}

//----------------------------------------------------------------------------------------------------
const TrackingParticle* doRecoToTpMatch(reco::RecoToSimCollection recSimColl, const reco::TrackRef &in){

    //if(in.status()!=1) return NULL;
    reco::RecoToSimCollection::const_iterator matchedSim = recSimColl.find(edm::RefToBase<reco::Track>(in));
    if(matchedSim==recSimColl.end()) return NULL;
    else{
        const TrackingParticle *tparticle = matchedSim->val[0].first.get();
        return tparticle;
    }
}

//-------------------------------------------------------------------------------------------------

vector<int> matchTpToGen(const edm::Event& iEvent, const TrackingParticle* tparticle, edm::EDGetTokenT<reco::GenParticleCollection> genCollection){

    vector<int> retArr;
    if(!tparticle){ retArr.push_back(-999); return retArr; }
    vector<reco::GenParticle> tempStore;
    for(TrackingParticle::genp_iterator igen=tparticle->genParticle_begin(); igen!=tparticle->genParticle_end(); igen++){
        const reco::GenParticle* temp = igen->get();
        tempStore.push_back(*temp); //temp->momentum().px()*temp->momentum().py()*temp->momentum().pz(); //store HepMC barcode for unique id
    }

    //now figure out the array number of the associated genParticles
    bool *tripwire = new bool[tempStore.size()];
    for(unsigned int ii=0; ii<tempStore.size(); ii++){ tripwire[ii]=0; } //reset tripwires

    edm::Handle<reco::GenParticleCollection> parts;
    iEvent.getByToken(genCollection,parts);
    for(unsigned int igenCand=0; igenCand<tempStore.size(); igenCand++){
        for(UInt_t igenP = 0; igenP < parts->size(); ++igenP){
            const reco::GenParticle& p = (*parts)[igenP];
            if(tripwire[igenCand]) continue;
            //there's no equivalence operator for genParticle!! This is a workaround since the stupid things are floats
            //if(p.px()*p.py()*p.pz() == tempStore.at(igenCand).momentum().px()*tempStore.at(igenCand).momentum().py()*tempStore.at(igenCand).momentum().pz()){
            if(fabs((p.px()+p.py()+p.pz()) - (tempStore.at(igenCand).px()+tempStore.at(igenCand).py()+tempStore.at(igenCand).pz())) < 0.001){
                retArr.push_back(igenP);  //genParticle instance number;
                tripwire[igenCand]=1;
            }
        }
        if(!tripwire[igenCand]) retArr.push_back(-999);
    }
    delete[] tripwire;

    return retArr;
}

#endif
