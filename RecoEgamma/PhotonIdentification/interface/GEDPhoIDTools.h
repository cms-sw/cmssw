//
//  GEDPhoIDTools.cc
//  
//
//  Created by Rishi Patel on 5/13/14.
//
//
#ifndef __GEDPhoIDTools_H__
#define __GEDPhoIDTools_H__
//#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TLorentzVector.h"
#include <vector>
#include <memory>
using namespace edm;
using namespace reco;
using namespace std;
class GEDPhoIDTools{
public:
    GEDPhoIDTools(const Event& iEvent, bool doFPRemoval=true, bool isEle=false ){
            //unload PFCandidate handle        

        InputTag label("particleFlow");
        iEvent.getByLabel(label, collection_);
        candidates_ = (*collection_.product());
        FPRemoval_=doFPRemoval;
        edm::Handle<double> rhoHandle;
        InputTag rhoLabel("fixedGridRhoFastjetAll");
        iEvent.getByLabel(rhoLabel,rhoHandle);
        rho_    = *(rhoHandle.product());
        if(FPRemoval_){
            Handle<ValueMap<std::vector<reco::PFCandidateRef > > > ParticleBasedIsoMapHandle;
            InputTag particleBase(string("particleBasedIsolation"),string("gedPhotons"));
            iEvent.getByLabel(particleBase, ParticleBasedIsoMapHandle);
            if(isEle){
                InputTag particleBase(string("particleBasedIsolation"),string("gedGsfElectrons"));
                iEvent.getByLabel(particleBase, ParticleBasedIsoMapHandle);
            }
            ParticleBasedIsoMap_ =  *(ParticleBasedIsoMapHandle.product());
            
        }
	//default const and slope 
    };
//    ~GEDPhoIDTools(){delete vtx_; };
    enum WP {
        Loose=0,     // based on Cut-based 2012 Photon ID
        Medium,
        Tight
    };
    void setPhotonP4(reco::PhotonRef pho, reco::Vertex vtx);
    bool CutBasedID(WP wp, bool UsedefaultCuts);//
    double SolidConeIso(float conesize, reco::PFCandidate::ParticleType pftype);
    void FrixioneIso(float conesize, int nrings, reco::PFCandidate::ParticleType pftype, std::vector<double>&IsoRings);
    //double FootPrint(reco::PFCandidate::ParticleType pftype);
    void setConstSlope(float c, float s, reco::PFCandidate::ParticleType, WP wp);
        void defaultCuts(){
    for(int i=0; i<3; ++i){ Chgs[i]=0.0;}
	bool isEB=pho_->isEB();
        if(isEB){

        Chgc[0]=2.6; Chgc[1]=1.5;Chgc[2]=0.7;

        Neuc[0]=3.5; Neuc[1]=1.0; Neuc[2]=0.4;
        Neus[0]=0.04; Neus[1]=0.04; Neus[2]=0.04;

        Phoc[0]=1.3; Phoc[1]=0.7; Phoc[2]=0.3;
        Phos[0]=0.005; Phos[1]=0.005; Phos[2]=0.005;
      }
        else{
        Chgc[0]=2.3; Chgc[1]=1.2;Chgc[2]=0.5;

        Neuc[0]=2.9; Neuc[1]=1.5; Neuc[2]=1.5;
        Neus[0]=0.04; Neus[1]=0.04; Neus[2]=0.04;

        Phoc[0]=0; Phoc[1]=1.0; Phoc[2]=1.0;
        Phos[0]=0; Phos[1]=0.005; Phos[2]=0.005;
      }

    }
private:
    void getEffArea(float eta){
	if(fabs(eta)<1.0 ){
	   EAChg_=0.012; EANeu_=0.030; EAPho_=0.148; 	
        }
	if(fabs(eta)>1.0 &&  fabs(eta)<1.479 ){
	EAChg_=0.010; EANeu_=0.057; EAPho_= 0.130;  
        }
	if(fabs(eta)>1.479 &&  fabs(eta)<2.0 ){
	EAChg_=0.014; EANeu_=0.039; EAPho_= 0.112; 
        }
        if(fabs(eta)>2.0 &&  fabs(eta)<2.2 ){
        EAChg_=0.012; EANeu_=0.015; EAPho_= 0.216; 
        }
        if(fabs(eta)>2.2 &&  fabs(eta)<2.3 ){
        EAChg_=0.016; EANeu_=0.024; EAPho_= 0.262; 
        }
        if(fabs(eta)>2.3 &&  fabs(eta)<2.4 ){
        EAChg_=0.020; EANeu_=0.039; EAPho_= 0.260; 
        }
	 if(fabs(eta)>2.4){
        EAChg_=0.012; EANeu_=0.072; EAPho_= 0.266; 
        }
    }
    float Chgc[3]; float Chgs[3];
    float Phoc[3]; float Phos[3];
    float Neuc[3]; float Neus[3];
    float EAChg_, EAPho_, EANeu_;//effective areas for PU subtraction
    reco::PhotonRef pho_;
    reco::Vertex* vtx_;
    float rho_;
    bool FPRemoval_;
    Handle<reco::PFCandidateCollection> collection_;
    std::vector<reco::PFCandidate> candidates_;
    ValueMap<std::vector<reco::PFCandidateRef > >ParticleBasedIsoMap_;
    math::XYZVector photon_directionWrtVtx_;
};
#endif
