//
//  GEDPhoIDTools.cc
//
//
//  Created by Rishi Patel on 5/13/14.
//
//
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoEgamma/PhotonIdentification/interface/GEDPhoIDTools.h"

bool GEDPhoIDTools::CutBasedID(WP wp, bool UsedefaultCuts=true){
    if(UsedefaultCuts)defaultCuts();
    bool passID=false;
    float sieie=pho_->sigmaIetaIeta();
    int pixVeto=pho_->hasPixelSeed();
//    bool isEB=pho_->isEB();
    float HoE=pho_->hadTowOverEm();
    float chgIso=SolidConeIso(0.3, reco::PFCandidate::h);
    float phoIso=SolidConeIso(0.3, reco::PFCandidate::gamma); 
    float neuIso=SolidConeIso(0.3, reco::PFCandidate::h0);  
    getEffArea(pho_->superCluster()->eta());
    
    float rhoSubChg= std::max(chgIso - rho_*EAChg_, (float)0.);
    float rhoSubNeu= std::max(neuIso - rho_*EANeu_, (float)0.);
    float rhoSubPho= std::max(phoIso - rho_*EAPho_, (float)0.);
    float scRawE=pho_->superCluster()->rawEnergy();
 
    float neuCut=Neuc[0]+Neus[0]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
    float phoCut=Phoc[0]+Phos[0]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
    float chgCut=Chgc[0]+Chgs[0]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
    switch(wp){
        case GEDPhoIDTools::Loose:
	neuCut=Neuc[0]+Neus[0]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
        phoCut=Phoc[0]+Phos[0]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
	chgCut=Chgc[0]+Chgs[0]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
	if(((sieie<0.012 && fabs(photon_directionWrtVtx_.Eta())<1.479) || (sieie<0.034 && fabs(photon_directionWrtVtx_.Eta())>=1.479))
	  && pixVeto<1 && HoE<0.05 && rhoSubChg<chgCut && rhoSubPho<phoCut 
	  && rhoSubNeu<neuCut)passID=true;        
	break;
        case GEDPhoIDTools::Medium:
        neuCut=Neuc[1]+Neus[1]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
        phoCut=Phoc[1]+Phos[1]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
        chgCut=Chgc[1]+Chgs[1]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
        if(((sieie<0.011 && fabs(photon_directionWrtVtx_.Eta())<1.479) || (sieie<0.033 && fabs(photon_directionWrtVtx_.Eta())>=1.479))
          && pixVeto<1 && HoE<0.05 && rhoSubChg<chgCut && rhoSubPho<phoCut
          && rhoSubNeu<neuCut)passID=true;
        break;
        case GEDPhoIDTools::Tight:
        neuCut=Neuc[2]+Neus[2]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
        phoCut=Phoc[2]+Phos[2]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
        chgCut=Chgc[2]+Chgs[2]*(scRawE/cosh(photon_directionWrtVtx_.Eta()));
        if(((sieie<0.011 && fabs(photon_directionWrtVtx_.Eta())<1.479) || (sieie<0.031 && fabs(photon_directionWrtVtx_.Eta())>=1.479))
          && pixVeto<1 && HoE<0.05 && rhoSubChg<chgCut && rhoSubPho<phoCut
          && rhoSubNeu<neuCut)passID=true;
         break;
    }
    return passID;
}
double GEDPhoIDTools::SolidConeIso(float conesize, reco::PFCandidate::ParticleType pfType){
    double PFIso=0;
    unsigned int iin=0;//candidate index
    std::vector<reco::PFCandidate>::iterator it = candidates_.begin();
    for ( it = candidates_.begin(), iin=0; it != candidates_.end(); ++it,  ++iin ){
        
        if ( (*it).particleId()!=pfType)continue;//require chg particles from the photon vertex
        if((*it).superClusterRef()==pho_->superCluster())continue; //skip over the GED Photon
//        if((*it).superClusterRef().isNonnull())continue; //skip over all GED Photons
        if((*it).particleId()==reco::PFCandidate::h){
        float dz = fabs((*it).vz() - vtx_->z());
        if (dz > 0.2) continue;
        double dxy = ( -((*it).vx() - vtx_->x())*(*it).py() + ((*it).vy() - vtx_->y())*(*it).px()) / (*it).pt();
        if(fabs(dxy) > 0.1) continue;
        }
        
        float dR=deltaR(photon_directionWrtVtx_.Eta(),photon_directionWrtVtx_.Phi(),(*it).eta(), (*it).phi());
        if(FPRemoval_){
            bool inFP=false;
            reco::PFCandidateRef pfCandRef(collection_, iin);
            for( std::vector<reco::PFCandidateRef>::const_iterator ipf = ParticleBasedIsoMap_[pho_].begin();
                ipf != ParticleBasedIsoMap_[pho_].end(); ++ipf ) {
                if(*ipf == pfCandRef){ inFP = true; break;}
            }
            if(inFP)continue;
        }
        if(dR>conesize)continue;
        PFIso=PFIso+(*it).pt();
     }
     
    return PFIso;
}

void GEDPhoIDTools::FrixioneIso(float conesize, int nrings, reco::PFCandidate::ParticleType pfType, std::vector<double>&IsoRings){
    //float ringsize=conesize/nrings;
    double PFIso[nrings];
    for(int i=0; i<nrings; ++i)PFIso[i]=0;//initialize
    unsigned int iin=0;//candidate index
    std::vector<reco::PFCandidate>::iterator it = candidates_.begin();
    for ( it = candidates_.begin(), iin=0; it != candidates_.end(); ++it,  ++iin ){

        
        if ( (*it).particleId()!=pfType)continue;//require chg particles from the photon vertex
	if((*it).superClusterRef()==pho_->superCluster())continue; //skip over the GED Photon
//        if((*it).superClusterRef().isNonnull())continue; //skip over GED Photons
        if((*it).particleId()==reco::PFCandidate::h){
            float dz = fabs((*it).vz() - vtx_->z());
            if (dz > 0.2) continue;
            double dxy = ( -((*it).vx() - vtx_->x())*(*it).py() + ((*it).vy() - vtx_->y())*(*it).px()) / (*it).pt();
	    if(fabs(dxy) > 0.1) continue;
	    //cout<<"off pointing "<<endl;
        }
        if(FPRemoval_){
            bool inFP=false;
            reco::PFCandidateRef pfCandRef(collection_, iin);
            for( std::vector<reco::PFCandidateRef>::const_iterator ipf = ParticleBasedIsoMap_[pho_].begin();
                ipf != ParticleBasedIsoMap_[pho_].end(); ++ipf ) {
                if(*ipf == pfCandRef){ inFP = true; break;}
            }
            if(inFP)continue;
        }
        float dR=deltaR(photon_directionWrtVtx_.Eta(),photon_directionWrtVtx_.Phi(),(*it).eta(), (*it).phi());
        if(dR>conesize*nrings)continue;
        for(int i=0; i<nrings; ++i){
            float lowbound=i*conesize;
            float upbound=(i+1)*conesize;
           if(dR<upbound && dR>lowbound)PFIso[i]=PFIso[i]+(*it).pt();
        }
    }
    IsoRings.resize(0);
    for(int i=0; i<nrings; ++i)IsoRings.push_back(PFIso[i]);
}

void GEDPhoIDTools::setConstSlope(float c, float s, reco::PFCandidate::ParticleType pftype, WP wp){
     switch(wp){

        case GEDPhoIDTools::Loose:
	if(pftype==PFCandidate::gamma){Phoc[0]=c; Phos[0]=s;}
	if(pftype==PFCandidate::h){Chgc[0]=c; Chgs[0]=s;}
	if(pftype==PFCandidate::h0){Neuc[0]=c; Neus[0]=s;}
        break;
        case GEDPhoIDTools::Medium:
        if(pftype==PFCandidate::gamma){Phoc[1]=c; Phos[1]=s;}
        if(pftype==PFCandidate::h){Chgc[1]=c; Chgs[1]=s;}
        if(pftype==PFCandidate::h0){Neuc[1]=c; Neus[1]=s;}    
	break;
        case GEDPhoIDTools::Tight:
        if(pftype==PFCandidate::gamma){Phoc[2]=c; Phos[2]=s;}
        if(pftype==PFCandidate::h){Chgc[2]=c; Chgs[2]=s;}
        if(pftype==PFCandidate::h0){Neuc[2]=c; Neus[2]=s;}    
	break;
    } 
}

void GEDPhoIDTools::setPhotonP4(reco::PhotonRef pho, reco::Vertex vtx){
    pho_=pho;
    vtx_=&vtx;
   math::XYZVector photon_directionWrtVtx(pho_->superCluster()->x() - vtx_->x(),
                                           pho_->superCluster()->y() - vtx_->y(),
                                           pho_->superCluster()->z() - vtx_->z());
    photon_directionWrtVtx_=photon_directionWrtVtx;
    
}

