#include "EgammaAnalysis/ElectronTools/interface/EGammaCutBasedEleId.h"
#include "EgammaAnalysis/ElectronTools/interface/ElectronEffectiveArea.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include <algorithm>

#ifndef STANDALONEID

bool EgammaCutBasedEleId::PassWP(WorkingPoint workingPoint,
    const reco::GsfElectron &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho)
{

    // get the mask
    unsigned int mask = TestWP(workingPoint, ele, conversions, beamspot, vtxs, iso_ch, iso_em, iso_nh, rho);

    // check if the desired WP passed
    if ((mask & PassAll) == PassAll) return true;
    return false;
}

bool EgammaCutBasedEleId::PassWP(WorkingPoint workingPoint,
    const reco::GsfElectronRef &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho)
{
  return PassWP(workingPoint,*ele,conversions,beamspot,vtxs,iso_ch,iso_em,iso_nh,rho);
}

bool EgammaCutBasedEleId::PassTriggerCuts(TriggerWorkingPoint triggerWorkingPoint, const reco::GsfElectron &ele)
{

    // get the variables
    bool isEB           = ele.isEB() ? true : false;
    float pt            = ele.pt();
    float dEtaIn        = ele.deltaEtaSuperClusterTrackAtVtx();
    float dPhiIn        = ele.deltaPhiSuperClusterTrackAtVtx();
    float sigmaIEtaIEta = ele.sigmaIetaIeta();
    float hoe           = ele.hadronicOverEm();
    float trackIso      = ele.dr03TkSumPt();
    float ecalIso       = ele.dr03EcalRecHitSumEt();
    float hcalIso       = ele.dr03HcalTowerSumEt();

    // test the trigger cuts
    return EgammaCutBasedEleId::PassTriggerCuts(triggerWorkingPoint, isEB, pt, dEtaIn, dPhiIn, sigmaIEtaIEta, hoe, trackIso, ecalIso, hcalIso);

}

bool EgammaCutBasedEleId::PassTriggerCuts(TriggerWorkingPoint triggerWorkingPoint, const reco::GsfElectronRef &ele) 
{
  return EgammaCutBasedEleId::PassTriggerCuts(triggerWorkingPoint, *ele);
}

bool EgammaCutBasedEleId::PassEoverPCuts(const reco::GsfElectron &ele)
{

    // get the variables
    float eta           = ele.superCluster()->eta();
    float eopin         = ele.eSuperClusterOverP();
    float fbrem         = ele.fbrem();    

    // test the eop/fbrem cuts
    return EgammaCutBasedEleId::PassEoverPCuts(eta, eopin, fbrem);

}

bool EgammaCutBasedEleId::PassEoverPCuts(const reco::GsfElectronRef &ele) {
  return PassEoverPCuts(*ele);
}


unsigned int EgammaCutBasedEleId::TestWP(WorkingPoint workingPoint,
    const reco::GsfElectron &ele,
    const edm::Handle<reco::ConversionCollection> &conversions,
    const reco::BeamSpot &beamspot,
    const edm::Handle<reco::VertexCollection> &vtxs,
    const double &iso_ch,
    const double &iso_em,
    const double &iso_nh,
    const double &rho)
{

    // get the ID variables from the electron object

    // kinematic variables
    bool isEB           = ele.isEB() ? true : false;
    float pt            = ele.pt();
    float eta           = ele.superCluster()->eta();

    // id variables
    float dEtaIn        = ele.deltaEtaSuperClusterTrackAtVtx();
    float dPhiIn        = ele.deltaPhiSuperClusterTrackAtVtx();
    float sigmaIEtaIEta = ele.sigmaIetaIeta();
    float hoe           = ele.hadronicOverEm();
    float ooemoop       = (1.0/ele.ecalEnergy() - ele.eSuperClusterOverP()/ele.ecalEnergy());

    // impact parameter variables
    float d0vtx         = 0.0;
    float dzvtx         = 0.0;
    if (vtxs->size() > 0) {
        reco::VertexRef vtx(vtxs, 0);    
        d0vtx = ele.gsfTrack()->dxy(vtx->position());
        dzvtx = ele.gsfTrack()->dz(vtx->position());
    } else {
        d0vtx = ele.gsfTrack()->dxy();
        dzvtx = ele.gsfTrack()->dz();
    }

    // conversion rejection variables
    bool vtxFitConversion = ConversionTools::hasMatchedConversion(ele, conversions, beamspot.position());
    float mHits = ele.gsfTrack()->trackerExpectedHitsInner().numberOfHits(); 

    // get the mask value
    unsigned int mask = EgammaCutBasedEleId::TestWP(workingPoint, isEB, pt, eta, dEtaIn, dPhiIn,
        sigmaIEtaIEta, hoe, ooemoop, d0vtx, dzvtx, iso_ch, iso_em, iso_nh, vtxFitConversion, mHits, rho);

    // return the mask value
    return mask;

}

unsigned int EgammaCutBasedEleId::TestWP(WorkingPoint workingPoint,
					 const reco::GsfElectronRef &ele,
					 const edm::Handle<reco::ConversionCollection> &conversions,
					 const reco::BeamSpot &beamspot,
					 const edm::Handle<reco::VertexCollection> &vtxs,
					 const double &iso_ch,
					 const double &iso_em,
					 const double &iso_nh,
					 const double &rho) {
  return TestWP(workingPoint,*ele,conversions,beamspot,vtxs,iso_ch,iso_em,iso_nh,rho);
}


#endif

bool EgammaCutBasedEleId::PassWP(WorkingPoint workingPoint, const bool isEB, const float pt, const float eta,
    const float dEtaIn, const float dPhiIn, const float sigmaIEtaIEta, const float hoe,
    const float ooemoop, const float d0vtx, const float dzvtx, const float iso_ch, const float iso_em, const float iso_nh, 
    const bool vtxFitConversion, const unsigned int mHits, const double rho)
{
    unsigned int mask = EgammaCutBasedEleId::TestWP(workingPoint, isEB, pt, eta, dEtaIn, dPhiIn,
        sigmaIEtaIEta, hoe, ooemoop, d0vtx, dzvtx, iso_ch, iso_em, iso_nh, vtxFitConversion, mHits, rho);

    if ((mask & PassAll) == PassAll) return true;
    return false;
}

bool EgammaCutBasedEleId::PassTriggerCuts(const TriggerWorkingPoint triggerWorkingPoint, 
    const bool isEB, const float pt, 
    const float dEtaIn, const float dPhiIn, const float sigmaIEtaIEta, const float hoe,
    const float trackIso, const float ecalIso, const float hcalIso)
{

   
    // choose cut if barrel or endcap
    unsigned int idx = isEB ? 0 : 1;

    if (triggerWorkingPoint == EgammaCutBasedEleId::TRIGGERTIGHT) {
        float cut_dEtaIn[2]         = {0.007, 0.009};
        float cut_dPhiIn[2]         = {0.15, 0.10};
        float cut_sigmaIEtaIEta[2]  = {0.01, 0.03};
        float cut_hoe[2]            = {0.12, 0.10};
        float cut_trackIso[2]       = {0.20, 0.20};
        float cut_ecalIso[2]        = {0.20, 0.20};
        float cut_hcalIso[2]        = {0.20, 0.20};
        if (fabs(dEtaIn) > cut_dEtaIn[idx])             return false;
        if (fabs(dPhiIn) > cut_dPhiIn[idx])             return false;
        if (sigmaIEtaIEta > cut_sigmaIEtaIEta[idx])     return false;
        if (hoe > cut_hoe[idx])                         return false;
        if (trackIso / pt > cut_trackIso[idx])          return false;
        if (ecalIso / pt > cut_ecalIso[idx])            return false;
        if (hcalIso / pt > cut_hcalIso[idx])            return false;
    }
    else if (triggerWorkingPoint == EgammaCutBasedEleId::TRIGGERWP70) {
        float cut_dEtaIn[2]         = {0.004, 0.005};
        float cut_dPhiIn[2]         = {0.03, 0.02};
        float cut_sigmaIEtaIEta[2]  = {0.01, 0.03};
        float cut_hoe[2]            = {0.025, 0.025};
        float cut_trackIso[2]       = {0.10, 0.10};
        float cut_ecalIso[2]        = {0.10, 0.05};
        float cut_hcalIso[2]        = {0.05, 0.05};
        if (fabs(dEtaIn) > cut_dEtaIn[idx])             return false;
        if (fabs(dPhiIn) > cut_dPhiIn[idx])             return false;
        if (sigmaIEtaIEta > cut_sigmaIEtaIEta[idx])     return false;
        if (hoe > cut_hoe[idx])                         return false;
        if (trackIso / pt > cut_trackIso[idx])          return false;
        if (ecalIso / pt > cut_ecalIso[idx])            return false;
        if (hcalIso / pt > cut_hcalIso[idx])            return false;
    }
    else {
        std::cout << "[EgammaCutBasedEleId::PassTriggerCuts] Undefined working point" << std::endl;
    }   

    return true; 
}

bool EgammaCutBasedEleId::PassEoverPCuts(const float eta, const float eopin, const float fbrem)
{
    if (fbrem > 0.15)                           return true;
    else if (fabs(eta) < 1.0 && eopin > 0.95)   return true;
    return false;
}

unsigned int EgammaCutBasedEleId::TestWP(WorkingPoint workingPoint, const bool isEB, const float pt, const float eta,
    const float dEtaIn, const float dPhiIn, const float sigmaIEtaIEta, const float hoe, 
    const float ooemoop, const float d0vtx, const float dzvtx, const float iso_ch, const float iso_em, const float iso_nh, 
    const bool vtxFitConversion, const unsigned int mHits, const double rho)
{

    unsigned int mask = 0;
    float cut_dEtaIn[2]         = {999.9, 999.9};
    float cut_dPhiIn[2]         = {999.9, 999.9};
    float cut_sigmaIEtaIEta[2]  = {999.9, 999.9};
    float cut_hoe[2]            = {999.9, 999.9};
    float cut_ooemoop[2]        = {999.9, 999.9};
    float cut_d0vtx[2]          = {999.9, 999.9};
    float cut_dzvtx[2]          = {999.9, 999.9};
    float cut_iso[2]            = {999.9, 999.9};
    bool cut_vtxFit[2]          = {false, false};
    unsigned int cut_mHits[2]   = {999, 999};

    if (workingPoint == EgammaCutBasedEleId::VETO) {
        cut_dEtaIn[0]        = 0.007; cut_dEtaIn[1]        = 0.010;
        cut_dPhiIn[0]        = 0.800; cut_dPhiIn[1]        = 0.700;
        cut_sigmaIEtaIEta[0] = 0.010; cut_sigmaIEtaIEta[1] = 0.030;
        cut_hoe[0]           = 0.150; cut_hoe[1]           = 999.9;
        cut_ooemoop[0]       = 999.9; cut_ooemoop[1]       = 999.9;
        cut_d0vtx[0]         = 0.040; cut_d0vtx[1]         = 0.040;
        cut_dzvtx[0]         = 0.200; cut_dzvtx[1]         = 0.200;
        cut_vtxFit[0]        = false; cut_vtxFit[1]        = false;
        cut_mHits[0]         = 999  ; cut_mHits[1]         = 999;
        cut_iso[0]           = 0.150; cut_iso[1]           = 0.150;
    } 
    else if (workingPoint == EgammaCutBasedEleId::LOOSE) {
        cut_dEtaIn[0]        = 0.007; cut_dEtaIn[1]        = 0.009;
        cut_dPhiIn[0]        = 0.150; cut_dPhiIn[1]        = 0.100;
        cut_sigmaIEtaIEta[0] = 0.010; cut_sigmaIEtaIEta[1] = 0.030;
        cut_hoe[0]           = 0.120; cut_hoe[1]           = 0.100;
        cut_ooemoop[0]       = 0.050; cut_ooemoop[1]       = 0.050;
        cut_d0vtx[0]         = 0.020; cut_d0vtx[1]         = 0.020;
        cut_dzvtx[0]         = 0.200; cut_dzvtx[1]         = 0.200;
        cut_vtxFit[0]        = true ; cut_vtxFit[1]        = true;
        cut_mHits[0]         = 1    ; cut_mHits[1]         = 1;
        if (pt >= 20.0) {
            cut_iso[0] = 0.150; cut_iso[1] = 0.150;
        }
        else {
            cut_iso[0] = 0.150; cut_iso[1] = 0.100;
        }
    } 
    else if (workingPoint == EgammaCutBasedEleId::MEDIUM) {
        cut_dEtaIn[0]        = 0.004; cut_dEtaIn[1]        = 0.007;
        cut_dPhiIn[0]        = 0.060; cut_dPhiIn[1]        = 0.030;
        cut_sigmaIEtaIEta[0] = 0.010; cut_sigmaIEtaIEta[1] = 0.030;
        cut_hoe[0]           = 0.120; cut_hoe[1]           = 0.100;
        cut_ooemoop[0]       = 0.050; cut_ooemoop[1]       = 0.050;
        cut_d0vtx[0]         = 0.020; cut_d0vtx[1]         = 0.020;
        cut_dzvtx[0]         = 0.100; cut_dzvtx[1]         = 0.100;
        cut_vtxFit[0]        = true ; cut_vtxFit[1]        = true;
        cut_mHits[0]         = 1    ; cut_mHits[1]         = 1;
        if (pt >= 20.0) {
            cut_iso[0] = 0.150; cut_iso[1] = 0.150;
        }
        else {
            cut_iso[0] = 0.150; cut_iso[1] = 0.100;
        }
    } 
    else if (workingPoint == EgammaCutBasedEleId::TIGHT) {
        cut_dEtaIn[0]        = 0.004; cut_dEtaIn[1]        = 0.005;
        cut_dPhiIn[0]        = 0.030; cut_dPhiIn[1]        = 0.020;
        cut_sigmaIEtaIEta[0] = 0.010; cut_sigmaIEtaIEta[1] = 0.030;
        cut_hoe[0]           = 0.120; cut_hoe[1]           = 0.100;
        cut_ooemoop[0]       = 0.050; cut_ooemoop[1]       = 0.050;
        cut_d0vtx[0]         = 0.020; cut_d0vtx[1]         = 0.020;
        cut_dzvtx[0]         = 0.100; cut_dzvtx[1]         = 0.100;
        cut_vtxFit[0]        = true ; cut_vtxFit[1]        = true;
        cut_mHits[0]         = 0    ; cut_mHits[1]         = 0;
        if (pt >= 20.0) {
            cut_iso[0] = 0.100; cut_iso[1] = 0.100;
        }
        else {
            cut_iso[0] = 0.100; cut_iso[1] = 0.070;
        }
    } 
    else {
        std::cout << "[EgammaCutBasedEleId::TestWP] Undefined working point" << std::endl;
    }

    // choose cut if barrel or endcap
    unsigned int idx = isEB ? 0 : 1;

    // effective area for isolation
    float AEff = ElectronEffectiveArea::GetElectronEffectiveArea(ElectronEffectiveArea::kEleGammaAndNeutralHadronIso03, eta, ElectronEffectiveArea::kEleEAData2011);

    // apply to neutrals
    double rhoPrime = std::max(rho, 0.0);
    double iso_n = std::max(iso_nh + iso_em - rhoPrime * AEff, 0.0);

    // compute final isolation
    double iso = (iso_n + iso_ch) / pt;

    // test cuts
    if (fabs(dEtaIn) < cut_dEtaIn[idx])             mask |= DETAIN;
    if (fabs(dPhiIn) < cut_dPhiIn[idx])             mask |= DPHIIN; 
    if (sigmaIEtaIEta < cut_sigmaIEtaIEta[idx])     mask |= SIGMAIETAIETA;
    if (hoe < cut_hoe[idx])                         mask |= HOE;
    if (fabs(ooemoop) < cut_ooemoop[idx])           mask |= OOEMOOP;
    if (fabs(d0vtx) < cut_d0vtx[idx])               mask |= D0VTX;
    if (fabs(dzvtx) < cut_dzvtx[idx])               mask |= DZVTX;
    if (!cut_vtxFit[idx] || !vtxFitConversion)      mask |= VTXFIT;
    if (mHits <= cut_mHits[idx])                    mask |= MHITS;
    if (iso < cut_iso[idx])                         mask |= ISO;

    // return the mask
    return mask;

}

void EgammaCutBasedEleId::PrintDebug(unsigned int mask)
{
    printf("detain(%i), ",  bool(mask & DETAIN));
    printf("dphiin(%i), ",  bool(mask & DPHIIN));
    printf("sieie(%i), ",   bool(mask & SIGMAIETAIETA));
    printf("hoe(%i), ",     bool(mask & HOE));
    printf("ooemoop(%i), ", bool(mask & OOEMOOP));
    printf("d0vtx(%i), ",   bool(mask & D0VTX));
    printf("dzvtx(%i), ",   bool(mask & DZVTX));
    printf("iso(%i), ",     bool(mask & ISO));
    printf("vtxfit(%i), ",  bool(mask & VTXFIT));
    printf("mhits(%i)\n",   bool(mask & MHITS));
}

