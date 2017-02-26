#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

using namespace l1t;


HGCalCluster::HGCalCluster( const LorentzVector p4, 
                            int pt,
                            int eta,
                            int phi)
    : L1Candidate(p4, pt, eta, phi)
{
    
    
}


HGCalCluster::~HGCalCluster()
{
}



bool HGCalCluster::isPertinent( const l1t::HGCalTriggerCell &tc, double dist_eta ) const {
    return false;
}

void HGCalCluster::addTC(const l1t::HGCalTriggerCell &tc) const{

    


}


double HGCalCluster::dist(const l1t::HGCalTriggerCell &tc) const{

//    return 0.;
    return reco::deltaR ( tc.p4().Eta(), tc.p4().Phi(), 
                          this->p4().Eta(), this->p4().Phi() );

}


bool HGCalCluster::operator<(const HGCalCluster& cl) const
{
    bool res = false;

    /* Prioratize high pT */
    if(hwPt()<cl.hwPt()) {
        res = true;
    }
    else if(hwPt()==cl.hwPt()) {
        if( abs(hwEta()) > abs( cl.hwEta() ) ) /* Prioratize central clusters */
            res = true;
        else if( abs(hwEta())==abs( cl.hwEta() ) )
            if( hwPhi() > cl.hwPhi())         /* Prioratize small phi (arbitrary) */
                res = true;
    }

    return res;

}

//
//double deltaPhi( double phi1, double phi2) {
//    
//    double dPhi(phi1-phi2);
//    double pi(acos(-1.0));
//    
//    if     ( dPhi <= -pi ) 
//        dPhi+=2.0*pi;
//    else if( dPhi > pi) 
//        dPhi -= 2.0*pi;
//    
//    return dPhi;
//
//}
//
//
//double deltaEta(double eta1, double eta2){
//    double dEta = eta1-eta2;
//    return dEta;
//}
//
//double deltaR(double eta1, double eta2, double phi1, double phi2) {
//    double dEta = deltaEta(eta1, eta2);
//    double dPhi = deltaPhi(phi1, phi2);
//    return sqrt(dEta*dEta+dPhi*dPhi);
//}
