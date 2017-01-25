#include "DataFormats/L1THGCal/interface/ClusterShapes.h"
#include <cmath>

using namespace l1t;

ClusterShapes ClusterShapes::operator+(const ClusterShapes& x)
{
    ClusterShapes cs(*this); // copy constructor
    cs += x;
    return cs;
}


void ClusterShapes::operator +=(const ClusterShapes &x){

	sum_e += x.sum_e;
	sum_e2 += x.sum_e2;
	sum_logE += x.sum_logE;
	n += x.n;

    sum_w += x.sum_w;

    emax = (emax> x.emax) ? emax: x.emax;

    // mid-point
    sum_eta += x.sum_eta;
    sum_phi_0 += x.sum_phi_0; //
    sum_phi_1 += x.sum_phi_1; //
    sum_r   += x.sum_r;

    // square
    sum_eta2 += x.sum_eta2;
    sum_phi2_0 += x.sum_phi2_0;
    sum_phi2_1 += x.sum_phi2_1;
    sum_r2 += x.sum_r2;

    // off diagonal
    sum_eta_r     += x.sum_eta_r    ;
    sum_r_phi_0   += x.sum_r_phi_0  ;
    sum_r_phi_1   += x.sum_r_phi_1  ;
    sum_eta_phi_0 += x.sum_eta_phi_0;
    sum_eta_phi_1 += x.sum_eta_phi_1;

}


// -------------- CLUSTER SHAPES ---------------
void ClusterShapes::Init(float e ,float eta, float phi, float r){
    if (e<=0 )  return;
    sum_e = e;
    sum_e2 = e*e;
    sum_logE = std::log(e);

    float w = e;

    n=1;
    
    sum_w = w;

    sum_phi_0 = w *( phi );
    sum_phi_1 = w* (phi + M_PI);
    sum_r = w * r;
    sum_eta = w * eta;

    //--
    sum_r2     += w * (r*r);
    sum_eta2   += w * (eta*eta);
    sum_phi2_0 += w * (phi*phi);
    sum_phi2_1 += w * (phi+M_PI)*(phi+M_PI);

    // -- off diagonal
    sum_eta_r += w * (r*eta);
    sum_r_phi_0 += w* (r *phi);
    sum_r_phi_1 += w* r *(phi + M_PI);
    sum_eta_phi_0 += w* (eta *phi);
    sum_eta_phi_1 += w* eta * (phi+M_PI);

}
// ------
float ClusterShapes::Eta()const { return sum_eta/sum_w;}
float ClusterShapes::R() const { return sum_r/sum_w;}

float ClusterShapes::SigmaEtaEta()const {return  sum_eta2/sum_w - Eta()*Eta();} 

float ClusterShapes::SigmaRR()const { return sum_r2/sum_w - R() *R();}


float ClusterShapes::SigmaPhiPhi()const { 
    float phi_0 = (sum_phi_0 / sum_w);
    float phi_1 = (sum_phi_1 / sum_w);
    float spp_0 = sum_phi2_0 / sum_w  - phi_0*phi_0;
    float spp_1 = sum_phi2_1 / sum_w  - phi_1*phi_1;

    if  (spp_0 < spp_1 )
    {
        float phi = phi_0;
        isPhi0=true;
        while (phi < - M_PI) phi += 2*M_PI;
        while (phi >   M_PI) phi -= 2*M_PI;
        return spp_0;
    }
    else 
    {
        float phi = phi_1 ;
        isPhi0=false;
        while (phi < - M_PI) phi += 2*M_PI;
        while (phi >   M_PI) phi -= 2*M_PI;
        return spp_1;
    }
}

float ClusterShapes::Phi()const {
    SigmaPhiPhi(); //update phi
    if (isPhi0) return (sum_phi_0 / sum_w);
    else return (sum_phi_1 / sum_w);
}


// off - diagonal
float ClusterShapes::SigmaEtaR() const { return -(sum_eta_r / sum_w - Eta() *R()) ;}

float ClusterShapes::SigmaEtaPhi()const {
    SigmaPhiPhi() ; // decide which phi use, update phi

    if (isPhi0)
        return -(sum_eta_phi_0 /sum_w - Eta()*(sum_phi_0 / sum_w));
    else
        return -(sum_eta_phi_1 / sum_w - Eta()*(sum_phi_1 / sum_w));
}

float ClusterShapes::SigmaRPhi()const {
    SigmaPhiPhi() ; // decide which phi use, update phi
    if (isPhi0)
        return -(sum_r_phi_0 / sum_w - R() *(sum_phi_0 / sum_w));
    else
        return -(sum_r_phi_1 / sum_w - R() * (sum_phi_1 / sum_w));
}

// -----------------------------------
// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
