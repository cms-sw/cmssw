#ifndef CLUSTER_SHAPES_H
#define CLUSTER_SHAPES_H
#include <vector>
#include <cmath>

namespace l1t{
    //  this class is design to contain and compute 
    //  efficiently the cluster shapes
    //  running only once on the cluster members.
    class ClusterShapes{
    private: 
        float sum_e = 0.0;
        float sum_e2 = 0.0;
        float sum_logE = 0.0;
        int n=0.0;

        float emax = 0.0;

        float sum_w =0.0; // just here for clarity
        float sum_eta = 0.0;
        float sum_r  = 0.0;
        // i will discriminate using the rms in -pi,pi or in 0,pi
        float sum_phi_0 = 0.0; // computed in -pi,pi
        float sum_phi_1 = 0.0; // computed in 0, 2pi

        float sum_eta2=0.0;
        float sum_r2 = 0.0;
        float sum_phi2_0=0.0; //computed in -pi,pi
        float sum_phi2_1=0.0; //computed in 0,2pi

        // off diagonal element of the tensor
        float sum_eta_r =0.0;
        float sum_r_phi_0 = 0.0;
        float sum_r_phi_1 = 0.0;
        float sum_eta_phi_0 = 0.0;
        float sum_eta_phi_1 = 0.0;

        // caching of informations
        mutable bool isPhi0 = true;
        mutable bool modified_ = false; // check wheneever i need 
    public:
        ClusterShapes(){}
        ClusterShapes(float e, float eta, float phi, float r) { Init(e,eta,phi,r);}
        ~ClusterShapes(){}
        ClusterShapes(const ClusterShapes&x)= default;
        //init an empty cluster
        void Init(float e, float eta, float phi, float r=0.);
        inline void Add(float e, float eta, float phi, float r=0.0)
        { (*this) += ClusterShapes(e,eta,phi,r);}
        

        // --- this is what I want out:
        float SigmaEtaEta() const ;
        float SigmaPhiPhi() const ;
        float SigmaRR() const ;
        // ----
        float Phi() const ;
        float R() const ;
        float Eta() const ;
        inline int N()const {return n;}
        // --
        float SigmaEtaR()const ;
        float SigmaEtaPhi() const ;
        float SigmaRPhi() const ;
        // -- 
        float LogEoverE() const { return sum_logE/sum_e;}
        float eD() const { return std::sqrt(sum_e2)/sum_e;}

        ClusterShapes operator+(const ClusterShapes &);
        void operator+=(const ClusterShapes &); 
        ClusterShapes& operator=(const ClusterShapes &) = default;
    };

}; // end namespace

#endif

// -----------------------------------
// Local Variables:
// mode:c++
// indent-tabs-mode:nil
// tab-width:4
// c-basic-offset:4
// End:
// vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4
