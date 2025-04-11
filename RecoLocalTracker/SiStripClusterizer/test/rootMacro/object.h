#pragma once

struct Track{

        int idx;
        float pt;
        float eta;
        float phi;
        float Dxy1;
        float DxyError1;
        float Dz1;
        float DzError1;

        Int_t    algo;
        Int_t    trkNHit;
        Int_t    trkNdof;
        Int_t    trkNlayer;
        float    trkChi2;
        float    trkPtError;

        Track(): idx(0),
        pt(0), eta(0), phi(0), Dxy1(0), DxyError1(0),
        Dz1(0), DzError1(0) {};

        Track(int& in_idx,
            const float& in_pt,
            const float& in_eta,
            const float& in_phi,
            const float& in_Dxy1,
            const float& in_DxyError1,
            const float& in_Dz1,
            const float& in_DzError1,
            const Int_t& in_algo,
            const Int_t&  in_trkNHit,
            const Int_t&  in_trkNdof,
            const Int_t&  in_trkNlayer,
            const float&  in_trkChi2,
            const float&  in_trkPtError
           ):
           idx(in_idx),
           pt(in_pt),
           eta(in_eta),
           phi(in_phi),
           Dxy1(in_Dxy1),
           DxyError1(in_DxyError1),
           Dz1(in_Dz1),
           DzError1(in_DzError1),
           algo(in_algo),
           trkNHit(in_trkNHit),
           trkNdof(in_trkNdof),
           trkNlayer(in_trkNlayer),
           trkChi2(in_trkChi2),
           trkPtError(in_trkPtError)

        {};

};

struct Jet{
        
        int   idx;
        float pt;
        float eta;
        float phi;
        float mass;
        
        Jet(): idx(0),
        pt(0), eta(0), phi(0), mass(0) {};
        
        Jet(const int& in_idx,
            const float& in_pt,
            const float& in_eta,
            const float& in_phi,
            const float& in_mass
           ):
           idx(in_idx),
           pt(in_pt),
           eta(in_eta),
           phi(in_phi),
           mass(in_mass)
        {};
};
