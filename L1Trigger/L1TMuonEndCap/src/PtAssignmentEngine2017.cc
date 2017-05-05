#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2017.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineAux2017.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PtLutVarCalc.h"

#include <cassert>
#include <iostream>
#include <sstream>

const PtAssignmentEngineAux2017& PtAssignmentEngine2017::aux() const {
  static const PtAssignmentEngineAux2017 instance; // KK: arguable design solution, but const qualifier makes it thread-safe anyway
  return instance;
}

PtAssignmentEngine::address_t PtAssignmentEngine2017::calculate_address(const EMTFTrack& track) const {
    address_t address = 0;

    EMTFPtLUT ptlut = track.PtLUT(); //KK: Who sets ptlut.address field?

    short dPh12 = ptlut.delta_ph[0] * ptlut.sign_ph[0];
    short dPh13 = ptlut.delta_ph[1] * ptlut.sign_ph[1];
    short dPh14 = ptlut.delta_ph[2] * ptlut.sign_ph[2];
    short dPh23 = ptlut.delta_ph[3] * ptlut.sign_ph[3];
    short dPh24 = ptlut.delta_ph[4] * ptlut.sign_ph[4];
    short dPh34 = ptlut.delta_ph[5] * ptlut.sign_ph[5];

    short dPhSign=0, dPhSum4=0, dPhSum4A=0, dPhSum3=0, dPhSum3A=0, outStPh=0;

    // compress dPhis
    CalcDeltaPhis( dPh12, dPh13, dPh14, dPh23, dPh24, dPh34, dPhSign,
                   dPhSum4, dPhSum4A, dPhSum3, dPhSum3A, outStPh,
                   ptlut.mode );

    short dTh12 = ptlut.delta_th[0] * ptlut.sign_th[0];
    short dTh13 = ptlut.delta_th[1] * ptlut.sign_th[1];
    short dTh14 = ptlut.delta_th[2] * ptlut.sign_th[2];
    short dTh23 = ptlut.delta_th[3] * ptlut.sign_th[3];
    short dTh24 = ptlut.delta_th[4] * ptlut.sign_th[4];
    short dTh34 = ptlut.delta_th[5] * ptlut.sign_th[5];
   
    // compress dThetas
    CalcDeltaThetas( dTh12, dTh13, dTh14, dTh23, dTh24, dTh34, ptlut.mode );


    EMTFHitCollection hits = track.Hits();
    EMTFHit stub[5];
    for(unsigned int h=0; h<hits.size(); h++){
        if( hits[h].Station()<1 || hits[h].Station()>4 ){
            continue;
        }
        stub[ hits[h].Station() ] = hits[h];
    }


    int st1_ring2 = (stub[1].Station()==1 ? (stub[1].Ring() == 2) : -99);

    int theta = CalcTrackTheta( ptlut.theta, st1_ring2, ptlut.mode );

    int clct1 = (stub[1].Station()==1 ? stub[1].Pattern() : -99);
    int clct2 = (stub[2].Station()==2 ? stub[2].Pattern() : -99);
    int clct3 = (stub[3].Station()==3 ? stub[3].Pattern() : -99);
    int clct4 = (stub[4].Station()==4 ? stub[4].Pattern() : -99);

    short theta_rpc_clct1 = 0; // CalcThetaRPCclct1(theta,clct1,clct2,clct3,clct4); //KK do be implemented

    int sign23 = (dPh12*dPh13>=0 ? 0 : 1);
    int sign34 = (dPh12*dPh34>=0 ? 0 : 1);

    switch (ptlut.mode) {
        case 3:   // 1-2
        break;
        case 5:   // 1-3
        break;
        case 9:   // 1-4
        break;
        case 6:   // 2-3
        break;
        case 10:  // 2-4
        break;
        case 12:  // 3-4
        break;
        case 7:   // 1-2-3
        break;
        case 11:  // 1-2-4
        break;
        case 13:  // 1-3-4
        break;
        case 14:  // 2-3-4
        break;
        case 15:  // 1-2-3-4
            address |= (dPh12      & ((1<<7)-1)) << (0);
            address |= (dPh23      & ((1<<5)-1)) << (0+7);
            address |= (dPh34      & ((1<<5)-1)) << (0+7+5);
            address |= (sign23     & ((1<<1)-1)) << (0+7+5+5);
            address |= (sign34     & ((1<<1)-1)) << (0+7+5+5+1);
            address |= (dTh14      & ((1<<2)-1)) << (0+7+5+5+1+1);
            address |= (theta_rpc_clct1 & ((1<<8)-1)) << (0+7+5+5+1+1+2);
            address |= (0x1        & ((1<<1)-1)) << (0+7+5+5+1+1+2+8);
        break;
        default:
        break;
    }

    return address;
}

float PtAssignmentEngine2017::calculate_pt_xml(const address_t& address) {
    float pt = 0.;

    if(address == 0)  // invalid address
        return -1;  // return pt;

    int mode = ((address >> (30-1)) ? 15 : 0);

    auto contain = [](const std::vector<int>& vec, int elem) {
        return (std::find(vec.begin(), vec.end(), elem) != vec.end());
    };

    bool is_good_mode = contain(allowedModes_, mode);

    if(!is_good_mode)  // invalid mode
        return -1;  // return pt;

    int theta     = 0;
    int St1_ring2 = 0;
    int dPhi12    = 0;
    int dPhi23    = 0;
    int dPhi34    = 0;
    int dPhi13    = 0;
    int dPhi14    = 0;
    int dPhi24    = 0;
    int FR1       = 0;
    int bend_1    = 0;
    int dPhiSum4  = 0;
    int dPhiSum4A = 0;
    int dPhiSum3  = 0;
    int dPhiSum3A = 0;
    int outStPhi  = 0;
    int dTh_14    = 0;
    int RPC_1     = 0;
    int RPC_2     = 0;
    int RPC_3     = 0;
    int RPC_4     = 0;

    switch (mode) {
        case 3:   // 1-2
        break;
        case 5:   // 1-3
        break;
        case 9:   // 1-4
        break;
        case 6:   // 2-3
        break;
        case 10:  // 2-4
        break;
        case 12:  // 3-4
        break;
        case 7:   // 1-2-3
        break;
        case 11:  // 1-2-4
        break;
        case 13:  // 1-3-4
        break;
        case 14:  // 2-3-4
        break;
        case 15: {  // 1-2-3-4
            int sign23 = 0;
            int sign34 = 0;
            int theta_rpc_clct1 = 0;

            dPhi12    = (address >> (0))                    & ((1<<7)-1);
            dPhi23    = (address >> (0+7))                  & ((1<<5)-1);
            dPhi34    = (address >> (0+7+5))                & ((1<<5)-1);
            sign23    = (address >> (0+7+5+5))              & ((1<<1)-1);
            sign34    = (address >> (0+7+5+5+1))            & ((1<<1)-1);
            dTh_14    = (address >> (0+7+5+5+1+1))          & ((1<<2)-1);
            theta_rpc_clct1 = (address >> (0+7+5+5+1+1+2))  & ((1<<8)-1);
            // decode the word above into the RPC1,2,3,4 ...
            RPC_1 = theta_rpc_clct1;

            // signs should be consistend with the above notations
            dPhi23 *= (sign23?1:-1);
            dPhi34 *= (sign34?1:-1);
            // need an equivalent of CalcDeltaPhis, for now port pieces of code from
            // https://github.com/abrinke1/EMTFPtAssign2017/blob/b14f84bd532f613098ceb8dab1546eb2353499f8/src/PtLutVarCalc.cc#L100
            dPhi13 = dPhi12 + dPhi23;
            dPhi14 = dPhi13 + dPhi34;
            dPhi24 = dPhi23 + dPhi34;
            dPhiSum4  = dPhi12 + dPhi13 + dPhi14 + dPhi23 + dPhi24 + dPhi34;
            dPhiSum4A = abs(dPhi12) + abs(dPhi13) + abs(dPhi14) + abs(dPhi23) + abs(dPhi24) + abs(dPhi34);
            int devSt1 = abs(dPhi12) + abs(dPhi13) + abs(dPhi14);
            int devSt2 = abs(dPhi12) + abs(dPhi23) + abs(dPhi24);
            int devSt3 = abs(dPhi13) + abs(dPhi23) + abs(dPhi34);
            int devSt4 = abs(dPhi14) + abs(dPhi24) + abs(dPhi34);
    
            if      (devSt4 > devSt3 && devSt4 > devSt2 && devSt4 > devSt1)  outStPhi = 4;
            else if (devSt3 > devSt4 && devSt3 > devSt2 && devSt3 > devSt1)  outStPhi = 3;
            else if (devSt2 > devSt4 && devSt2 > devSt3 && devSt2 > devSt1)  outStPhi = 2;
            else if (devSt1 > devSt4 && devSt1 > devSt3 && devSt1 > devSt2)  outStPhi = 1;
            else                                                             outStPhi = 0;
    
            if      (outStPhi == 4) {
                dPhiSum3  = dPhi12 + dPhi13 + dPhi23;
                dPhiSum3A = abs(dPhi12) + abs(dPhi13) + abs(dPhi23);
            } else if (outStPhi == 3) {
                dPhiSum3  = dPhi12 + dPhi14 + dPhi24;
                dPhiSum3A = abs(dPhi12) + abs(dPhi14) + abs(dPhi24);
            } else if (outStPhi == 2) {
                dPhiSum3  = dPhi13 + dPhi14 + dPhi34;
                dPhiSum3A = abs(dPhi13) + abs(dPhi14) + abs(dPhi34);
            } else {
                dPhiSum3  = dPhi23 + dPhi24 + dPhi34;
                dPhiSum3A = abs(dPhi23) + abs(dPhi24) + abs(dPhi34);
            }
        } break;
        default:
        break;
    }

    // KK: sequence of variables here should exaclty match <Variables> block produced by TMVA

    std::vector<int> predictors = {
            theta, St1_ring2, dPhi12, dPhi23, dPhi34, dPhi13, dPhi14, dPhi24, FR1, bend_1,
            dPhiSum4, dPhiSum4A, dPhiSum3, dPhiSum3A, outStPhi, dTh_14, RPC_1, RPC_2, RPC_3, RPC_4
    };

    std::vector<Double_t> tree_data(predictors.cbegin(),predictors.cend());

    auto tree_event = std::make_unique<emtf::Event>();
    tree_event->predictedValue = 0;
    tree_event->data = tree_data;

    forests_.at(mode).predictEvent(tree_event.get(), 400);

    float log2pt = tree_event->predictedValue;  // is actually log2(pT)

    if (verbose_ > 1) {
        std::cout << "mode_inv: " << mode << " log2(pT[GeV]): " << log2pt << std::endl;
        std::cout << "dPhi12: " << dPhi12 << " dPhi13: " << dPhi13 << " dPhi14: " << dPhi14
            << " dPhi23: " << dPhi23 << " dPhi24: " << dPhi24 << " dPhi34: " << dPhi34 << std::endl;
        std::cout << " dTheta14: " << dTh_14 << std::endl;
        std::cout << "FR1: " << FR1 << std::endl;
    }

    assert(log2pt > 0);

    pt = pow(2,log2pt);

    return pt;
}
