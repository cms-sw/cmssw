// -*- C++ -*-
//
// Package:    EcalDeadChannelRecoveryAlgos
// Class:      EBDeadChannelRecoveryAlgos
//
/**\class EBDeadChannelRecoveryAlgos EBDeadChannelRecoveryAlgos.cc RecoLocalCalo/EcalDeadChannelRecoveryAlgos/src/EBDeadChannelRecoveryAlgos.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
// 
//  Original Author:   Stilianos Kesisoglou - Institute of Nuclear and Particle Physics NCSR Demokritos (Stilianos.Kesisoglou@cern.ch)
//          Created:   Wed Nov 21 11:24:39 EET 2012
// 
//      Nov 21 2012:   First version of the code. Based on the old "EcalDeadChannelRecoveryAlgos.cc" code
//      Feb 14 2013:   Implementation of the criterion to select the "correct" max. cont. crystal.
//


// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapHardcodedTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//#include "Geometry/Vector/interface/GlobalPoint.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EBDeadChannelRecoveryAlgos.h"
#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/CorrectEBDeadChannelsNN.cc"
#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/CrystalMatrixProbabilityEB.cc"

#include <string>
using namespace cms;
using namespace std;


EBDeadChannelRecoveryAlgos::EBDeadChannelRecoveryAlgos(const CaloTopology  * theCaloTopology)
{
    // now do what ever initialization is needed
    calotopo = theCaloTopology;
}




//
// member functions
//

// ------------ method called to for each event  ------------
EcalRecHit EBDeadChannelRecoveryAlgos::correct(const EBDetId Id, const EcalRecHitCollection* hit_collection, std::string algo_, double Sum8Cut, bool* AcceptFlag)
{
    //  Enumeration to switch from custom names within the 3x3 matrix.
    enum { CC=0, UU=1, DD=2, LL=3, LU=4, LD=5, RR=6, RU=7, RD=8 } ;

    double NewEnergy = 0.0;

    double NewEnergy_RelMC = 0.0;
    double NewEnergy_RelDC = 0.0;

    double MNxN_RelMC[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } ;
    double MNxN_RelDC[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 } ;

    double sum8 = 0.0;

    double sum8_RelMC = MakeNxNMatrice_RelMC(Id,hit_collection,MNxN_RelMC,AcceptFlag);
    double sum8_RelDC = MakeNxNMatrice_RelDC(Id,hit_collection,MNxN_RelDC,AcceptFlag);

    //  Only if "AcceptFlag" is true call the ANN
    if ( *AcceptFlag ) {
        if (algo_=="NeuralNetworks") {
            if (sum8_RelDC > Sum8Cut && sum8_RelMC > Sum8Cut) {
            
                NewEnergy_RelMC = CorrectEBDeadChannelsNN(MNxN_RelMC);
                NewEnergy_RelDC = CorrectEBDeadChannelsNN(MNxN_RelDC);
                
                //  Matrices "MNxN_RelMC" and "MNxN_RelDC" have now the full set of energies, the original ones plus 
                //  whatever "estimates" by the ANN for the "dead" xtal. Use those full matrices and calculate probabilities.
                //  
                double SumMNxN_RelMC =  MNxN_RelMC[LU] + MNxN_RelMC[UU] + MNxN_RelMC[RU] + 
                                        MNxN_RelMC[LL] + MNxN_RelMC[CC] + MNxN_RelMC[RR] + 
                                        MNxN_RelMC[LD] + MNxN_RelMC[DD] + MNxN_RelMC[RD] ;
                
                double frMNxN_RelMC[9];  for (int i=0; i<9; i++) { frMNxN_RelMC[i] = MNxN_RelMC[i] / SumMNxN_RelMC ; }
                
                double prMNxN_RelMC  =  EBDiagonal(  frMNxN_RelMC[LU] ) * EBUpDown(  frMNxN_RelMC[UU] ) * EBDiagonal(  frMNxN_RelMC[RU] ) * 
                                        EBReftRight( frMNxN_RelMC[LL] ) * EBCentral( frMNxN_RelMC[CC] ) * EBReftRight( frMNxN_RelMC[RR] ) * 
                                        EBDiagonal(  frMNxN_RelMC[LD] ) * EBUpDown(  frMNxN_RelMC[DD] ) * EBDiagonal(  frMNxN_RelMC[RD] ) ;
                
                double SumMNxN_RelDC =  MNxN_RelDC[LU] + MNxN_RelDC[UU] + MNxN_RelDC[RU] + 
                                        MNxN_RelDC[LL] + MNxN_RelDC[CC] + MNxN_RelDC[RR] + 
                                        MNxN_RelDC[LD] + MNxN_RelDC[DD] + MNxN_RelDC[RD] ;
                
                double frMNxN_RelDC[9];  for (int i=0; i<9; i++) { frMNxN_RelDC[i] = MNxN_RelDC[i] / SumMNxN_RelDC ; }
                
                double prMNxN_RelDC  =  EBDiagonal(  frMNxN_RelDC[LU] ) * EBUpDown(  frMNxN_RelDC[UU] ) * EBDiagonal(  frMNxN_RelDC[RU] ) * 
                                        EBReftRight( frMNxN_RelDC[LL] ) * EBCentral( frMNxN_RelDC[CC] ) * EBReftRight( frMNxN_RelDC[RR] ) * 
                                        EBDiagonal(  frMNxN_RelDC[LD] ) * EBUpDown(  frMNxN_RelDC[DD] ) * EBDiagonal(  frMNxN_RelDC[RD] ) ;
                
                if ( prMNxN_RelDC > prMNxN_RelMC )  { NewEnergy = NewEnergy_RelDC ; sum8 = sum8_RelDC ; } 
                if ( prMNxN_RelDC <= prMNxN_RelMC ) { NewEnergy = NewEnergy_RelMC ; sum8 = sum8_RelMC ; } 
                
                
                //  If the return value of "CorrectDeadChannelsNN" is one of the followin negative values then
                //  it corresponds to an error condition. See "CorrectDeadChannelsNN.cc" for possible values.
                if ( NewEnergy == -1000000.0 ||
                     NewEnergy == -1000001.0 ||
                     NewEnergy == -2000000.0 ||
                     NewEnergy == -3000000.0 ||
                     NewEnergy == -3000001.0 ) { *AcceptFlag=false ; NewEnergy = 0.0 ; }
            }
        }
    }
    
    // Protect against non physical high values
    // From the distribution of (max.cont.xtal / Sum8) we get as limit 5 (hard) and 10 (softer)
    // Choose 10 as highest possible energy to be assigned to the dead channel under any scenario.
    uint32_t flag = 0;
    
    if ( NewEnergy > 10.0 * sum8 ) { *AcceptFlag=false ; NewEnergy = 0.0 ; }

    EcalRecHit NewHit(Id,NewEnergy,0, flag);
    
    return NewHit;

}


// ------------ MakeNxNMatrice_RelMC  ------------
double EBDeadChannelRecoveryAlgos::MakeNxNMatrice_RelMC(EBDetId itID,const EcalRecHitCollection* hit_collection, double *MNxN_RelMC, bool* AcceptFlag) {

    //  Since ANN corrects within a 3x3 window, the possible candidate 3x3 windows that contain 
    //  the "dead" crystal form a 5x5 window around it (totaly eight 3x3 windows overlapping).
    //  Get this 5x5 and locate the Max.Contain.Crystal within.
    const CaloSubdetectorTopology* topology=calotopo->getSubdetectorTopology(DetId::Ecal,EcalBarrel);

    std::vector<DetId> NxNaroundDC = topology->getWindow(itID,5,5);     //  Get the 5x5 window around the "dead" crystal -> vector "NxNaroundDC"

    EBDetId EBCellMax ;         //  Create a null EBDetId
    double EnergyMax = 0.0;

    //  Loop over all cells in the vector "NxNaroundDC", and for each cell find it's energy
    //  (from the EcalRecHits collection). Use this energy to detect the Max.Cont.Crystal.
    std::vector<DetId>::const_iterator theCells;

    for (theCells = NxNaroundDC.begin(); theCells != NxNaroundDC.end(); ++theCells) {
    
        EBDetId EBCell = EBDetId(*theCells);

        if (!EBCell.null()) {
        
            EcalRecHitCollection::const_iterator goS_it = hit_collection->find(EBCell);
            
            if ( goS_it !=  hit_collection->end() && goS_it->energy() >= EnergyMax ) {
            
                EnergyMax = goS_it->energy();
                EBCellMax = EBCell;
                
            }
            
        } else {
        
            continue; 
                       
        }
        
    }
    
    //  No Max.Cont.Crystal found, return back with no changes.
    if ( EBCellMax.null() ) { *AcceptFlag=false ; return 0.0 ; }

    //  If the Max.Cont.Crystal found is at the EB boundary (+/- 85) do nothing since we need a full 3x3 around it.
    if ( EBCellMax.ieta() == 85 || EBCellMax.ieta() == -85 ) { *AcceptFlag=false ; return 0.0 ; }

    //  Take the Max.Cont.Crystal as reference and get the 3x3 around it.
    std::vector<DetId> NxNaroundMaxCont = topology->getWindow(EBCellMax,3,3);
    
    //  Check that the "dead" crystal belongs to the 3x3 around  Max.Cont.Crystal
    bool dcIn3x3 = false ;
    
    std::vector<DetId>::const_iterator testCell;
    
    for (testCell = NxNaroundMaxCont.begin(); testCell != NxNaroundMaxCont.end(); ++testCell) {
    
        EBDetId EBtestCell = EBDetId(*testCell);
        
        if ( itID == EBtestCell ) { dcIn3x3 = true ; } 
    
    }
    
    //  If the "dead" crystal is outside the 3x3 then do nothing.
    if (!dcIn3x3) { *AcceptFlag=false ; return 0.0 ; }
    
    //  Define the ieta and iphi steps (zero, plus, minus)
    int ietaZ = EBCellMax.ieta() ;
    int ietaP = ( ietaZ == -1 ) ?  1 : ietaZ + 1 ;
    int ietaN = ( ietaZ ==  1 ) ? -1 : ietaZ - 1 ;

    int iphiZ = EBCellMax.iphi() ;
    int iphiP = ( iphiZ == 360 ) ?   1 : iphiZ + 1 ;
    int iphiN = ( iphiZ ==   1 ) ? 360 : iphiZ - 1 ;
    
    enum { CC=0, UU=1, DD=2, LL=3, LU=4, LD=5, RR=6, RU=7, RD=8 } ;

    for (int i=0; i<9; i++) { MNxN_RelMC[i] = 0.0 ; }
    
    //  Loop over all cells in the vector "NxNaroundMaxCont", and fill the MNxN_RelMC matrix
    //  to be passed to the ANN for prediction.
    std::vector<DetId>::const_iterator itCells;

    for (itCells = NxNaroundMaxCont.begin(); itCells != NxNaroundMaxCont.end(); ++itCells) {
    
        EBDetId EBitCell = EBDetId(*itCells);

        if (!EBitCell.null()) {

            EcalRecHitCollection::const_iterator goS_it = hit_collection->find(EBitCell);
            
            if ( goS_it !=  hit_collection->end() ) { 
            
                if       ( EBitCell.ieta() == ietaP && EBitCell.iphi() == iphiP ) { MNxN_RelMC[RU] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaP && EBitCell.iphi() == iphiZ ) { MNxN_RelMC[RR] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaP && EBitCell.iphi() == iphiN ) { MNxN_RelMC[RD] = goS_it->energy(); }
                
                else if  ( EBitCell.ieta() == ietaZ && EBitCell.iphi() == iphiP ) { MNxN_RelMC[UU] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaZ && EBitCell.iphi() == iphiZ ) { MNxN_RelMC[CC] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaZ && EBitCell.iphi() == iphiN ) { MNxN_RelMC[DD] = goS_it->energy(); }
                
                else if  ( EBitCell.ieta() == ietaN && EBitCell.iphi() == iphiP ) { MNxN_RelMC[LU] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaN && EBitCell.iphi() == iphiZ ) { MNxN_RelMC[LL] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaN && EBitCell.iphi() == iphiN ) { MNxN_RelMC[LD] = goS_it->energy(); }

                else { *AcceptFlag=false ; return 0.0 ;}
            
            }
            
        } else {
        
            continue; 
                       
        }

    }

    //  Get the sum of 8
    double ESUMis = 0.0 ; 
    
    for (int i=0; i<9; i++) { ESUMis = ESUMis + MNxN_RelMC[i] ; }

    *AcceptFlag=true ;
    
    return ESUMis;
}


// ------------ MakeNxNMatrice_RelDC  ------------
double EBDeadChannelRecoveryAlgos::MakeNxNMatrice_RelDC(EBDetId itID,const EcalRecHitCollection* hit_collection, double *MNxN_RelDC, bool* AcceptFlag) {

    //  Since ANN corrects within a 3x3 window, get the 3x3 the "dead" crystal.
    //  This method works exactly as the "MakeNxNMatrice_RelMC" but doesn't scans to locate
    //  the Max.Contain.Crystal around the "dead" crystal. It simply gets the 3x3 centered around the "dead" crystal.
    const CaloSubdetectorTopology* topology=calotopo->getSubdetectorTopology(DetId::Ecal,EcalBarrel);

    //  If the "dead" crystal is at the EB boundary (+/- 85) do nothing since we need a full 3x3 around it.
    if ( itID.ieta() == 85 || itID.ieta() == -85 ) { *AcceptFlag=false ; return 0.0 ; }

    //  Take the "dead" crystal as reference and get the 3x3 around it.
    std::vector<DetId> NxNaroundRefXtal = topology->getWindow(itID,3,3);
    
    //  Define the ieta and iphi steps (zero, plus, minus)
    int ietaZ = itID.ieta() ;
    int ietaP = ( ietaZ == -1 ) ?  1 : ietaZ + 1 ;
    int ietaN = ( ietaZ ==  1 ) ? -1 : ietaZ - 1 ;

    int iphiZ = itID.iphi() ;
    int iphiP = ( iphiZ == 360 ) ?   1 : iphiZ + 1 ;
    int iphiN = ( iphiZ ==   1 ) ? 360 : iphiZ - 1 ;
    
    enum { CC=0, UU=1, DD=2, LL=3, LU=4, LD=5, RR=6, RU=7, RD=8 } ;

    for (int i=0; i<9; i++) { MNxN_RelDC[i] = 0.0 ; }
    
    //  Loop over all cells in the vector "NxNaroundRefXtal", and fill the MNxN_RelDC matrix
    //  to be passed to the ANN for prediction.
    std::vector<DetId>::const_iterator itCells;

    for (itCells = NxNaroundRefXtal.begin(); itCells != NxNaroundRefXtal.end(); ++itCells) {
    
        EBDetId EBitCell = EBDetId(*itCells);

        if (!EBitCell.null()) {

            EcalRecHitCollection::const_iterator goS_it = hit_collection->find(EBitCell);
            
            if ( goS_it !=  hit_collection->end() ) { 
            
                if       ( EBitCell.ieta() == ietaP && EBitCell.iphi() == iphiP ) { MNxN_RelDC[RU] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaP && EBitCell.iphi() == iphiZ ) { MNxN_RelDC[RR] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaP && EBitCell.iphi() == iphiN ) { MNxN_RelDC[RD] = goS_it->energy(); }
                
                else if  ( EBitCell.ieta() == ietaZ && EBitCell.iphi() == iphiP ) { MNxN_RelDC[UU] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaZ && EBitCell.iphi() == iphiZ ) { MNxN_RelDC[CC] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaZ && EBitCell.iphi() == iphiN ) { MNxN_RelDC[DD] = goS_it->energy(); }
                
                else if  ( EBitCell.ieta() == ietaN && EBitCell.iphi() == iphiP ) { MNxN_RelDC[LU] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaN && EBitCell.iphi() == iphiZ ) { MNxN_RelDC[LL] = goS_it->energy(); }
                else if  ( EBitCell.ieta() == ietaN && EBitCell.iphi() == iphiN ) { MNxN_RelDC[LD] = goS_it->energy(); }

                else { *AcceptFlag=false ; return 0.0 ;}
            
            }
            
        } else {
        
            continue; 
                       
        }

    }

    //  Get the sum of 8
    double ESUMis = 0.0 ; 
    
    for (int i=0; i<9; i++) { ESUMis = ESUMis + MNxN_RelDC[i] ; }

    *AcceptFlag=true ;
    
    return ESUMis;
}

