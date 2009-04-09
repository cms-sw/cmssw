// -*- C++ -*-
//
// Package:    EcalDeadChannelRecoveryAlgos
// Class:      EcalDeadChannelRecoveryAlgos
// 
/**\class EcalDeadChannelRecoveryAlgos EcalDeadChannelRecoveryAlgos.cc RecoLocalCalo/EcalDeadChannelRecoveryAlgos/src/EcalDeadChannelRecoveryAlgos.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Georgios Daskalakis
//         Created:  Thu Apr 12 17:02:06 CEST 2007
// $Id: EcalDeadChannelRecoveryAlgos.cc,v 1.4 2007/05/18 09:38:08 gdaskal Exp $
//
// May 4th 2007 S. Beauceron : modification of MakeNxNMatrice in order to use vectors
//



// Geometry
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
//#include "Geometry/Vector/interface/GlobalPoint.h"

// Reconstruction Classes
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryAlgos.h"
#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/CorrectDeadChannelsClassic.cc"
#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/CorrectDeadChannelsNN.cc"

#include <string>
using namespace cms;
using namespace std;


EcalDeadChannelRecoveryAlgos::EcalDeadChannelRecoveryAlgos(const CaloTopology theCaloTopology)
//EcalDeadChannelRecoveryAlgos::EcalDeadChannelRecoveryAlgos(const edm::ESHandle<CaloTopology> & theCaloTopology)
{
  //now do what ever initialization is needed
  calotopo = theCaloTopology;
}


EcalDeadChannelRecoveryAlgos::~EcalDeadChannelRecoveryAlgos()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
EcalRecHit EcalDeadChannelRecoveryAlgos::correct(const EBDetId Id, const EcalRecHitCollection* hit_collection, string algo_, double Sum8Cut)
{
  double NewEnergy=0.0;
  
  double MNxN[121];

  if(algo_=="Spline"){
     
    if(MakeNxNMatrice(Id,hit_collection,MNxN)>Sum8Cut){
      NewEnergy = CorrectDeadChannelsClassic(MNxN,Id.ieta());
    }
  }else if(algo_=="NeuralNetworks"){
   
    if(MakeNxNMatrice(Id,hit_collection,MNxN)>Sum8Cut){
      NewEnergy = CorrectDeadChannelsNN(MNxN);
    }
  }
  

  EcalRecHit NewHit(Id,NewEnergy,0);
  return NewHit;

}

// FIXME -- temporary backward compatibility
EcalRecHit EcalDeadChannelRecoveryAlgos::Correct(const EBDetId Id, const EcalRecHitCollection* hit_collection, string algo_, double Sum8Cut)
{
        return correct(Id, hit_collection, algo_, Sum8Cut);
}

//==============================================================================================================
//==============================================================================================================
//==============================================================================================================
//==============================================================================================================



double EcalDeadChannelRecoveryAlgos::MakeNxNMatrice(EBDetId itID,const EcalRecHitCollection* hit_collection, double *MNxN){



  //Build NxN around a given cristal
  for(int i=0; i<121;i++)MNxN[i]=0.0;
  
  //cout<<"===================================================================="<<endl;
  //cout<<" Dead Cell CENTRAL  eta = "<< itID.ieta()<<" ,  phi = "<< itID.iphi()<<endl;

  const CaloSubdetectorTopology* topology=calotopo.getSubdetectorTopology(DetId::Ecal,EcalBarrel);
  int size =5;
  std::vector<DetId> NxNaroundDC = topology->getWindow(itID,size,size);
  
  
  //cout<<"NxNaroundDC size is = "<<NxNaroundDC.size()<<endl;
  vector<DetId>::const_iterator theCells;
  
  EBDetId EBCellMax = itID;
  double EnergyMax = 0.0;

  for(theCells=NxNaroundDC.begin();theCells<NxNaroundDC.end();theCells++){
    EBDetId EBCell = EBDetId(*theCells);
      
    // We Will look for the cristal with maximum energy 
    if(!EBCell.null()){
      EcalRecHitCollection::const_iterator goS_it = hit_collection->find(EBCell);
      if( goS_it !=  hit_collection->end() && goS_it->energy()>=EnergyMax){EnergyMax=goS_it->energy(); EBCellMax = EBCell;}
    }else{
      continue; 
    }
  } 
  if(EBCellMax.null()){cout<<" Error No maximum found around dead channel, no corrections applied"<<endl;return 0;}
  //cout << " Max Cont Crystal eta phi E = " << EBCellMax.ieta() <<" "<< EBCellMax.iphi() <<" "<< EnergyMax<<endl;


  
  //NxNaroundMaxCont with N==11
  // 000 is now the maximum containement one:
  ////////////////////////////
  //Any modification of the following parameters will require changes in the Correction algos
  // The window is large enought to avoid modification of the number.
  int FixedSize =11;
  std::vector<DetId> NxNaroundMaxCont = topology->getWindow(EBCellMax,FixedSize,FixedSize);

  double ESUMis=0.0;
  int theIndex=0;

  vector<DetId>::const_iterator itCells;
  for(itCells=NxNaroundMaxCont.begin();itCells<NxNaroundMaxCont.end();itCells++){
    EBDetId EBitCell = EBDetId(*itCells);
    int CReta = EBitCell.ieta();
    int CRphi = EBitCell.iphi();

    double Energy = 0.0;
    if(!EBitCell.null()){
      EcalRecHitCollection::const_iterator goS_it = hit_collection->find(EBitCell);
      if( goS_it !=  hit_collection->end() ) Energy=goS_it->energy();
    }

    //cout<<"Around DC we have eta,phi,E "<<CReta<<" "<<CRphi<<" "<<Energy<<endl;

    //============
    int ietaCorr = 0;
    if((CReta * EBCellMax.ieta()) < 0 && EBCellMax.ieta()>0 )ietaCorr= -1;
    if((CReta * EBCellMax.ieta()) < 0 && EBCellMax.ieta()<0 )ietaCorr=  1;
    if((CReta * EBCellMax.ieta()) > 0 )ietaCorr= 0;
    int ieta = -( (CReta -ietaCorr) - EBCellMax.ieta() ) + int((FixedSize - 1)/2);

    int iphiCorr = 0;
    if((CRphi - EBCellMax.iphi())> 50)iphiCorr= 360;
    if((CRphi - EBCellMax.iphi())<-50)iphiCorr=-360;
    int iphi = CRphi - EBCellMax.iphi() - iphiCorr + int((FixedSize - 1)/2);

    int MIndex = ieta+iphi*FixedSize;
    if(abs(CReta)<=85)
    MNxN[MIndex]=Energy;
    
    
    //============



    
    //We add up the energy in 5x5 around the MaxCont to decide if we will correct for DCs
    if(theIndex>=36 && theIndex<=40)ESUMis +=  Energy;
    if(theIndex>=47 && theIndex<=51)ESUMis +=  Energy;
    if(theIndex>=58 && theIndex<=62)ESUMis +=  Energy;
    if(theIndex>=69 && theIndex<=73)ESUMis +=  Energy;
    if(theIndex>=80 && theIndex<=84)ESUMis +=  Energy;
    theIndex++;
  }
  //cout<<"Around MaxCont Collected Energy in 5x5 is =  "<< ESUMis  <<endl;
  //So now, we have a vector which is ordered around the Maximum containement and which contains a dead channel as:
    //Filling of the vector : NxNaroundDC with N==11 Typo are possible...
    // 000 is Maximum containement which is in +/- 5 from DC
    //
    // 120 119 118 117 116 115 114 113 112 111 110  
    // 109 108 107 106 105 104 103 102 101 100 099 
    // 098 097 096 095 094 093 092 091 090 089 088
    // 087 086 085 084 083 082 081 080 079 078 077
    // 076 075 074 073 072 071 070 069 068 067 066
    // 065 064 063 062 061 060 059 058 057 056 055
    // 054 053 052 051 050 049 048 047 046 045 044
    // 043 042 041 040 039 038 037 036 035 034 033
    // 032 031 030 029 028 027 026 025 024 023 022
    // 021 020 019 018 017 016 015 014 013 012 011
    // 010 009 008 007 006 005 004 003 002 001 000
    //////////////////////////////////////////////
  

  
  //================

  return ESUMis;
}

