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
// $Id$
//
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
#include "Geometry/Vector/interface/GlobalPoint.h"

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
EcalRecHit 
EcalDeadChannelRecoveryAlgos::Correct(const EBDetId Id, const EcalRecHitCollection* hit_collection, string algo_)
{
  double NewEnergy=0.0;
  

  if(algo_=="Spline"){
    const int Msize = 9;
    const int Msize2 = Msize*Msize;
    double MNxN[Msize2];
    if(MakeNxNMatrice(Id,hit_collection,Msize,MNxN)){
      NewEnergy = CorrectDeadChannelsClassic(MNxN,Id.ieta());
    }
  }else if(algo_=="NeuralNetworks"){
    const int Msize = 7;
    const int Msize2 = Msize*Msize;
    double MNxN[Msize2];
    if(MakeNxNMatrice(Id,hit_collection,Msize,MNxN)){
      NewEnergy = CorrectDeadChannelsNN(MNxN);
    }
  }
  

  EcalRecHit NewHit(Id,NewEnergy,0);
  return NewHit;

}


//==============================================================================================================
//==============================================================================================================
//==============================================================================================================
//==============================================================================================================



bool EcalDeadChannelRecoveryAlgos::MakeNxNMatrice(EBDetId itID,const EcalRecHitCollection* hit_collection,const int size, double *MNxN){



  //Build NxN around a given cristal
  const int Nsize2 = size*size;
  double N[Nsize2];
  for(int i=0;i<Nsize2;i++){
    N[i]=0.000001; 
    MNxN[i]=N[i];
  }

  cout<<" Cell CENTRAL  eta = "<< itID.ieta()<<" ,  phi = "<< itID.iphi()<<endl;




  const CaloSubdetectorTopology* topology=calotopo.getSubdetectorTopology(DetId::Ecal,EcalBarrel);
  std::vector<DetId> NxNaroundDC = topology->getWindow(itID,size,size);
  
  
  cout<<"NxNaroundDC size is = "<<NxNaroundDC.size()<<endl;
  vector<DetId>::const_iterator theCells;
  
  double ESUMis=0.0;
  for(theCells=NxNaroundDC.begin();theCells<NxNaroundDC.end();theCells++){
    
    EBDetId theCell = EBDetId(*theCells);
    
    int CReta = theCell.ieta();
    int CRphi = theCell.iphi();
    
    double Energy = 0.0;
    if(!theCell.null()){
      EcalRecHitCollection::const_iterator goS_it = hit_collection->find(theCell);
      if( goS_it !=  hit_collection->end() ) Energy=goS_it->energy();
    }
    //cout<<"Around DC we have eta,phi,E "<<CReta<<" "<<CRphi<<" "<<Energy<<endl;


    int ietaCorr = 0;
    if((CReta * itID.ieta()) < 0 && itID.ieta()>0 )ietaCorr= -1;
    if((CReta * itID.ieta()) < 0 && itID.ieta()<0 )ietaCorr=  1;
    if((CReta * itID.ieta()) > 0 )ietaCorr= 0;
    int ieta = CReta - itID.ieta() - ietaCorr + int((size - 1)/2);

    int iphiCorr = 0;
    if((CRphi - itID.iphi())> 50)iphiCorr= 360;
    if((CRphi - itID.iphi())<-50)iphiCorr=-360;
    int iphi = CRphi - itID.iphi() - iphiCorr + int((size - 1)/2);

    int theIndex = ieta*size+iphi;
    if(abs(CReta)<=85)
    MNxN[theIndex]=Energy;

    
    if( abs( int((size - 1)/2)-ieta ) <= 2  && abs( int((size - 1)/2)-iphi ) <= 2  )
      ESUMis +=  Energy;
    
  }
  cout<<"Around DC Collected Energy is =  "<< ESUMis  <<endl;
  
  
  if(ESUMis>4.0){
    //print the N x N matrix around the DC
    for(int iphi=0;iphi<size;iphi++){
      for(int ieta=0;ieta<size;ieta++){
	int theIndex = ieta*size+size-iphi-1;
	cout<<setw(12)<<MNxN[theIndex];
      }
      cout<<endl;
    }
    return 1;
  }
  
  
  
  //================

  return 0;
}

