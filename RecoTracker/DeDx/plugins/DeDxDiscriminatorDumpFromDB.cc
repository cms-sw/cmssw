// -*- C++ -*-
//
// Package:    DeDxDiscriminatorDumpFromDB
// Class:      DeDxDiscriminatorDumpFromDB
// 
/**\class DeDxDiscriminatorDumpFromDB DeDxDiscriminatorDumpFromDB.cc RecoTracker/DeDxDiscriminatorDumpFromDB/src/DeDxDiscriminatorDumpFromDB.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  andrea
//         Created:  Thu May 31 14:09:02 CEST 2007
//    Code Updates:  loic Quertenmont (querten)
//         Created:  Thu May 10 14:09:02 CEST 2008
// $Id: DeDxDiscriminatorDumpFromDB.cc,v 1.2 2012/01/17 13:46:51 innocent Exp $
//
//


// system include files
#include <memory>
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"

#include "RecoTracker/DeDx/plugins/DeDxDiscriminatorDumpFromDB.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "CondFormats/DataRecord/interface/SiStripDeDxMip_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxElectron_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxPion_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxKaon_3D_Rcd.h"


#include "TFile.h"

using namespace reco;
using namespace std;
using namespace edm;

DeDxDiscriminatorDumpFromDB::DeDxDiscriminatorDumpFromDB(const edm::ParameterSet& iConfig)
{
   Reccord             = iConfig.getUntrackedParameter<string>  ("Reccord"            , "SiStripDeDxMip_3D_Rcd");
   HistoFile           = iConfig.getUntrackedParameter<string>  ("HistoFile"        ,  "out.root");
}


DeDxDiscriminatorDumpFromDB::~DeDxDiscriminatorDumpFromDB(){}

// ------------ method called once each job just before starting event loop  ------------
void  DeDxDiscriminatorDumpFromDB::beginRun(edm::Run & run, const edm::EventSetup& iSetup)
{
   edm::ESHandle<PhysicsTools::Calibration::HistogramD3D> DeDxMapHandle_;    
   if(      strcmp(Reccord.c_str(),"SiStripDeDxMip_3D_Rcd")==0){
      iSetup.get<SiStripDeDxMip_3D_Rcd>() .get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxPion_3D_Rcd")==0){
      iSetup.get<SiStripDeDxPion_3D_Rcd>().get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxKaon_3D_Rcd")==0){
      iSetup.get<SiStripDeDxKaon_3D_Rcd>().get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxProton_3D_Rcd")==0){
      iSetup.get<SiStripDeDxProton_3D_Rcd>().get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxElectron_3D_Rcd")==0){
      iSetup.get<SiStripDeDxElectron_3D_Rcd>().get(DeDxMapHandle_);
   }else{
//      printf("The reccord %s is not known by the DeDxDiscriminatorDumpFromDB\n", Reccord.c_str());
//      printf("Program will exit now\n");
      exit(0);
   }
   DeDxMap_ = *DeDxMapHandle_.product();

   double xmin = DeDxMap_.rangeX().min;
   double xmax = DeDxMap_.rangeX().max;
   double ymin = DeDxMap_.rangeY().min;
   double ymax = DeDxMap_.rangeY().max;
   double zmin = DeDxMap_.rangeZ().min;
   double zmax = DeDxMap_.rangeZ().max;

   TH3D* Prob_ChargePath  = new TH3D ("Prob_ChargePath"     , "Prob_ChargePath" , DeDxMap_.numberOfBinsX(), xmin, xmax, DeDxMap_.numberOfBinsY() , ymin, ymax, DeDxMap_.numberOfBinsZ(), zmin, zmax);

   for(int i=0;i<=Prob_ChargePath->GetXaxis()->GetNbins()+1;i++){
      for(int j=0;j<=Prob_ChargePath->GetYaxis()->GetNbins()+1;j++){
         for(int k=0;k<=Prob_ChargePath->GetZaxis()->GetNbins()+1;k++){
            Prob_ChargePath->SetBinContent (i, j, k, DeDxMap_.binContent(i,j,k));
         }
      }
   }

  TFile* Output = new TFile(HistoFile.c_str(), "RECREATE");
  Prob_ChargePath->Write();
  Output->Write();
  Output->Close();
}

void  DeDxDiscriminatorDumpFromDB::endJob()
{
}

void DeDxDiscriminatorDumpFromDB::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeDxDiscriminatorDumpFromDB);
