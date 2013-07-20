// -*- C++ -*-
//
// Package:    MagneticFieldPlotter
// Class:      MagneticFieldPlotter
// 
/**\class MagneticFieldPlotter MagneticFieldPlotter.cc MyAnalyzers/MagneticFieldPlotter/src/MagneticFieldPlotter.cc

 Description: Plots Magnetic Field Components in the Tracker Volume

 Implementation:
     This Analyzer fills some histograms with the Magnetic Field components in the tracker volume. It's mainly aimed to look
     at differences between the Veikko parametrized field and the VolumeBased one.
*/
//
// Original Author:  Massimiliano Chiorboli
//         Created:  Mon Jun 11 17:20:15 CEST 2007
// $Id: MagneticFieldPlotter.cc,v 1.4 2009/12/14 22:23:22 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"



#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticFieldPlotter.h"
#include "DataFormats/GeometryVector/interface/Pi.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <TH1.h>
#include <TH2.h>

using namespace edm;
using namespace std;


MagneticFieldPlotter::MagneticFieldPlotter(const edm::ParameterSet& iConfig):
  HistoFileName(iConfig.getUntrackedParameter("histoFileName",
			        std::string("magneticFieldPlotterHistos.root")))
{
  theHistoFile = 0;
  
  nZstep   = 1;
  nPhistep = 1;
  zHalfLength = 280;

}


MagneticFieldPlotter::~MagneticFieldPlotter()
{
 
}


void
MagneticFieldPlotter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   ESHandle<MagneticField> ESMGField;
   iSetup.get<IdealMagneticFieldRecord>().get(ESMGField);
   theMGField = &(*ESMGField);

   
   //   float radius[5] = {10, 20, 30, 50, 100};
   float radius[1] = {100};
   //   for(int iR=0; iR<5; iR++) {
   for(int iR=0; iR<1; iR++) {
       for(int iPhi=0; iPhi<nPhistep; iPhi++) {
     for(int iZ=0; iZ<nZstep; iZ++) {
       float zCoordinate = (float)iZ*zHalfLength*2/(float)nZstep - zHalfLength;
	 float phiCoordinate = (float)iPhi*(float)Geom::pi()*2/(float)nPhistep - (float)Geom::pi();
	 float rCoordinate = radius[iR];
	 GlobalPoint gp(GlobalPoint::Cylindrical(rCoordinate,phiCoordinate,zCoordinate));
	 GlobalVector myFieldVector = theMGField->inTesla(gp);
	 float Br   = myFieldVector.x()*cos(gp.phi()) + myFieldVector.y()*sin(gp.phi());
	 float Bphi = - myFieldVector.x()*sin(gp.phi()) + myFieldVector.y()*cos(gp.phi());
	 cout << "Radius  = " << rCoordinate       ;
	 cout << ", Z     = " << zCoordinate      ;
	 cout << ", Phi    = " << phiCoordinate     <<endl;
	 cout << "Bz     = " << myFieldVector.z() ;
	 cout << ", Br     = " << Br                ;
	 cout << ", Bphi   = " << Bphi              << endl;
	 gBz[iR]  ->Fill(phiCoordinate,zCoordinate,myFieldVector.z());
	 gBr[iR]  ->Fill(phiCoordinate,zCoordinate,Br               );
	 gBphi[iR]->Fill(phiCoordinate,zCoordinate,Bphi             );
       }
     }
   }

}


void 
MagneticFieldPlotter::beginJob()
{
   theHistoFile = new TFile(HistoFileName.c_str(), "RECREATE");
  

   float phiStepWidth = Geom::pi()*2/(float)nPhistep;
   float zStepWidth   = zHalfLength*2/(float)nZstep;

   for(int i=0; i<5; i++) {
     stringstream iSS;
     string iS;
     iSS << i;
     iSS >> iS;
     string gBzName    = "gBz" + iS;
     string gBzTitle   = "B_{z}" + iS;
     string gBphiName  = "gBphi" + iS;
     string gBphiTitle = "B_{#phi}" + iS;
     string gBrName    = "gBr" + iS;
     string gBrTitle   = "B_{r}" + iS;
     gBz[i]    = new TH2D(gBzName.c_str()   , gBzTitle.c_str()   , 
			  nPhistep , -Geom::pi()-phiStepWidth/2 , Geom::pi()-phiStepWidth/2, 
			  nZstep   , -zHalfLength-zStepWidth/2  , zHalfLength-zStepWidth/2  );
     gBphi[i]  = new TH2D(gBphiName.c_str() , gBphiTitle.c_str() , 
			  nPhistep , -Geom::pi()-phiStepWidth/2 , Geom::pi()-phiStepWidth/2, 
			  nZstep   , -zHalfLength-zStepWidth/2  , zHalfLength-zStepWidth/2  );
     gBr[i]    = new TH2D(gBrName.c_str()   , gBrTitle.c_str()   , 
			  nPhistep , -Geom::pi()-phiStepWidth/2 , Geom::pi()-phiStepWidth/2, 
			  nZstep   , -zHalfLength-zStepWidth/2  , zHalfLength-zStepWidth/2  );

   }
   
}

void 
MagneticFieldPlotter::endJob() {
  theHistoFile->Write();
  theHistoFile->Close() ;
}

DEFINE_FWK_MODULE(MagneticFieldPlotter);
