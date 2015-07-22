#ifndef RecoHI_HiEvtPlaneAlgos_LoadEPDB_h
#define RecoHI_HiEvtPlaneAlgos_LoadEPDB_h

// system include files
#include <memory>
#include <iostream>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneFlatten.h"
#include "RecoHI/HiEvtPlaneAlgos/interface/HiEvtPlaneList.h"
#include <vector>

//using namespace hi;

class LoadEPDB {
 public:

  explicit LoadEPDB(const edm::ESHandle<RPFlatParams> flatparmsDB_, HiEvtPlaneFlatten ** flat)
  {
    int Hbins;
    int Obins;
    int flatTableSize = flatparmsDB_->m_table.size();
    genFlatPsi_ = kTRUE;
    if(flatTableSize<flat[0]->getHBins()+2*flat[0]->getOBins()) {
      genFlatPsi_ = kFALSE;
    } else {
      Hbins = flat[0]->getHBins();
      Obins = flat[0]->getOBins();

      for(int i = 0; i<flatTableSize; i++) {
	const RPFlatParams::EP* thisBin = &(flatparmsDB_->m_table[i]);
	for(int j = 0; j<hi::NumEPNames; j++) {
	  int indx = thisBin->RPNameIndx[j];
	  if(indx<0||indx>=hi::NumEPNames) {
	    genFlatPsi_ = kFALSE;
	    break;
	  }
	  if(indx>=0) {
	    if(i<Hbins) {
	      flat[indx]->setXDB(i, thisBin->x[j]);
	      flat[indx]->setYDB(i, thisBin->y[j]);
	    } else if(i>=Hbins && i<Hbins+Obins) {
	      flat[indx]->setXoffDB(i - Hbins, thisBin->x[j]);
	      flat[indx]->setYoffDB(i - Hbins, thisBin->y[j]);

	    } else if (i>=Hbins+Obins && i<Hbins+2*Obins) {
	      flat[indx]->setPtDB(i - Hbins- Obins, thisBin->x[j]);
	      flat[indx]->setPt2DB(i - Hbins- Obins, thisBin->y[j]);
	    }
	  }
	}
      }
      int cbins = 0;
      while(flatTableSize>Hbins + 2*Obins + cbins) {
	const RPFlatParams::EP* thisBin = &(flatparmsDB_->m_table[Hbins+2*Obins +cbins]);
	double centbinning = thisBin->x[0];
	int ncentbins = (int) thisBin->y[0]+0.01;
	if(ncentbins==0) break;
	for(int j = 0; j< ncentbins; j++) {
	  const RPFlatParams::EP* thisBin = &(flatparmsDB_->m_table[Hbins+2*Obins +cbins+j+1]);
	  if(fabs(centbinning-1.)<0.01) {
	    for(int i = 0; i<hi::NumEPNames; i++) flat[i]->setCentRes1(j,thisBin->x[i],thisBin->y[i]);
	  }
	  if(fabs(centbinning-2.)<0.01) {
	    for(int i = 0; i<hi::NumEPNames; i++) flat[i]->setCentRes2(j,thisBin->x[i],thisBin->y[i]);
	  }
	  if(fabs(centbinning-5.)<0.01) {
	    for(int i = 0; i<hi::NumEPNames; i++) flat[i]->setCentRes5(j,thisBin->x[i],thisBin->y[i]);
	  }
	  if(fabs(centbinning-10.)<0.01) {
	    for(int i = 0; i<hi::NumEPNames; i++) flat[i]->setCentRes10(j,thisBin->x[i],thisBin->y[i]);
	  }
	  if(fabs(centbinning-20.)<0.01) {
	    for(int i = 0; i<hi::NumEPNames; i++) flat[i]->setCentRes20(j,thisBin->x[i],thisBin->y[i]);
	  }
	  if(fabs(centbinning-25.)<0.01) {
	    for(int i = 0; i<hi::NumEPNames; i++) flat[i]->setCentRes25(j,thisBin->x[i],thisBin->y[i]);
	  }
	  if(fabs(centbinning-30.)<0.01) {
	    for(int i = 0; i<hi::NumEPNames; i++) flat[i]->setCentRes30(j,thisBin->x[i],thisBin->y[i]);
	  }
	  if(fabs(centbinning-40.)<0.01) {
	    for(int i = 0; i<hi::NumEPNames; i++) flat[i]->setCentRes40(j,thisBin->x[i],thisBin->y[i]);
	  }
	}

	cbins+=ncentbins+1;
      }

    }

  }

  bool IsSuccess(){return genFlatPsi_;}
  ~LoadEPDB(){}

 private:
  bool genFlatPsi_;

};

#endif
