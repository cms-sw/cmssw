
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2010/02/16 10:03:23 $
 *  $Revision: 1.3 $
 *  \author S. Maselli - INFN Torino
 */

#include "CalibMuon/DTCalibration/plugins/DTTTrigCorrectionFirst.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTSuperLayer.h"

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include <iostream>
#include <fstream>

using namespace edm;
using namespace std;

DTTTrigCorrectionFirst::DTTTrigCorrectionFirst(const ParameterSet& pset) {
 
  debug = pset.getUntrackedParameter<bool>("debug",false);
  ttrigMin = pset.getUntrackedParameter<double>("ttrigMin",0);
  ttrigMax = pset.getUntrackedParameter<double>("ttrigMax",5000);
  rmsLimit = pset.getUntrackedParameter<double>("rmsLimit",5.);

  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");
}

DTTTrigCorrectionFirst::~DTTTrigCorrectionFirst(){}

void DTTTrigCorrectionFirst::beginRun( const edm::Run& run, const edm::EventSetup& setup ) {
  ESHandle<DTTtrig> tTrig;
  setup.get<DTTtrigRcd>().get(dbLabel,tTrig);
  tTrigMap = &*tTrig;
  cout << "[DTTTrigCorrection]: TTrig version: " << tTrig->version() << endl;

  setup.get<MuonGeometryRecord>().get(muonGeom);
}

void DTTTrigCorrectionFirst::endJob() {
  // Create the object to be written to DB
  DTTtrig* tTrigNewMap = new DTTtrig();  
  //Get the superlayers list
  vector<DTSuperLayer*> dtSupLylist = muonGeom->superLayers();

  //Loop on all superlayers to compute the mean 
  double average = 0.;
  double average2 = 0.;
  double rms = 0.;
  double averageSigma = 0.;
  double average2Sigma = 0.;
  double rmsSigma = 0.;
  double counter = 0.;
  double averagekfactor = 0;
  float kfactor = 0;

  for (vector<DTSuperLayer*>::const_iterator sl = dtSupLylist.begin();
       sl != dtSupLylist.end(); sl++) {
    float ttrigMean = 0;
    float ttrigSigma = 0;   
    tTrigMap->get((*sl)->id(),
		  ttrigMean,
		  ttrigSigma,
                  kfactor,
		  DTTimeUnits::ns);
    if( ttrigMean < ttrigMax && ttrigMean > ttrigMin ) {
      average += ttrigMean;
      averageSigma += ttrigSigma;
      if(ttrigMean > 0)
	counter +=  1.;
    }
  }//End of loop on superlayers 

  average =  average/ counter;
  averageSigma = averageSigma/counter;
  averagekfactor = kfactor;

  //  cout << " average counter "<< average << " "<< counter <<endl;

  for (vector<DTSuperLayer*>::const_iterator sl = dtSupLylist.begin();
       sl != dtSupLylist.end(); sl++) {
    float ttrigMean = 0;
    float ttrigSigma = 0;
    float kfactor = 0;
    tTrigMap->get((*sl)->id(),
		  ttrigMean,
		  ttrigSigma,
                  kfactor,
		  DTTimeUnits::ns);
    if( ttrigMean < ttrigMax && ttrigMean > ttrigMin ) {
      average2 += (ttrigMean-average)*(ttrigMean-average);
      average2Sigma += (ttrigSigma-averageSigma)*(ttrigSigma-averageSigma);
    }
  }//End of loop on superlayers 

     rms = average2 /(counter-1);
     rmsSigma = average2Sigma /(counter-1);
     rms = sqrt(rms);
     rmsSigma = sqrt(rmsSigma);
  cout << "average averageSigma counter rms "<< average <<" " << averageSigma << " " << counter << " " << rms  <<endl;


  for (vector<DTSuperLayer*>::const_iterator sl = dtSupLylist.begin();
       sl != dtSupLylist.end(); sl++) {
    //Compute new ttrig
    double newTTrigMean =  0;
    double newTTrigSigma =  0;
    double newkfactor =0;
    float tempttrigMean = 0;
    float tempttrigSigma = 0;
    float tempkfactor = 0;
    float ttrigMean = 0;
    float ttrigSigma = 0;
    float kfactor = 0;
    tTrigMap->get((*sl)->id(),
		  ttrigMean,
		  ttrigSigma,
                  kfactor,
		  DTTimeUnits::ns);
    int chamber = (*sl)->id().chamberId().station();

    //check if ttrigMean is similar to the mean
    if (abs(ttrigMean - average) < rmsLimit*rms ){
      newTTrigMean = ttrigMean;
      newTTrigSigma = ttrigSigma;
      newkfactor = kfactor;
    } else {
      // do not consider if ttrig == 0
      if(ttrigMean > 0) {
	//cout << "ttrig chamber " << ttrigMean <<" "<<chamber<<endl;
	if(((*sl)->id().superlayer()) == 1){
	  //cout << " superlayer " << ((*sl)->id().superlayer()) <<endl; 
	  DTSuperLayerId slId((*sl)->id().chamberId(),3);
	  tTrigMap->get(slId,
			tempttrigMean,
			tempttrigSigma,
                        tempkfactor,
			DTTimeUnits::ns);
	  if (abs(tempttrigMean - average) < rmsLimit*rms) {
	    newTTrigMean = tempttrigMean;
	    newTTrigSigma = tempttrigSigma;
            newkfactor = tempkfactor;
	    cout <<"Chamber "<< chamber << " sl " << ((*sl)->id().superlayer()) << "has ttrig "<< ttrigMean  
		 << " -> takes value of sl 3 " <<  newTTrigMean <<endl;
	  } else if(chamber == 4){
	    cout << "No correction possible within same chamber (sl1) "  <<endl; 
	    newTTrigMean = average;
	    newTTrigSigma = averageSigma;
            newkfactor = averagekfactor;
	    cout<<"####### Bad SL: " << (*sl)->id() << " from "<< ttrigMean <<" to "<< newTTrigMean <<endl;
	  } else if(chamber != 4){	  
	    DTSuperLayerId slId((*sl)->id().chamberId(),2);
	    tTrigMap->get(slId,
			  tempttrigMean,
			  tempttrigSigma,
                          tempkfactor,
			  DTTimeUnits::ns);
	    if (abs(tempttrigMean - average) < rmsLimit*rms) {
	      newTTrigMean = tempttrigMean;
	      newTTrigSigma = tempttrigSigma;
              newkfactor = tempkfactor;
	      cout <<"Chamber "<< chamber << " sl " << ((*sl)->id().superlayer()) << "has ttrig "<< ttrigMean  
		   <<" -> takes value of sl 2 " <<  newTTrigMean <<endl;
	    } else {
	      cout << "No correction possible within same chamber (sl1) "  <<endl;
	      newTTrigMean = average;
	      newTTrigSigma = averageSigma;
              newkfactor = averagekfactor;
	      cout<<"####### Bad SL: " << (*sl)->id() << " from "<< ttrigMean <<" to "<< newTTrigMean <<endl;
	    }
	  }
	} else if (((*sl)->id().superlayer()) == 2) {
	  //cout << " superlayer " << ((*sl)->id().superlayer()) <<endl; 
	  DTSuperLayerId slId((*sl)->id().chamberId(),1);
	  tTrigMap->get(slId,
			tempttrigMean,
			tempttrigSigma,
                        tempkfactor,
			DTTimeUnits::ns);
	  if (abs(tempttrigMean - average) < rmsLimit*rms) {
	    newTTrigMean = tempttrigMean;
	    newTTrigSigma = tempttrigSigma;
            newkfactor = tempkfactor;
	    cout <<"Chamber "<< chamber << " sl " << ((*sl)->id().superlayer()) << "has ttrig "<< ttrigMean  
		 << " -> takes value of sl 1 " <<  newTTrigMean <<endl;
	  } else {
	    DTSuperLayerId slId((*sl)->id().chamberId(),3);
	    tTrigMap->get(slId,
			  tempttrigMean,
			  tempttrigSigma,
                          tempkfactor,
			  DTTimeUnits::ns);
	    if (abs(tempttrigMean - average) < rmsLimit*rms) {
	      newTTrigMean = tempttrigMean;
	      newTTrigSigma = tempttrigSigma;
              newkfactor = tempkfactor;
	      cout <<"Chamber "<< chamber << " sl " << ((*sl)->id().superlayer()) << "has ttrig "<< ttrigMean  
		   << "-> takes value of sl 3 " <<  newTTrigMean <<endl;
	      } else {
	      cout << "No correction possible within same chamber (sl2)  "  <<endl;
	      newTTrigMean = average;
	      newTTrigSigma = averageSigma;
              newkfactor = averagekfactor;
	      cout<<"####### Bad SL: " << (*sl)->id() << " from "<< ttrigMean <<" to "<< newTTrigMean <<endl;
	    }
	  } 
	  
	  
	} else if (((*sl)->id().superlayer()) == 3) {
	  //cout << " superlayer " << ((*sl)->id().superlayer()) <<endl; 
	  DTSuperLayerId slId((*sl)->id().chamberId(),1);
	  tTrigMap->get(slId,
			tempttrigMean,
			tempttrigSigma,
                        tempkfactor,
			DTTimeUnits::ns);
	  if (abs(tempttrigMean - average) < rmsLimit*rms) {
	    newTTrigMean = tempttrigMean;
	    newTTrigSigma = tempttrigSigma;
            newkfactor = tempkfactor;
	    cout <<"Chamber "<< chamber << " sl " << ((*sl)->id().superlayer()) << "has ttrig "<< ttrigMean  
		 << " -> takes value of sl 1 " <<  newTTrigMean <<endl;
	  } else if(chamber == 4) {
	    cout << "No correction possible within same chamber (sl3)"  <<endl;
	    newTTrigMean = average;
	    newTTrigSigma = averageSigma;
            newkfactor = averagekfactor;
	    cout<<"####### Bad SL: " << (*sl)->id() << " from "<< ttrigMean <<" to "<< newTTrigMean <<endl;
	  } else if(chamber != 4){
	    DTSuperLayerId slId((*sl)->id().chamberId(),2);
	    tTrigMap->get(slId,
			  tempttrigMean,
			  tempttrigSigma,
                          tempkfactor,
			  DTTimeUnits::ns);
	    if (abs(tempttrigMean - average) < rmsLimit*rms) {
	      newTTrigMean = tempttrigMean;
	      newTTrigSigma = tempttrigSigma;
              newkfactor = tempkfactor;
	      cout <<"Chamber "<< chamber << " sl " << ((*sl)->id().superlayer()) << "has ttrig "<< ttrigMean  
		   << " -> takes value of sl 2 " <<  newTTrigMean <<endl;
	    } else {
	      cout << "No correction possible within same chamber (sl3) "  <<endl;
	      newTTrigMean = average;
	      newTTrigSigma = averageSigma;
              newkfactor = averagekfactor;
	      cout<<"####### Bad SL: " << (*sl)->id() << " from "<< ttrigMean <<" to "<< newTTrigMean <<endl;
	    }
	  } 
	  
	}
      } else {
	//	cout << "SL not present " << ttrigMean << " " <<((*sl)->id().superlayer()) <<endl;
	newTTrigMean = average;
	newTTrigSigma = averageSigma;
        newkfactor = averagekfactor;
	cout<<"####### NotPresent SL: " << (*sl)->id() << " from "<< ttrigMean <<" to "<< newTTrigMean <<endl;
      }
    }

    //Store new ttrig in the new map
    tTrigNewMap->set((*sl)->id(), newTTrigMean, newTTrigSigma, newkfactor, DTTimeUnits::ns);
    if(debug){
      cout<<"New tTrig: " << (*sl)->id()
    	  << " from "<< ttrigMean <<" to "<< newTTrigMean <<endl;
      cout<<"New tTrigSigma: " << (*sl)->id()
    	  << " from "<<ttrigSigma  <<" to "<< newTTrigSigma <<endl;

    }
  }//End of loop on superlayers 

  
  //Write object to DB
  cout << "[DTTTrigCorrection]: Writing ttrig object to DB!" << endl;
  string record = "DTTtrigRcd";
   DTCalibDBUtils::writeToDB<DTTtrig>(record, tTrigNewMap);
} 



