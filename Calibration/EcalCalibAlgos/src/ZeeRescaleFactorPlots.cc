
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/DataRecord/interface/EcalIntercalibConstantsRcd.h"
#include "Calibration/Tools/interface/calibXMLwriter.h"
#include "Calibration/Tools/interface/CalibrationCluster.h"
#include "Calibration/Tools/interface/HouseholderDecomposition.h"
#include "Calibration/Tools/interface/MinL3Algorithm.h"
#include "Calibration/EcalCalibAlgos/interface/ZeeRescaleFactorPlots.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TRandom.h"


#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

ZeeRescaleFactorPlots::ZeeRescaleFactorPlots( char* fileName )
{

  fileName_ = fileName;
  file_ = new TFile(fileName_, "RECREATE");
}


ZeeRescaleFactorPlots::~ZeeRescaleFactorPlots()
{

  file_->Close();

  delete file_;

}

//========================================================================

void ZeeRescaleFactorPlots::writeHistograms(ZIterativeAlgorithmWithFit* theAlgorithm_){

  file_ -> cd();

  
  const ZIterativeAlgorithmWithFit::ZIterativeAlgorithmWithFitPlots* algoHistos = theAlgorithm_->getHistos();

  for (int iIteration=0;iIteration<theAlgorithm_->getNumberOfIterations();iIteration++)
    for (int iChannel=0;iChannel<theAlgorithm_->getNumberOfChannels();iChannel++)
      {

	if(iChannel%20==0){
	  
	  file_ -> cd();
	  
	  algoHistos->weightedRescaleFactor[iIteration][iChannel]->Write();
	  algoHistos->unweightedRescaleFactor[iIteration][iChannel]->Write();
	  algoHistos->weight[iIteration][iChannel]->Write();
	}


      }


}


