#ifndef ElectronLikelihood_H
#define ElectronLikelihood_H

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEgamma/ElectronIdentification/interface/ElectronIDAlgo.h"
#include "RecoEgamma/ElectronIdentification/interface/LikelihoodSwitches.h"
#include "RecoEgamma/ElectronIdentification/interface/LikelihoodPdfProduct.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "CondFormats/DataRecord/interface/ElectronLikelihoodRcd.h"
#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCalibration.h"
#include <TDirectory.h>
#include <vector>


class ElectronLikelihood {

 public:
  
  //! ctor, not used for this algo (need initialization from ES)
  ElectronLikelihood () {} ;

  //! ctor
  ElectronLikelihood (const ElectronLikelihoodCalibration *calibration,
		      LikelihoodSwitches eleIDSwitches,
		      std::string signalWeightSplitting,
		      std::string backgroundWeightSplitting,
		      bool splitSignalPdfs,
		      bool splitBackgroundPdfs) ;

  //! dtor
  virtual ~ElectronLikelihood () ;

  //! not used for this algo
  void setup (const edm::ParameterSet& conf) {} ;

  //! get the result of the algorithm
  float result (const reco::GsfElectron &electron, 
                const EcalClusterLazyTools&) const ;
  //! get the log-expanded result of the algorithm
  float resultLog (const reco::GsfElectron &electron, 
                   const EcalClusterLazyTools&) const ;

 private:

  //! build the likelihood model from histograms 
  //! in Barrel file and Endcap file
  void Setup (const ElectronLikelihoodCalibration *calibration,
	      std::string signalWeightSplitting,
	      std::string backgroundWeightSplitting,
	      bool splitSignalPdfs,
	      bool splitBackgroundPdfs) ;


  //! get the input variables from the electron and the e-Setup
  void getInputVar (const reco::GsfElectron &electron, 
                    std::vector<float> &measuremnts, 
                    const EcalClusterLazyTools&) const ;

  //! likelihood below 15GeV/c
  LikelihoodPdfProduct *_EB0lt15lh, *_EB1lt15lh, *_EElt15lh;
  //! likelihood above 15GeV/c
  LikelihoodPdfProduct *_EB0gt15lh, *_EB1gt15lh, *_EEgt15lh;

  //! general parameters of all the ele id algorithms
  LikelihoodSwitches m_eleIDSwitches ;

  //! splitting rule for PDF's
  std::string m_signalWeightSplitting;
  std::string m_backgroundWeightSplitting;
  bool m_splitSignalPdfs;
  bool m_splitBackgroundPdfs;

};

#include "FWCore/Framework/interface/data_default_record_trait.h"
EVENTSETUP_DATA_DEFAULT_RECORD (ElectronLikelihood, ElectronLikelihoodRcd)

#endif // ElectronLikelihood_H
