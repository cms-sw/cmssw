#ifndef EcalCreateTimeCalibrations_hh
#define EcalCreateTimeCalibrations_hh

// -*- C++ -*-
//
// Package:   EcalCreateTimeCalibrations
// Class:     EcalCreateTimeCalibrations
//
/**\class EcalCreateTimeCalibrations EcalCreateTimeCalibrations.h

Description: <one line class summary>

Implementation:
<Notes on implementation>
 */
//
// Authors:                              Seth Cooper (Minnesota)
//         Created:  Tu Apr 26  10:46:22 CEST 2011
// $Id: EcalCreateTimeCalibrations.h,v 1.9 2012/06/14 18:58:27 jared Exp $
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibErrors.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CalibCalorimetry/EcalTiming/interface/EcalTimeTreeContent.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DQM/EcalCommon/interface/Numbers.h"

#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile2D.h"
#include "TGraph.h"

#include <vector>
#include <string>

#include <boost/tokenizer.hpp>

class EcalCreateTimeCalibrations : public edm::EDAnalyzer {

        public:
                explicit EcalCreateTimeCalibrations(const edm::ParameterSet& ps);
                ~EcalCreateTimeCalibrations();

        protected:
                virtual void beginJob();
                virtual void analyze(edm::Event const&, edm::EventSetup const&);
                virtual void endJob();
                virtual void beginRun(edm::EventSetup const&);

        private:
                bool includeEvent(int* triggers,
                    int numTriggers,
                    std::vector<std::vector<double> > includeVector,
                    std::vector<std::vector<double> > excludeVector);
                bool includeEvent(double eventParameter,
                    std::vector<std::vector<double> > includeVector,
                    std::vector<std::vector<double> > excludeVector);
                void genIncludeExcludeVectors(std::string optionString,
                    std::vector<std::vector<double> >& includeVector,
                    std::vector<std::vector<double> >& excludeVector);
                std::vector<std::string> split(std::string msg, std::string separator);
                std::string intToString(int num);
                void initEBHists(edm::Service<TFileService>& fileService_);
                void initEEHists(edm::Service<TFileService>& fileService_);
                void set(edm::EventSetup const&);

                edm::ESHandle<EcalTimeCalibConstants> origTimeCalibConstHandle;
                edm::ESHandle<EcalTimeOffsetConstant> origTimeOffsetConstHandle;
                std::vector<std::string> inputFiles_;
                std::string fileName_; // beginning of file name of txt output, etc.
		std::string timeCalibFileName_; 
		std::string timeOffsetFileName_; 
                TChain* myInputTree_;
                bool produce_;
                EcalTimeTreeContent treeVars_;
                int numTotalCrys_;
                int numTotalCrysEB_;
                int numTotalCrysEE_;
                // For selection cuts
                bool disableGlobalShift_;
                bool subtractDBcalibs_;
                std::string inBxs_, inOrbits_, inTrig_, inTTrig_, inLumis_, inRuns_;
                float avgTimeMin_, avgTimeMax_;
                float minHitAmpEB_, minHitAmpEE_;
                float maxSwissCrossNoise_;  // EB only, no spikes seen in EE
                float maxHitTimeEB_, minHitTimeEB_;
                float maxHitTimeEE_, minHitTimeEE_;
		int eventsUsedFractionNum_, eventsUsedFractionDen_;
                // vectors for skipping selections
                std::vector<std::vector<double> > bxIncludeVector;
                std::vector<std::vector<double> > bxExcludeVector;
                std::vector<std::vector<double> > orbitIncludeVector;
                std::vector<std::vector<double> > orbitExcludeVector;
                std::vector<std::vector<double> > trigIncludeVector;
                std::vector<std::vector<double> > trigExcludeVector;
                std::vector<std::vector<double> > ttrigIncludeVector;
                std::vector<std::vector<double> > ttrigExcludeVector;
                std::vector<std::vector<double> > lumiIncludeVector;
                std::vector<std::vector<double> > lumiExcludeVector;
                std::vector<std::vector<double> > runIncludeVector;
                std::vector<std::vector<double> > runExcludeVector;

                // hists EB
                TH1F* calibHistEB_;
                TH1F* calibAfterSubtractionHistEB_;
                TH1F* calibErrorHistEB_;
                TH2F* calibsVsErrors_;
                TH1F* expectedStatPresHistEB_;
                TH2F* expectedStatPresVsObservedMeanErrHistEB_;
                TH1F* expectedStatPresEachEventHistEB_;
                TH2F* errorOnMeanVsNumEvtsHist_;
                TH1F* hitsPerCryHistEB_;
                TH2F* hitsPerCryMapEB_;
                TProfile2D* ampProfileMapEB_;
                TProfile* ampProfileEB_;
                TH1F* sigmaHistEB_;
                TH2F* calibMapEB_;
                TH2F* calibAfterSubtractionMapEB_;
                TH2F* sigmaMapEB_;
                TH2F* calibErrorMapEB_;
                TProfile2D* calibTTMapEB_;
                TH1F* cryTimingHistsEB_[61200];
                TH1F* superModuleTimingHistsEB_[36]; 
                TH1F* triggerTowerTimingHistsEB_[2488];
                // hists EE
                TH1F* calibHistEE_;
                TH1F* calibAfterSubtractionHistEE_;
                TH1F* calibErrorHistEE_;
                TH2F* calibsVsErrorsEE_;
                //TH2F* calibMapEE_;
                //TH2F* calibMapEEFlip_;
                //TH2F* calibMapEEPhase_;
                //TH2F* calibMapEtaAvgEE_;
                //TH1F* calibHistEtaAvgEE_;
                TH2F* hitsPerCryMapEEM_;
                TH2F* hitsPerCryMapEEP_;
                TH1F* hitsPerCryHistEEM_;
                TH1F* hitsPerCryHistEEP_;
                //TH1C* eventsEEMHist_;
                //TH1C* eventsEEPHist_;
                TProfile* ampProfileEEM_;
                TProfile* ampProfileEEP_;
                TProfile2D* ampProfileMapEEP_;
                TProfile2D* ampProfileMapEEM_;
                //TH1F* eventsEEHist_;
                //TH1F* calibSigmaHist_;
                TH1F* sigmaHistEE_;
                //TH1F* chiSquaredEachEventHist_;
                //TH2F* chiSquaredVsAmpEachEventHist_;
                //TH2F* chiSquaredHighMap_;
                //TH1F* chiSquaredTotalHist_;
                //TH1F* chiSquaredSingleOverTotalHist_;
                //TH1F* ampEachEventHist_;
                //TH1F* numPointsErasedHist_;
                //TProfile2D* myAmpProfile_;
                TH1F* expectedStatPresHistEEM_;
                TH2F* expectedStatPresVsObservedMeanErrHistEEM_;
                TH1F* expectedStatPresEachEventHistEEM_;
                TH1F* expectedStatPresHistEEP_;
                TH2F* expectedStatPresVsObservedMeanErrHistEEP_;
                TH1F* expectedStatPresEachEventHistEEP_;
                TH2F* errorOnMeanVsNumEvtsHistEE_;
                TH1F* calibHistEEM_;
                TH1F* calibHistEEP_;
                TH1F* calibAfterSubtractionHistEEM_;
                TH1F* calibAfterSubtractionHistEEP_;
                TH1F* calibErrorHistEEM_;
                TH1F* calibErrorHistEEP_;
                TH2F* calibMapEEM_;
                TH2F* calibMapEEP_;
                TH2F* calibAfterSubtractionMapEEM_;
                TH2F* calibAfterSubtractionMapEEP_;
                TH2F* sigmaMapEEM_;
                TH2F* sigmaMapEEP_;
                TH2F* calibErrorMapEEM_;
                TH2F* calibErrorMapEEP_;
                TH1F* cryTimingHistsEEP_[100][100]; // [0][0] = ix 1, iy 1
                TH1F* cryTimingHistsEEM_[100][100]; // [0][0] = ix 1, iy 1
                TH1F* superModuleTimingHistsEEM_[9];
                TH1F* superModuleTimingHistsEEP_[9];
                TH1F* etaSlicesTimingHistsEEM_[5];
                TH1F* etaSlicesTimingHistsEEP_[5];
};
#endif


