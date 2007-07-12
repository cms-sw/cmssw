//----------Author's Names: B.Fabbro, FX Gentit DSM/DAPNIA/SPP CEA-Saclay
//---------Copyright: Those valid for CEA sofware
//----------Modified: 08/06/2007
// ROOT include files

// user's include files

#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBParameters.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TEBNumbering.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaParameters.h"
#include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaRunEB.h"

R__EXTERN TCnaRootFile *gCnaRootFile;

ClassImp(TCnaRunEB)
//___________________________________________________________________________
//
// TCnaRunEB + instructions for use of the CNA (Correlated Noise Analysis)
//             in the framework of CMSSW.
//
//==============> INTRODUCTION
//
//    The present documentation contains:
//
//    [1] a brief description of the CNA package and instructions for use
//        this package in the framework of the CMS Software
//
//    [2] the documentation for the class TCnaRunEB
//
//
//==[1]=====================================================================================
//
//
//         DOCUMENTATION FOR THE INTERFACE: CNA package / CMSSW / SCRAM
//
//
//==========================================================================================
//
//  The CNA software consists in 2 packages named: EcalCorrelatedNoiseAnalysisModules and
//  EcalCorrelatedNoiseAnalysisAlgos.
//
//  The directory tree is the following:
//
//      <local path>/CMSSW_a_b_c/src/----CalibCalorimetry/---EcalCorrelatedNoiseAnalysisModules/BuildFile
//                              |   |                    |                                     |---interface/
//                              |   |                    |                                     |---src/
//                                  |                    |                                     |---data/
//                                  |                    |
//                                  |                    |---EcalCorrelatedNoiseAnalysisAlgos/BuildFile
//                                  |                    |                                   |---interface/
//                                  |                    |                                   |---src/
//                                  |                    |                                   |---test/
//                                  |                    |
//                                  |                    |
//                                  |                    \--- <other packages of CalibCalorimetry> 
//                                  |
//                                  \----<other subsystems...>
//
//
//    The package EcalCorrelatedNoiseAnalysisModules contains one analyzer
//    (EcalCorrelatedNoisePedestalRunAnalyzer). The user has to edit
//    this analyzer. A detailed description is given here after in the class TCnaRunEB
//    documentation. An analyzer skeleton can be obtained by means of the
//    SkeletonCodeGenerator "mkedanlzr" (see the CMSSW Framework/Edm web page).
//
//    The package EcalCorrelatedNoiseAnalysisAlgos contains the basic classes of the CNA 
//    (in src and interface) and two executables (in directory test):
//
//    1) EcalCorrelatedNoiseExample:  a simple example using the class TCnaViewEB
//
//    2) InteractiveCNAForEcalBarrel: a GUI dialog box with buttons, menus...
//       This program uses the classes TCnaDialogEB and  TCnaViewEB.
//       See the documentation of these classes for more details.
//
// 
//==[2]======================================================================================
//
//
//                         DOCUMENTATION FOR THE CLASS TCnaRunEB
//
//
//===========================================================================================
//TCnaRunEB.
//
//
//
//================> Brief and general description
//                  -----------------------------
//
//   This class allows the user to calculate expectation values, variances,
//   covariances, correlations and other quantities of interest for correlated
//   noise studies on the CMS/ECAL BARREL (EB).
//
//   Three main operations are performed by the class TCnaRunEB. Each of them is
//   associated with a specific method of the analyzer EcalCorrelatedNoisePedestalRunAnalyzer:
//
//    (1) Initialization and calls to "preparation methods" of the CNA.
//        This task is done in the constructor of the analyzer:
//        EcalCorrelatedNoiseAnalysisModules::EcalCorrelatedNoisePedestalRunAnalyzer
//                                                          (const edm::ParameterSet& pSet)
//
//    (2) Building of the event distributions (distributions of the ADC
//        values for each sample, each channel, etc...)
//        This task is done in the method "analyze" of the analyzer:
//        EcalCorrelatedNoisePedestalRunAnalyzer::analyze
//                                     (const edm::Event& iEvent, const edm::EventSetup& iSetup)
//
//    (3) Calculation of the different quantities (correlations, covariances, etc...)
//        from the distributions obtained in (2) and writing of these quantities
//        in results ROOT files and also in ASCII files.
//        This task is done in the destructor of the analyzer:
//        EcalCorrelatedNoisePedestalRunAnalyzer::~EcalCorrelatedNoisePedestalRunAnalyzer()
//   
//
//========> Use of the class TCnaRunEB by the analyzer EcalCorrelatedNoisePedestalRunAnalyzer
//          ---------------------------------------------------------------------------------
//
//     In the following, the parts of the code concerning TCnaRunEB
//     are emphasized by special comments with strings: "###############"
//
//     +-----------------------------------------------------------------+
//     |                                                                 |
//     |  (0) The class EcalCorrelatedNoisePedestalRunAnalyzer.          |
//     |      Example of EcalCorrelatedNoisePedestalRunAnalyzer.h file:  |
//     |                                                                 |
//     +-----------------------------------------------------------------+
//
//     #ifndef CNASUB_RUNEVT_H
//     #define CNASUB_RUNEVT_H
//     
//     // -*- C++ -*-
//     //
//     // Package:    EcalCorrelatedNoisePedestalRunAnalyzer
//     // Class:      EcalCorrelatedNoisePedestalRunAnalyzer
//     // 
//     /**\class EcalCorrelatedNoisePedestalRunAnalyzer
//               EcalCorrelatedNoisePedestalRunAnalyzer.cc
//               CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/src/EcalCorrelatedNoisePedestalRunAnalyzer.cc
//     
//      Description: <one line class summary>
//     
//      Implementation:
//          <Notes on implementation>
//     */
//     //
//     // Original Author:  Bernard Fabbro
//     //         Created:  Fri Jun  2 10:27:01 CEST 2006
//     // $Id$
//     //
//     //
//     
//     // system include files
//     #include <memory>
//     #include <iostream>
//     #include <fstream>
//     #include <iomanip>
//     #include <string>
//     #include <vector>
//     #include <time.h>
//     #include "Riostream.h"
//     
//     // ROOT include files
//     #include "TObject.h"
//     #include "TSystem.h"
//     #include "TString.h"
//     #include "TVectorD.h"
//     
//     // user include files
//     //################ include of TCnaRunEB.h #################
//     #include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisAlgos/interface/TCnaRunEB.h"
//     
//     //
//     // class declaration
//     //
//     
//     class EcalCorrelatedNoisePedestalRunAnalyzer: public edm::EDAnalyzer {
//     
//      public:
//      
//       enum { kChannels = 1700, kGains = 3, kFirstGainId = 1 };
//
//       // CONSTRUCTOR             
//       explicit EcalCorrelatedNoisePedestalRunAnalyzer(const edm::ParameterSet&);
// 
//       // DESTRUCTOR       
//       ~EcalCorrelatedNoisePedestalRunAnalyzer();  
//       
//       // "analyze" METHOD   
//       virtual void analyze(const edm::Event&, const edm::EventSetup&);
//    
//     
//      private:
//       // ----------member data ---------------------------
//       unsigned int verbosity_;
//       Int_t    nChannels_;
//       Int_t    iEvent_;
//     
//       Int_t    fEvtNumber;
//     
//       //############### DECLARATION of the TCnaRunEB OBJECT: fMyCnaRun #################
//       TCnaRunEB*       fMyCnaRun; 
//
//       TEBNumbering* fMyCnaNumbering; // another object used to recover some quantities
//                                      // related to the channel numbers, crystal numbers, etc...
//     };
//     
//     #endif
//     
//
//     +----------------------------------------------------------------------------+
//     |                                                                            |
//     |  (1) EXAMPLE OF CONSTRUCTOR                                                |
//     |      EcalCorrelatedNoisePedestalRunAnalyzer::                              |
//     |      EcalCorrelatedNoisePedestalRunAnalyzer(const edm::ParameterSet& pSet) |
//     |                                                                            |
//     +----------------------------------------------------------------------------+
//
//     // -*- C++ -*-
//     //
//     // Package:    EcalCorrelatedNoiseAnalysisModules
//     // Class:      EcalCorrelatedNoisePedestalRunAnalyzer
//     // 
//     /**\class EcalCorrelatedNoisePedestalRunAnalyzer
//               EcalCorrelatedNoisePedestalRunAnalyzer.cc
//               CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/src/EcalCorrelatedNoisePedestalRunAnalyzer.cc
//     
//      Description: <one line class summary>
//     
//     Implementation:
//         <Notes on implementation>
//     */
//
//     // Original Author:  Bernard Fabbro
//     //         Created:  Fri Jun  2 10:27:01 CEST 2006
//     // $Id$
//     //
//     //          Update:  02/10/2006  
//     
//     // CMSSW include files
//     
//     #include "Riostream.h"
//     
//     #include "FWCore/Framework/interface/EDAnalyzer.h"
//     #include "FWCore/Framework/interface/Event.h"
//     #include "FWCore/Framework/interface/Handle.h"
//     
//     #include "FWCore/Framework/interface/Frameworkfwd.h"
//     #include "FWCore/ParameterSet/interface/ParameterSet.h"
//     
//     #include "DataFormats/EcalDigi/interface/EBDataFrame.h"
//     #include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
//     #include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
//     
//     #include "DataFormats/Common/interface/EventID.h"
//     
//     //################## include of EcalCorrelatedNoisePedestalRunAnalyzer.h ##############
//     #include "CalibCalorimetry/EcalCorrelatedNoiseAnalysisModules/interface/EcalCorrelatedNoisePedestalRunAnalyzer.h"
//     
//     // constants, enums and typedefs
//
//     // static data member definitions
//
//     //
//     // constructors
//     //
//
//     EcalCorrelatedNoisePedestalRunAnalyzer::EcalCorrelatedNoisePedestalRunAnalyzer(const edm::ParameterSet& pSet) : 
//       verbosity_(pSet.getUntrackedParameter("verbosity", 1U)),
//       nChannels_(0), iEvent_(0)
//     {
//       //now do what ever initialization is needed
//     
//       using namespace edm;       
//
//       fEvtNumber=0;
//     
//     //#################################################################################
//     //
//     //                          INIT OF the C.N.A.
//     //
//     // First of all, you have to declare fMyCnaRun as an object of the class TCnaRunEB.
//     // This declaration is done by calling the constructor without argument:
//     //#################################################################################
//
//       fMyCnaRun = new TCnaRunEB();
//       fMyCnaRun->PrintComments();  // => optional
//     
//     //##########################################################################
//     // Hence, you have to call the preparation method: "GetReadyToReadData"
//     // This methods needs some arguments.
//     //##########################################################################
//
//       TString AnalysisName    = "cosmics"
//       Int_t   RunNumber       = 22770;
//       Int_t   FirstTakenEvt   = 300;   
//       Int_t   NbOfTakenEvents = 150; // 150 = nb of events in one burst (1 burst = 1 gain)    
//       Int_t   SuperModule     = 22;
//       Int_t   Nentries        = 3*NbOfTakenEvents;       //  3 bursts of 150 events   
//
//
//       MyCnaRun->GetReadyToReadData(AnalysisName,   RunNumber,
//	                              FirstTakenEvt,  NbOfTakenEvents,
//                                    SuperModule,    Nentries);
//
//  * Remark:
//
//       The argument Nentries is used to check that Nentries-FirstTakenEvt is positive.
//       You can call the method without this argument (then default value = 9999999 is used):
//
//       MyCnaRun->GetReadyToReadData(AnalysisName,   RunNumber,
//	                              FirstTakenEvt,  NbOfTakenEvents,
//                                    SuperModule);
//    
//  ! IMPORTANT REMARK ====> In pedestal runs, the number of taken events must not be larger
//                           than the number of events by burst (NBURST = 150).
//                            > 1rst burst: gain  1, event   0 to 149 (       0 to   NBURST-1)
//                            > 2nd  burst: gain  6, event 150 to 299 (  NBURST to 2*NBURST-1)
//                            > 3rd  burst: gain 12. event 300 to 449 (2*NBURST to 3*NBURST-1)
//                           There is a change of gain every NBURST events during a pedestal run.
//                           Therefore, if the calculations are done on more than NBURST events, you
//                           will find very strong correlations because of an artificial effect due
//                           to of the change of gain.
//        
//       fMyCnaNumbering = new TEBNumbering();    // just to recover tower number in SM
//                                                   and channel number in tower in the
//                                                   method "analyze"
//     }
//
//     +-------------------------------------------------------------------------------+
//     |                                                                               |
//     |  (2) EXAMPLE OF METHOD "analyze"                                              |
//     |      EcalCorrelatedNoisePedestalRunAnalyzer::                                 |
//     |              analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) |
//     |                                                                               |
//     +-------------------------------------------------------------------------------+
//
//     void EcalCorrelatedNoisePedestalRunAnalyzer::
//                                analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
//     {
//        using namespace std;
//        using namespace edm;
//     
//        edm::Handle<EBDigiCollection> digis;
//        iEvent.getByLabel("ecalEBunpacker", digis);
//     
//       // Initialize vectors if not already done
//       if ( int(digis->size()) > nChannels_ )
//         {
//           nChannels_ = digis->size();
//         }
//       
//       if( fEvtNumber >= fMyCnaRun->GetFirstTakenEvent() )
//         { 
//           Int_t TakenEventNumber = fEvtNumber - fMyCnaRun->GetFirstTakenEvent();
//           
//           if( Int_t(digis->end()-digis->begin()) >= 0 ||
//               Int_t(digis->end()-digis->begin()) <  Int_t(digis->size()) )
//             {
//             // Loop over Ecal barrel digis
//             for (EBDigiCollection::const_iterator digiItr = digis->begin(); 
//                 digiItr != digis->end(); ++digiItr) 
//       	    {
//       	      Int_t iChannel = Int_t(digiItr - digis->begin());     
//       	      Int_t iSMCrys  = iChannel + 1;
//         	      Int_t smTower  = fMyCnaNumbering->GetSMTowFromSMCrys(iSMCrys);
//         	      Int_t iTowEcha = fMyCnaNumbering->GetTowEchaFromSMCrys(iSMCrys);
//         	      
//         	      Int_t nSample = digiItr->size();
//
//         	      // Loop over the samples
//         	      for (Int_t iSample = 0; iSample < nSample; ++iSample)
//         		{
//         		  Double_t adc = Double_t((*digiItr).sample(iSample).adc());
//
//         		  //####################################################
//                        //
//                        //    CALL TO THE METHOD "BuildEventDistributions"
//                        //
//                        //####################################################
//
//         		  fMyCnaRun->BuildEventDistributions
//         		               (TakenEventNumber,SMTower,iTowEcha,iSample,adc);
//         	      }
//         	  }
//             }      
//         }
//           
//       fEvtNumber++;
//       iEvent_++;
//     }
//
//     +---------------------------------------------------------+
//     |                                                         |
//     |  (3) EXAMPLE OF DESTRUCTOR                              |
//     |      EcalCorrelatedNoisePedestalRunAnalyzer::           |
//     |            ~EcalCorrelatedNoisePedestalRunAnalyzer()    |
//     |                                                         |
//     +---------------------------------------------------------+
//
//     EcalCorrelatedNoisePedestalRunAnalyzer::~EcalCorrelatedNoisePedestalRunAnalyzer()
//     {
//        // do anything here that needs to be done at destruction time
//        // (e.g. close files, deallocate resources etc.)
// 
//     //##########> CALL TO THE METHOD: GetReadyToCompute() (MANDATORY)   
//       fMyCnaRun->GetReadyToCompute();
//     
//     //##########> CALLS TO METHODS WHICH COMPUTES THE QUANTITIES. EXAMPLES:
//       fMyCnaRun->ComputeExpectationValuesOfSamples();
//       fMyCnaRun->ComputeVariancesOfSamples();
//       fMyCnaRun->ComputeCorrelationsBetweenSamples();
//     
//       fMyCnaRun->ComputeExpectationValuesOfExpectationValuesOfSamples();
//       fMyCnaRun->ComputeExpectationValuesOfSigmasOfSamples();
//       fMyCnaRun->ComputeExpectationValuesOfCorrelationsBetweenSamples();
// 
//     // etc...  
//
//     //##########> CALLS TO THE METHODS FOR WRITING THE RESULTS IN THE ROOT FILE:    
//
//       fMyCnaRun->GetPathForResultsRootFiles();
//     
//       if(fMyCnaRun->WriteRootFile() == kTRUE )
//        {
//           cout << "*EcalCorrelatedNoisePedestalRunAnalyzer> Write ROOT file OK" << endl;
//         }
//       else 
//         {
//           cout << "!EcalCorrelatedNoisePedestalRunAnalyzer> PROBLEM with write ROOT file." << endl;
//         }
//     
//       delete fMyCnaRun;
//       delete fMyCnaNumbering;
//     }
//
//
//
//================> More detailled description of the class TCnaRunEB
//                  -----------------------------------------------
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//                     Declaration and Print Methods
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//     Just after the declaration with the constructor without arguments,
//     you can set a "Print Flag" by means of the following "Print Methods":
//
//     TCnaRunEB* MyCnaRun = new TCnaRunEB(); // declaration of the object MyCnaRun
//
//    // Print Methods: 
//
//    MyCnaRun->PrintNoComment();  // Set flag to forbid printing of all the comments
//                                 // except ERRORS.
//
//    MyCnaRun->PrintWarnings();   // (DEFAULT)
//                                 // Set flag to authorize printing of some warnings.
//                                 // WARNING/INFO: information on something unusual
//                                 // in the data is pointed out.
//                                 // WARNING/CORRECTION: something wrong (but not too serious)
//                                 // in the value of some argument is pointed out and
//                                 // automatically modified to a correct value.
//
//   MyCnaRun->PrintComments();    // Set flag to authorize printing of infos
//                                 // and some comments concerning initialisations
//
//   MyCnaRun->PrintAllComments(); // Set flag to authorize printing of all the comments
//
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//           Method GetReadyToReadData(...) and associated methods
//
//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
//
//      MyCnaRun->GetReadyToReadData(AnalysisName,      RunNumber,
//		      FirstTakenEvt, NbOfTakenEvents,   SuperModule,   Nentries);
//
//
//   Explanations for the arguments (all of them are input arguments):
//
//      TString  AnalysisName: code for the analysis. This code is
//                             necessary to distinguish between different
//                             analyses on the same events of a same run.
//                             
//                             The string ana_name is automatically
//                             included in the name of the results files
//                             (see below: ROOT and ASCII results files paragraph)
//
//      Int_t     RunNumber:        run number
//
//      Int_t     FirstTakenEvent:  first taken event
//      Int_t     NbOfTakenEvents:  number of taken events
//      Int_t     SuperModule:      super-module number
//      
//        The different quantities (correlations, etc...) will be calculated
//        from the event numbered: FirstTakenEvent
//        to the event numbered:   FirstTakenEvent + NbOfTakenEvents - 1.
//
//      Int_t    Nentries:     number of entries of the run
//
//==============> Method to set the start and stop times of the analysis (optional)
//
//  A method can be used to set the fStartDate and fStopDate attributes
//  of the class TCnaHeaderEB from start and stop time given by the user provided
//  these values have been recovered from the event reading:
//
//      void  MyCnaRun->StartStopDate(TString StartDate, TString StopDate);
// 
//     // TString StartDate, StopDate:  start and stop time of the run
//     //                               in "date" format. Example: 
//     //                               Wed Oct  8 04:14:23 2003
//
//     If the method is not called, the values of the attributes
//     fStartDate and fStopDate are set to: "no info" at the level of
//     the Init() method of the class TCnaHeaderEB.
//     The values of StartDate and StopDate are writen in the header of
//     the .root result file.
//
//
//  Another similar method exists, with time_t type arguments:
//
//     void  MyCnaRun->StartStopTime(time_t  StartTime, time_t  StopTime);
//
//
//
//==============> CALCULATION METHODS
//
//    The "calculation methods" are methods which compute the different
//    quantities of interest. 
//
//    List of the calculation methods (still in evolution):
//          
//  void  ComputeExpectationValuesOfSamples(); // expectation values of the samples
//                                             // for all the channels
//
//  void  ComputeVariancesOfSamples();         // variances of the samples
//                                             // for all the channels
//
//  void  MakeHistosOfSampleDistributions();   // Making of the histos of the ADC distributions
//                                             // for all the samples and all the channels
//
//  void  MakeHistosOfSamplesAsFunctionOfEvent(); // Histogram making of the sample ADC value as a function
//                                                // of the event for all the pairs (channel,sample)
//
//  void  ComputeCovariancesBetweenSamples();                  // Cov(s,s) for all the channels 
//  void  ComputeCorrelationsBetweenSamples();                 // Cor(s,s) for all the channels
//
//  void  ComputeCovariancesBetweenChannelsMeanOverSamples();  // Cov(c,c) for all the samples
//  void  ComputeCorrelationsBetweenChannelsMeanOverSamples(); // Cor(c,c) for all the samples 
//
//  void  ComputeCovariancesBetweenTowersMeanOverSamplesAndChannels();
//        // Calculation of the cov(channel,channel) averaged over the samples and
//        // calculation of the mean of these cov(channel,channel) for all the towers
//
//  void  ComputeCorrelationsBetweenTowersMeanOverSamplesAndChannels();
//        // Calculation of the cor(channel,channel) averaged over the samples and
//        // calculation of the mean of these cor(channel,channel) for all the towers
//
//  void  ComputeExpectationValuesOfExpectationValuesOfSamples(); 
//        // Calc. of the exp.val. of the exp.val. of the samples for all the channels
//  void  ComputeExpectationValuesOfSigmasOfSamples(); 
//        // Calc. of the exp.val. of the sigmas of the samples for all the channels
//  void  ComputeExpectationValuesOfCorrelationsBetweenSamples(); 
//        // Calc. of the exp.val. of the (sample,sample) correlations for all the channels
//
//  void  ComputeSigmasOfExpectationValuesOfSamples();
//        // Calc. of the sigmas of the exp.val. of the samples for all the channels
//  void  ComputeSigmasOfSigmasOfSamples();
//        // Calc. of the sigmas of the of the sigmas of the samples for all the channels
//  void  ComputeSigmasOfCorrelationsBetweenSamples();
//        // Calc. of the sigmas of the (sample,sample) correlations for all the channels
//
//==============> RESULTS FILES
//
//  The calculation methods above provide results which can be used directly
//  in the user's code. However, these results can also be written in results
//  files by appropriate methods. In TCnaRunEB, there are such methods which
//  write the results according to two options:
//
//   (a) writting in a ROOT file
//   (b) writting in ASCII files
//
//  The names of the results files are automaticaly generated by the methods.
//
//  In the following, we describe:
//
//     (a1) The method which gets the path for the results ROOT file
//          from a "cna-configuration" file
//     (a2) The codification for the name of the ROOT file
//     (a3) The method which writes the results in the ROOT file
//
//     (b1) The method which gets the path for the results ASCII files
//          from a "cna-configuration" file
//     (b2) The codification for the names of the ASCII files
//     (b3) The methods which writes the results in the ASCII files
//
//    ++++++++++++++++  (a) WRITING IN THE ROOT FILE  ++++++++++++++++++++
//
// (a1)-----------> Method to get the path for the results ROOT file
//
//      void  MyCnaRun->GetPathForResultsRootFiles(pathname);
//            TString pathname = name of a "cna-configuration" file located
//            in the user's HOME directory and containing one line which
//            specifies the path where must be writen the .root result files.
//            (no slash at the end of the string)
//    
//   DEFAULT: 
//            void MyCnaRun->GetPathForResultsRootFiles();
//            If there is no argument, the "cna-configuration" file must be named
//            "cna_results_root.cfg" and must be located in the user's HOME
//            directory
//
//
// (a2)-----------> Codification for the name of the ROOT file:
//
//  The name of the ROOT file is the following:
//
//       aaa_rrr_fff_ttt_SMsss.root
//
//  with:
//       aaa = Analysis name
//       rrr = Run number
//       fff = First event number
//       ttt = Number of taken events
//       sss = Super-module number
//
//  This name is automatically generated from the values of the arguments
//  of the method "GetReadyToReadData".
//
// (a3)-----------> Method which writes the result in the ROOT file:
//
//       Bool_t  MyCnaRun->WriteRootFile();
//
//
//    ++++++++++++++++  (b) WRITING IN THE ASCII FILES  ++++++++++++++++++++
//
// (b1)-----------> Method to set the path for the results ASCII files
//
//      void  MyCnaRun->GetPathForResultsAsciiFiles(pathname);
//
//            TString pathname = name of a "cna-config" file located
//            in the user's HOME directory and containing one line which
//            specifies the path where must be writen the .ascii result files.
//            (no slash at the end of the string)
//    
//   DEFAULT: 
//            void  MyCnaRun->GetPathForResultsAsciiFiles();
//            If there is no argument, the "cna-config" file must be named
//            "cna_results_ascii.cfg" and must be located in the user's HOME
//            directory
//
//
// (b2)-----------> Codification for the names of the ASCII files (examples):
//
//       aaa_rrr_ev_fff_ttt_SMsss.ascii
//       aaa_rrr_cor_ss_cCCC_fff_ttt_SMsss.ascii
//       aaa_rrr_cov_cc_sSSS_fff_ttt_SMsss.ascii
//       etc...  
//
//  with:
//       aaa = Analysis name
//       rrr = Run number
//       fff = First event number
//       ttt = Number of taken (required) events
//       SSS = Sample number
//       CCC = Electronic Channel number in SM
//       sss = Super-module number
//
//  Examples:
//       cut1_2208_ev_0_500_SM10.ascii  :
//        ASCII file containing the expectation values (code "ev")
//        for the analysis "cut1", run 2208, first event: 0,
//        500 events from the event 0 (=> last evt = 499) and super-module 10
//
//       cut2_33559_cor_ss_c1264_2000_1500_SM10.ascii  :
//        ASCII file containing the correlations between samples ("cor_ss")
//        for the channel 1264 ("c1264"), analysis "cut2", run 33559,
//        first event: 2000, 1500 events from the event 2000
//        (=> last evt = 3499) and super-module 10 
//
//       std_10_cov_cc_s0_0_2000_SM8.ascii  :
//        ASCII file containing the covariances between channels ("cov_cc")
//        for the sample 0 ("s0"), analysis "std", run 10,
//        first event: 0, 2000 events from the event 0
//        (=> last evt = 1999) and super-module 8
//
//
// (b3)-----------> Methods which write the ASCII files:
//
//  The methods which write the ASCII files are the following:
//
//      void  WriteAsciiExpectationValuesOfSamples();    
//      void  WriteAsciiVariancesOfSamples();
//      void  WriteAsciiCovariancesBetweenChannelsMeanOverSamples();
//      void  WriteAsciiCovariancesBetweenSamples(n_channel);
//      void  WriteAsciiCorrelationsBetweenChannelsMeanOverSamples();
//      void  WriteAsciiCorrelationsBetweenSamples(n_channel);
//
//  Each of these methods corresponds to a "calculation method" presented
//  above. The calculation method must be called before the writing method.
//
//-------------------------------------------------------------------------
//
//        For more details on other classes of the CNA package:
//
//                 http://www.cern.ch/cms-fabbro/cna
//
//-------------------------------------------------------------------------
//

//------------------------------ TCnaRunEB.cxx -----------------------------
//  
//   Creation (first version): 03 Dec 2002
//
//   For questions or comments, please send e-mail to Bernard Fabbro:
//             
//   fabbro@hep.saclay.cea.fr 
//
//------------------------------------------------------------------------

TCnaRunEB::TCnaRunEB()
{
//Constructor without argument: call to Init()
  Init();
}

void TCnaRunEB::Init()
{
//Initialisation

  fCnew       = 0;
  fCdelete    = 0;
  fCnaCommand = 0;
  fCnaError   = 0;

  fTTBELL = '\007';   //   "BIP!"

  //........................... TString file names init
  fgMaxCar  = (Int_t)512;

  Int_t MaxCar = fgMaxCar;
  fRootFileName.Resize(MaxCar);
  fRootFileName      = "?";

  MaxCar = fgMaxCar;
  fRootFileNameShort.Resize(MaxCar);
  fRootFileNameShort = "?";

  MaxCar = fgMaxCar;
  fAsciiFileName.Resize(MaxCar);
  fAsciiFileName      = "?";

  MaxCar = fgMaxCar;
  fAsciiFileNameShort.Resize(MaxCar);
  fAsciiFileNameShort = "?";

  fDim_name = fgMaxCar;

  //................................................... Code Print

  TCnaParameters* MyParameters = new TCnaParameters();      fCnew++;

  fCodePrintNoComment   = MyParameters->GetCodePrint("NoComment");
  fCodePrintWarnings    = MyParameters->GetCodePrint("Warnings ");      // => default value
  fCodePrintComments    = MyParameters->GetCodePrint("Comments");
  fCodePrintAllComments = MyParameters->GetCodePrint("AllComments");

  delete MyParameters;                                      fCdelete++;

  fMiscDiag = 0;

  //................ MiscDiag counters .................
  fMaxMsgIndexForMiscDiag = (Int_t)10;
  fNbOfMiscDiagCounters   = (Int_t)50;

  fMiscDiag   = new Int_t[fNbOfMiscDiagCounters];       fCnew++;
  for (Int_t iz=0; iz<fNbOfMiscDiagCounters; iz++){fMiscDiag[iz] = (Int_t)0;}

  //............................. init pointers

  fT3d_distribs  = 0;
  fT3d2_distribs = 0;
  fT3d1_distribs = 0;

  fVal_data  = 0;
  fVal_dat2  = 0;

  fT2d_EvtNbInLoop = 0;
  fT1d_EvtNbInLoop = 0;

  fT1d_SMtowFromIndex = 0;

  fT2d_LastEvtNumber = 0;
  fT1d_LastEvtNumber = 0; 

  fT2d_ev        = 0;
  fT1d_ev        = 0;
  fT2d_var       = 0;
  fT1d_var       = 0;

  fT3d_his_s     = 0;
  fT2d_his_s     = 0;
  fT1d_his_s     = 0;

  fT2d_xmin      = 0;
  fT1d_xmin      = 0;

  fT2d_xmax      = 0;
  fT1d_xmax      = 0;

  fT3d_cov_ss    = 0;
  fT3d2_cov_ss   = 0;
  fT3d1_cov_ss   = 0;

  fT3d_cor_ss    = 0;
  fT3d2_cor_ss   = 0;
  fT3d1_cor_ss   = 0;

  fT3d_cov_cc    = 0;
  fT3d2_cov_cc   = 0;
  fT3d1_cov_cc   = 0;

  fT3d_cor_cc    = 0;
  fT3d2_cor_cc   = 0;
  fT3d1_cor_cc   = 0;

  fT2d_cov_cc_mos  = 0;
  fT2d1_cov_cc_mos = 0;

  fT2d_cor_cc_mos  = 0;
  fT2d1_cor_cc_mos = 0;

  fT2d_cov_moscc_mot  = 0;
  fT2d1_cov_moscc_mot = 0;

  fT2d_cor_moscc_mot  = 0;
  fT2d1_cor_moscc_mot = 0;

  fT1d_ev_ev     = 0;
  fT1d_ev_sig    = 0;
  fT1d_ev_cor_ss = 0;

  fT1d_sig_ev     = 0;
  fT1d_sig_sig    = 0;
  fT1d_sig_cor_ss = 0;

  fT2d_sv_correc_covss_s  = 0;
  fT2d1_sv_correc_covss_s = 0;

  fT3d_cov_correc_covss_s  = 0;
  fT3d2_cov_correc_covss_s = 0;
  fT3d1_cov_correc_covss_s = 0;

  fT3d_cor_correc_covss_s  = 0;
  fT3d2_cor_correc_covss_s = 0;
  fT3d1_cor_correc_covss_s = 0;

  fT2dCrysNumbersTable = 0;
  fT1dCrysNumbersTable = 0;

  fjustap_2d_ev  = 0;
  fjustap_1d_ev  = 0;

  fjustap_2d_var = 0;
  fjustap_1d_var = 0;

  fjustap_2d_cc  = 0;
  fjustap_1d_cc  = 0;

  fjustap_2d_ss  = 0;
  fjustap_1d_ss  = 0;

  //............................... codes (all the values must be different)

  fCodeHeaderAscii     =  0;
  fCodeRoot            =  1; 
  fCodeCorresp         =  2; 

  fCodeSampTime        =  3;

  fCodeEv              =  4;  
  fCodeVar             =  5;  
  fCodeEvts            =  6;

  fCodeCovCss          =  7;
  fCodeCorCss          =  8;
  fCodeCovScc          =  9;
  fCodeCorScc          = 10;
  fCodeCovSccMos       = 11;
  fCodeCorSccMos       = 12;

  fCodeCovMosccMot     = 13;
  fCodeCorMosccMot     = 14;

  fCodeEvCorCss        = 15; 
  fCodeSigCorCss       = 16; 
  
  fCodeSvCorrecCovCss  = 17;
  fCodeCovCorrecCovCss = 18;
  fCodeCorCorrecCovCss = 19;

  //.............................................................................. codes (end)

  //................................ tags
  fTagTowerNumbers    = 0;
  fTagLastEvtNumber   = 0;
  fTagEvtNbInLoop     = 0;

  fTagSampTime        = 0;

  fTagEv              = 0;
  fTagVar             = 0;
  fTagEvts            = 0;

  fTagCovCss          = 0;
  fTagCorCss          = 0;

  fTagCovScc          = 0;
  fTagCorScc          = 0;
  fTagCovSccMos       = 0;
  fTagCorSccMos       = 0;

  fTagCovMosccMot     = 0;
  fTagCorMosccMot     = 0;

  fTagEvEv            = 0;
  fTagEvSig           = 0;
  fTagEvCorCss        = 0;

  fTagSigEv           = 0;
  fTagSigSig          = 0;
  fTagSigCorCss       = 0;

  fTagSvCorrecCovCss  = 0;
  fTagCovCorrecCovCss = 0;
  fTagCorCorrecCovCss = 0;

  //.............................................. Miscellaneous
  
  fFlagPrint = fCodePrintWarnings;

  fFileHeader = new TCnaHeaderEB();    fCnew++;
  
  fOpenRootFile  = kFALSE;
  
  fReadyToReadData = 0;

  fSectChanSizeX = 0;
  fSectChanSizeY = 0;
  fSectSampSizeX = 0;
  fSectSampSizeY = 0;

  fUserSamp   = 0;
  fUserSMEcha = 0;

  fSpecialSMTowerNotIndexed = -1;

  fTowerIndexBuilt = 0;

}
// end of Init()
//=========================================== private copy ==========

void  TCnaRunEB::fCopy(const TCnaRunEB& rund)
{
//Private copy

#define NOCO
#ifndef NOCO
  fFileHeader   = rund.fFileHeader;
  fOpenRootFile = rund.fOpenRootFile;

  fUserSamp     = rund.fUserSamp;
  fUserSMEcha     = rund.fUserSMEcha;

  fSectChanSizeX = rund.fSectChanSizeX;
  fSectChanSizeY = rund.fSectChanSizeY;
  fSectSampSizeX = rund.fSectSampSizeX;
  fSectSampSizeY = rund.fSectSampSizeY;

  fT1d_SMtowFromIndex = rund.fT1d_SMtowFromIndex;

  fT2d_EvtNbInLoop = rund.fT2d_EvtNbInLoop;
  fT1d_EvtNbInLoop = rund.fT1d_EvtNbInLoop;

  fT3d_distribs  = rund.fT3d_distribs;
  fT3d2_distribs = rund.fT3d2_distribs;
  fT3d1_distribs = rund.fT3d1_distribs;

  fVal_data     = rund.fVal_data;   
  fVal_dat2     = rund.fVal_dat2;

  fT2d_LastEvtNumber = rund.fT2d_LastEvtNumber;
  fT1d_LastEvtNumber = rund.fT1d_LastEvtNumber;

  fT2d_ev      = rund.fT2d_ev;
  fT1d_ev      = rund.fT1d_ev;

  fT2d_var     = rund.fT2d_var;
  fT1d_var     = rund.fT1d_var;

  fT3d_his_s   = rund.fT3d_his_s;
  fT2d_his_s   = rund.fT2d_his_s;
  fT1d_his_s   = rund.fT1d_his_s;

  fT2d_xmin    = rund.fT2d_xmin;
  fT1d_xmin    = rund.fT1d_xmin;
  fT2d_xmax    = rund.fT2d_xmax; 
  fT1d_xmax    = rund.fT1d_xmax;   

  fT3d_cov_ss  = rund.fT3d_cov_ss;
  fT3d2_cov_ss = rund.fT3d2_cov_ss;
  fT3d1_cov_ss = rund.fT3d1_cov_ss;

  fT3d_cor_ss  = rund.fT3d_cor_ss;
  fT3d2_cor_ss = rund.fT3d2_cor_ss;
  fT3d1_cor_ss = rund.fT3d1_cor_ss;

  fT3d_cov_cc  = rund.fT3d_cov_cc;
  fT3d2_cov_cc = rund.fT3d2_cov_cc;
  fT3d1_cov_cc = rund.fT3d1_cov_cc;

  fT3d_cor_cc  = rund.fT3d_cor_cc;
  fT3d2_cor_cc = rund.fT3d2_cor_cc;
  fT3d1_cor_cc = rund.fT3d1_cor_cc;

  fT2d_cov_cc_mos  = rund.fT2d_cov_cc_mos;
  fT2d1_cov_cc_mos = rund.fT2d1_cov_cc_mos;

  fT2d_cor_cc_mos  = rund.fT2d_cor_cc_mos;
  fT2d1_cor_cc_mos = rund.fT2d1_cor_cc_mos;

  fT2d_cov_moscc_mot  = rund.fT2d_cov_moscc_mot;
  fT2d1_cov_moscc_mot = rund.fT2d1_cov_moscc_mot;

  fT2d_cor_moscc_mot  = rund.fT2d_cor_moscc_mot;
  fT2d1_cor_moscc_mot = rund.fT2d1_cor_moscc_mot;

  fT1d_ev_ev     = rund.fT1d_ev_ev;
  fT1d_ev_sig    = rund.fT1d_ev_sig;
  fT1d_ev_cor_ss = rund.fT1d_ev_cor_ss;

  fT1d_sig_ev     = rund.fT1d_sig_ev;
  fT1d_sig_sig    = rund.fT1d_sig_sig;
  fT1d_sig_cor_ss = rund.fT1d_sig_cor_ss;

  fT2d_sv_correc_covss_s = rund.fT2d_sv_correc_covss_s;
  fT2d1_sv_correc_covss_s = rund.fT2d1_sv_correc_covss_s;

  fT3d_cov_correc_covss_s  = rund.fT3d_cov_correc_covss_s;
  fT3d2_cov_correc_covss_s = rund.fT3d2_cov_correc_covss_s;
  fT3d1_cov_correc_covss_s = rund.fT3d1_cov_correc_covss_s;

  fT3d_cor_correc_covss_s  = rund.fT3d_cor_correc_covss_s;
  fT3d2_cor_correc_covss_s = rund.fT3d2_cor_correc_covss_s;
  fT3d1_cor_correc_covss_s = rund.fT3d1_cor_correc_covss_s;

  fT2dCrysNumbersTable  = rund.fT2dCrysNumbersTable;
  fT1dCrysNumbersTable  = rund.fT1dCrysNumbersTable;

  fjustap_2d_ev  = rund.fjustap_2d_ev;
  fjustap_1d_ev  = rund.fjustap_1d_ev;

  fjustap_2d_var = rund.fjustap_2d_var;
  fjustap_1d_var = rund.fjustap_1d_var;

  fjustap_2d_cc  = rund.fjustap_2d_cc;
  fjustap_1d_cc  = rund.fjustap_1d_cc;

  fjustap_2d_ss  = rund.fjustap_2d_ss;
  fjustap_1d_ss  = rund.fjustap_1d_ss;

  //........................................ Codes   
  fCodeHeaderAscii     = rund.fCodeHeaderAscii;
  fCodeRoot            = rund.fCodeRoot;
  fCodeSampTime        = rund.fCodeSampTime;  
  fCodeEv              = rund.fCodeEv;  
  fCodeVar             = rund.fCodeVar;  
  fCodeEvts            = rund.fCodeEvts;
  fCodeCovCss          = rund.fCodeCovCss;
  fCodeCorCss          = rund.fCodeCorCss;
  fCodeCovScc          = rund.fCodeCovScc;
  fCodeCorScc          = rund.fCodeCorScc;
  fCodeCovSccMos       = rund.fCodeCovSccMos;
  fCodeCorSccMos       = rund.fCodeCorSccMos;
  fCodeCovMosccMot     = rund.fCodeCovMosccMot;
  fCodeCorMosccMot     = rund.fCodeCorMosccMot;
  fCodeEvCorCss        = rund.fCodeEvCorCss;
  fCodeSigCorCss       = rund.fCodeSigCorCss;
  fCodeSvCorrecCovCss  = rund.fCodeSvCorrecCovCss;
  fCodeCovCorrecCovCss = rund.fCodeCovCorrecCovCss;
  fCodeCorCorrecCovCss = rund.fCodeCorCorrecCovCss;
  fCodePrintComments    = rund.fCodePrintComments;
  fCodePrintWarnings    = rund.fCodePrintWarnings;
  fCodePrintAllComments = rund.fCodePrintAllComments;
  fCodePrintNoComment   = rund.fCodePrintNoComment;

  //.................................................. Tags
  fTagTowerNumbers  = rund.fTagTowerNumbers;
  fTagLastEvtNumber = rund.fTagLastEvtNumber;
  fTagEvtNbInLoop   = rund.fTagEvtNbInLoop;
  fTagEv            = rund.fTagEv;
  fTagVar           = rund.fTagVar;
  fTagEvts          = rund.fTagEvts;
  fTagCovCss        = rund.fTagCovCss;
  fTagCorCss        = rund.fTagCorCss;
  fTagCovScc        = rund.fTagCovScc;
  fTagCorScc        = rund.fTagCorScc;
  fTagCovSccMos     = rund.fTagCovSccMos;
  fTagCorSccMos     = rund.fTagCorSccMos;
  fTagCovMosccMot   = rund.fTagCovMosccMot;
  fTagCorMosccMot   = rund.fTagCorMosccMot;
  fTagEvEv          = rund.fTagEvEv;
  fTagEvSig         = rund.fTagEvSig;
  fTagEvCorCss      = rund.fTagEvCorCss;
  fTagSigEv         = rund.fTagSigEv;
  fTagSigSig        = rund.fTagSigSig;
  fTagSigCorCss     = rund.fTagSigCorCss;
  fTagSvCorrecCovCss  = rund.fTagSvCorrecCovCss;
  fTagCovCorrecCovCss = rund.fTagCovCorrecCovCss;
  fTagCorCorrecCovCss = rund.fTagCorCorrecCovCss;
  fFlagPrint          = rund.fFlagPrint;

  fRootFileName         = rund.fRootFileName;
  fRootFileNameShort    = rund.fRootFileNameShort;
  fAsciiFileName        = rund.fAsciiFileName;
  fAsciiFileNameShort   = rund.fAsciiFileNameShort;

  fDim_name             = rund.fDim_name;

  fCfgResultsRootFilePath  = rund.fCfgResultsRootFilePath;
  fCfgResultsAsciiFilePath = rund.fCfgResultsAsciiFilePath;

  fFileForResultsRootFilePath  = rund.fFileForResultsRootFilePath; 
  fFileForResultsAsciiFilePath = rund.fFileForResultsAsciiFilePath; 

  fCnew    = rund.fCnew;
  fCdelete = rund.fCdelete;
#endif // NOCO
}
//  end of private copy

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    copy constructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TCnaRunEB::TCnaRunEB(const TCnaRunEB& dcop)
{
  cout << "*TCnaRunEB::TCnaRunEB(const TCnaRunEB& dcop)> "
       << " It is time to write a copy constructor" << endl
       << " type an integer value and then RETURN to continue"
       << endl;
  
  { Int_t cintoto;  cin >> cintoto; }
  
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                    overloading of the operator=
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TCnaRunEB& TCnaRunEB::operator=(const TCnaRunEB& dcop)
{
//Overloading of the operator=

  fCopy(dcop);
  return *this;
}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                            destructor
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

TCnaRunEB::~TCnaRunEB()
{
  //Destructor
  
  if(fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments ||
     fFlagPrint == fCodePrintWarnings )
    {
      cout << "*TCnaRunEB::~TCnaRunEB()> Entering destructor." << endl;
    }

  if(fFlagPrint == fCodePrintAllComments)
    {
      Int_t misc_czero = 0;
      for(Int_t i = 0; i < fNbOfMiscDiagCounters; i++)
	{
	  if( fMiscDiag[i] != 0 )
	    {
	      cout << "                          fMiscDiag Counter "
		   << setw(3) << i << " = " << setw(9) << fMiscDiag[i]
		   << " (INFO: alloc on non zero freed zone) " << endl;
	    }
	  else
	    {
	      misc_czero++;
	    }
	}
      cout << "                          Nb of fMiscDiag counters at zero: "
	   << misc_czero << " (total nb of counters: "
	   << fNbOfMiscDiagCounters << ")" << endl;
    }

  if (fMiscDiag                != 0){delete [] fMiscDiag;                 fCdelete++;}
  
  if (fFileHeader              != 0){delete fFileHeader;                  fCdelete++;}
  
  if (fT1d_SMtowFromIndex      != 0){delete [] fT1d_SMtowFromIndex;       fCdelete++;}
  
  if (fVal_data                != 0){delete [] fVal_data;                 fCdelete++;}
  if (fVal_dat2                != 0){delete [] fVal_dat2;                 fCdelete++;}

  if (fT2d_LastEvtNumber       != 0){delete [] fT2d_LastEvtNumber;        fCdelete++;}
  if (fT1d_LastEvtNumber       != 0){delete [] fT1d_LastEvtNumber;        fCdelete++;}

  if (fT2d_EvtNbInLoop         != 0){delete [] fT2d_EvtNbInLoop;          fCdelete++;}
  if (fT1d_EvtNbInLoop         != 0){delete [] fT1d_EvtNbInLoop;          fCdelete++;}

  if (fT3d_distribs            != 0){delete [] fT3d_distribs;             fCdelete++;}
  if (fT3d2_distribs           != 0){delete [] fT3d2_distribs;            fCdelete++;}
  if (fT3d1_distribs           != 0){delete [] fT3d1_distribs;            fCdelete++;}

  if (fT2d_ev                  != 0){delete [] fT2d_ev;                   fCdelete++;}
  if (fT1d_ev                  != 0){delete [] fT1d_ev;                   fCdelete++;}

  if (fT2d_var                 != 0){delete [] fT2d_var;                  fCdelete++;}
  if (fT1d_var                 != 0){delete [] fT1d_var;                  fCdelete++;}

  if (fT3d_his_s               != 0){delete [] fT3d_his_s;                fCdelete++;}
  if (fT2d_his_s               != 0){delete [] fT2d_his_s;                fCdelete++;}
  if (fT1d_his_s               != 0){delete [] fT1d_his_s;                fCdelete++;}

  if (fT2d_xmin                != 0){delete [] fT2d_xmin;                 fCdelete++;}
  if (fT1d_xmin                != 0){delete [] fT1d_xmin;                 fCdelete++;}
  if (fT2d_xmax                != 0){delete [] fT2d_xmax;                 fCdelete++;}
  if (fT1d_xmax                != 0){delete [] fT1d_xmax;                 fCdelete++;}

  if (fT3d_cov_ss              != 0){delete [] fT3d_cov_ss;               fCdelete++;}
  if (fT3d2_cov_ss             != 0){delete [] fT3d2_cov_ss;              fCdelete++;}
  if (fT3d1_cov_ss             != 0){delete [] fT3d1_cov_ss;              fCdelete++;}

  if (fT3d_cor_ss              != 0){delete [] fT3d_cor_ss;               fCdelete++;}
  if (fT3d2_cor_ss             != 0){delete [] fT3d2_cor_ss;              fCdelete++;}
  if (fT3d1_cor_ss             != 0){delete [] fT3d1_cor_ss;              fCdelete++;}

  if (fT3d_cov_cc              != 0){delete [] fT3d_cov_cc;               fCdelete++;}
  if (fT3d2_cov_cc             != 0){delete [] fT3d2_cov_cc;              fCdelete++;}
  if (fT3d1_cov_cc             != 0){delete [] fT3d1_cov_cc;              fCdelete++;}

  if (fT3d_cor_cc              != 0){delete [] fT3d_cor_cc;               fCdelete++;}
  if (fT3d2_cor_cc             != 0){delete [] fT3d2_cor_cc;              fCdelete++;}
  if (fT3d1_cor_cc             != 0){delete [] fT3d1_cor_cc;              fCdelete++;}

  if (fT2d_cov_cc_mos          != 0){delete [] fT2d_cov_cc_mos;           fCdelete++;}
  if (fT2d1_cov_cc_mos         != 0){delete [] fT2d1_cov_cc_mos;          fCdelete++;}

  if (fT2d_cor_cc_mos          != 0){delete [] fT2d_cor_cc_mos;           fCdelete++;}
  if (fT2d1_cor_cc_mos         != 0){delete [] fT2d1_cor_cc_mos;          fCdelete++;}

  if (fT2d_cov_moscc_mot       != 0){delete [] fT2d_cov_moscc_mot;        fCdelete++;}
  if (fT2d1_cov_moscc_mot      != 0){delete [] fT2d1_cov_moscc_mot ;      fCdelete++;}

  if (fT2d_cor_moscc_mot       != 0){delete [] fT2d_cor_moscc_mot ;       fCdelete++;}
  if (fT2d1_cor_moscc_mot      != 0){delete [] fT2d1_cor_moscc_mot;       fCdelete++;}

  if (fT1d_ev_ev               != 0){delete [] fT1d_ev_ev;                fCdelete++;}
  if (fT1d_ev_sig              != 0){delete [] fT1d_ev_sig;               fCdelete++;}
  if (fT1d_ev_cor_ss           != 0){delete [] fT1d_ev_cor_ss;            fCdelete++;}

  if (fT1d_sig_ev              != 0){delete [] fT1d_sig_ev;               fCdelete++;}
  if (fT1d_sig_sig             != 0){delete [] fT1d_sig_sig;              fCdelete++;}
  if (fT1d_sig_cor_ss          != 0){delete [] fT1d_sig_cor_ss;           fCdelete++;}

  if (fT2d_sv_correc_covss_s   != 0){delete [] fT2d_sv_correc_covss_s;    fCdelete++;}
  if (fT2d1_sv_correc_covss_s  != 0){delete [] fT2d1_sv_correc_covss_s;   fCdelete++;}

  if (fT3d_cov_correc_covss_s  != 0){delete [] fT3d_cov_correc_covss_s;   fCdelete++;}
  if (fT3d2_cov_correc_covss_s != 0){delete [] fT3d2_cov_correc_covss_s;  fCdelete++;}
  if (fT3d1_cov_correc_covss_s != 0){delete [] fT3d1_cov_correc_covss_s;  fCdelete++;}

  if (fT3d_cor_correc_covss_s  != 0){delete [] fT3d_cor_correc_covss_s;   fCdelete++;}
  if (fT3d2_cor_correc_covss_s != 0){delete [] fT3d2_cor_correc_covss_s;  fCdelete++;}
  if (fT3d1_cor_correc_covss_s != 0){delete [] fT3d1_cor_correc_covss_s;  fCdelete++;} 

  if (fT2dCrysNumbersTable     != 0){delete [] fT2dCrysNumbersTable;      fCdelete++;}
  if (fT1dCrysNumbersTable     != 0){delete [] fT1dCrysNumbersTable;      fCdelete++;}

  if (fjustap_2d_ev            != 0){delete [] fjustap_2d_ev;             fCdelete++;}
  if (fjustap_1d_ev            != 0){delete [] fjustap_1d_ev;             fCdelete++;}
  if (fjustap_2d_var           != 0){delete [] fjustap_2d_var;            fCdelete++;}
  if (fjustap_1d_var           != 0){delete [] fjustap_1d_var;            fCdelete++;}
  if (fjustap_2d_cc            != 0){delete [] fjustap_2d_cc;             fCdelete++;}
  if (fjustap_1d_cc            != 0){delete [] fjustap_1d_cc;             fCdelete++;}
  if (fjustap_2d_ss            != 0){delete [] fjustap_2d_ss;             fCdelete++;}
  if (fjustap_1d_ss            != 0){delete [] fjustap_1d_ss;             fCdelete++;}

  if (fTagTowerNumbers         != 0){delete [] fTagTowerNumbers;          fCdelete++;}
  if (fTagLastEvtNumber        != 0){delete [] fTagLastEvtNumber;         fCdelete++;}
  if (fTagEvtNbInLoop          != 0){delete [] fTagEvtNbInLoop;           fCdelete++;}
  if (fTagSampTime             != 0){delete [] fTagSampTime;              fCdelete++;}
  if (fTagEv                   != 0){delete [] fTagEv;                    fCdelete++;}
  if (fTagVar                  != 0){delete [] fTagVar;                   fCdelete++;}
  if (fTagEvts                 != 0){delete [] fTagEvts;                  fCdelete++;}

  if (fTagCovCss               != 0){delete [] fTagCovCss;                fCdelete++;}
  if (fTagCorCss               != 0){delete [] fTagCorCss;                fCdelete++;}

  if (fTagCovScc               != 0){delete [] fTagCovScc;                fCdelete++;}
  if (fTagCorScc               != 0){delete [] fTagCorScc;                fCdelete++;}
  if (fTagCovSccMos            != 0){delete [] fTagCovSccMos;             fCdelete++;}
  if (fTagCorSccMos            != 0){delete [] fTagCorSccMos;             fCdelete++;}

  if (fTagCovMosccMot          != 0){delete [] fTagCovMosccMot;           fCdelete++;}
  if (fTagCorMosccMot          != 0){delete [] fTagCorMosccMot;           fCdelete++;}

  if (fTagEvEv                 != 0){delete [] fTagEvEv;                  fCdelete++;}
  if (fTagEvSig                != 0){delete [] fTagEvSig;                 fCdelete++;}
  if (fTagEvCorCss             != 0){delete [] fTagEvCorCss;              fCdelete++;}

  if (fTagSigEv                != 0){delete [] fTagSigEv;                 fCdelete++;}
  if (fTagSigSig               != 0){delete [] fTagSigSig;                fCdelete++;}
  if (fTagSigCorCss            != 0){delete [] fTagSigCorCss;             fCdelete++;}

  if (fTagSvCorrecCovCss       != 0){delete [] fTagSvCorrecCovCss;        fCdelete++;}
  if (fTagCovCorrecCovCss      != 0){delete [] fTagCovCorrecCovCss;       fCdelete++;}
  if (fTagCorCorrecCovCss      != 0){delete [] fTagCorCorrecCovCss;       fCdelete++;}

  if ( fCnew != fCdelete )
    {
      cout << "!TCnaRunEB::~TCnaRunEB()> WRONG MANAGEMENT OF MEMORY ALLOCATIONS: fCnew = "
	   << fCnew << ", fCdelete = " << fCdelete << fTTBELL << endl;
    }
  else
    {
      // cout << "*TCnaRunEB::~TCnaRunEB()> Management of memory allocations: OK. fCnew = "
      //   << fCnew << ", fCdelete = " << fCdelete << endl;
    }
  
  if(fFlagPrint == fCodePrintComments || fFlagPrint == fCodePrintWarnings )
    {
      cout << "*TCnaRunEB::~TCnaRunEB()> Exiting destructor." << endl;
    }
  if(fFlagPrint == fCodePrintAllComments)
    {
      cout << "*TCnaRunEB::~TCnaRunEB()> Exiting destructor (this = " << this << ")." << endl
	   << "~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#"
	   << endl;
    }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
//                             M  E  T  H  O  D  S
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

//============================================================================
//
//                           GetReadyToReadData(...)
//                  
//============================================================================

void TCnaRunEB::GetReadyToReadData(TString typ_ana,      const Int_t& run_number,
				 Int_t&  nfirst,             Int_t& nevents,
				 const Int_t&  super_module)
{
//Preparation of the data reading. Set part of the header. No Nentries as argument.
//Use default value = 999999 and call method with all the arguments (see below)

  Int_t nentries = 999999999;

  GetReadyToReadData(typ_ana, run_number, nfirst, nevents, super_module, nentries);

}
//--------------------------------------------------------------------------------

void TCnaRunEB::GetReadyToReadData( TString typ_ana,      const Int_t& run_number,
				  Int_t&  nfirst,             Int_t& nevents,
			    const Int_t&  super_module, const Int_t& nentries)
{
//Preparation of the data reading. Set part of the header

  //************** CHECK OF ARGUMENTS: nfirst_arg and nevents_arg

  if ( nfirst < nentries )
    {
      //--------------------- test of positivity of nfirst_arg
      if (nfirst < 0 )
	{
	  if (fFlagPrint != fCodePrintNoComment){
	    cout << "!TCnaRunEB::GetReadyToReadData(...) > WARNING/CORRECTION:" << endl
		 << "! The fisrt taken event number is negative ( = " << nfirst << "). Forced to zero."
		 << fTTBELL << endl;}
	  nfirst = 0;
	}

      //-------- test of compatibility between the last taken event number
      //         and the number of entries

      Int_t   last_taken_event = nfirst + nevents - 1;
      Int_t   nevents_anc      = nevents;    

      if( last_taken_event > (nentries - 1) )
	{
	  nevents = nentries - nfirst -1;
	  if (fFlagPrint != fCodePrintNoComment){
	    cout << endl << "!TCnaRunEB::GetReadyToReadData(...)> WARNING/CORRECTION:" << endl
		 << "! The number of events (nevents = " << nevents_anc << ") is too large." << endl
		 << "! Number of last event = nfirst + nevents - 1 = " << nfirst << " + " << nevents_anc
		 << " - 1 = " << last_taken_event << " > number of entries - 1 = " << nentries-1 << ". "
		 << endl << "! Number of events forced to the value: nb of events ="
		 << " nb of entries - first event number - 1 = "
		 << nevents << "." << endl
		 << "! WARNING ===> THIS MODIFICATION WILL BE TRANSFERED TO THE RESULT FILE NAME"
		 << " (ROOT and ASCII FILES)."
		 << fTTBELL << endl << endl;}
	}
      
      //-----------------------------------------------------------------------------
 
      Text_t *h_name  = "CnaHeader";   //==> voir cette question avec FXG
      Text_t *h_title = "CnaHeader";   //==> voir cette question avec FXG

      TEBParameters* MyEcal = new TEBParameters();  fCnew++;
      Int_t n_tow  = MyEcal->fMaxTowInSM;
      Int_t n_crys = MyEcal->fMaxCrysInTow;
      Int_t n_samp = MyEcal->fMaxSampADC; 
      delete MyEcal;                                  fCdelete++;

      if ( n_tow > 0  &&  n_crys > 0  &&  n_samp> 0 ) 
	{
	  fFileHeader = new TCnaHeaderEB(h_name,       h_title ,
					 typ_ana,      run_number,  nfirst,    nevents,
					 super_module, nentries);
	  
	  // After this call to TCnaHeaderEB, we have:
	  //     fFileHeader->fTypAna        = typ_ana
	  //     fFileHeader->fRunNumber     = run_number
	  //     fFileHeader->fFirstEvt      = nfirst
	  //     fFileHeader->fNbOfTakenEvts = nevents
	  //     fFileHeader->fSuperModule   = super_module
	  //     fFileHeader->fNentries      = nentries
	  
	  // fFileHeader->Print();

	  // {Int_t cintoto; cout << "taper 0 pour continuer" << endl; cin >> cintoto;}
	  
	  //  fFileHeader->SetName("CnaHeader");              *======> voir FXG
	  //  fFileHeader->SetTitle("CnaHeader");
	  
	  //......................................... allocation tags + init of them
	  
	  fTagTowerNumbers    = new Int_t[1]; fCnew++; fTagTowerNumbers[0]    = (Int_t)0;
	  fTagLastEvtNumber   = new Int_t[1]; fCnew++; fTagLastEvtNumber[0]   = (Int_t)0;
	  fTagEvtNbInLoop     = new Int_t[1]; fCnew++; fTagEvtNbInLoop[0]     = (Int_t)0;

	  fTagSampTime        = new Int_t[fFileHeader->fMaxCrysInSM]; fCnew++;
          for (Int_t iz=0; iz<fFileHeader->fMaxCrysInSM; iz++){fTagSampTime[iz] = (Int_t)0;}

	  fTagEv              = new Int_t[1]; fCnew++; fTagEv[0]              = (Int_t)0;
	  fTagVar             = new Int_t[1]; fCnew++; fTagVar[0]             = (Int_t)0;
 
	  fTagEvts            = new Int_t[fFileHeader->fMaxCrysInSM]; fCnew++;
          for (Int_t iz=0; iz<fFileHeader->fMaxCrysInSM; iz++){fTagEvts[iz]   = (Int_t)0;}

	  fTagCovCss          = new Int_t[fFileHeader->fMaxCrysInSM]; fCnew++;
	  for (Int_t iz=0; iz<fFileHeader->fMaxCrysInSM; iz++){fTagCovCss[iz] = (Int_t)0;}

	  fTagCorCss          = new Int_t[fFileHeader->fMaxCrysInSM]; fCnew++;
          for (Int_t iz=0; iz<fFileHeader->fMaxCrysInSM; iz++){fTagCorCss[iz] = (Int_t)0;}

	  fTagCovScc          = new Int_t[fFileHeader->fMaxSampADC];  fCnew++;
          for (Int_t iz=0; iz<fFileHeader->fMaxSampADC; iz++){fTagCovScc[iz]  = (Int_t)0;}

	  fTagCorScc          = new Int_t[fFileHeader->fMaxSampADC];  fCnew++; 
          for (Int_t iz=0; iz<fFileHeader->fMaxSampADC; iz++){fTagCorScc[iz]  = (Int_t)0;}

	  fTagCovSccMos       = new Int_t[1]; fCnew++;	   fTagCovSccMos[0]    = (Int_t)0;
	  fTagCorSccMos       = new Int_t[1]; fCnew++; 	   fTagCorSccMos[0]    = (Int_t)0;
          fTagCovMosccMot     = new Int_t[1]; fCnew++; 	   fTagCovMosccMot[0]  = (Int_t)0;
          fTagCorMosccMot     = new Int_t[1]; fCnew++; 	   fTagCorMosccMot[0]  = (Int_t)0;

	  fTagEvEv            = new Int_t[1]; fCnew++;	   fTagEvEv[0]         = (Int_t)0;
	  fTagEvSig           = new Int_t[1]; fCnew++;	   fTagEvSig[0]        = (Int_t)0;
	  fTagEvCorCss        = new Int_t[1]; fCnew++;	   fTagEvCorCss[0]     = (Int_t)0; 

	  fTagSigEv           = new Int_t[1]; fCnew++;	   fTagSigEv[0]        = (Int_t)0; 
	  fTagSigSig          = new Int_t[1]; fCnew++;	   fTagSigSig[0]       = (Int_t)0; 
	  fTagSigCorCss       = new Int_t[1]; fCnew++;	   fTagSigCorCss[0]    = (Int_t)0; 

	  fTagSvCorrecCovCss  = new Int_t[1]; fCnew++;	   fTagSvCorrecCovCss[0]        = (Int_t)0; 

	  fTagCovCorrecCovCss = new Int_t[fFileHeader->fMaxCrysInSM]; fCnew++;
           for (Int_t iz=0; iz<fFileHeader->fMaxCrysInSM; iz++){fTagCovCorrecCovCss[iz] = (Int_t)0;}

	  fTagCorCorrecCovCss = new Int_t[fFileHeader->fMaxCrysInSM]; fCnew++;
           for (Int_t iz=0; iz<fFileHeader->fMaxCrysInSM; iz++){fTagCorCorrecCovCss[iz] = (Int_t)0;}

	  //====================================================================================
	  //
	  //   allocation for fT1d_SMtowFromIndex[] and init to fSpecialSMTowerNotIndexed
	  //
	  //====================================================================================

	  if(fT1d_SMtowFromIndex == 0)
	    {
	      fT1d_SMtowFromIndex = new Int_t[fFileHeader->fMaxTowInSM];          fCnew++;
	    }
	  for ( Int_t i_tow = 0; i_tow < fFileHeader->fMaxTowInSM; i_tow++ )
	    {
	      fT1d_SMtowFromIndex[i_tow] = fSpecialSMTowerNotIndexed;
	    }
	  
	  //------------------------------------------------------------------
	  
	  //====================================================================================
	  //
	  //   allocation of the 3D array fT3d_distribs[channel][sample][events] (ADC values)
	  //
	  //   This array is filled in the BuildEventDistributions(...) method
	  //
	  //====================================================================================
	  
	  if(fT3d_distribs == 0)
	    {
	      //............ Allocation for the 3d array 
	      fT3d_distribs   =
		new Double_t**[fFileHeader->fMaxCrysInSM];                         fCnew++;  
	      fT3d2_distribs  =
		new  Double_t*[fFileHeader->fMaxCrysInSM*
			       fFileHeader->fMaxSampADC];                          fCnew++;  
	      fT3d1_distribs  =
		new   Double_t[fFileHeader->fMaxCrysInSM*
			       fFileHeader->fMaxSampADC*
			       fFileHeader->fNbOfTakenEvts];                       fCnew++;
	      
	      for(Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++){
		fT3d_distribs[i_SMEcha] = &fT3d2_distribs[0] + i_SMEcha*fFileHeader->fMaxSampADC;
		for(Int_t j_samp = 0 ; j_samp < fFileHeader->fMaxSampADC ; j_samp++){
		  fT3d2_distribs[fFileHeader->fMaxSampADC*i_SMEcha + j_samp] = &fT3d1_distribs[0]+
		    fFileHeader->fNbOfTakenEvts*(fFileHeader->fMaxSampADC*i_SMEcha+j_samp);}}
	    }
	  
	  //................................. Init to zero
          for (Int_t iza=0; iza<fFileHeader->fMaxCrysInSM; iza++)
	    {
	      for (Int_t izb=0; izb<fFileHeader->fMaxSampADC; izb++)
	  	{
		  for (Int_t izc=0; izc<fFileHeader->fNbOfTakenEvts; izc++)
		    {
		      if( fT3d_distribs[iza][izb][izc] != (Double_t)0 )
			{
			  fMiscDiag[0]++;
			  fT3d_distribs[iza][izb][izc] = (Double_t)0;
			}
		    }
		}
	    }	  
	  
	  //====================================================================================
	  //
	  //   allocation of the 2D array fT2d_LastEvtNumber[channel][sample] (Max nb of evts)
	  //
	  //====================================================================================
	  
	  if (fT2d_LastEvtNumber == 0)
	    {
	      fT2d_LastEvtNumber  = new Int_t*[fFileHeader->fMaxCrysInSM];           fCnew++;
	      fT1d_LastEvtNumber  = new  Int_t[fFileHeader->fMaxCrysInSM*
					       fFileHeader->fMaxSampADC];            fCnew++;
	      
	      for(Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
		{
		  fT2d_LastEvtNumber[i_SMEcha] =
		    &fT1d_LastEvtNumber[0] + i_SMEcha*fFileHeader->fMaxSampADC;
		}
	      
	      //................ Init the array to -1
	      for(Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
		{
		  for(Int_t i_samp = 0 ; i_samp < fFileHeader->fMaxSampADC ; i_samp++)
		    {
		      fT2d_LastEvtNumber[i_SMEcha][i_samp] = -1;
		    }
		}
	    }
	  else
	    {
	      cerr << "!TCnaRunEB::GetReadyToReadData(...)> *** ERROR *** No allocation for fT2d_LastEvtNumber!"
		   << " Pointer already not NULL " << fTTBELL << endl;
	      {Int_t cintoto; cout << "Enter: 0 and RETURN to continue or: CTRL C to exit"
				   << endl; cin >> cintoto;}
	    }
	  
	  //========================================================================================
	  //
	  //   allocation of the 2D array fT2d_EvtNbInLoop[tower][cna evt number] (evt nb in loop)
	  //
	  //========================================================================================
	  
	  if (fT2d_EvtNbInLoop == 0)
	    {
	      fT2d_EvtNbInLoop  = new Int_t*[fFileHeader->fMaxTowInSM];          fCnew++;
	      fT1d_EvtNbInLoop  = new  Int_t[fFileHeader->fMaxTowInSM*
					     fFileHeader->fNbOfTakenEvts];       fCnew++;
	      
	      for(Int_t i_tow = 0 ; i_tow < fFileHeader->fMaxTowInSM ; i_tow++)
		{
		  fT2d_EvtNbInLoop[i_tow] =
		    &fT1d_EvtNbInLoop[0] + i_tow*fFileHeader->fNbOfTakenEvts;
		}
	      
	      //................ init the array to -1
	      for(Int_t i_tow = 0 ; i_tow < fFileHeader->fMaxTowInSM ; i_tow++)
		{
		  for(Int_t i_cna_evt = 0 ; i_cna_evt < fFileHeader->fNbOfTakenEvts ; i_cna_evt++)
		    {
		      fT2d_EvtNbInLoop[i_tow][i_cna_evt] = -1;
		    }
		}	      
	    }
	  else
	    {
	      cerr << "!TCnaRunEB::GetReadyToReadData(...)> *** ERROR *** No allocation for fT2d_EvtNbInLoop!"
		   << " Pointer already not NULL " << fTTBELL << endl;
	      {Int_t cintoto; cout << "Enter: 0 and RETURN to continue or: CTRL C to exit"
				   << endl; cin >> cintoto;}
	    }
	}
      else
	{
	  cerr << endl
	       << "!TCnaRunEB::GetReadyToReadData(...)> "
	       << " *** ERROR *** " << endl
	       << " --------------------------------------------------"
	       << endl
	       << " NULL or NEGATIVE values for arguments" << endl
	       << " with expected positive values:"        << endl
	       << " Number of towers in SuperModule = " << fFileHeader->fMaxTowInSM  << endl
	       << " Number of crystals in tower     = " << fFileHeader->fMaxCrysInTow << endl
	       << " Number of samples by channel    = " << fFileHeader->fMaxSampADC << endl
	       << endl
	       << endl
	       << " hence, no memory allocation for array member has been performed." << endl;
	  
	  cout << "Enter: 0 and RETURN to continue or: CTRL C to exit";
	  Int_t toto;
	  cin >> toto;
	}

      
      //    CI-DESSOUS: REPRENDRE LES VALEURS NUMERIQUES  A PARTIR DE TEBParameters

      //#####################################################################################
      //
      //..................... (for ASCII files writing methods only) ..............
      //
      //                DEFINITION OF THE SECTOR SIZES
      //       FOR THE CORRELATION AND COVARIANCE MATRICES DISPLAY
      //
      //            MUST BE A DIVISOR OF THE TOTAL NUMBER.
      //            ======================================
      //
      //     Examples:
      //      
      //      (1)       25 channels => size = 25 or 5 (divisors of 25)
      //
      //                25 => matrix = 1 x 1 sector  of size (25 x 25)
      //                             = (1 x 1) x (25 x 25) = 1 x 625 = 625 
      //                 5 => matrix = 5 x 5 sectors of size (5 x 5)
      //                             = (5 x 5) x ( 5 x  5) = 25 x 25 = 625 
      //
      //      (2)       10 samples  => size = 10, 5 or 2 (divisors of 10)
      //
      //                10 => matrix = 1 X 1 sectors of size (10 x 10) 
      //                             = (1 x 1) x (10 x 10) =  1 x 100 = 100
      //                 5 => matrix = 2 x 2 sectors of size (5 x 5) 
      //                             = (2 x 2) x ( 5 x  5) =  4 x  25 = 100
      //                 2 => matrix = 5 x 5 sectors of size (2 x 2) 
      //                             = (5 x 5) x ( 2 x  2) = 25 x  4  = 100
      //
      //........................................................................
      fSectChanSizeX = 5;      // => 25 crystals by tower
      fSectChanSizeY = 5;
      fSectSampSizeX = 10;      // test beam Nov 2004 => 10 samples
      fSectSampSizeY = 10;
      
      //........................................................................
      //
      //                DEFINITION OF THE NUMBER OF VALUES BY LINE
      //                for the Expectation Values, Variances and.
      //                Event distributions by (channel,sample)
      //
      //               MUST BE A DIVISOR OF THE TOTAL NUMBER.
      //               ======================================
      //
      //     Examples: 
      //                1) For expectation values and variances:
      //
      //                25 channels => size = 5
      //                => sample sector = 5 lines of 5 values
      //                                 = 5 x 5 = 25 values 
      //
      //                10 samples  => size = 10
      //                => channel sector = 1 lines of 10 values
      //                                  = 1 x 10 = 10 values
      //
      //                2) For event distributions:
      //
      //                100 bins  => size = 10
      //                => sample sector = 10 lines of 10 values
      //                                 = 10 x 10 = 100 values
      //
      //........................................................................
      fNbChanByLine = 5;   // => 25 crystals by tower
      fNbSampByLine = 10;  // => 10 samples     
      
      //.............. (for ASCII files writing methods - END - ) ..................
    }
  else
    {
      cout << "!TCnaRunEB::GetReadyToReadData(...) *** ERROR ***> "
	   << " The first taken event number is greater than the number of entries - 1."
	   << fTTBELL << endl;
    }
  
  if(fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments ){
    cout << endl;
    cout << "*TCnaRunEB::GetReadyToReadData(...)>" << endl
	 << "          The method has been called with the following argument values:"  << endl
	 << "          Analysis name          = "
	 << fFileHeader->fTypAna << endl
	 << "          Run number             = "
	 << fFileHeader->fRunNumber << endl
	 << "          Number of entries      = "
	 << fFileHeader->fNentries << endl
	 << "          First taken event      = "
	 << fFileHeader->fFirstEvt << endl
	 << "          Number of taken events = "
	 << fFileHeader->fNbOfTakenEvts << endl
	 << "          SuperModule number     = "
	 << fFileHeader->fSuperModule << endl
	 << "          Number of towers in SM = "
	 << fFileHeader->fMaxTowInSM   << endl
	 << "          Number of crystals in tower  = "
	 << fFileHeader->fMaxCrysInTow  << endl
	 << "          Number of samples by channel = "
	 << fFileHeader->fMaxSampADC  << endl
	 << endl;}

  fReadyToReadData = 1;                        // set flag

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::GetReadyToReadData(...)> Leaving the method"
  	 << endl; }
} // end of GetReadyToReadData

//====================================================================================================
//
//                  Building of the arrays for CNA
//    (tower number <-> tower index correspondance, ADC sample values)
//
//        THIS METHOD IS CALLED INSIDE THE LOOPS OVER:
//          ( EVENTS (TOWERS (CRYSTAL IN TOWER (SAMPLES)))) 
//
//  Arguments: ievent   = event number.                Range = [ 0, fFileHeader->fNbOfTakenEvts - 1 ]
//             SMtower  = tower number in SuperModule. Range = [ 1, fFileHeader->fMaxTowInSM ]
//             TowEcha  = channel number in tower.     Range = [ 0, fFileHeader->fMaxCrysInTow - 1 ]     
//             sample   = ADC sample number.           Range = [ 0, fFileHeader->fMaxSampADC - 1 ]
//             adcvalue = ADC sample value.
//
//====================================================================================================
Bool_t TCnaRunEB::BuildEventDistributions(const Int_t&    ievent,   const Int_t& SMtower,
					const Int_t&    TowEcha,  const Int_t& sample,
					const Double_t& adcvalue)
{
  //Building of the arrays fT1d_SMtowFromIndex[] and fT3d_distribs[][][]
  
  Bool_t ret_code = kFALSE;
  Int_t  i_SMtow  = SMtower-1;  // INDEX FOR SMTower = Number_of_the_tower_in_SM - 1
  Int_t  i_trouve = 0;

  if(fReadyToReadData == 1)  
    {
      if( SMtower>= 1 && SMtower <= fFileHeader->fMaxTowInSM )
	{      
	  if( TowEcha >= 0 && TowEcha < fFileHeader->fMaxCrysInTow )
	    {
	      if( sample >= 0 && sample < fFileHeader->fMaxSampADC )
		{
		  //..... Put the SMtower number in 1D array fT1d_SMtowFromIndex[] = tower index + 1
		  if(fT1d_SMtowFromIndex != 0)       // table fT1d_SMtowFromIndex[index] already allocated
		    {
		      ret_code = kTRUE;

		      // SMtower already indexed
		      if( SMtower == fT1d_SMtowFromIndex[i_SMtow] )
			{
			  i_trouve = 1;
			}
		  
		      // SMtower index not found: new SMtower
		      if (i_trouve != 1 )
			{
			  if ( fT1d_SMtowFromIndex[i_SMtow] == fSpecialSMTowerNotIndexed )
			    {
			      fT1d_SMtowFromIndex[i_SMtow] = SMtower;
			      fFileHeader->fTowerNumbersCalc = 1;
			      fTagTowerNumbers[0] = 1;
			      fTowerIndexBuilt++;                      //  number of found towers
			  
			      if(fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments){
				cout << "*TCnaRunEB::BuildEventDistributions(...)> event " << ievent
				     << " : first event for tower " << fT1d_SMtowFromIndex[i_SMtow]
				     << " (" << fTowerIndexBuilt << " towers found)" << endl;}
			      ret_code = kTRUE;
			    }
			  else
			    {
			      cout << "!TCnaRunEB::BuildEventDistributions(...)> *** ERROR ***> IMPOSSIBILITY. " 
				   << " SMtower= " << SMtower << ", fT1d_SMtowFromIndex[" << i_SMtow << "] = "
				   << fT1d_SMtowFromIndex[i_SMtow]
				   << ", fTowerIndexBuilt = " << fTowerIndexBuilt
				   << fTTBELL << endl;
			      ret_code = kFALSE;
			    }
			}
		    }
		  else
		    {
		      cout << "!TCnaRunEB, BuildEventDistributions *** ERROR ***> "
			   << " fT1d_SMtowFromIndex = " << fT1d_SMtowFromIndex
			   << " fT1d_SMtowFromIndex[] ALLOCATION NOT DONE" << fTTBELL << endl;
		      ret_code = kFALSE;
		    }
		}
	      else
		{
		  cout << "!TCnaRunEB::BuildEventDistributions(...) *** ERROR ***> "
		       << " sample number = " << sample << ". OUT OF BOUNDS"
		       << " (max = " << fFileHeader->fMaxSampADC << ")"
		       << fTTBELL << endl;
		  ret_code = kFALSE;
		}
	    }
	  else
	    {
	      cout << "!TCnaRunEB::BuildEventDistributions(...) *** ERROR ***> "
		   << " TowEcha number = " << TowEcha << ". OUT OF BOUNDS"
		   << " (max = " << fFileHeader->fMaxCrysInTow << ")"
		   << fTTBELL << endl;
	      ret_code = kFALSE;
	    }
	}
      else
	{
	  cout << "!TCnaRunEB::BuildEventDistributions(...) *** ERROR ***> "
	       << " SMtower number = " << SMtower << ". OUT OF BOUNDS"
	       << " (max = " << fFileHeader->fMaxTowInSM << ")"
	       << fTTBELL << endl;
	  ret_code = kFALSE;
	}

      //........ Filling of the 2D array of the event numbers in the data reading loop and 
      //         filling of the 3D array of the ADC sample values

      if( ret_code == kTRUE )
	{
	  //............ 1) Conversion (tower,TowEcha) -> SMEcha
	  Int_t SMEcha = i_SMtow*fFileHeader->fMaxCrysInTow + TowEcha;
	  
	  if( SMEcha >= 0 && SMEcha < fFileHeader->fMaxCrysInSM )
	    {
	      //............ 2) Increase of the nb of evts for (SMEcha,sample)
	      (fT2d_LastEvtNumber[SMEcha][sample])++;     // value after first incrementation = 0
	      fTagLastEvtNumber[0] = 1;
	      fFileHeader->fLastEvtNumberCalc = 1;
	      //if(fBuildEvtDistrib == 0){fBuildEvtDistrib = 1;} // set flag of building
	      
	      //............ 3) Filling of the array fT2d_EvtNbInLoop[tower][cna event number]
	      Int_t k_event = fT2d_LastEvtNumber[SMEcha][sample];

	      if ( k_event >= 0 && k_event < fFileHeader->fNbOfTakenEvts )
		{ 
		  fT2d_EvtNbInLoop[i_SMtow][k_event] = ievent;
		}
	      else
		{
		  cout << "!TCnaRunEB::BuildEventDistributions(...) *** ERROR ***> "
		       << " last event number = " << k_event << ". OUT OF BOUNDS"
		       << " (range = [0," << fFileHeader->fNbOfTakenEvts-1
		       << "]). Channel# = " << SMEcha << ", sample = " << sample
		       << fTTBELL << endl;
		  ret_code = kFALSE;
		}
	      fTagEvtNbInLoop[0] = 1;
	      fFileHeader->fEvtNbInLoopCalc = 1;
	      
	      //............ 4) Filling of the 3D array of the ADC values
	      if ( ievent >= 0 && ievent < fFileHeader->fNbOfTakenEvts )
		{  
		  fT3d_distribs[SMEcha][sample][ievent] = adcvalue;
		}
	      else
		{
		  cout << "!TCnaRunEB::BuildEventDistributions(...) *** ERROR ***> "
		       << " event number = " << ievent << ". OUT OF BOUNDS"
		       << " (max = " << fFileHeader->fNbOfTakenEvts << ")"
		       << fTTBELL << endl;
		  ret_code = kFALSE;
	      	}
	    }
	  else
	    {
	      cout << "!TCnaRunEB::BuildEventDistributions(...) *** ERROR ***> "
		   << " CHANNEL NUMBER OUT OF BOUNDS" << endl
		   << " SMEcha number = " << SMEcha
		   << " , SMtower = " << SMtower
		   << " , TowEcha = " << TowEcha
		   << " , fFileHeader->fMaxCrysInSM = " << fFileHeader->fMaxCrysInSM 
		   << fTTBELL << endl; 
	      ret_code = kFALSE;
	      // {Int_t cintoto; cout << "TAPER 0 POUR CONTINUER" << endl; cin >> cintoto;}
	    }
	}
      else
	{
	  cout << "!TCnaRunEB::BuildEventDistributions(...) *** ERROR ***> ret_code = kFALSE "
	       << fTTBELL << endl;
	}
    }
  else
    {
      cout << "!TCnaRunEB::BuildEventDistributions(...) *** ERROR ***> GetReadyToReadData(...) not called."
	   << fTTBELL << endl;
      ret_code = kFALSE;
    }

  if (ret_code == kFALSE)
    {
      cout << "*> ievent: " << ievent
	   << ", SMtower: " << SMtower
	   << ", TowEcha: " << TowEcha
	   << ", sample: "  << sample
	   << ", adcvalue: " << adcvalue << endl;
    } 

  return ret_code;
}
//###################################################################################################
//
// THE FOLLOWING METHODS ARE CALLED AFTER THE LOOPS OVER EVENTS, TOWERS, CRYSTALS AND SAMPLES
//
//###################################################################################################

//=========================================================================
//
//                         GetReadyToCompute()
//          Building of the TDistrib objects for CNA calculations
//
//=========================================================================
void TCnaRunEB::GetReadyToCompute()
{
//Building of the TDistrib objects for CNA calculations

  //----------------------------------------------------------------------
  //
  //       recuperation des valeurs et mise dans les objets
  //       "distributions d'evenements"  ( classe TDistrib )
  //
  //  fT3d_distribs[SMEcha][sample][event] -> fVal_Data[SMEcha][sample]
  //
  //----------------------------------------------------------------------  

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
   if(fFlagPrint == fCodePrintAllComments)
    { 
      cout << endl << "%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%~%"
	   << endl << "*TCnaRunEB::GetReadyToCompute()> (this = " << this << ") " << endl;
    }
#define TOBJ
#ifndef TOBJ 

  //......... Essai utilisation TClonesArray --> NE MARCHE PAS ENCORE:
  //
  //TCnaRunEB.cxx: In method `void TCnaRunEB::GetReadyToCompute ()':
  //<internal>:1333: too many arguments to function `void *operator new 
  //(unsigned int)'
  //TCnaRunEB.cxx:1333: at this point in file
  // 1333 = ancien numero de la ligne ou il y a : new(fVal_dat2[i_new]) etc...

  Int_t maxichan = fFileHeader->fMaxCrysInSM*fFileHeader->fMaxSampADC;
  TObjArray fVal_dat2(maxichan);
  
  for (Int_t i_chan = 0 ; i_chan < fFileHeader->fMaxCrysInSM ; i_chan++)
    { 
      for (Int_t n_sampl = 0 ; n_sampl < fFileHeader->fMaxSampADC ; n_sampl++)
	{       
	  Int_t i_new = (fFileHeader->fMaxSampADC)*i_chan + n_sampl;
	  fVal_dat2[i_new] = new
	    TDistrib(fFileHeader->fNbOfTakenEvts, fT3d2_distribs[i_new]);  fCnew++;
	}
    }

#endif // TOBJ

//&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

  //-------------------- declaration des objets fVal_data[][]
   
  //....................................................................
  //
  //    TDistrib = "distribution d'evenements" = objet comprenant:
  //
  //    (1) un nombre d'evenements (Int_t nmax) 
  //    (2) un tableau de valeurs de dimension le nombre
  //        d'evenements ( Double_t valeurs[nmax] ).
  //
  //    Il y a une distribution d'evenements par echantillon et par voie.
  //    On a donc un tableau 2d fVal_data[][] de distributions qu'on va
  //    initialiser ci-dessous a partir du tableau 3d
  //    fT3d_distribs[][][].
  //
  //    PS: on est oblige de passer par un objet temporaire (pdata_temp)
  //        pour pouvoir appeler le constructeur avec arguments et faire
  //        ensuite fVal_data[i_SMEcha][n_sampl] = pdata_temp;
  //        (utilisation de la surcharge de l'operateur=
  //         de la classe TDistrib)
  //......................................................................
  
  if(fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments ){
    cout << "*TCnaRunEB::GetReadyToCompute()> Starting TDistrib object allocations"
	 << endl;}
  
  // !!!!! ALLOC DYNAMIQUE D'UN TABLEAU 2D d'objets de classe "TDistrib"
  //  alloc dynamique de fVal_data** ou fVal_data[d1][d2]
  //  sous forme de tableau 1D de valeurs fVal_dat2[d1xd2]
  //  et de tableau d'adresses 1D fVal_data[d1]  
  
  fVal_data = new TDistrib*[fFileHeader->fMaxCrysInSM];                          fCnew++; 
  fVal_dat2 = new  TDistrib[fFileHeader->fMaxCrysInSM*fFileHeader->fMaxSampADC]; fCnew++;
  
  // calcul des pointeurs du tableau 1D fVal_data[d1]    ( GetReadyToCompute() )
  
  for( Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++ )
    {
      fVal_data[i_SMEcha] = &fVal_dat2[0] + i_SMEcha*fFileHeader->fMaxSampADC;
    }
  
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      ( GetReadyToCompute() )
  
  for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    { 
      for (Int_t n_sampl = 0 ; n_sampl < fFileHeader->fMaxSampADC ; n_sampl++)
	{ 	
	  //.............. standard C++
	  //	  TDistrib  pdata_temp( fFileHeader->fNbOfTakenEvts,
	  //		*((fT3d_distribs)+
	  //		  (fFileHeader->fMaxSampADC)*i_SMEcha +n_sampl) ); 
	  
	  //................  CINT-ROOT

	  TDistrib*	pdata_temp =
	    new TDistrib( fFileHeader->fNbOfTakenEvts,
			  fT3d2_distribs[(fFileHeader->fMaxSampADC)*i_SMEcha
					 + n_sampl] );                           fCnew++;
	  
	  // Utilisation de la surcharge de l'operateur= de la classe TDistrib
	  // fVal_data[i_SMEcha][n_sampl] = pdata_temp;  // standard C++
	  
	  fVal_data[i_SMEcha][n_sampl] = pdata_temp[0];  // CINT-ROOT
	  delete pdata_temp;                fCdelete++;  // CINT-ROOT
	}
    }

  if(fFlagPrint == fCodePrintAllComments || fFlagPrint == fCodePrintComments ){
    cout << "*TCnaRunEB::GetReadyToCompute()> TDistrib object allocations done."
	 << endl;}

#define RESZ
#ifndef RESZ
  //------------- Resize of the TDistrib fVal_data to the effective max nb of evts    ( GetReadyToCompute() )
  //
  //   Finally, no resize since it masks cases with wrong numbers of events
  //
  Int_t  nb_of_resize = 0;
  Bool_t ok_resize    = kFALSE;

  for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    { 
      for (Int_t n_sampl = 0 ; n_sampl < fFileHeader->fMaxSampADC ; n_sampl++)
	{
	  ok_resize = kFALSE;
	  Int_t  nb_evts   = fT2d_LastEvtNumber[i_SMEcha][n_sampl]+1;
	  ok_resize = fVal_data[i_SMEcha][n_sampl].Resize(nb_evts);
	  if ( ok_resize == kTRUE ) {nb_of_resize++;}
 	}
    }

  if (nb_of_resize != 0)
    {
      cout << "!TCnaRunEB::GetReadyToCompute()> " 
	   << "INFO/warning =!=!=!=!=!=> a resize of TDistrib distribution has been done in "
	   << nb_of_resize << " cases " << endl
	   << "                               (could be due to incorrect numbers of events in data)"
	   << fTTBELL << endl;
    }
  else
    {
      if(fFlagPrint == fCodePrintAllComments){
	cout << "!TCnaRunEB::GetReadyToCompute()> " 
	     << "INFO =+=+=+=+=+=> TDistrib distributions: no resize has been done." << endl;}
    }

  //-----------------------------------------------------------------------------------------------
#endif // RESZ

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}  
//  end of GetReadyToCompute()

//=========================================================================
//
//     Set start time, stop time, StartDate, StopDate
//
//=========================================================================
void TCnaRunEB::StartStopTime(time_t t_startime, time_t t_stoptime)
{
// Put the start an stop time (if they exist) in class attributes.

  fFileHeader->fStartTime = t_startime;
  fFileHeader->fStopTime  = t_stoptime;
}

void TCnaRunEB::StartStopDate(TString c_startdate, TString c_stopdate)
{
// Put the start an stop date (if they exist) in class attributes.

  fFileHeader->fStartDate = c_startdate;
  fFileHeader->fStopDate  = c_stopdate;
}

//=========================================================================
//
//               C A L C U L A T I O N    M E T H O D S
//
//     fTag... => Calculation done. OK for writing on result file
//     ...Calc => Incrementation for result file size. 
//
//=========================================================================
//-------------------------------------------------------------------
//
//             Making of the histograms of the sample value
//             as a funtion of the event number
//             for all the pairs (SMEcha, samples)
//
//-------------------------------------------------------------------
void TCnaRunEB::MakeHistosOfSamplesAsFunctionOfEvent()
{
//Making of the histograms of the sample value as a funtion of the event number
//for all the pairs (SMEcha, samples)

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::MakeHistosOfSamplesAsFunctionOfEvent()>"
         << " Making of the histograms of the sample value as a funtion of"
	 << " the event number for all the pairs (SMEcha, samples)" << endl;}

  // In fact, the histo is already in fT3d_distribs[][][fFileHeader->fNbOfTakenEvts]
  // this method sets only the "Tag" and increment the "Calc" (and must be kept for that)

  for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      fTagSampTime[i_SMEcha] = 1;        fFileHeader->fSampTimeCalc++;
    }
}

//----------------------------------------------------------------
//
//     Calculation of the expectation values of the samples
//                 for all the SMEchas
//
//----------------------------------------------------------------

void TCnaRunEB::ComputeExpectationValuesOfSamples()
{
//Calculation of the expectation values of the samples for all the SMEchas

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeExpectationValuesOfSamples()>"
         << " Calculation of the expectation values of the samples"
	 << " for all the SMEchas" << endl;}

  //................... Allocation ev + init to zero
  if ( fT2d_ev == 0 ){
    Int_t n_samp = fFileHeader->fMaxSampADC;
    Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    fT2d_ev = new Double_t*[n_SMEcha];             fCnew++;  
    fT1d_ev = new  Double_t[n_SMEcha*n_samp];      fCnew++;   
    for(Int_t i = 0 ; i < n_SMEcha ; i++){
      fT2d_ev[i] = &fT1d_ev[0] + i*n_samp;}
  }
  
  for(Int_t i_SMEcha=0; i_SMEcha<fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      for(Int_t i_samp=0; i_samp<fFileHeader->fMaxSampADC; i_samp++)
 	{
 	  if( fT2d_ev[i_SMEcha][i_samp] != (Double_t)0 )
	    {fMiscDiag[1]++; fT2d_ev[i_SMEcha][i_samp] = (Double_t)0;}
 	} 
    }
  
  //................... Calculation
  for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      for (Int_t i_samp = 0 ; i_samp < fFileHeader->fMaxSampADC ; i_samp++)
	{	  
	  fT2d_ev[i_SMEcha][i_samp] = 
	    ((fVal_data)[i_SMEcha][i_samp]).ExpectationValue();
	}
    }
  fTagEv[0] = 1;        fFileHeader->fEvCalc++;
}

//--------------------------------------------------------
//
//      Calculation of the variances of the samples
//                 for all the SMEchas
//
//--------------------------------------------------------
  
void TCnaRunEB::ComputeVariancesOfSamples() 
{
//Calculation of the variances of the samples for all the SMEchas
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeVariancesOfSamples()>"
	 << " Calculation of the variances of the samples"
	 << " for all the SMEchas" << endl;}
  
  //................... Allocation var + init to zero
  if( fT2d_var == 0){
    Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    Int_t n_samp = fFileHeader->fMaxSampADC;
    fT2d_var = new Double_t*[n_SMEcha];                fCnew++;        
    fT1d_var = new  Double_t[n_SMEcha*n_samp];         fCnew++;  
    for(Int_t i_SMEcha = 0 ; i_SMEcha < n_SMEcha ; i_SMEcha++){
      fT2d_var[i_SMEcha] = &fT1d_var[0] + i_SMEcha*n_samp;}
  }
  
  for(Int_t i_SMEcha=0; i_SMEcha<fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      for(Int_t i_samp=0; i_samp<fFileHeader->fMaxSampADC; i_samp++)
 	{
	  if( fT2d_var[i_SMEcha][i_samp] != (Double_t)0 )
	    {fMiscDiag[2]++; fT2d_var[i_SMEcha][i_samp] = (Double_t)0;}
 	} 
    }
  
  //................... Calculation
  for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      for (Int_t i_samp = 0 ; i_samp < fFileHeader->fMaxSampADC ; i_samp++)
	{
	  fT2d_var[i_SMEcha][i_samp] = 
	    fVal_data[i_SMEcha][i_samp].VarianceValue();
	}
    }
  fTagVar[0] = 1;        fFileHeader->fVarCalc++;
}

//-------------------------------------------------------------------
//
//             Making of the histograms of the ADC distributions
//             for all the pairs (SMEcha, samples)
//
//-------------------------------------------------------------------

void TCnaRunEB::MakeHistosOfSampleDistributions()
{
//Histograms of the ADC distributions for all the pairs (SMEcha,sample)

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::MakeHistosOfSampleDistributions()>"
	 << " Histograms of ADC distributions for all the pairs (SMEcha,sample)"
	 << endl;}

  //................... Allocation his_s + init to zero

  if (fT3d_his_s == 0 )
    {
      //............ Allocation fT3d_his_s 
      fT3d_his_s = new Double_t**[fFileHeader->fMaxCrysInSM];                          fCnew++;   
      fT2d_his_s = new  Double_t*[fFileHeader->fMaxCrysInSM*fFileHeader->fMaxSampADC]; fCnew++;   
      fT1d_his_s = new   Double_t[fFileHeader->fMaxCrysInSM*
				  fFileHeader->fMaxSampADC*
				  fFileHeader->fNbBinsADC];                            fCnew++;

      for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++){
	  fT3d_his_s[i_SMEcha] = &fT2d_his_s[0] + i_SMEcha*fFileHeader->fMaxSampADC;
	  for(Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++){
	    fT2d_his_s[i_SMEcha*fFileHeader->fMaxSampADC + i_samp] = &fT1d_his_s[0]
	      + (i_SMEcha*fFileHeader->fMaxSampADC + i_samp)*fFileHeader->fNbBinsADC;}}
    }

  for(Int_t i_SMEcha=0; i_SMEcha<fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      for(Int_t i_samp=0; i_samp<fFileHeader->fMaxSampADC; i_samp++)
 	{
 	  for(Int_t i_bin=0; i_bin<fFileHeader->fNbBinsADC; i_bin++)
 	    {
	      if( fT3d_his_s[i_SMEcha][i_samp][i_bin] != (Double_t)0 )
		{fMiscDiag[3]++; fT3d_his_s[i_SMEcha][i_samp][i_bin] = (Double_t)0;}
 	    }
 	} 
    }
  
  //.................................. Allocations xmin, xmax + init to zero
  if( fT1d_xmin == 0 )
    {
      fT2d_xmin = new Double_t*[fFileHeader->fMaxCrysInSM];                           fCnew++;
      fT1d_xmin = new  Double_t[fFileHeader->fMaxCrysInSM*fFileHeader->fMaxSampADC];  fCnew++;
      for (Int_t i1 = 0; i1 < fFileHeader->fMaxCrysInSM; i1++){
	fT2d_xmin[i1] = &fT1d_xmin[0] + i1*fFileHeader->fMaxSampADC;}
    }
  
  if( fT1d_xmax == 0 )
    {
      fT2d_xmax = new Double_t*[fFileHeader->fMaxCrysInSM];                           fCnew++;
      fT1d_xmax = new  Double_t[fFileHeader->fMaxCrysInSM*fFileHeader->fMaxSampADC];  fCnew++;
      for (Int_t i1 = 0; i1 < fFileHeader->fMaxCrysInSM; i1++){
	fT2d_xmax[i1] = &fT1d_xmax[0] + i1*fFileHeader->fMaxSampADC;}
    } 
  
  for(Int_t i_SMEcha=0; i_SMEcha<fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      for(Int_t i_samp=0; i_samp<fFileHeader->fMaxSampADC; i_samp++)
 	{
	  if( fT2d_xmin[i_SMEcha][i_samp] != (Double_t)0 )
	    {fMiscDiag[4]++; fT2d_xmin[i_SMEcha][i_samp] = (Double_t)0;}
 
	  if( fT2d_xmax[i_SMEcha][i_samp] != (Double_t)0 )
	    {fMiscDiag[5]++; fT2d_xmax[i_SMEcha][i_samp] = (Double_t)0;}  
 	} 
    }
  
  Double_t xmin = 0;
  Double_t xmax = 0;

  //............. alloc dynamique de s_histo + init to zero
  Double_t*  s_histo = new Double_t[fFileHeader->fNbBinsADC];  fCnew++;  
  for(Int_t i_bin=0; i_bin<fFileHeader->fNbBinsADC; i_bin++)
    {
      if( s_histo[i_bin] != (Double_t)0 ){fMiscDiag[6]++; s_histo[i_bin] = (Double_t)0;}
    }

  Int_t total_underflow = 0;
  Int_t total_overflow  = 0;

  Int_t nb_null_sample_chan = 0;
  Int_t nb_null_sample_all  = 0;

  TEBNumbering* MyNumbering = new TEBNumbering();            fCnew++;

  for (Int_t SMEcha = 0; SMEcha < fFileHeader->fMaxCrysInSM; SMEcha++)
    {
      Int_t n_underflow = 0;
      Int_t n_overflow  = 0;
      Int_t nb_null_sample_cases = 0;
      
      for (Int_t i_samp = 0 ; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  Int_t range_null = 0;
	  ((fVal_data)[SMEcha][i_samp]).
	    HistoDistrib(fFileHeader->fNbBinsADC, xmin, xmax,
	   		 &s_histo[0], range_null, n_underflow, n_overflow);

	  if ( n_underflow !=0 ){total_underflow++;}
	  if ( n_overflow  !=0 ){total_overflow++;}
 
	  if (range_null != 0 )
	    {
	      nb_null_sample_cases++;
	      nb_null_sample_all++;
	    }

	  fT2d_xmin[SMEcha][i_samp]= xmin;
	  fT2d_xmax[SMEcha][i_samp]= xmax;
	  
	  for ( Int_t i_bin = 0 ; i_bin < fFileHeader->fNbBinsADC ; i_bin++)
	    {
	      fT3d_his_s[SMEcha][i_samp][i_bin] = s_histo[i_bin];
	    }
	} // end of loop on samples
      
      if ( nb_null_sample_cases != 0 )
	{
	  nb_null_sample_chan++;
	  if(fFlagPrint == fCodePrintAllComments)
	    {
	      Int_t SMTow   = MyNumbering->GetSMTowFromSMEcha(SMEcha);
	      Int_t TowEcha = MyNumbering->GetTowEchaFromSMEcha(SMEcha);

	      cerr << "!TCnaRunEB::MakeHistosOfSampleDistributions()> WARNING/INFO:"
		   << " possibility of empty histo in " << nb_null_sample_cases << " case(s)"
		   << " for SMEcha " << SMEcha << " (tower: " << SMTow
		   << ", TowEcha: " << TowEcha << ")"
		   << fTTBELL << endl;
	    }
	} 
      fTagEvts[SMEcha] = 1;     fFileHeader->fEvtsCalc++;
    } // end of loop on SMEchas
  
  if ( nb_null_sample_all != 0 )
    {	 
      if(fFlagPrint != fCodePrintNoComment)
	{
	  cerr << "!TCnaRunEB::MakeHistosOfSampleDistributions()> WARNING/INFO:"
	       << " possibility of empty histo in " << nb_null_sample_all << " case(s)"
	       << " concerning " << nb_null_sample_chan << " SMEcha(s)" << fTTBELL << endl;
	} 
    }

  if (total_underflow != 0)
    {
      if(fFlagPrint != fCodePrintNoComment)
	{
	  cerr << "!TCnaRunEB::MakeHistosOfSampleDistributions()> WARNING/INFO: "
	       << total_underflow << " underflow(s) have been detected"
	       << fTTBELL << endl;
	}
    }

  if (total_overflow != 0)
    {
      if(fFlagPrint != fCodePrintNoComment)
	{
	  cerr << "!TCnaRunEB::MakeHistosOfSampleDistributions()> WARNING/INFO: "
	       << total_overflow << " overflow(s) have been detected"
	       << fTTBELL << endl;
	}
    }

  delete [] s_histo;        fCdelete++;
  delete MyNumbering;       fCdelete++;      
}

//-----------------------------------------------------------
//
//      Calculation of the covariances between samples
//      for all the SMEchas
//
//-----------------------------------------------------------
void TCnaRunEB::ComputeCovariancesBetweenSamples()
{
  //Calculation of the covariances between samples for all the SMEchas
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCovariancesBetweenSamples()>"
	 << " Calculation of the covariances between samples"
         << " for all the SMEchas" << endl;}
  
  //................... Allocations cov_ss
  if( fT3d_cov_ss == 0 ){
    const Int_t n_samp = fFileHeader->fMaxSampADC;
    const Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    fT3d_cov_ss  = new Double_t**[n_SMEcha];                fCnew++;  
    fT3d2_cov_ss = new  Double_t*[n_SMEcha*n_samp];         fCnew++;  
    fT3d1_cov_ss = new   Double_t[n_SMEcha*n_samp*n_samp];  fCnew++;  
    for(Int_t i = 0 ; i < n_SMEcha ; i++){
      fT3d_cov_ss[i] = &fT3d2_cov_ss[0] + i*n_samp;
      for(Int_t j = 0 ; j < n_samp ; j++){
	fT3d2_cov_ss[n_samp*i+j] = &fT3d1_cov_ss[0]+n_samp*(n_samp*i+j);}}
  }
  
  //.................. Calculation (= init)
  //.................. computation of half of the matrix, diagonal included)
  for (Int_t j_SMEcha = 0 ; j_SMEcha < fFileHeader->fMaxCrysInSM ; j_SMEcha++)
    {
      for (Int_t i_samp = 0 ; i_samp < fFileHeader->fMaxSampADC ; i_samp++)
	{
	  for (Int_t j_samp = 0 ; j_samp <= i_samp; j_samp++)
	    {
	      fT3d_cov_ss[j_SMEcha][i_samp][j_samp] =
		(fVal_data[j_SMEcha][i_samp]).Covariance(fVal_data[j_SMEcha][j_samp]);
	      fT3d_cov_ss[j_SMEcha][j_samp][i_samp] = fT3d_cov_ss[j_SMEcha][i_samp][j_samp];
	    }
	}
      fTagCovCss[j_SMEcha] = 1;     fFileHeader->fCovCssCalc++;
    }
}

//-----------------------------------------------------------
//
//      Calculation of the correlations between samples
//      for all the SMEchas
//
//-----------------------------------------------------------

void TCnaRunEB::ComputeCorrelationsBetweenSamples()
{
//Calculation of the correlations between samples for all the SMEchas

  //... preliminary calculation of the covariances if not done yet.
  //    Test only the first tag since the cov are computed globaly
  //    but set all the tags to 0 because we want not to write
  //    the covariances in the result ROOT file    
  if ( fTagCovCss[0] != 1 ){ComputeCovariancesBetweenSamples();
  for (Int_t j_SMEcha = 0 ; j_SMEcha < fFileHeader->fMaxCrysInSM ; j_SMEcha++)
    {fTagCovCss[j_SMEcha] = 0;}}
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCorrelationsBetweenSamples()>"
	 << " Calculation of the correlations between samples"
	 << " for all the SMEchas" << endl;}

  //................... Allocations cor_ss
  if( fT3d_cor_ss == 0){
    const Int_t n_samp = fFileHeader->fMaxSampADC;
    const Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    fT3d_cor_ss  = new Double_t**[n_SMEcha];                fCnew++;  
    fT3d2_cor_ss = new  Double_t*[n_SMEcha*n_samp];         fCnew++;  
    fT3d1_cor_ss = new   Double_t[n_SMEcha*n_samp*n_samp];  fCnew++;  
    for(Int_t i = 0 ; i < n_SMEcha ; i++){
      fT3d_cor_ss[i] = &fT3d2_cor_ss[0] + i*n_samp;
      for(Int_t j = 0 ; j < n_samp ; j++){
	fT3d2_cor_ss[n_samp*i+j] = &fT3d1_cor_ss[0]+n_samp*(n_samp*i+j);}}
  }
  
  //..................... calculation of the correlations (=init)
  //......................computation of half of the matrix, diagonal included (verif = 1)
    
 for (Int_t j_SMEcha = 0 ; j_SMEcha < fFileHeader->fMaxCrysInSM ; j_SMEcha++)
   {
     for (Int_t i_samp = 0 ; i_samp < fFileHeader->fMaxSampADC ; i_samp++)
       {
	 for (Int_t j_samp = 0 ; j_samp <= i_samp ; j_samp++)
	   {
	     if((fT3d_cov_ss)[j_SMEcha][i_samp][i_samp] > 0
		&& (fT3d_cov_ss)[j_SMEcha][j_samp][j_samp] > 0 )
	       {
		 fT3d_cor_ss[j_SMEcha][i_samp][j_samp] =
		   fT3d_cov_ss[j_SMEcha][i_samp][j_samp]/
		   (fVal_data[j_SMEcha][i_samp].StandardDeviation("corss1") *
		    fVal_data[j_SMEcha][j_samp].StandardDeviation("corss2"));
	       }
	     else
	       {
		 (fT3d_cor_ss)[j_SMEcha][i_samp][j_samp] = (Double_t)0; // prevoir compteur + fTTBELL
	       }
	     fT3d_cor_ss[j_SMEcha][j_samp][i_samp] = fT3d_cor_ss[j_SMEcha][i_samp][j_samp];
	   }
       }
     fTagCorCss[j_SMEcha] = 1;          fFileHeader->fCorCssCalc++;
   }
}

//------------------------------------------------------------------
//
//      Calculation of the covariances between SMEchas
//      for all the samples
//
//------------------------------------------------------------------

void TCnaRunEB::ComputeCovariancesBetweenChannels()
{
//Calculation of the covariances between SMEchas for all the samples

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCovariancesBetweenChannels()>"
	 << " Calculation of the covariances between SMEchas"
         << " for each sample." << endl;}

  //................... Allocations cov_cc
  if( fT3d_cov_cc == 0 ){
    const Int_t n_samp = fFileHeader->fMaxSampADC;
    const Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    fT3d_cov_cc  = new Double_t**[n_samp];                   fCnew++;  
    fT3d2_cov_cc = new  Double_t*[n_samp*n_SMEcha];          fCnew++;  
    fT3d1_cov_cc = new   Double_t[n_samp*n_SMEcha*n_SMEcha]; fCnew++;  
    for(Int_t i = 0 ; i < n_samp ; i++){
      fT3d_cov_cc[i] = &fT3d2_cov_cc[0] + i*n_SMEcha;
      for(Int_t j = 0 ; j < n_SMEcha ; j++){
	fT3d2_cov_cc[n_SMEcha*i+j] = &fT3d1_cov_cc[0]+n_SMEcha*(n_SMEcha*i+j);}}
  }

  //........ Calculation for each sample (=init)
  //...........................computation of half of the matrix, diagonal included

  for (Int_t k_samp = 0 ; k_samp < fFileHeader->fMaxSampADC ; k_samp++)
    {
      for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
	{
	  for (Int_t j_SMEcha = 0 ; j_SMEcha <= i_SMEcha ; j_SMEcha++)
	    {
	      fT3d_cov_cc[k_samp][i_SMEcha][j_SMEcha]
	      	= (fVal_data[i_SMEcha][k_samp]).Covariance(fVal_data[j_SMEcha][k_samp]);
	      fT3d_cov_cc[k_samp][j_SMEcha][i_SMEcha] = fT3d_cov_cc[k_samp][i_SMEcha][j_SMEcha]; 	      
	    }
	}
      fTagCovScc[k_samp] = 1;              fFileHeader->fCovSccCalc++;
    }
}

//------------------------------------------------------------------
//
//      Calculation of the covariances between SMEchas
//      for all the samples, averaged over the samples
//
//------------------------------------------------------------------

void TCnaRunEB::ComputeCovariancesBetweenChannelsMeanOverSamples()
{
//Calculation of the covariances between SMEchas for all the samples
//and averaged over the samples

  //... preliminary calculation of the covariances if not done yet.
  //    Test only the first tag since the cov are computed globaly
  //    but set all the tags to 0 because we want not to write
  //    the covariances in the result ROOT file    
  if ( fTagCovScc[0] != 1 ){ComputeCovariancesBetweenChannels();
  for (Int_t k_samp = 0 ; k_samp < fFileHeader->fMaxSampADC ; k_samp++)
    {fTagCovScc[k_samp] = 0;}}
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCovariancesBetweenChannelsMeanOverSamples()>"
	 << " Calculation of the covariances between SMEchas,"
         << " Calculation of the average over the samples." << endl;}

  //.................. Mean over the samples
  //................. allocation cov_cc_mos + init to zero (mandatory)
  if( fT2d_cov_cc_mos == 0 ){
    const Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    fT2d_cov_cc_mos   = new Double_t*[n_SMEcha];                   fCnew++;
    fT2d1_cov_cc_mos  = new  Double_t[n_SMEcha*n_SMEcha];          fCnew++;
    for(Int_t i_SMEcha = 0 ; i_SMEcha < n_SMEcha ; i_SMEcha++){
	fT2d_cov_cc_mos[i_SMEcha] = &fT2d1_cov_cc_mos[0] + i_SMEcha*n_SMEcha;} 
  }

  for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      for (Int_t j_SMEcha = 0 ; j_SMEcha < fFileHeader->fMaxCrysInSM ; j_SMEcha++)
  	{
	  if( fT2d_cov_cc_mos[i_SMEcha][j_SMEcha] != (Double_t)0 )
	    {fMiscDiag[7]++; fT2d_cov_cc_mos[i_SMEcha][j_SMEcha] = (Double_t)0;}
  	}
    }
  
  //................. Calculation of the mean over the samples  
  for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      for (Int_t j_SMEcha = 0 ; j_SMEcha < fFileHeader->fMaxCrysInSM ; j_SMEcha++)
	{
	  for (Int_t k_samp = 0 ; k_samp < fFileHeader->fMaxSampADC ; k_samp++)
	    {
	      fT2d_cov_cc_mos[i_SMEcha][j_SMEcha] =
		fT2d_cov_cc_mos[i_SMEcha][j_SMEcha] + fT3d_cov_cc[k_samp][i_SMEcha][j_SMEcha];
	    }
	  fT2d_cov_cc_mos[i_SMEcha][j_SMEcha] =
	    fT2d_cov_cc_mos[i_SMEcha][j_SMEcha]/(Double_t)(fFileHeader->fMaxSampADC);
	} 
    }
  fTagCovSccMos[0] = 1;    fFileHeader->fCovSccMosCalc++;
}

//----------------------------------------------------------------------
//
//      Calculation of the correlations between SMEchas
//      for all the samples
//
//----------------------------------------------------------------------
void TCnaRunEB::ComputeCorrelationsBetweenChannels()
{
//Calculation of the correlations between SMEchas for all the samples
  
  //... preliminary calculation of the covariances if not done yet.
  //    Test only the first tag since the cov are computed globaly
  //    but set all the tags to 0 because we want not to write
  //    the covariances in the result ROOT file  
  if ( fTagCovScc[0] != 1 ){ComputeCovariancesBetweenChannels();
  for (Int_t k_samp = 0 ; k_samp < fFileHeader->fMaxSampADC ; k_samp++)
    {fTagCovScc[k_samp] = 0;}}
  
  //............ calculation of the correlations from the covariances
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCorrelationsBetweenChannels()>"
	 << " Calculation of the correlations between SMEchas"
         << " for all the samples." << endl;}
 
  //................... Allocations cor_cc
  if ( fT3d_cor_cc == 0 ){ 
    const Int_t n_samp = fFileHeader->fMaxSampADC;
    const Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    fT3d_cor_cc  = new Double_t**[n_samp];                    fCnew++;  
    fT3d2_cor_cc = new  Double_t*[n_samp*n_SMEcha];           fCnew++;  
    fT3d1_cor_cc = new   Double_t[n_samp*n_SMEcha*n_SMEcha];  fCnew++;  
    for(Int_t i = 0 ; i < n_samp ; i++){
      fT3d_cor_cc[i] = &fT3d2_cor_cc[0] + i*n_SMEcha;
      for(Int_t j = 0 ; j < n_SMEcha ; j++){
	fT3d2_cor_cc[n_SMEcha*i+j] = &fT3d1_cor_cc[0]+n_SMEcha*(n_SMEcha*i+j);}}
  }
  
  //........................... Calculation (=init)
  //........................... computation of half of the matrix, diagonal included (verif=1)
  for (Int_t n_samp = 0 ; n_samp < fFileHeader->fMaxSampADC ; n_samp++)
    {
      for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
	{
	  for (Int_t j_SMEcha = 0 ; j_SMEcha <= i_SMEcha ; j_SMEcha++)
	    {
	      if(  fT3d_cov_cc[n_samp][i_SMEcha][i_SMEcha] > 0
		   && fT3d_cov_cc[n_samp][j_SMEcha][j_SMEcha] > 0 )
		{
		  fT3d_cor_cc[n_samp][i_SMEcha][j_SMEcha] =
		    fT3d_cov_cc[n_samp][i_SMEcha][j_SMEcha]/
		    (  (Double_t)sqrt(fT3d_cov_cc[n_samp][i_SMEcha][i_SMEcha]) *
		       (Double_t)sqrt(fT3d_cov_cc[n_samp][j_SMEcha][j_SMEcha])  );
		}
	      else
		{
		  fT3d_cor_cc[n_samp][i_SMEcha][j_SMEcha] = (Double_t)0.;
		}

	      fT3d_cor_cc[n_samp][j_SMEcha][i_SMEcha] = fT3d_cor_cc[n_samp][i_SMEcha][j_SMEcha];
	    }
	}
      fTagCorScc[n_samp] = 1;    fFileHeader->fCorSccCalc++;
    }
}

//----------------------------------------------------------------------
//
//      Calculation of the correlations between SMEchas
//      for all the samples, averaged over the samples
//
//----------------------------------------------------------------------
void TCnaRunEB::ComputeCorrelationsBetweenChannelsMeanOverSamples()
{
//Calculation of the correlations between SMEchas for all the samples
// averaged over the samples

  //... preliminary calculation of the covariances if not done yet.
  //    Test only the first tag since the cov are computed globaly
  //    but set all the tags to 0 because we want not to write
  //    the covariances in the result ROOT file  
  if ( fTagCorScc[0] != 1 ){ComputeCorrelationsBetweenChannels();
  for (Int_t k_samp = 0 ; k_samp < fFileHeader->fMaxSampADC ; k_samp++)
    {fTagCorScc[k_samp] = 0;}}

  //.................. Mean over the samples  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCorrelationsBetweenChannelsMeanOverSamples()>"
	 << " Calculation of the correlations between SMEchas"
         << " Calculation of the average over the samples." << endl;}

  //................. allocation cor_cc_mos + init to zero (mandatory)
  if( fT2d_cor_cc_mos == 0 ){
    const Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    fT2d_cor_cc_mos   = new Double_t*[n_SMEcha];                   fCnew++;  
    fT2d1_cor_cc_mos  = new  Double_t[n_SMEcha*n_SMEcha];          fCnew++;  
    for(Int_t i_SMEcha = 0 ; i_SMEcha < n_SMEcha ; i_SMEcha++){
	fT2d_cor_cc_mos[i_SMEcha] = &fT2d1_cor_cc_mos[0] + i_SMEcha*n_SMEcha;} 
  }

  for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      for (Int_t j_SMEcha = 0 ; j_SMEcha < fFileHeader->fMaxCrysInSM ; j_SMEcha++)
	{
	  if( fT2d_cor_cc_mos[i_SMEcha][j_SMEcha] != (Double_t)0 )
	    {fMiscDiag[8]++; fT2d_cor_cc_mos[i_SMEcha][j_SMEcha] = (Double_t)0;}
	}
    }
  
  //................. Calculation of the mean over the samples
  
  for (Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      for (Int_t j_SMEcha = 0 ; j_SMEcha < fFileHeader->fMaxCrysInSM ; j_SMEcha++)
	{
	  for (Int_t k_samp = 0 ; k_samp < fFileHeader->fMaxSampADC ; k_samp++)
	    {
	      fT2d_cor_cc_mos[i_SMEcha][j_SMEcha] += fT3d_cor_cc[k_samp][i_SMEcha][j_SMEcha];
	    }
	  fT2d_cor_cc_mos[i_SMEcha][j_SMEcha] =
	    fT2d_cor_cc_mos[i_SMEcha][j_SMEcha]/(Double_t)(fFileHeader->fMaxSampADC);
	} 
    }
  fTagCorSccMos[0] = 1;    fFileHeader->fCorSccMosCalc++;
}

//-----------------------------------------------------------------------------
//      Calculation of the covariances between SMEchas
//      for all the samples, averaged over the samples
//      and calculation of the mean of these covariances
//      (relevant ones) for all the towers
//-----------------------------------------------------------------------------

void  TCnaRunEB::ComputeCovariancesBetweenTowersMeanOverSamplesAndChannels()
{
//Calculation of the covariances between SMEchas
//for all the samples, averaged over the samples and
//calculation of the mean of these covariances for all the towers


  //... preliminary calculation of the covariances averaged over samples (cov_moscc_mot) if not done yet
  //    Only one tag (dim=1) to set to 0 (no write in the result ROOT file)
  if(fTagCovSccMos[0] != 1){ComputeCovariancesBetweenChannelsMeanOverSamples(); fTagCovSccMos[0]=0;}

  //..... mean of the cov_moscc_mot for each pair (tower_X,tower_Y)
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCovariancesBetweenTowersMeanOverSamplesAndChannels()>"
	 << " Calculation of the averaged covariances between towers,"
         << " (average over the samples and over the channels)." << endl;}

  //................. allocation cov_moscc_mot + init to zero (mandatory)
  if( fT2d_cov_moscc_mot == 0 ){
    const Int_t n_tow = fFileHeader->fMaxTowInSM;
    fT2d_cov_moscc_mot  = new Double_t*[n_tow];                 fCnew++;
    fT2d1_cov_moscc_mot = new  Double_t[n_tow*n_tow];           fCnew++;  
    for(Int_t i_tow = 0 ; i_tow < n_tow ; i_tow++){
	fT2d_cov_moscc_mot[i_tow] = &fT2d1_cov_moscc_mot[0] + i_tow*n_tow;} 
  }
  
  for(Int_t i_tow=0; i_tow<fFileHeader->fMaxTowInSM; i_tow++)
    {
      for(Int_t j_tow=0; j_tow<fFileHeader->fMaxTowInSM; j_tow++)
  	{
  	  if( fT2d_cov_moscc_mot[i_tow][j_tow] != (Double_t)0 )
	    {fMiscDiag[9]++; fT2d_cov_moscc_mot[i_tow][j_tow] = (Double_t)0;}
  	}
    }
  
  //..... Calculation of the mean of the averaged cov(c,c) over samples for each pair (tower_X,tower_Y)    
  for(Int_t i_tow=0; i_tow<fFileHeader->fMaxTowInSM; i_tow++)
    {
      for(Int_t j_tow=0; j_tow<fFileHeader->fMaxTowInSM; j_tow++)
	{
	  for(Int_t i_crys=0; i_crys<fFileHeader->fMaxCrysInTow; i_crys++)
	    {
	      Int_t i_SMEcha = i_tow*fFileHeader->fMaxCrysInTow + i_crys;
	      for(Int_t j_crys=0; j_crys<fFileHeader->fMaxCrysInTow; j_crys++)
		{
		  Int_t j_SMEcha = j_tow*fFileHeader->fMaxCrysInTow + j_crys;
		  fT2d_cov_moscc_mot[i_tow][j_tow] += fT2d_cov_cc_mos[i_SMEcha][j_SMEcha];
		}
	    }
	  fT2d_cov_moscc_mot[i_tow][j_tow] = fT2d_cov_moscc_mot[i_tow][j_tow]
	    /((Double_t)(fFileHeader->fMaxCrysInTow*fFileHeader->fMaxCrysInTow));
	}
    }
  fTagCovMosccMot[0] = 1;                   fFileHeader->fCovMosccMotCalc++;
}

//-----------------------------------------------------------------------------
//
//      Calculation of the correlations between SMEchas
//      for all the samples, averaged over the samples
//      and calculation of the mean of these correlations
//      (relevant ones) for all the towers
//-----------------------------------------------------------------------------

void  TCnaRunEB::ComputeCorrelationsBetweenTowersMeanOverSamplesAndChannels()
{
//Calculation of the correlations between SMEchas
//for all the samples, averaged over the samples and
//calculation of the mean of these correlations for all the towers

  //... preliminary calculation of the correlations averaged over samples (cor_moscc_mot) if not done yet
  //    Only one tag (dim=1) to set to 0 (no write in the result ROOT file)

  if(fTagCorSccMos[0] != 1){ComputeCorrelationsBetweenChannelsMeanOverSamples(); fTagCorSccMos[0]= 0;}

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCorrelationsBetweenTowersMeanOverSamplesAndChannels()>"
	 << " Calculation of the averaged correlations between towers,"
         << " (average over the samples and over the channels)." << endl;}

  //................. allocation cor_moscc_mot + init to zero (mandatory)
  if( fT2d_cor_moscc_mot == 0 ){
    const Int_t n_tow = fFileHeader->fMaxTowInSM;
    fT2d_cor_moscc_mot  = new Double_t*[n_tow];                 fCnew++;
    fT2d1_cor_moscc_mot = new  Double_t[n_tow*n_tow];           fCnew++;  
    for(Int_t i_tow = 0 ; i_tow < n_tow ; i_tow++){
      fT2d_cor_moscc_mot[i_tow] = &fT2d1_cor_moscc_mot[0] + i_tow*n_tow;}
  }
  
  for(Int_t i_tow=0; i_tow<fFileHeader->fMaxTowInSM; i_tow++)
    {
      for(Int_t j_tow=0; j_tow<fFileHeader->fMaxTowInSM; j_tow++)
  	{
  	  if( fT2d_cor_moscc_mot[i_tow][j_tow] != (Double_t)0 )
	    {fMiscDiag[10]++; fT2d_cor_moscc_mot[i_tow][j_tow] = (Double_t)0;}
  	}
    }
  
  //..... Calculation of the mean of the averaged over samples cor(c,c) for each pair (tower_X,tower_Y)
  
  for(Int_t i_tow=0; i_tow<fFileHeader->fMaxTowInSM; i_tow++)
    {
      for(Int_t j_tow=0; j_tow<fFileHeader->fMaxTowInSM; j_tow++)
	{
	  //..... Calculation of the average values over the channels for the current (i_tow, j_tow)
	  for(Int_t i_crys=0; i_crys<fFileHeader->fMaxCrysInTow; i_crys++)
	    {
	      Int_t i_SMEcha = i_tow*fFileHeader->fMaxCrysInTow + i_crys;
	      for(Int_t j_crys=0; j_crys<fFileHeader->fMaxCrysInTow; j_crys++)
		{
		  Int_t j_SMEcha = j_tow*fFileHeader->fMaxCrysInTow + j_crys; 
		  fT2d_cor_moscc_mot[i_tow][j_tow] += fT2d_cor_cc_mos[i_SMEcha][j_SMEcha];
		}
	    }

	  fT2d_cor_moscc_mot[i_tow][j_tow] = fT2d_cor_moscc_mot[i_tow][j_tow]
	    /((Double_t)(fFileHeader->fMaxCrysInTow*fFileHeader->fMaxCrysInTow));
	}
    }
  fTagCorMosccMot[0] = 1;    fFileHeader->fCorMosccMotCalc++;
}

//-------------------------------------------------------------------------
//
//      Calculation of the expectation values of the expectation values
//      of the samples for all the SMEchas
//
//-------------------------------------------------------------------------
void  TCnaRunEB::ComputeExpectationValuesOfExpectationValuesOfSamples()
{
//Calculation of the expectation values of the expectation values
// of the samples for all the SMEchas 

  //... preliminary calculation of the expectation values if not done yet
  if ( fTagEv[0] != 1 ){ComputeExpectationValuesOfSamples(); fTagEv[0]=0; }

  //................... Allocation ev_ev + init to zero (mandatory)
  if( fT1d_ev_ev == 0 ){
    fT1d_ev_ev = new Double_t[fFileHeader->fMaxCrysInSM];                   fCnew++;
  }
   for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
     {
       if( fT1d_ev_ev[i_SMEcha] != (Double_t)0 )
	 {fMiscDiag[11]++; fT1d_ev_ev[i_SMEcha] = (Double_t)0;}
     }

  //................... Allocations ch_ev, amplit
    TVectorD ch_ev(fFileHeader->fMaxSampADC);
    TDistrib* amplit   = new TDistrib(fFileHeader->fMaxSampADC, ch_ev);     fCnew++;  
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeExpectationValuesOfExpectationValuesOfSamples()>" << endl
	 << "          Calculation of the expectation values of the"
	 << " expectation values of the samples for all the SMEchas" << endl;}

  //..................... Calculation
  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {     
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  ch_ev(i_samp) = fT2d_ev[i_SMEcha][i_samp];
	}

      Bool_t ok_refill = amplit->Refill(fFileHeader->fMaxSampADC, ch_ev);
      if(ok_refill == kTRUE){fT1d_ev_ev[i_SMEcha] = amplit->ExpectationValue();}
    }
  delete amplit;                        fCdelete++;
  fTagEvEv[0] = 1;                      fFileHeader->fEvEvCalc++;
}

//-------------------------------------------------------------------------
//
//      Calculation of the expectation values of the sigmas
//      of the samples for all the SMEchas
//
//-------------------------------------------------------------------------
void  TCnaRunEB::ComputeExpectationValuesOfSigmasOfSamples()
{
//Calculation of the expectation values of the sigmas
// of the samples for all the SMEchas 

  //... preliminary calculation of the variances if not done yet
  if ( fTagVar[0] != 1 ){ComputeVariancesOfSamples(); fTagVar[0]=0; }
 
  //................... Allocation ev_sig + init to zero (mandatory)
  if( fT1d_ev_sig == 0 ){
    fT1d_ev_sig = new Double_t[fFileHeader->fMaxCrysInSM];              fCnew++;  
  }
   for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
     {
       if( fT1d_ev_sig[i_SMEcha] != (Double_t)0 )
	 {fMiscDiag[12]++; fT1d_ev_sig[i_SMEcha] = (Double_t)0;}
     }

  //................... Allocations ch_sig, amplit
  TVectorD  ch_sig(fFileHeader->fMaxSampADC);  
  TDistrib* amplit   = new TDistrib(fFileHeader->fMaxSampADC, ch_sig);  fCnew++; 
 
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeExpectationValuesOfSigmasOfSamples()>" << endl
	 << "          Calculation of the expectation values of the"
	 << " sigmas of the samples for all the SMEchas" << endl;}

  //..................... Calculation
  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {     
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  if (ch_sig(i_samp) >= (Double_t)0.)
	    {
	      ch_sig(i_samp) = (Double_t)sqrt(fT2d_var[i_SMEcha][i_samp]);
	    }
	  else
	    {
	      ch_sig(i_samp) = (Double_t)(-1.);
	      cout << "!TCnaRunEB::ComputeExpectationValuesOfSigmasOfSamples() *** ERROR ***> " << endl
		   << "          Negative variance! Sigma forced to -1" << fTTBELL << endl;
	    }
	}

      Bool_t ok_refill = amplit->Refill(fFileHeader->fMaxSampADC, ch_sig);
      if(ok_refill == kTRUE){fT1d_ev_sig[i_SMEcha] = amplit->ExpectationValue();}
    }
  delete amplit;                         fCdelete++;
  fTagEvSig[0] = 1;                      fFileHeader->fEvSigCalc++;
}

//-------------------------------------------------------------------------
//
//      Calculation of the expectation values of the (sample,sample)
//      correlations for all the SMEchas
//
//-------------------------------------------------------------------------
void  TCnaRunEB::ComputeExpectationValuesOfCorrelationsBetweenSamples()
{
  //Calculation of the expectation values of the (sample,sample) correlations for all the SMEchas 
  
  //... preliminary calculation of the correlationss if not done yet
  //    (test only the first element since the cor are computed globaly)
  if ( fTagCorCss[0] != 1 ){ComputeCorrelationsBetweenSamples(); fTagCorCss[0]=0;}

  //................... Allocations ev_cor_ss + init to zero (mandatory)
  if( fT1d_ev_cor_ss == 0 ){
    Int_t n_SMEcha =  fFileHeader->fMaxCrysInSM;
    fT1d_ev_cor_ss = new Double_t[n_SMEcha];               fCnew++;  
  }
  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      if( fT1d_ev_cor_ss[i_SMEcha] != (Double_t)0 )
	{fMiscDiag[13]++; fT1d_ev_cor_ss[i_SMEcha] = (Double_t)0;}
    }
  
  //.......... 1D array half_cor_ss[N(N-1)/2] to put the N (sample,sample) correlations
  //           (half of them minus the diagonal) 
  Int_t ndim = (Int_t)(fFileHeader->fMaxSampADC*(fFileHeader->fMaxSampADC - 1)/2);

  TVectorD  half_cor_ss(ndim);

  TDistrib* amplit = new TDistrib(ndim, half_cor_ss);      fCnew++;
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeExpectationValuesOfCorrelationsBetweenSamples()>" << endl
	 << "          Calculation of the expectation values of the"
	 << " (sample,sample) correlations for all the SMEchas" << endl;}

  //..................... Calculation
  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {     
      Int_t i_count = 0;
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  for (Int_t j_samp = 0; j_samp < i_samp; j_samp++)
	    {
	      half_cor_ss(i_count) = fT3d_cor_ss[i_SMEcha][i_samp][j_samp];
	      i_count++;
	    }
	}
      
      Bool_t ok_refill = amplit->Refill(ndim, half_cor_ss);
      if(ok_refill == kTRUE){fT1d_ev_cor_ss[i_SMEcha] = amplit->ExpectationValue();}
    }
  delete amplit;                    fCdelete++;
  fTagEvCorCss[0] = 1;              fFileHeader->fEvCorCssCalc++;
}

//-------------------------------------------------------------------------
//
//      Calculation of the sigmas of the expectation values
//      of the samples for all the SMEchas
//
//-------------------------------------------------------------------------
void  TCnaRunEB::ComputeSigmasOfExpectationValuesOfSamples()
{
//Calculation of the sigmas of the expectation values
// of the samples for all the SMEchas 
  
  //... preliminary calculation of the expectation values if not done yet
  if ( fTagEv[0] != 1 ){ComputeExpectationValuesOfSamples(); fTagEv[0]=0; }

  //................... Allocation sig_ev + init to zero (mandatory)
  if( fT1d_sig_ev == 0 ){
    fT1d_sig_ev = new Double_t[fFileHeader->fMaxCrysInSM];               fCnew++;  
  }

  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      if( fT1d_sig_ev[i_SMEcha] != (Double_t)0 )
	{fMiscDiag[14]++; fT1d_sig_ev[i_SMEcha] = (Double_t)0;}
    }
  
  //................... Allocations ch_ev, amplit
  TVectorD  ch_ev(fFileHeader->fMaxSampADC); 
  TDistrib* amplit   = new TDistrib(fFileHeader->fMaxSampADC, ch_ev);    fCnew++;
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeSigmasOfExpectationValuesOfSamples()>" << endl
	 << "          Calculation of the sigmas of the"
	 << " expectation values of the samples for all the SMEchas" << endl;}

  //..................... Calculation
  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {     
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  ch_ev(i_samp) = fT2d_ev[i_SMEcha][i_samp];
	}

      Bool_t ok_refill = amplit->Refill(fFileHeader->fMaxSampADC, ch_ev);
      if(ok_refill == kTRUE){fT1d_sig_ev[i_SMEcha] = amplit->StandardDeviation();}
    }
  delete amplit;                        fCdelete++;
  fTagSigEv[0] = 1;                     fFileHeader->fSigEvCalc++;
}

//-------------------------------------------------------------------------
//
//      Calculation of the sigmas of the sigmas
//      of the samples for all the SMEchas
//
//-------------------------------------------------------------------------
void  TCnaRunEB::ComputeSigmasOfSigmasOfSamples()
{
//Calculation of the sigmas of the sigmas
// of the samples for all the SMEchas 

  //... preliminary calculation of the variances if not done yet
  if ( fTagVar[0] != 1 ){ComputeVariancesOfSamples(); fTagVar[0]=0; }

  //................... Allocation sig_sig + init to zero (mandatory)
  if( fT1d_sig_sig == 0 ){
    fT1d_sig_sig = new Double_t[fFileHeader->fMaxCrysInSM];                fCnew++;
  } 
  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      if( fT1d_sig_sig[i_SMEcha] != (Double_t)0 )
	{fMiscDiag[15]++; fT1d_sig_sig[i_SMEcha] = (Double_t)0;}
    }
  
  //................... Allocations ch_sig, amplit
  TVectorD  ch_sig(fFileHeader->fMaxSampADC);
  TDistrib* amplit = new TDistrib(fFileHeader->fMaxSampADC, ch_sig);       fCnew++;
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeSigmasOfSigmasOfSamples()>" << endl
	 << "          Calculation of the sigmas of the"
	 << " sigmas of the samples for all the SMEchas" << endl;}

  //..................... Calculation
  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {     
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  if (ch_sig(i_samp) >= (Double_t)0.)
	    {
	      ch_sig(i_samp) = (Double_t)sqrt(fT2d_var[i_SMEcha][i_samp]);
	    }
	  else
	    {
	      ch_sig(i_samp) = (Double_t)(-1.);
	      cout << "!TCnaRunEB::ComputeSigmasOfSigmasOfSamples() *** ERROR ***> " << endl
		   << "          Negative variance! Sigma forced to -1" << fTTBELL << endl;
	    }
	}

      Bool_t ok_refill = amplit->Refill(fFileHeader->fMaxSampADC, ch_sig);
      if(ok_refill == kTRUE){fT1d_sig_sig[i_SMEcha] = amplit->StandardDeviation();}
    }
  delete amplit;                         fCdelete++;
  fTagSigSig[0] = 1;                     fFileHeader->fSigSigCalc++;
}

//-------------------------------------------------------------------------
//
//      Calculation of the sigmas of the (sample,sample) correlations
//      for all the SMEchas
//
//--------------------------------------------------------------------------
void  TCnaRunEB::ComputeSigmasOfCorrelationsBetweenSamples()
{
  //Calculation of the sigmas of the (sample,sample) correlations for all the SMEchas
 
  //... preliminary calculation of the correlationss if not done yet
  //    (test only the first element since the cor are computed globaly)
  if ( fTagCorCss[0] != 1 ){ComputeCorrelationsBetweenSamples(); fTagCorCss[0]=0;}

  //................... Allocations sig_cor_ss + init to zero
  if( fT1d_sig_cor_ss == 0 ){
    Int_t n_SMEcha =  fFileHeader->fMaxCrysInSM;
    fT1d_sig_cor_ss = new Double_t[n_SMEcha];                fCnew++;  
  }
  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      if( fT1d_sig_cor_ss[i_SMEcha] != (Double_t)0 )
	{fMiscDiag[16]++; fT1d_sig_cor_ss[i_SMEcha] = (Double_t)0;}
    }

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeSigmasOfCorrelationsBetweenSamples()>" << endl
	 << "          Calculation of the sigmas of the (sample,sample)"
	 << " correlations for all the cna_SMEchas" << endl;}

  //.......... 1D array half_cor_ss[N(N-1)/2] to put the N (sample,sample) correlations
  //           (half of them minus the diagonal)
  Int_t ndim = (Int_t)(fFileHeader->fMaxSampADC*(fFileHeader->fMaxSampADC - 1)/2);
  //Int_t ndim = (int)(fFileHeader->fMaxSampADC*(fFileHeader->fMaxSampADC - 1.)/2.);

  TVectorD  half_cor_ss(ndim);

  TDistrib* amplit      = new TDistrib(ndim, half_cor_ss);   fCnew++;

  //.................. Calculation
  for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {     
      Int_t i_count = 0;
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  for (Int_t j_samp = 0; j_samp < i_samp; j_samp++)
	    {
	      half_cor_ss(i_count) = fT3d_cor_ss[i_SMEcha][i_samp][j_samp];
	      i_count++;
	    }
	}

      Bool_t ok_refill = amplit->Refill(ndim, half_cor_ss);
      if(ok_refill == kTRUE){fT1d_sig_cor_ss[i_SMEcha] = amplit->StandardDeviation("SigmaCorss");}
    }
  delete amplit;                     fCdelete++;
  fTagSigCorCss[0] = 1;              fFileHeader->fSigCorCssCalc++;
}

//##################################### CORRECTION METHODS ##############################
//---------------------------------------------------------------------------------
//
//     Calculation of the correction coefficients to sample values
//     for all the SMEchas from the (sample,sample) covariances
//
//--------------------------------------------------------------------------------
void  TCnaRunEB::ComputeCorrectionsToSamplesFromCovss(const Int_t& nb_first_samples)
{
// Calulation of the corrections coefficients to the sample values
// for all the SMEchas from cov(s,s)
  
  //................... Allocations correction samples/covss + init to zero (mandatory)
  if( fT2d_sv_correc_covss_s == 0 ){
    Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    Int_t n_samp = fFileHeader->fMaxSampADC;
    fT2d_sv_correc_covss_s = new Double_t*[n_SMEcha];                 fCnew++;
    fT2d1_sv_correc_covss_s = new  Double_t[n_SMEcha*n_samp];         fCnew++;  
    for(Int_t i = 0 ; i < n_SMEcha ; i++){
      fT2d_sv_correc_covss_s[i] = &fT2d1_sv_correc_covss_s[0] + i*n_samp;}
  }

  for(Int_t i_SMEcha = 0; i_SMEcha <fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {  
      for(Int_t i_samp = 0; i_samp <fFileHeader->fMaxSampADC; i_samp++)
 	{
 	  if( fT2d_sv_correc_covss_s[i_SMEcha][i_samp] != (Double_t)0 )
	    {fMiscDiag[17]++; fT2d_sv_correc_covss_s[i_SMEcha][i_samp] = (Double_t)0;}
 	}
    }
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCorrectionsToSamplesFromCovss()>" << endl
	 << "          Calculation of the correction coefficients to sample values"
	 << " for all the SMEchas from the (sample,sample) covariances" << endl;}

  //.. preliminary calculation of the covariances if not done yet
  if ( fTagCovCss[0] != 1 ) { ComputeCovariancesBetweenSamples(); }
  
  //................. Calculation
  //----------------- Loop on the SMEchas 
  for(Int_t i_SMEcha = 0; i_SMEcha <fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      // to be done...     
    }

  fTagSvCorrecCovCss[0] = 1;            fFileHeader->fSvCorrecCovCssCalc++;
}


//-------------------------------------------------------------------------
//
//          Calculation of the corrections factors
//  to the (sample,sample) covariances for all the SMEchas  (OLD? CHECK)
//
//-------------------------------------------------------------------------
void  TCnaRunEB::ComputeCorrectionFactorsToCovss()
{
//Calculation of the corrections factors
//to the (sample,sample) covariances for all the SMEchas
  
  //................... Allocations correction covariances/covss + init to zero
  if (fT3d_cov_correc_covss_s == 0 ){
    Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    Int_t n_samp = fFileHeader->fMaxSampADC;
    fT3d_cov_correc_covss_s  = new Double_t**[n_SMEcha];               fCnew++;
    fT3d2_cov_correc_covss_s = new  Double_t*[n_SMEcha*n_samp];        fCnew++;
    fT3d1_cov_correc_covss_s = new   Double_t[n_SMEcha*n_samp*n_samp]; fCnew++;
    for(Int_t i = 0 ; i < n_SMEcha ; i++){
      fT3d_cov_correc_covss_s[i] = &fT3d2_cov_correc_covss_s[0] + i*n_samp;
      for(Int_t j = 0 ; j < n_samp ; j++){
	fT3d2_cov_correc_covss_s[n_samp*i+j] =
	  &fT3d1_cov_correc_covss_s[0]+n_samp*(n_samp*i+j);}}
  }
  
  for(Int_t i_SMEcha = 0; i_SMEcha <fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {  
      for(Int_t i_samp = 0; i_samp <fFileHeader->fMaxSampADC; i_samp++)
 	{
 	  for(Int_t j_samp = 0; j_samp <fFileHeader->fMaxSampADC; j_samp++)
 	    {
 	      if( fT3d_cov_correc_covss_s[i_SMEcha][i_samp][j_samp] != (Double_t)0 )
		{fMiscDiag[18]++; fT3d_cov_correc_covss_s[i_SMEcha][i_samp][j_samp] = (Double_t)0;}
 	    }
 	}
    }
  
  //.. preliminary calculation of the covariances if not done yet
  //    (test only the first elt since the cov ara computed globaly) 
  
  if ( fTagCovCss[0] != 1 ) { ComputeCovariancesBetweenSamples(); }
 
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCorrectionFactorsToCovss>" << endl
	 << "          Calculation of the correction factors to the"
	 << " (sample,sample) covariances for all the SMEchas" << endl;}

 
  //.. Calculation of the sum (on mp_samp) of the covariances
  //   (sample m_samp , sample mp_samp) as a function of m_samp
  //   (array sum_cov[mp_samp])

  //........ ALLOCATION + init to zero (mandatory)
  Double_t** sum_cov   = new Double_t*[fFileHeader->fMaxCrysInSM];                         fCnew++;
  Double_t*  sum_cov12 = new Double_t[fFileHeader->fMaxCrysInSM*fFileHeader->fMaxSampADC];  fCnew++;
  
  for(Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {sum_cov[i_SMEcha] = &sum_cov12[0] + i_SMEcha*fFileHeader->fMaxSampADC;}
  
  for(Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      for(Int_t i_samp = 0 ; i_samp < fFileHeader->fMaxCrysInSM ; i_samp++)
 	{
 	  if( sum_cov[i_SMEcha][i_samp] != (Double_t)0 )
	    {fMiscDiag[19]++; sum_cov[i_SMEcha][i_samp] = (Double_t)0;}
 	}
    }
  
  //........ CALCULATION
  for (Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      // to be done...
    }

  //..... Calculation of the Double_t sum (on m_samp and mp_samp)
  //      of the covariances (sample m_samp , sample mp_samp)
  //      (number sum_sum_cov)

  //........ ALLOCATION + init to zerop (mandatory)
  Double_t*  sum_sum_cov = new Double_t[fFileHeader->fMaxCrysInSM];     fCnew++;
  for (Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {sum_sum_cov[i_SMEcha] = (Double_t)0;}

  //........ CALCULATION
  for (Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {     
      for (Int_t m_samp = 0; m_samp < fFileHeader->fMaxSampADC; m_samp++)
        {     
          sum_sum_cov[i_SMEcha] = sum_sum_cov[i_SMEcha] + sum_cov[i_SMEcha][m_samp];
        }
    }
  
  //... Calculation of the correction factors to the covariances (f_{jj'})
  //      f_{jj'} = 1 - c_{jj'}  

  //........ ALLOCATION + init to zero (mandatory)
  Double_t*** sum_cov_num1   = new Double_t**[fFileHeader->fMaxCrysInSM];                          fCnew++;
  Double_t**  sum_cov_num12  = new  Double_t*[fFileHeader->fMaxCrysInSM*fFileHeader->fMaxSampADC]; fCnew++;
  Double_t*   sum_cov_num123 =
    new   Double_t[fFileHeader->fMaxCrysInSM*fFileHeader->fMaxSampADC*fFileHeader->fMaxSampADC];   fCnew++;

  for(Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      sum_cov_num1[i_SMEcha] = &sum_cov_num12[0] + i_SMEcha*fFileHeader->fMaxSampADC;
      for (Int_t i_samp = 0 ; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  sum_cov_num12[i_SMEcha*fFileHeader->fMaxSampADC + i_samp] =
	    &sum_cov_num123[0] + (i_SMEcha*fFileHeader->fMaxSampADC + i_samp)*fFileHeader->fMaxSampADC;
	}
    }

  for(Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      for(Int_t i_samp = 0 ; i_samp < fFileHeader->fMaxSampADC ; i_samp++)
 	{
 	  for(Int_t j_samp = 0 ; j_samp < fFileHeader->fMaxSampADC ; j_samp++)
 	    {
 	      if( sum_cov_num1[i_SMEcha][i_samp][j_samp] != (Double_t)0 )
		{fMiscDiag[20]++; sum_cov_num1[i_SMEcha][i_samp][j_samp] = (Double_t)0;}
 	    }
 	}
    }
  
  //........ CALCULATION
  for(Int_t i_SMEcha = 0 ; i_SMEcha < fFileHeader->fMaxCrysInSM ; i_SMEcha++)
    {
      // to be done...
      fTagCovCorrecCovCss[i_SMEcha] = 1;           fFileHeader->fCovCorrecCovCssCalc++;
    }

  delete [] sum_cov;                                fCdelete++;
  delete [] sum_cov12;                              fCdelete++;
  delete [] sum_sum_cov;                            fCdelete++;
  delete [] sum_cov_num1;                           fCdelete++;
  delete [] sum_cov_num12;                          fCdelete++;
  delete [] sum_cov_num123;                         fCdelete++;
}
//-----------------------------------------------------------------------------
//
//             Calculation of the corrections factors
//     to the (sample,sample) correlations for all the SMEchas (OLD? CHECK)
//
//-----------------------------------------------------------------------------
void  TCnaRunEB::ComputeCorrectionFactorsToCorss()
{
// Calculation of the corrections factors to the (sample,sample)
// correlations for all the SMEchas
  
  //................... Allocations correction correlations/covss + init to zero (mandatory)
  if( fT3d_cor_correc_covss_s == 0 ){
    Int_t n_SMEcha = fFileHeader->fMaxCrysInSM;
    Int_t n_samp = fFileHeader->fMaxSampADC;
    fT3d_cor_correc_covss_s  = new Double_t**[n_SMEcha];               fCnew++;
    fT3d2_cor_correc_covss_s = new  Double_t*[n_SMEcha*n_samp];        fCnew++;
    fT3d1_cor_correc_covss_s = new   Double_t[n_SMEcha*n_samp*n_samp]; fCnew++;
    for(Int_t i = 0 ; i < n_SMEcha ; i++){
      fT3d_cor_correc_covss_s[i] = &fT3d2_cor_correc_covss_s[0] + i*n_samp;
      for(Int_t j = 0 ; j < n_samp ; j++){
	fT3d2_cor_correc_covss_s[n_samp*i+j] =
	  &fT3d1_cor_correc_covss_s[0]+n_samp*(n_samp*i+j);}}   
  }
 
  for(Int_t i_SMEcha = 0; i_SMEcha <fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {  
      for(Int_t i_samp = 0; i_samp <fFileHeader->fMaxSampADC; i_samp++)
 	{
 	  for(Int_t j_samp = 0; j_samp <fFileHeader->fMaxSampADC; j_samp++)
 	    {
 	      if( fT3d_cor_correc_covss_s[i_SMEcha][i_samp][j_samp] != (Double_t)0 )
		{fMiscDiag[21]++; fT3d_cor_correc_covss_s[i_SMEcha][i_samp][j_samp] = (Double_t)0;}
	    }
	}
    }
 
  //.. preliminary calculation of the covariances if not done yet
  //    (test only the first elt since the cov ara computed globaly)
  
  if ( fTagCovCss[0] != 1 ) { ComputeCovariancesBetweenSamples(); }
   
  //.. Calculation of the correction factor to the covariances (f_{jj'})
  //   if not done yet
  
  if ( fTagCovCorrecCovCss[0] != 1 ) ComputeCorrectionFactorsToCovss();

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB::ComputeCorrectionFactorsToCorss>" << endl
	 << "          Calculation of the correction factors to the"
	 << " (sample,sample) correlations for all the SMEchas" << endl;}

  //...... Calculation of the correction factor to the correlations (g_{jj'})
  //       g_{jj'} = f_{jj'}/ ( sqrt(f_{jj} sqrt(f_{j'j'}) )

  for ( Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      // to be done...
      fTagCorCorrecCovCss[i_SMEcha] = 1;         fFileHeader->fCorCorrecCovCssCalc++;
    }
}
//########################## (END OF CORRECTION METHODS) ####################################

//=========================================================================
//
//                  W R I T I N G     M E T H O D S
//
//=========================================================================

//=======================================================================
//
//        M E T H O D S    T O    G E T    T H E   P A T H S
//
//        O F    T H E    R E S U L T S    F I L E S
//
//=======================================================================
void TCnaRunEB::GetPathForResultsRootFiles()
{
  GetPathForResultsRootFiles("");
}

void TCnaRunEB::GetPathForResultsRootFiles(const TString argFileName)
{
  // Init fCfgResultsRootFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "cna_results_root.cfg" and file located in $HOME user's directory (default)

  Int_t MaxCar = fgMaxCar;
  fCfgResultsRootFilePath.Resize(MaxCar);
  fCfgResultsRootFilePath            = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForResultsRootFilePath

  if ( argFileName == "" )
    {
      string cFileNameForCnaPaths = "cna_results_root.cfg";     // config file name
      TString s_file_name = cFileNameForCnaPaths.c_str();
      Text_t *t_file_name = (Text_t *)s_file_name.Data();
      
      TString s_path_name = gSystem->Getenv("HOME");       // get user's home directory path
      
      fFileForResultsRootFilePath = s_path_name;
      fFileForResultsRootFilePath.Append('/');
      fFileForResultsRootFilePath.Append(t_file_name);
    }
  else
    {
      fFileForResultsRootFilePath = argFileName.Data();
    }

  //........ Reading of the paths in the file named fFileForResultsRootFilePath.Data()
  //

  fFcin_rr.open(fFileForResultsRootFilePath.Data());
  if(fFcin_rr.fail() == kFALSE)
    {
      fFcin_rr.clear();
      string xResultsFileP;
      fFcin_rr >> xResultsFileP;
      fCfgResultsRootFilePath = xResultsFileP.c_str();

      fCnaCommand++;
      cout << "   *CNA [" << fCnaCommand << "]> Automatic registration of cna paths -> " << endl
	   << "    Results .root files: " << fCfgResultsRootFilePath.Data() << endl;
      fFcin_rr.close();

      //.............. Making of the results .root file name (TCnaRunEB specific)
      fMakeResultsFileName(fCodeRoot);
    }
  else
    {
      fFcin_rr.clear();
      fCnaError++;
      cout << fTTBELL << endl
	   << " ***************************************************************************** " << endl;
      cout << "   !CNA(TCnaRunEB) (" << fCnaError << ") *** ERROR *** " << endl << endl
	   << "    "
	   << fFileForResultsRootFilePath.Data() << ": file not found. " << endl
	   << "    "
	   << endl << endl
	   << "    "
           << " The file " << fFileForResultsRootFilePath.Data()
	   << " is a configuration file for the CNA and"
	   << " must contain one line with the following syntax:" << endl << endl
	   << "    "
	   << "   path of the results .root files ($HOME/etc...) " << endl
	   << "    "
	   << "          (without slash at the end of line)" << endl
	   << endl << endl
	   << "    "
	   << " EXAMPLE:" << endl << endl
	   << "    "
	   << "  $HOME/scratch0/cna/results_root_files" << endl << endl
	   << " ***************************************************************************** "
	   << fTTBELL << endl;

      fFcin_rr.close();
    }
}

//================================================================================================

void TCnaRunEB::GetPathForResultsAsciiFiles()
{
  GetPathForResultsAsciiFiles("");
}

void TCnaRunEB::GetPathForResultsAsciiFiles(const TString argFileName)
{
  // Init fCfgResultsAsciiFilePath and get it from the file named argFileName
  // argFileName = complete name of the file (/afs/cern.ch/...)
  // if string is empty, file name = "cna_results_ascii.cfg"
  // and file located in $HOME user's directory (default)

  Int_t MaxCar = fgMaxCar;
  fCfgResultsAsciiFilePath.Resize(MaxCar);
  fCfgResultsAsciiFilePath = "?";

  //..... put the name of the ascii file (containing the paths for CNA treatements)
  //      in the string cFileNameForCnaPaths and in class attribute fFileForResultsAsciiFilePath

  if ( argFileName == "" )
    {
      string cFileNameForCnaPaths = "cna_results_ascii.cfg";     // config file name
      TString s_file_name = cFileNameForCnaPaths.c_str();
      Text_t *t_file_name = (Text_t *)s_file_name.Data();
      
      TString s_path_name = gSystem->Getenv("HOME");       // get user's home directory path
      
      fFileForResultsAsciiFilePath = s_path_name;
      fFileForResultsAsciiFilePath.Append('/');
      fFileForResultsAsciiFilePath.Append(t_file_name);
    }
  else
    {
      fFileForResultsAsciiFilePath = argFileName.Data();
    }

  //........ Reading of the paths in the file named fFileForResultsAsciiFilePath.Data()
  //

  fFcin_ra.open(fFileForResultsAsciiFilePath.Data());
  if(fFcin_ra.fail() == kFALSE)
    {
      fFcin_ra.clear();
      string xResultsFileP;
      fFcin_ra >> xResultsFileP;
      fCfgResultsAsciiFilePath = xResultsFileP.c_str();

      fCnaCommand++;
      cout << "   *CNA [" << fCnaCommand << "]> Automatic registration of cna paths -> " << endl
	   << "    Results .ascii files: " << fCfgResultsAsciiFilePath.Data() << endl;
      fFcin_ra.close();
      
      //.........................Making of the results .ascii file name (TCnaRunEB specific)
      fMakeResultsFileName(fCodeHeaderAscii);
    }
  else
    {
      fFcin_ra.clear();
      fCnaError++;
      cout << fTTBELL << endl
	   << " ***************************************************************************** " << endl;
      cout << "   !CNA(TCnaRunEB) (" << fCnaError << ") *** ERROR *** " << endl << endl
	   << "    "
	   << fFileForResultsAsciiFilePath.Data() << ": file not found. " << endl
	   << "    "
	   << endl << endl
	   << "    "
           << " The file " << fFileForResultsAsciiFilePath.Data()
	   << " is a configuration file for the CNA and"
	   << " must contain one line with the following syntax:" << endl << endl
	   << "    "
	   << "   path of the results .ascii files ($HOME/etc...) " << endl
	   << "    "
	   << "          (without slash at the end of line)" << endl
	   << endl << endl
	   << "    "
	   << " EXAMPLE:" << endl << endl
	   << "    "
	   << "  $HOME/scratch0/cna/results_ascii_files" << endl << endl
	   << " ***************************************************************************** "
	   << fTTBELL << endl;

      fFcin_ra.close();
    }
}

//=========================================================================
//
//         W R I T I N G   M E T H O D S :    R O O T    F I L E S
//
//=========================================================================
//-------------------------------------------------------------
//
//                      OpenRootFile
//
//-------------------------------------------------------------
Bool_t TCnaRunEB::OpenRootFile(Text_t *name, TString status) {
//Open the Root file

  TString s_path;
  s_path = fCfgResultsRootFilePath;
  s_path.Append('/');
  s_path.Append(name);
  
  gCnaRootFile   = new TCnaRootFile(s_path.Data(), status);     fCnew++;

  Bool_t ok_open = kFALSE;

  if ( gCnaRootFile->fRootFileStatus == "RECREATE" )
    {
      ok_open = gCnaRootFile->OpenW();
    }
  if ( gCnaRootFile->fRootFileStatus == "READ"     )
    {
      ok_open = gCnaRootFile->OpenR();
    }

  if (!ok_open) // unable to open file
    {
      cout << "TCnaRunEB::OpenRootFile> Cannot open file " << s_path.Data() << endl;

    }
  else
    {
      if(fFlagPrint == fCodePrintAllComments){
	cout << "*TCnaRunEB::OpenRootFile> Open ROOT file OK for file " << s_path.Data() << endl;}
      
      fOpenRootFile  = kTRUE;
    }
  
  return ok_open;
}

//-------------------------------------------------------------
//
//                      CloseRootFile
//
//-------------------------------------------------------------
Bool_t TCnaRunEB::CloseRootFile(Text_t *name) {
//Close the Root file
 
  Bool_t ok_close = kFALSE;

  if (fOpenRootFile == kTRUE ) 
    {
      gCnaRootFile->CloseFile();

      if(fFlagPrint == fCodePrintAllComments){
	cout << "*TCnaRunEB::CloseRootFile> Close ROOT file OK " << endl;}

      delete gCnaRootFile;                                     fCdelete++;
      fOpenRootFile = kFALSE;
      ok_close      = kTRUE;
    }
  else
    {
      cout << "*TCnaRunEB::CloseRootFile(...)> no close since no file is open"
	   << fTTBELL << endl;
    }

  return ok_close;
}
//-------------------------------------------------------------
//
//            WriteRootFile without arguments.
//            Call WriteRootFile WITH argument (file name)
//            after an automatic generation of the file name.
//
//            Codification for the file name:
//
//                  AAA_RRR_FFF_LLL_SSS.root
//
//            AAA: Analysis name
//            RRR: Run number
//            FFF: First analysed event number
//            LLL: Last  analysed event number
//            SSS: SuperModule number
//
//-------------------------------------------------------------
Bool_t TCnaRunEB::WriteRootFile() {
//Write the Root file. File name automatically generated in fMakeResultsFileName.

  Bool_t ok_write = kFALSE;
  fMakeResultsFileName(fCodeRoot);  // set fRootFileName, fRootFileNameShort,
                                    // fAsciiFileName, fAsciiFileNameShort,
                                    // fResultsFileName and fResultsFileNameShort

  Text_t *s_name = (Text_t *)fRootFileNameShort.Data();

  if(fFlagPrint != fCodePrintNoComment){
    cout << "*TCnaRunEB::WriteRootFile()> Results are going to be written in the ROOT file: " << endl
	 << "                           " <<  fRootFileName.Data() << endl;}

  ok_write = WriteRootFile(s_name);
  
  return ok_write;
}
//-------------------------------------------------------------
//
//               WriteRootFile with argument
//
//-------------------------------------------------------------
Bool_t TCnaRunEB::WriteRootFile(Text_t *name) {
//Write the Root file

  Text_t *file_name = name;

  Bool_t ok_open  = kFALSE;
  Bool_t ok_write = kFALSE;

  if ( fOpenRootFile )
    {
      cout << "!TCnaRunEB::WriteRootFile(...) *** ERROR ***> Writing on file already open."
	   << fTTBELL << endl;
    }
  else
    {
      //..... List of the different element types and associated parameters as ordered in the ROOT file
      //
      //   Nb of   Type of element          Type      Type                                      Size    Comment
      // elements                           Number    Name
      //
      //        1  fMatHis(1,tower)         ( 0)  cTypTowerNumbers        1*(   1,  68) =         68

      //        1  fMatHis(1,SMEcha)        (16)  cTypEvEv                1*(   1,1700) =      1 700
      //        1  fMatHis(1,SMEcha)        (17)  cTypEvSig               1*(   1,1700) =      1 700
      //        1  fMatHis(1,SMEcha)        (10)  cTypEvCorCss            1*(   1,1700) =      1 700

      //        1  fMatHis(1,SMEcha)        (18)  cTypSigEv               1*(   1,1700) =      1 700
      //        1  fMatHis(1,SMEcha)        (19)  cTypSigSig              1*(   1,1700) =      1 700
      //        1  fMatHis(1,SMEcha)        (11)  cTypSigCorCss           1*(   1,1700) =      1 700

      //        1  fMatMat(tower,tower)     (23)  cTypCovMosccMot         1*(  68,  68) =      4 624
      //        1  fMatMat(tower,tower)     (24)  cTypCorMosccMot         1*(  68,  68) =      4 624

      //        1  fMatHis(SMEcha, sample)  (15)  cTypLastEvtNumber       1*(1700,  10) =     17 000
      //        1  fMatHis(SMEcha, sample)  ( 1)  cTypEv                  1*(1700,  10) =     17 000
      //        1  fMatHis(SMEcha, sample)  ( 2)  cTypVar                 1*(1700,  10) =     17 000

      //        1  fMatHis(SMEcha, sample)  (12)  cTypSvCorrecCovCss      1*(1700,  10) =     17 000

      //   SMEcha  fMatMat(sample, sample)  ( 8)  cTypCovCss           1700*(  10,  10) =    170 000
      //   SMEcha  fMatMat(sample, sample   ( 9)  cTypCorCss           1700*(  10,  10) =    170 000

      //   SMEcha  fMatHis(sample, bin_adc) ( 3)  cTypEvts,            1700*(  10, 100) =  1 700 000
      //        1  fMatHis(SMEcha, sample)  ( 4)  cTypEvtsXmin            1*(1700,  10) =     17 000
      //        1  fMatHis(SMEcha, sample   ( 5)  cTypEvtsXmax            1*(1700,  10) =     17 000

      //   SMEcha  fMatHis(sample, bin_evt) (20)  cTypSampTime,        1700*(  10, 150) =  2 550 000

      //        1  fMatMat(SMEcha, SMEcha)  (21)  cTypCovSccMos           1*(1700,1700) =  2 890 000
      //        1  fMatMat(SMEcha, SMEcha)  (22)  cTypCorSccMos           1*(1700,1700) =  2 890 000

      //   SMEcha  fMatMat(sample, sample)  (13)  cTypCovCorrecCovCss  1700*(  10,  10) =    170 000
      //   SMEcha  fMatMat(sample, sample)  (14)  cTypCorCorrecCovCss  1700*(  10,  10) =    170 000

      //......................................................................................................

      ok_open = OpenRootFile(file_name, "RECREATE");

      TString typ_name = "?";
      Int_t v_nb_times = 0;
      Int_t v_dim_one  = 0;
      Int_t v_dim_two  = 0;
      Int_t v_size     = 0;
      Int_t v_tot      = 0;
      Int_t v_tot_writ = 0;

      //-------------------------- Tower numbers 
      //       1   fMatHis(1,tower)           ( 0)  cTypTowerNumbers        1*(   1,  68) =         68

      Int_t MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "TowerNumbers";
      v_nb_times = fFileHeader->fTowerNumbersCalc;
      v_dim_one  = 1;
      v_dim_two  = fFileHeader->fMaxTowInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagTowerNumbers[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypTowerNumbers;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(1,fFileHeader->fMaxTowInSM);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootTowerNumbers();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;

      //-------------------------- Expectation values of the expectation values the samples
      //       1   fMatHis(1,SMEcha)         (16)  cTypEvEv                1*(   1,1700) =      1 700

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "EvEv";
      v_nb_times = fFileHeader->fEvEvCalc;
      v_dim_one  = 1;
      v_dim_two  = fFileHeader->fMaxCrysInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagEvEv[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypEvEv;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(1,fFileHeader->fMaxCrysInSM);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootEvEv();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;
                  
      //-------------------------- Expectation values of the sigmas the samples
      //       1   fMatHis(1,SMEcha)         (17)  cTypEvSig               1*(   1,1700) =      1 700

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "EvSig";
      v_nb_times = fFileHeader->fEvSigCalc;
      v_dim_one  = 1;
      v_dim_two  = fFileHeader->fMaxCrysInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagEvSig[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypEvSig;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(1,fFileHeader->fMaxCrysInSM);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootEvSig();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	} 
      cout << endl;
                                     
      //-------------------------- Expectation values of the correlations between the samples
      //       1   fMatHis(1,SMEcha)         (10)  cTypEvCorCss            1*(   1,1700) =      1 700

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "EvCorCss";
      v_nb_times = fFileHeader->fEvCorCssCalc;
      v_dim_one  = 1;
      v_dim_two  = fFileHeader->fMaxCrysInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagEvCorCss[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypEvCorCss;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(1,fFileHeader->fMaxCrysInSM);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootEvCorCss();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	} 
      cout << endl;
      
      //-------------------------- Sigmas of the expectation values of the samples  
      //       1   fMatHis(1,SMEcha)         (18)  cTypSigEv               1*(   1,1700) =      1 700

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "SigEv";
      v_nb_times = fFileHeader->fSigEvCalc;
      v_dim_one  = 1;
      v_dim_two  = fFileHeader->fMaxCrysInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagSigEv[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypSigEv;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(1,fFileHeader->fMaxCrysInSM);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootSigEv();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	} 
      cout << endl;
      
      //-------------------------- Sigmas of the sigmas of the samples  
      //       1   fMatHis(1,SMEcha)         (19)  cTypSigSig              1*(   1,1700) =      1 700

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "SigSig";
      v_nb_times = fFileHeader->fSigSigCalc;
      v_dim_one  = 1;
      v_dim_two  = fFileHeader->fMaxCrysInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}
 
      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagSigSig[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypSigSig;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(1,fFileHeader->fMaxCrysInSM);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootSigSig();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl; 
                    
      //-------------------------- Sigmas of the correlations between the samples  
      //       1   fMatHis(1,SMEcha)         (11)  cTypSigCorCss           1*(   1,1700) =      1 700

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "SigCorCss";
      v_nb_times = fFileHeader->fSigCorCssCalc;
      v_dim_one  = 1;
      v_dim_two  = fFileHeader->fMaxCrysInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}
   
      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagSigCorCss[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypSigCorCss;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(1,fFileHeader->fMaxCrysInSM);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootSigCorCss();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	} 
      cout << endl;
            
      //----- Mean Covariances between SMEchas (averaged over samples) for all (tower_X,tower_Y)
      //       1   fMatMat(tower,tower)       (23)  cTypCovMosccMot         1*(  68,  68) =      4 624

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "CovMosccMot";
      v_nb_times = fFileHeader->fCovMosccMotCalc;
      v_dim_one  = fFileHeader->fMaxTowInSM;
      v_dim_two  = fFileHeader->fMaxTowInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagCovMosccMot[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCovMosccMot;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeMat(fFileHeader->fMaxTowInSM,fFileHeader->fMaxTowInSM);
	      gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1,1);
	      TRootCovMosccMot();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	} 
      cout << endl;
            
      //----- Mean Correlations between SMEchas (averaged over samples) for all (tower_X,tower_Y)
      //       1   fMatMat(tower,tower)       (24)  cTypCorMosccMot         1*(  68,  68) =      4 624

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "CorMosccMot";
      v_nb_times = fFileHeader->fCorMosccMotCalc;
      v_dim_one  = fFileHeader->fMaxTowInSM;
      v_dim_two  = fFileHeader->fMaxTowInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagCorMosccMot[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCorMosccMot;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeMat(fFileHeader->fMaxTowInSM,fFileHeader->fMaxTowInSM);
	      gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1,1);
	      TRootCorMosccMot();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	} 
      cout << endl;
      
      //-------------------------- Numbers of found events (LastEvtNumber)
      //       1   fMatHis(SMEcha, sample)   (15)  cTypLastEvtNumber       1*(1700,  10) =     17 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "LastEvtNumber";
      v_nb_times = fFileHeader->fLastEvtNumberCalc;
      v_dim_one  = fFileHeader->fMaxCrysInSM;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagLastEvtNumber[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypLastEvtNumber;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(fFileHeader->fMaxCrysInSM,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootLastEvtNumber();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;
                  
      //-------------------------- Expectation values of the samples
      //       1   fMatHis(SMEcha, sample)   ( 1)  cTypEv                  1*(1700,  10) =     17 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "Ev";
      v_nb_times = fFileHeader->fEvCalc;
      v_dim_one  = fFileHeader->fMaxCrysInSM;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagEv[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypEv;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(fFileHeader->fMaxCrysInSM,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootEv();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;
      
      //-------------------------- Variances of the samples     
      //       1   fMatHis(SMEcha, sample)   ( 2)  cTypVar                 1*(1700,  10) =     17 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "Var";
      v_nb_times = fFileHeader->fVarCalc;
      v_dim_one  = fFileHeader->fMaxCrysInSM;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagVar[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypVar;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(fFileHeader->fMaxCrysInSM,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootVar();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;

      //-------------------------- Corrections to the sample values from cov(s,s)
      //       1   fMatHis(SMEcha, sample)   (12)  cTypSvCorrecCovCss      1*(1700,  10) =     17 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "SvCorrecCovCss";
      v_nb_times = fFileHeader->fSvCorrecCovCssCalc;
      v_dim_one  = fFileHeader->fMaxCrysInSM;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagSvCorrecCovCss[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypSvCorrecCovCss;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(fFileHeader->fMaxCrysInSM,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootSvCorrecCovCss();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	} 
      cout << endl;
      
      //-------------------------- Covariances between samples

      // SMEcha   fMatMat(sample,  sample)   ( 8)  cTypCovCss           1700*(  10,  10) =    170 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "CovCss";
      v_nb_times = fFileHeader->fCovCssCalc;
      v_dim_one  = fFileHeader->fMaxSampADC;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i_SMEcha = 0; i_SMEcha < v_nb_times; i_SMEcha++)
	{
	  if ( fTagCovCss[i_SMEcha] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCovCss;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i_SMEcha;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeMat(fFileHeader->fMaxSampADC,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1,1);
	      TRootCovCss(i_SMEcha);
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i_SMEcha == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;
      
      //-------------------------- Correlations between samples   
      // SMEcha   fMatMat(sample,  sample)   ( 9)  cTypCorCss           1700*(  10,  10) =    170 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "CorCss";
      v_nb_times = fFileHeader->fCorCssCalc;
      v_dim_one  = fFileHeader->fMaxSampADC;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}
 
      for (Int_t i_SMEcha = 0; i_SMEcha < v_nb_times; i_SMEcha++)
	{
	  if ( fTagCorCss[i_SMEcha] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCorCss;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i_SMEcha;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeMat(fFileHeader->fMaxSampADC,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1,1);
	      TRootCorCss(i_SMEcha);
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i_SMEcha == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }	  
	}
      cout << endl;

      //-------------------------- Event distributions
      // SMEcha   fMatHis(sample,  bins)     ( 3)  cTypEvts,            1700*(  10, 100) =  1 700 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "Evts";
      v_nb_times = fFileHeader->fEvtsCalc;
      v_dim_one  = fFileHeader->fMaxSampADC;
      v_dim_two  = fFileHeader->fNbBinsADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i_SMEcha = 0; i_SMEcha < v_nb_times; i_SMEcha++)
	{
	  if ( fTagEvts[i_SMEcha] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypEvts;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i_SMEcha;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(fFileHeader->fMaxSampADC,fFileHeader->fNbBinsADC);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootEvts(i_SMEcha);
	      gCnaRootFile->fCnaResultsTree->Fill(); 
	      if( i_SMEcha == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;

      //-------------------------- Event distributions Xmin
      //       1   fMatHis(cna_SMEcha, sample)   ( 4)  cTypEvtsXmin            1*(1700,  10) =     17 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "EvtsXmin";
      v_nb_times = fFileHeader->fEvtsCalc;
      v_dim_one  = 1;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}
      
      for (Int_t i_SMEcha = 0; i_SMEcha < v_nb_times; i_SMEcha++)
      	{
      	  if ( fTagEvts[i_SMEcha] == 1 )
      	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypEvtsXmin;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i_SMEcha;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(1,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootEvtsXmin(i_SMEcha);
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i_SMEcha == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
      	    }
      	}
      cout << endl;
      
      //-------------------------- Event distributions Xmax
      //       1   fMatHis(SMEcha, sample)   ( 5)  cTypEvtsXmax            1*(1700,  10) =     17 000

      MaxCar = fgMaxCar;      
      typ_name.Resize(MaxCar);
      typ_name   = "EvtsXmax";
      v_nb_times = fFileHeader->fEvtsCalc;
      v_dim_one  = 1;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}
      
      for (Int_t i_SMEcha = 0; i_SMEcha < v_nb_times; i_SMEcha++)
     	{
      	  if ( fTagEvts[i_SMEcha] == 1 )
      	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypEvtsXmax;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i_SMEcha;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(1,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootEvtsXmax(i_SMEcha);
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i_SMEcha == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
      	    }
      	}
      cout << endl;

      //-------------------------- Samples as a function of event
      // SMEcha   fMatHis(sample,  bins)     (20)  cTypSampTime,        1700*(  10, 200) =  3 400 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "SampTime";
      v_nb_times = fFileHeader->fSampTimeCalc;
      v_dim_one  = fFileHeader->fMaxSampADC;
      v_dim_two  = fFileHeader->fNbOfTakenEvts;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i_SMEcha = 0; i_SMEcha < v_nb_times; i_SMEcha++)
	{
	  if ( fTagSampTime[i_SMEcha] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypSampTime;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i_SMEcha;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeHis(fFileHeader->fMaxSampADC,fFileHeader->fNbOfTakenEvts);
	      gCnaRootFile->fCnaIndivResult->fMatMat.ReSet(1,1);
	      TRootSampTime(i_SMEcha);
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i_SMEcha == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;

      //-------------------------- Covariances between SMEchas (Mean Over Samples)
      //  sample   fMatMat(SMEcha, SMEcha)  (21)  cTypCovSccMos           1*(1700,1700) =  2 890 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "CovSccMos";
      v_nb_times = fFileHeader->fCovSccMosCalc;
      v_dim_one  = fFileHeader->fMaxCrysInSM;
      v_dim_two  = fFileHeader->fMaxCrysInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagCovSccMos[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCovSccMos;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeMat(fFileHeader->fMaxCrysInSM,fFileHeader->fMaxCrysInSM);
	      gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1,1);
	      TRootCovSccMos();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	} 
      cout << endl;
      
      //-------------------------- Correlations between SMEchas (Mean Over Samples)
      //  sample   fMatMat(SMEcha, SMEcha)  (22)  cTypCorSccMos           1*(1700,1700) =  2 890 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "CorSccMos";
      v_nb_times = fFileHeader->fCorSccMosCalc;
      v_dim_one  = fFileHeader->fMaxCrysInSM;
      v_dim_two  = fFileHeader->fMaxCrysInSM;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;

      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i = 0; i < v_nb_times; i++)
	{
	  if ( fTagCorSccMos[0] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCorSccMos;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeMat(fFileHeader->fMaxCrysInSM,fFileHeader->fMaxCrysInSM);
	      gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1,1);
	      TRootCorSccMos();
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;

      //-------------------------- Correction factors to the covariances from cov(s,s)
      // SMEcha   fMatMat(sample,  sample)   (13)  cTypCovCorrecCovCss  1700*(  10,  10) =    170 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "CovCorrecCovCss";
      v_nb_times = fFileHeader->fCovCorrecCovCssCalc;
      v_dim_one  = fFileHeader->fMaxSampADC;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i_SMEcha = 0; i_SMEcha < v_nb_times; i_SMEcha++)
	{
	  if ( fTagCovCorrecCovCss[i_SMEcha] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCovCorrecCovCss;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i_SMEcha;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeMat(fFileHeader->fMaxSampADC,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1,1);
	      TRootCovCorrecCovCss(i_SMEcha);
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i_SMEcha == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	} 
      cout << endl;
      
      //-------------------------- Correction factors to the correlations from cov(s,s)
      // SMEcha   fMatMat(sample,  sample)   (14)  cTypCorCorrecCovCss  1700*(  10,  10) =    170 000

      MaxCar = fgMaxCar;
      typ_name.Resize(MaxCar);
      typ_name   = "CorCorrecCovCss";
      v_nb_times = fFileHeader->fCorCorrecCovCssCalc;
      v_dim_one  = fFileHeader->fMaxSampADC;
      v_dim_two  = fFileHeader->fMaxSampADC;
      v_size     = v_nb_times*v_dim_one*v_dim_two;
      v_tot     += v_size;
 
      if(fFlagPrint != fCodePrintNoComment){
      cout << "*TCnaRunEB::WriteRootFile()> " << setw(18) << typ_name << ": " << setw(4) << v_nb_times
	   << " * ("  << setw(4) << v_dim_one << ","  << setw(4) << v_dim_two  << ") = "
	   << setw(9) << v_size;}

      for (Int_t i_SMEcha = 0; i_SMEcha < v_nb_times; i_SMEcha++)
	{
	  if ( fTagCorCorrecCovCss[i_SMEcha] == 1 )
	    {
	      gCnaRootFile->fCnaIndivResult->fTypOfCnaResult = cTypCorCorrecCovCss;
	      gCnaRootFile->fCnaIndivResult->fIthElement     = i_SMEcha;
	      gCnaRootFile->fCnaIndivResult->
		SetSizeMat(fFileHeader->fMaxSampADC,fFileHeader->fMaxSampADC);
	      gCnaRootFile->fCnaIndivResult->fMatHis.ReSet(1,1);
	      TRootCorCorrecCovCss(i_SMEcha);
	      gCnaRootFile->fCnaResultsTree->Fill();
	      if( i_SMEcha == 0 ){cout << " => WRITTEN ON FILE "; v_tot_writ += v_size;}
	    }
	}
      cout << endl;

      //---------------------------------------------- WRITING 
      //...................................... file 
      gCnaRootFile->fRootFile->Write();
      //...................................... header
      fFileHeader->Write();

      //...................................... status message
      if(fFlagPrint != fCodePrintNoComment){
	cout << "*TCnaRunEB::WriteRootFile()> " << setw(20) << "TOTAL: "
	     << setw(21) << "CALCULATED = " << setw(9) <<  v_tot
	     << " => WRITTEN ON FILE = "    << setw(9) << v_tot_writ << endl;}

      if(fFlagPrint != fCodePrintNoComment){
	cout << "*TCnaRunEB::WriteRootFile()> Write OK in file " << file_name << " in directory:" << endl
	     << "                           " << fCfgResultsRootFilePath
	     << endl;}

      ok_write = kTRUE;

      //...................................... close
      CloseRootFile(file_name);
    }
  return ok_write;
}

//======================== "PREPA FILL" METHODS ===========================

//-------------------------------------------------------------------------
//
//  Prepa Fill Tower numbers as a function of the tower index 
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootTowerNumbers()
{
  if (fTagTowerNumbers[0] == 1 )
    {
      for (Int_t j_tow = 0; j_tow < fFileHeader->fMaxTowInSM; j_tow++)
	{
	  gCnaRootFile->fCnaIndivResult->fMatHis(0, j_tow) =
	    fT1d_SMtowFromIndex[j_tow];
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill last evt numbers for all the (SMEcha,sample)
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootLastEvtNumber()
{
  if (fTagLastEvtNumber[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatHis( j_SMEcha, i_samp) =
	      	fT2d_LastEvtNumber[j_SMEcha][i_samp] + 1;
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill histogram of samples as a function of event
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootSampTime(const Int_t& user_SMEcha)
{
  if (fTagSampTime[user_SMEcha] == 1 )
    {
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  //...................... all the bins set to zero
	  for (Int_t j_bin = 0; j_bin < fFileHeader->fNbOfTakenEvts; j_bin++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatHis(i_samp, j_bin) = (Double_t)0.;
	    }
	  //...................... fill the non-zero bins 
	  for (Int_t j_bin = 0; j_bin < fFileHeader->fNbOfTakenEvts; j_bin++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatHis(i_samp, j_bin) =
		fT3d_distribs[user_SMEcha][i_samp][j_bin];  
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill expectation values of the samples for all the SMEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootEv()
{
  if (fTagEv[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatHis( j_SMEcha, i_samp) =
	      	fT2d_ev[j_SMEcha][i_samp];
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill variances of the samples for all the SMEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootVar()
{
  if (fTagVar[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatHis( j_SMEcha, i_samp) =
	      	fT2d_var[j_SMEcha][i_samp];
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill mean covariances between SMEchas, mean over samples
//  for all (tower_X, tower_Y)
//                           (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootCovMosccMot()
{
  if (fTagCovMosccMot[0] == 1 )
    {
      for (Int_t i_tow = 0; i_tow < fFileHeader->fMaxTowInSM; i_tow++)
	{
	  for (Int_t j_tow = 0; j_tow < fFileHeader->fMaxTowInSM; j_tow++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatMat(i_tow, j_tow) =
	      	fT2d_cov_moscc_mot[i_tow][j_tow];
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill mean correlations between SMEchas, mean over samples
//  for all (tower_X, tower_Y)
//                           (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootCorMosccMot()
{
  if (fTagCorMosccMot[0] == 1 )
    {
      for (Int_t i_tow = 0; i_tow < fFileHeader->fMaxTowInSM; i_tow++)
	{
	  for (Int_t j_tow = 0; j_tow < fFileHeader->fMaxTowInSM; j_tow++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatMat(i_tow, j_tow) =
	      	fT2d_cor_moscc_mot[i_tow][j_tow];
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill ADC distributions of the samples for all the SMEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootEvts(const Int_t& user_SMEcha)
{
  if (fTagEvts[user_SMEcha] == 1 )
    {
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  for (Int_t j_bin = 0; j_bin < fFileHeader->fNbBinsADC; j_bin++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatHis(i_samp, j_bin) =
	      	fT3d_his_s[user_SMEcha][i_samp][j_bin];
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill ADC distributions xmin of the samples for all the SMEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootEvtsXmin(const Int_t& user_SMEcha)
{
  if (fTagEvts[user_SMEcha] == 1 )
    {
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  gCnaRootFile->fCnaIndivResult->fMatHis(0, i_samp) =
	    fT2d_xmin[user_SMEcha][i_samp];  
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill ADC distributions xmax of the samples for all the SMEchas
//                       (for writing in the ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootEvtsXmax(const Int_t& user_SMEcha)
{
  if (fTagEvts[user_SMEcha] == 1 )
    {
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  gCnaRootFile->fCnaIndivResult->fMatHis(0, i_samp) =
	    fT2d_xmax[user_SMEcha][i_samp];  
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill covariances between SMEchas, mean over samples
//                           (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootCovSccMos()
{
  if (fTagCovSccMos[0] == 1 )
    {
      for (Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
	{
	  for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatMat(i_SMEcha, j_SMEcha) =
	      	fT2d_cov_cc_mos[i_SMEcha][j_SMEcha];
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill correlations between SMEchas, mean over samples
//                         (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootCorSccMos()
{
  if (fTagCorSccMos[0] == 1 )
    {
      for (Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
	{
	  for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatMat(i_SMEcha, j_SMEcha) =
	      	fT2d_cor_cc_mos[i_SMEcha][j_SMEcha];
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill covariances between samples for a given SMEcha
//                      (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootCovCss(const Int_t& user_SMEcha)
{
   if (fTagCovCss[user_SMEcha] == 1 )
     {
       for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	 {
	   for (Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++)
	     {
	       gCnaRootFile->fCnaIndivResult->fMatMat(i_samp, j_samp) =
	       	 fT3d_cov_ss[user_SMEcha][i_samp][j_samp];
	     }
	 }
     }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill correlations between samples for a given SMEcha
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootCorCss(const Int_t& user_SMEcha)
{
   if (fTagCorCss[user_SMEcha] == 1 )
     {
       for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	 {
	   for (Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++)
	     {
	       gCnaRootFile->fCnaIndivResult->fMatMat(i_samp, j_samp) =
	       	 fT3d_cor_ss[user_SMEcha][i_samp][j_samp];
	     }
	 }
     }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill expectation values of the expectation values of the samples
//  for all the SMEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootEvEv()
{
  if (fTagEvEv[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  gCnaRootFile->fCnaIndivResult->fMatHis(0, j_SMEcha) =
	    fT1d_ev_ev[j_SMEcha];
	}      
    }
}
//-------------------------------------------------------------------------
//
//  Prepa Fill expectation values of the sigmas of the samples
//  for all the SMEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootEvSig()
{
  if (fTagEvSig[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  gCnaRootFile->fCnaIndivResult->fMatHis(0, j_SMEcha) =
	    fT1d_ev_sig[j_SMEcha];
	}      
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill expectation values of the (sample,sample) correlations
//  for all the SMEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootEvCorCss()
{
  if (fTagEvCorCss[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  gCnaRootFile->fCnaIndivResult->fMatHis(0, j_SMEcha) =
	    fT1d_ev_cor_ss[j_SMEcha];
	}      
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill sigmas of the expectation values of the samples
//  for all the SMEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootSigEv()
{
  if (fTagSigEv[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  gCnaRootFile->fCnaIndivResult->fMatHis(0, j_SMEcha) =
	    fT1d_sig_ev[j_SMEcha];
	}      
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill sigmas of the expectation values of the sigmas
//  for all the SMEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootSigSig()
{
  if (fTagSigSig[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  gCnaRootFile->fCnaIndivResult->fMatHis(0, j_SMEcha) =
	    fT1d_sig_sig[j_SMEcha];
	}      
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill sigmas of the (sample,sample) correlations
//  for all the SMEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootSigCorCss()
{
  if (fTagSigCorCss[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  gCnaRootFile->fCnaIndivResult->fMatHis(0, j_SMEcha) =
	    fT1d_sig_cor_ss[j_SMEcha];
	}      
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill sample values correction coefficients
//  for all the SMEchas
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootSvCorrecCovCss()
{
  if (fTagSvCorrecCovCss[0] == 1 )
    {
      for (Int_t j_SMEcha = 0; j_SMEcha < fFileHeader->fMaxCrysInSM; j_SMEcha++)
	{
	  for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatHis(j_SMEcha, i_samp) =
	      	fT2d_sv_correc_covss_s[j_SMEcha][i_samp];
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill 
//  
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootCovCorrecCovCss(const Int_t& user_SMEcha)
{
  if (fTagCovCorrecCovCss[0] == 1 )   // test 1st elt only since global calc
    {
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  for (Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatMat(i_samp, j_samp) =
		fT3d_cov_correc_covss_s[user_SMEcha][i_samp][j_samp];
	    }
	}
    }
}

//-------------------------------------------------------------------------
//
//  Prepa Fill 
//  
//                        (for writing in ROOT file)
//
//-------------------------------------------------------------------------
void TCnaRunEB::TRootCorCorrecCovCss(const Int_t& user_SMEcha)
{
  if (fTagCorCorrecCovCss[0] == 1 )   // test 1st elt only since global calc
    {
      for (Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	{
	  for (Int_t j_samp = 0; j_samp < fFileHeader->fMaxSampADC; j_samp++)
	    {
	      gCnaRootFile->fCnaIndivResult->fMatMat(i_samp, j_samp) =
		fT3d_cor_correc_covss_s[user_SMEcha][i_samp][j_samp];
	    }
	}
    }
}

//=========================================================================
//
//    W R I T I N G   M E T H O D S :    A S C I I    F I L E S   
//
//=========================================================================

//------------------------------------------------------------
//
//      Writing of the expectation values in an ASCII file
//
//------------------------------------------------------------

void TCnaRunEB::WriteAsciiExpectationValuesOfSamples()
{
//Writing of the expectation values in an ASCII file

  if ( fTagEv[0] == 1)
    {
      Int_t i_code = fCodeEv;
      fMakeResultsFileName(i_code);
      fAsciiFileWriteHeader(i_code);
      
      Int_t i_lic1 = fNbChanByLine; 
      Int_t i_lic2 = fNbSampByLine;

      fT1dWriteAscii(i_code, i_lic1, i_lic2);
    }
  else
    {
      cout << "!TCnaRunEB::WriteAsciiExpectationValuesOfSamples()> "
	   << " Quantities not calculated. No reason to write an ASCII file."
	   << fTTBELL << endl;
    }
}

//-------------------------------------------------------
//
//      Writing of the variances in an ASCII file
//
//-------------------------------------------------------

void TCnaRunEB::WriteAsciiVariancesOfSamples()
{
//Writing of the variances in an ASCII file

  if ( fTagVar[0] == 1)
    { 
      Int_t i_code = fCodeVar;  // code for variance   
      fMakeResultsFileName(i_code);
      fAsciiFileWriteHeader(i_code);
      
      Int_t i_lic1 = fNbChanByLine;
      Int_t i_lic2 = fNbSampByLine;
      
      fT1dWriteAscii(i_code, i_lic1, i_lic2);
    }
  else
    {
      cout << "!TCnaRunEB::WriteAsciiVariancesOfSamples()> "
	   << " Quantities not calculated. No reason to write an ASCII file."
	   << fTTBELL << endl;
    }
}

//--------------------------------------------------------------------------------
//
//      Writing of the covariances between samples
//      for a given SMEcha in an ASCII file
//
//--------------------------------------------------------------------------------
void   TCnaRunEB::WriteAsciiCovariancesBetweenSamples(const Int_t& user_SMEcha)
{
//Writing of the covariances between samples for a given SMEcha in an ASCII file

  if (fTagCovCss[user_SMEcha] != 1 ) {ComputeCovariancesBetweenSamples();}
  fUserSMEcha = user_SMEcha;
  Int_t i_code = fCodeCovCss;  // code for covariances between samples
  fMakeResultsFileName(i_code);  
  fAsciiFileWriteHeader(i_code);

  Int_t i_pasx = fSectSampSizeX;
  Int_t i_pasy = fSectSampSizeY;

  fT2dWriteAscii(i_code, i_pasx, i_pasy);
}

//---------------------------------------------------------------------------------
//
//   Writing of the correlations between samples
//   for a given SMEcha in an ASCII file
//
//---------------------------------------------------------------------------------
void      TCnaRunEB::WriteAsciiCorrelationsBetweenSamples(const Int_t& user_SMEcha)
{
//Writing of the correlations between samples for a given SMEcha in an ASCII file

  if (fTagCorCss[user_SMEcha] != 1 ) {ComputeCorrelationsBetweenSamples();}

  fUserSMEcha = user_SMEcha;
  Int_t i_code = fCodeCorCss; // code for correlations between samples
  fMakeResultsFileName(i_code); 
  fAsciiFileWriteHeader(i_code);

  Int_t i_pasx = fSectSampSizeX;
  Int_t i_pasy = fSectSampSizeY;
 
  fT2dWriteAscii(i_code, i_pasx, i_pasy);
}

//---------------------------------------------------------------------------
//
//   Writing of the expectation values of the correlations between samples
//   for all the SMEchas in an ASCII file
//
//---------------------------------------------------------------------------

void     TCnaRunEB::WriteAsciiExpectationValuesOfCorrelationsBetweenSamples()
{
//Write the expectation values of the correlations between samples for all the SMEchas in an ASCII file

  if ( fTagEvCorCss[0] == 1)
    {
      Int_t i_code = fCodeEvCorCss;  // code for expectation values of ss correlations   
      fMakeResultsFileName(i_code);
      fAsciiFileWriteHeader(i_code);
      
      Int_t i_lic1 = fNbChanByLine;
      Int_t i_lic2 = 0;
      
      fT1dWriteAscii(i_code, i_lic1, i_lic2);
    }
  else
    {
      cout << "!TCnaRunEB::WriteAsciiExpectationValuesOfCorrelationsBetweenSamples()> "
	   << " Quantities not calculated. No reason to write an ASCII file."
	   << fTTBELL << endl;
    }
}
//---------------------------------------------------------------------
//
//   Writing of the sigmas of the correlations between samples
//   for all the SMEchas in an ASCII file
//
//---------------------------------------------------------------------

void     TCnaRunEB::WriteAsciiSigmasOfCorrelationsBetweenSamples()
{
//Writing of the amplitudes of the correlations between samples for all the SMEchas in an ASCII file

  if ( fTagSigCorCss[0] == 1)
    {
      Int_t i_code = fCodeSigCorCss;  // code for amplitudes   
      fMakeResultsFileName(i_code);
      fAsciiFileWriteHeader(i_code);
      
      Int_t i_lic1 = fNbChanByLine;
      Int_t i_lic2 = 0;
      
      fT1dWriteAscii(i_code, i_lic1, i_lic2);
    }
  else
    {
      cout << "!TCnaRunEB::WriteAsciiSigmasOfCorrelationsBetweenSamples()> "
	   << " Quantities not calculated. No reason to write an ASCII file."
	   << fTTBELL << endl;
    }
}

//------------------------------------------------------------
//
//      Writing of the sample value correction coefficients
//      in an ASCII file
//
//------------------------------------------------------------

void  TCnaRunEB::WriteAsciiSvCorrecCovCss()
{
//Writing of the sample value correction coefficients in an ASCII file

  if ( fTagSvCorrecCovCss[0] == 1)
    {
      Int_t i_code = fCodeSvCorrecCovCss;
      fMakeResultsFileName(i_code);
      fAsciiFileWriteHeader(i_code);
      
      Int_t i_lic1 = fNbChanByLine;    
      Int_t i_lic2 = fNbSampByLine;
      
      fT1dWriteAscii(i_code, i_lic1, i_lic2);
    }
  else
    {
      cout << "!TCnaRunEB::WriteAsciiSvCorrecCovCss()> "
	   << " Quantities not calculated. No reason to write an ASCII file."
	   << fTTBELL << endl;
    }
}

//------------------------------------------------------------
//
//      Writing of the  correction factors to the
//      (sample,sample) covariances in an ASCII file
//
//------------------------------------------------------------

void  TCnaRunEB::WriteAsciiCovCorrecCovCss(const Int_t& user_SMEcha)
{
//Writing of the correction factors to the (sample,sample) covariances in ASCII file

  if ( fTagCovCorrecCovCss[0] == 1)
    {
      Int_t i_code = fCodeCovCorrecCovCss; // code for correction factors to ss covariances
      fUserSMEcha = user_SMEcha;
      fMakeResultsFileName(i_code); 
      fAsciiFileWriteHeader(i_code);
      
      Int_t i_pasx = fSectSampSizeX;
      Int_t i_pasy = fSectSampSizeY;
      
      fT2dWriteAscii(i_code, i_pasx, i_pasy);
    }
  else
    {
      cout << "!TCnaRunEB::WriteAsciiCovCorrecCovCss()> "
	   << " Quantities not calculated. No reason to write an ASCII file."
	   << fTTBELL << endl;
    }
}

//------------------------------------------------------------
//
//      Writing of the  correction factors to the
//      (sample,sample) correlations in an ASCII file
//
//------------------------------------------------------------

void  TCnaRunEB::WriteAsciiCorCorrecCovCss(const Int_t& user_SMEcha)
{
//Writing of the correction factors to the (sample,sample) correlations in ASCII file

  if ( fTagCorCorrecCovCss[0] == 1)
    {
      Int_t i_code = fCodeCorCorrecCovCss; // code for correction factors to ss correlations
      fUserSMEcha = user_SMEcha;
      fMakeResultsFileName(i_code);
      fAsciiFileWriteHeader(i_code);
      
      Int_t i_pasx = fSectSampSizeX;
      Int_t i_pasy = fSectSampSizeY;
      
      fT2dWriteAscii(i_code, i_pasx, i_pasy);
    }
  else
    {
      cout << "!TCnaRunEB::WriteAsciiCorCorrecCovCss()> "
	   << " Quantities not calculated. No reason to write an ASCII file."
	   << fTTBELL << endl;
    }
}

//------------------------------------------------------------
//
//             WriteAsciiCnaChannelTable()   
//
//------------------------------------------------------------
void  TCnaRunEB::WriteAsciiCnaChannelTable()
{
//Write the correspondance table: CNA-channel <-> (SMTow,TowEcha), SMcrystal

  TEBNumbering* MyNumbering = new TEBNumbering();           fCnew++;
  // BuildCrysTable() is called in the method Init() which is called by the constructor

  if(fFlagPrint != fCodePrintNoComment){
    cout << "*TCnaRunEB::WriteAsciiCnaChannelTable()> The correspondance table is made."
	 << " Number of CNA-channels = " << fFileHeader->fMaxCrysInSM << endl;}

  Int_t i_code = fCodeCorresp; // code for cna correspondance table
  fMakeResultsFileName(i_code);
  fAsciiFileWriteHeader(i_code);

  for (Int_t i_SMEcha=0; i_SMEcha<fFileHeader->fMaxCrysInSM; i_SMEcha++)
    {
      Int_t sm_tower = MyNumbering->GetSMTowFromSMEcha(i_SMEcha);
      Int_t TowEcha  = MyNumbering->GetTowEchaFromSMEcha(i_SMEcha);
      if ( TowEcha == 0 )
	{ fFcout_f << endl;
	  fFcout_f << "  CNA     tower#   channel#   crystal#    Number of events"  << endl;
	  fFcout_f << "channel    in SM   in tower     in SM     found  (required)" << endl << endl;	  
	}
      Int_t SMcrys      = MyNumbering->GetSMCrysFromSMTowAndTowEcha(sm_tower, TowEcha);
      Int_t sample_z    = 0;
      Int_t nb_of_evts  = PickupNumberOfEvents(i_SMEcha, sample_z);
      fFcout_f  << setw(7)  << i_SMEcha
		<< setw(9)  << sm_tower
		<< setw(11) << TowEcha
		<< setw(10) << SMcrys
		<< setw(10) << nb_of_evts 
		<< setw(4) << "(" << setw(6) << fFileHeader->fNbOfTakenEvts << ")" << endl;
    }
  fFcout_f.close();

  if(fFlagPrint != fCodePrintNoComment){
    cout << "*TCnaRunEB::WriteAsciiCnaChannelTable()> The CNA correspondance table "
	 << "has been written in file: " << fAsciiFileName << endl;}

  delete MyNumbering;                       fCdelete++;
}

//=========================================================================
//
//         METHODS TO SET FLAGD TO PRINT (OR NOT) COMMENTS (DEBUG)
//
//=========================================================================

void  TCnaRunEB::PrintComments()
{
// Set flags to authorize printing of some comments concerning initialisations (default)

  fFlagPrint = fCodePrintComments;
  cout << "*TCnaRunEB::PrintComments()> Warnings and some comments on init will be printed" << endl;
}

void  TCnaRunEB::PrintWarnings()
{
// Set flags to authorize printing of warnings

  fFlagPrint = fCodePrintWarnings;
  cout << "*TCnaRunEB::PrintWarnings()> Warnings will be printed" << endl;
}

void  TCnaRunEB::PrintAllComments()
{
// Set flags to authorize printing of the comments of all the methods

  fFlagPrint = fCodePrintAllComments;
  cout << "*TCnaRunEB::PrintAllComments()> All the comments will be printed" << endl;
}

void  TCnaRunEB::PrintNoComment()
{
// Set flags to forbid the printing of all the comments

  fFlagPrint = fCodePrintNoComment;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
//                 +--------------------------------------+
//                 |    P R I V A T E     M E T H O D S   |
//                 +--------------------------------------+
//
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


//=========================================================================
//
//                 Results Filename Making  (private)
//
//=========================================================================

void  TCnaRunEB::fMakeResultsFileName(const Int_t&  i_code)
{
//Results filename making  (private)
  
  //----------------------------------------------------------------------
  //
  //     making of the name of the results file.
  //
  //     Put the names in the following class attributes:
  //
  //     fRootFileName,  fRootFileNameShort,
  //     fAsciiFileName, fAsciiFileNameShort
  //
  //     (Short means: without the directory path)
  //
  //     set indications (run number, type of quantity, ...)
  //     and add the extension ".ascii" or ".root"
  //     
  //     ROOT:  only one ROOT file:  i_code = fCodeRoot.
  //                                          All the types of quantities
  //
  //     ASCII: several ASCII files: i_code = code for one type of quantity
  //            each i_code which is not equal to fCodeRoot is also implicitly
  //            a code "fCodeAscii" (this last attribute is not in the class)
  //     
  //----------------------------------------------------------------------
  
  char* f_in       = new char[fDim_name];                 fCnew++;
  char* f_in_short = new char[fDim_name];                 fCnew++;
  
  //  switch (i_code){  
  
  //===================================  R O O T  =====================================
  if (i_code == fCodeRoot)
    {
      if(fCfgResultsRootFilePath == "?")
	{
	  cout << "#TCnaRunEB::fMakeResultsFileName>  * * * W A R N I N G * * * " << endl << endl
	       << "    Path for results .root file not defined. Default option will be used here:" << endl
	       << "    your results files will be writen in your HOME directory." << endl << endl
	       << "    In order to write the results file in the file FILE_NAME," << endl
               << "    you have to call the method GetPathForResultsRootFiles(TString FILE_NAME)" << endl 
	       << "    where FILE_NAME is the complete name (/afs/etc...) of a configuration file" << endl 
	       << "    which must have only one line containing the path of the .root result files." << endl
	       << "    If the TString FILE_NAME is empty, the configuration file must be " << endl
	       << "    in your HOME directory and must be called: cna_results_root.cfg" << endl
	       << endl;

	  TString home_path = gSystem->Getenv("HOME");
	  fCfgResultsRootFilePath = home_path;      
	}

      if(fCfgResultsRootFilePath.BeginsWith("$HOME"))
	{
	  fCfgResultsRootFilePath.Remove(0,5);
	  Text_t *t_file_nohome = (Text_t *)fCfgResultsRootFilePath.Data(); //  /scratch0/cna/...
	  
	  TString home_path = gSystem->Getenv("HOME");
	  fCfgResultsRootFilePath = home_path;             //  /afs/cern.ch/u/USER
	  fCfgResultsRootFilePath.Append(t_file_nohome);   //  /afs/cern.ch/u/USER/scratch0/cna/...
	}

      sprintf(f_in, "%s/%s_%d_%d_%d_SM%d",
	      fCfgResultsRootFilePath.Data(), fFileHeader->fTypAna.Data(),  fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }

  //===================================  A S C I I  ===================================  

  if (i_code == fCodeHeaderAscii)
    {
      if(fCfgResultsAsciiFilePath == "?")
	{
	  cout << "#TCnaRunEB::fMakeResultsFileName>  * * * W A R N I N G * * * " << endl << endl
	       << "    Path for results .ascii file not defined. Default option will be used here:" << endl
	       << "    your results files will be writen in your HOME directory." << endl << endl
	       << "    In order to write the results file in the file FILE_NAME," << endl
               << "    you have to call the method GetPathForResultsAsciiFiles(TString FILE_NAME)" << endl 
	       << "    where FILE_NAME is the complete name (/afs/etc...) of a configuration file" << endl 
	       << "    which must have only one line containing the path of the .ascii result files." << endl
	       << "    If the TString FILE_NAME is empty, the configuration file must be " << endl
	       << "    in your HOME directory and must be called: cna_results_ascii.cfg" << endl
	       << endl;

	  TString home_path = gSystem->Getenv("HOME");
	  fCfgResultsAsciiFilePath = home_path;  
	} 
      
      if(fCfgResultsAsciiFilePath.BeginsWith("$HOME"))
	{
	  fCfgResultsAsciiFilePath.Remove(0,5);
	  Text_t *t_file_nohome = (Text_t *)fCfgResultsAsciiFilePath.Data(); //  /scratch0/cna/...
	  
	  TString home_path = gSystem->Getenv("HOME");
	  fCfgResultsAsciiFilePath = home_path;             //  /afs/cern.ch/u/USER
	  fCfgResultsAsciiFilePath.Append(t_file_nohome);   //  /afs/cern.ch/u/USER/scratch0/cna/...
	}
     
      sprintf(f_in, "%s/%s_%d_header_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_header_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }

  //------------------------------------------------------------------------------------------------------
  if (i_code == fCodeCorresp)
    {
      sprintf(f_in, "%s/%s_%d_%d_%d_SM%d_cna",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_%d_%d_SM%d_cna",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }

  if (i_code == fCodeEv)
    {
      sprintf(f_in, "%s/%s_%d_ev_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_ev_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeVar)
    {
      sprintf(f_in, "%s/%s_%d_var_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_var_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if( i_code == fCodeEvts)
    {
      sprintf(f_in, "%s/%s_%d_evts_s_c%d_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber, fUserSMEcha,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_evts_s_c%d_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber, fUserSMEcha,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if(  i_code == fCodeCovSccMos)
    {
      sprintf(f_in, "%s/%s_%d_cov_cc_mos_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_cov_cc_mos_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if ( i_code == fCodeCorSccMos)
    {
      sprintf(f_in, "%s/%s_%d_cor_cc_mos_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_cor_cc_mos_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code ==  fCodeCovCss)
    {
      sprintf(f_in, "%s/%s_%d_cov_ss_c%d_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber, fUserSMEcha,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_cov_ss_c%d_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber, fUserSMEcha,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeCorCss)
    {
      sprintf(f_in, "%s/%s_%d_cor_ss_c%d_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fUserSMEcha, fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_cor_ss_c%d_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fUserSMEcha, fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeEvCorCss)
    {
      sprintf(f_in, "%s/%s_%d_ev_cor_ss_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_ev_cor_ss_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeSigCorCss)
    {
      sprintf(f_in, "%s/%s_%d_sig_cor_ss_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_sig_cor_ss_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }

  if (i_code == fCodeSvCorrecCovCss)
    {
      sprintf(f_in, "%s/%s_%d_sv_correc_covss_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_sv_correc_covss_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }
  
  if (i_code == fCodeCovCorrecCovCss)
    {
      sprintf(f_in, "%s/%s_%d_cov_correc_covss_c%d_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fUserSMEcha, fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_cov_correc_covss_c%d_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fUserSMEcha, fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    }

  if (i_code == fCodeCorCorrecCovCss)
    {
      sprintf(f_in, "%s/%s_%d_cor_correc_covss_c%d_%d_%d_SM%d",
	      fCfgResultsAsciiFilePath.Data(), fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fUserSMEcha, fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
      sprintf(f_in_short, "%s_%d_cor_correc_covss_c%d_%d_%d_SM%d",
	      fFileHeader->fTypAna.Data(), fFileHeader->fRunNumber,
	      fUserSMEcha, fFileHeader->fFirstEvt, fFileHeader->fNbOfTakenEvts, fFileHeader->fSuperModule);
    } 

  // default:
  //    cout << "*TCnaRunEB::fMakeResultsFileName(const Int_t&  i_code)> "
  //	 << "wrong header code , i_code = " << i_code << endl; 
  //  }

  //======================================= f_name
  
  char* f_name = new char[fDim_name];                   fCnew++;
  
  for (Int_t i = 0 ; i < fDim_name ; i++)
    {
      f_name[i] = '\0';
    }
  
  Int_t ii = 0;
  for (Int_t i = 0 ; i < fDim_name ; i++)
    {
      if ( f_in[i] != '\0' ){f_name[i] = f_in[i]; ii++;}
      else {break;}  // va directement a f_name[ii] = '.';
    }
 
  if ( ii+5 < fDim_name )
    {
      //.......... writing of the file extension (.root or .ascii)
      
      //------------------------------------------- extension .ascii
      if ( i_code != fCodeRoot  || i_code == fCodeCorresp )
	{
	  f_name[ii] = '.';   f_name[ii+1] = 'a';
	  f_name[ii+2] = 's'; f_name[ii+3] = 'c';
	  f_name[ii+4] = 'i'; f_name[ii+5] = 'i';
	  
	  fAsciiFileName = f_name;
	}
      //------------------------------------------- extension .root
      if ( i_code == fCodeRoot )
	{
	  f_name[ii] = '.';   f_name[ii+1] = 'r';
	  f_name[ii+2] = 'o'; f_name[ii+3] = 'o';  f_name[ii+4] = 't';
	  
	  fRootFileName = f_name;
	}
    }
  else
    {
      cout << "*TCnaRunEB::fMakeResultsFileName(...)> Name too long (for f_name)."
	   << " No room enough for the extension. (ii = " << ii << ")"
	   << fTTBELL << endl; 
    }


  //====================================== f_name_short
  
  char* f_name_short = new char[fDim_name];          fCnew++;

  for (Int_t i = 0 ; i < fDim_name ; i++)
    {
      f_name_short[i] = '\0';
    }
  
  ii = 0;
  for (Int_t i = 0 ; i < fDim_name ; i++)
    {
      if ( f_in_short[i] != '\0' ){f_name_short[i] = f_in_short[i]; ii++;}
      else {break;}  // va directement a f_name_short[ii] = '.';
    }
 
  if ( ii+5 < fDim_name )
    {
      //.......... writing of the file extension (.root or .ascii)
      
      //-------------------------------------------extension .ascii
      if ( i_code != fCodeRoot || i_code == fCodeCorresp )
	{
	  f_name_short[ii] = '.';   f_name_short[ii+1] = 'a';
	  f_name_short[ii+2] = 's'; f_name_short[ii+3] = 'c';
	  f_name_short[ii+4] = 'i'; f_name_short[ii+5] = 'i';
	  
	  fAsciiFileNameShort = f_name_short;
	}
      
      //-------------------------------------------- extension .root
      if ( i_code == fCodeRoot )
	{
	  f_name_short[ii] = '.';   f_name_short[ii+1] = 'r';
	  f_name_short[ii+2] = 'o'; f_name_short[ii+3] = 'o';  f_name_short[ii+4] = 't';
	  
	  fRootFileNameShort = f_name_short;
	}
    }
  else
    {
      cout << "*TCnaRunEB::fMakeResultsFileName(...)> Name too long (for f_name_short)."
	   << " No room enough for the extension. (ii = " << ii << ")"
	   << fTTBELL  << endl; 
    }

    delete [] f_name;                                        fCdelete++;
    delete [] f_name_short;                                  fCdelete++;

    delete [] f_in;                                          fCdelete++;
    delete [] f_in_short;                                    fCdelete++;
}

//==========================================================================================
//
//
//
//==========================================================================================
void TCnaRunEB::fAsciiFileWriteHeader(const Int_t&  i_code)
{
//Ascii results file header writing  (private). Called by the WriteAscii...() methods
  
  //-----------------------------------------------
  //
  //     opening of the ASCII results file
  //     and writing of its header
  //
  //-----------------------------------------------
  
  fFcout_f.open(fAsciiFileName);
  
  fFcout_f << "*** File: " << fAsciiFileName
	   << " *** " << endl << endl;
  fFcout_f << "*Analysis name            : " << fFileHeader->fTypAna        << endl; 
  fFcout_f << "*Run number               : " << fFileHeader->fRunNumber     << endl;
  fFcout_f << "*First taken event        : " << fFileHeader->fFirstEvt      << endl;
  fFcout_f << "*Number of taken events   : " << fFileHeader->fNbOfTakenEvts << endl;
  fFcout_f << "*Super-module number      : " << fFileHeader->fSuperModule   << endl;
  fFcout_f << "*Time first taken event   : " << fFileHeader->fStartTime     << endl;
  fFcout_f << "*Time last  taken event   : " << fFileHeader->fStopTime      << endl;
  fFcout_f << "*Date first taken event   : " << fFileHeader->fStartDate     << endl;
  fFcout_f << "*Date last  taken event   : " << fFileHeader->fStopDate      << endl;
  fFcout_f << "*Number of entries        : " << fFileHeader->fNentries      << endl;
  fFcout_f << "*Max nb of towers in SM   : " << fFileHeader->fMaxTowInSM    << endl;
  fFcout_f << "*Max nb of Xtals in tower : " << fFileHeader->fMaxCrysInTow  << endl;
  fFcout_f << "*Max nb of samples ADC    : " << fFileHeader->fMaxSampADC    << endl;
  fFcout_f << endl; 

  //========================================================================= 
  //   closing of the results file if i_code = fCodeHeaderAscii only.
  //   closing is done in the fT1dWriteAscii() and fT2dWriteAscii() methods
  //   except for i_code = fCodeHeaderAscii
  //=========================================================================
  if(i_code == fCodeHeaderAscii) fFcout_f.close();
}

//===========================================================================
//
//         fT1dWriteAscii: writing of  a 2d array as a 1d array of 1d arrays
//
//                              (private)
//
//===========================================================================

void  TCnaRunEB::fT1dWriteAscii(const Int_t&  i_code,
			      const Int_t&  i_lic1, const Int_t&  i_lic2 )
{
  //Writing of the values of a 2d array as a 1d array of 1d arrays (private)
  
  //----------------------------------------------------------------
  //
  //     The 2D Array TAB[d1][d2] is writen like this:
  //
  //     TAB[0][j]  ; j=1,d2 by lines of i_lic1 values
  //     TAB[1][j]  ; j=1,d2 by lines of i_lic1 values
  //     TAB[2][j]  ; j=1,d2 by lines of i_lic1 values  
  // 
  //      etc... ,      hence (if i_lic2 > 0 ):
  //     
  //     TAB[i][0]  ; i=1,d1 by lines of i_lic2 values
  //     TAB[i][1]  ; i=1,d1 by lines of i_lic2 values
  //     TAB[i][2]  ; i=1,d1 by lines of i_lic2 values  
  //
  //----------------------------------------------------------------

  fFcout_f << setiosflags(ios::showpoint | ios::uppercase);
  fFcout_f << setprecision(3) << setw(6);
  fFcout_f.setf(ios::dec, ios::basefield);
  fFcout_f.setf(ios::fixed, ios::floatfield);
  fFcout_f.setf(ios::left, ios::adjustfield);
  fFcout_f.setf(ios::right, ios::adjustfield);
  
  cout << setiosflags(ios::showpoint | ios::uppercase);
  cout << setprecision(3) << setw(6);
  cout.setf(ios::dec, ios::basefield);
  cout.setf(ios::fixed, ios::floatfield);
  cout.setf(ios::left, ios::adjustfield);
  cout.setf(ios::right, ios::adjustfield);
  
  Int_t limit   = 0;
  Int_t nb_elts = 0;
  Int_t i_lico  = 0;
  
  TString g_nam0 = "?";
  TString g_nam1 = "?";
  TString g_nam2 = "?";
  TString g_nam3 = "?";
  TString g_nam4 = "?";
  
  Int_t nb_pass = 2;

  //------------------------------------------------------------------------
  //  Reservation dynamique de 2 arrays Double_t** (pour ev et var) de
  //  dimensions les multiples de 5 juste au-dessus des dimensions de
  //  l'array 2D a ecrire (array de dimensions
  // (fFileHeader->fMaxSampADC,fFileHeader->fMaxCrysInSM))
  //------------------------------------------------------------------------
  //.... Determination des tailles multiples de fNbChanByLine ou fNbSampByLine

  Int_t justap_samp = 0;
  if ( fFileHeader->fMaxSampADC%fNbSampByLine == 0 ){
      justap_samp = fFileHeader->fMaxSampADC;}
  else{
      justap_samp = ((fFileHeader->fMaxSampADC/fNbSampByLine)+1)*fNbSampByLine;}

  Int_t justap_chan = 0;  
  if ( fFileHeader->fMaxCrysInSM%fNbChanByLine == 0 ){
      justap_chan = fFileHeader->fMaxCrysInSM;}
  else{
      justap_chan = ((fFileHeader->fMaxCrysInSM/fNbChanByLine)+1)*fNbChanByLine;}
  
  //................................. allocation fjustap_2d_ev
  if ( i_code == fCodeEv )
    {
      if( fjustap_2d_ev == 0 )
	{
	  //................... Allocation
	  fjustap_2d_ev = new Double_t*[justap_chan];              fCnew++;  
	  fjustap_1d_ev = new  Double_t[justap_chan*justap_samp];  fCnew++;  
	  for(Int_t i = 0 ; i < justap_chan ; i++){
	    fjustap_2d_ev[i] = &fjustap_1d_ev[0] + i*justap_samp;}
	}
      //...................... Transfert des valeurs dans fjustap_2d_ev (=init)
      for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
	{ 
	  for(Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    { 
	      fjustap_2d_ev[i_SMEcha][i_samp]  = fT2d_ev[i_SMEcha][i_samp];
	    }
	}      
      //............................... mise a zero du reste de l'array (=init)
      for(Int_t i_SMEcha = fFileHeader->fMaxCrysInSM; i_SMEcha < justap_chan; i_SMEcha++)
	{
	  for(Int_t i_samp = fFileHeader->fMaxSampADC; i_samp < justap_samp; i_samp++)
	    {
	      fjustap_2d_ev[i_SMEcha][i_samp] = (Double_t)0;
	    }
	}      
    }

  //...... Allocation fjustap_2d_var (meme tableau pour corrections samples => a clarifier)

  if ( i_code == fCodeVar || i_code == fCodeSvCorrecCovCss )
    {
      if(fjustap_2d_var == 0)
	{
	  //................... Allocation
	  fjustap_2d_var = new Double_t*[justap_chan];             fCnew++;  
	  fjustap_1d_var = new  Double_t[justap_chan*justap_samp]; fCnew++;  
	  for(Int_t i = 0 ; i < justap_chan ; i++){
	    fjustap_2d_var[i] = &fjustap_1d_var[0] + i*justap_samp;}
	}
      //...................... Transfert des valeurs dans les fjustap_2d_var (=init)
      for(Int_t i_SMEcha = 0; i_SMEcha < fFileHeader->fMaxCrysInSM; i_SMEcha++)
	{ 
	  for(Int_t i_samp = 0; i_samp < fFileHeader->fMaxSampADC; i_samp++)
	    {
	      if (i_code == fCodeVar){
		fjustap_2d_var[i_SMEcha][i_samp] = fT2d_var[i_SMEcha][i_samp];}
	      if (i_code == fCodeSvCorrecCovCss){
		fjustap_2d_var[i_SMEcha][i_samp] =
		  fT2d_sv_correc_covss_s[i_SMEcha][i_samp];}
	    }
	}
      //............................... mise a zero du reste de l'array (=init)
      for(Int_t i_SMEcha = fFileHeader->fMaxCrysInSM; i_SMEcha < justap_chan; i_SMEcha++)
	{
	  for(Int_t i_samp = fFileHeader->fMaxSampADC; i_samp < justap_samp; i_samp++)
	    {
	      fjustap_2d_var[i_SMEcha][i_samp] = (Double_t)0;
	    }
	}
    }

  //------------------------------------------------------------------------

 //................ Ecriture de la taille des lignes
  if( i_code == fCodeEv || i_code == fCodeVar || i_code == fCodeSvCorrecCovCss )
    {
      fFcout_f << "*TCnaRunEB> sample sectors, nb of channels by line = "
	       << fNbChanByLine <<endl;
      fFcout_f << "*TCnaRunEB> channel sectors, nb of samples by line = "
	       << fNbSampByLine <<endl;
    }
  if( i_code == fCodeEvCorCss || i_code == fCodeSigCorCss)
    {
      fFcout_f << "*TCnaRunEB> sample sectors, nb of channels by line = "
	       << fNbChanByLine <<endl;
    }

  fFcout_f << endl;
  //.......................................................................

  if ( i_code == fCodeEvts || i_code == fCodeEvCorCss 
       || i_code == fCodeSigCorCss ) {nb_pass = 1;}
  
  for(Int_t i_pass = 0 ; i_pass < nb_pass ; i_pass++)
    {
      if(i_pass == 0)
	{
	  i_lico  = i_lic1;
	  limit   = justap_samp;
	  
	  if( i_code == fCodeEv  ||
	      i_code == fCodeVar ||
	      i_code == fCodeSvCorrecCovCss )
	    {
	      limit   = justap_samp;
	      nb_elts = justap_chan;
	      g_nam1  = "sample";
	      g_nam2  = "channel";
	    }
	  if(i_code == fCodeEvCorCss || i_code == fCodeSigCorCss )
	    {
	      limit   = 1;
	      nb_elts = justap_chan;
	      g_nam1  = "channel";
	    }
	  if ( i_code == fCodeEvts )
	    {
	      limit   = fFileHeader->fMaxSampADC;
	      nb_elts = fFileHeader->fNbBinsADC;
	      g_nam0  = "channel";
	      g_nam1  = "sample";
	      g_nam2  = "bins";
	      g_nam3  = "xmin=";
	      g_nam4  = "xmax=";
	    }
	}
      
      if(i_pass == 1)
	{
	  i_lico  = i_lic2;
	  limit   = justap_chan;
	  nb_elts = justap_samp;
	  g_nam1  = "channel";
	  g_nam2  = "sample";
	}
      
      if(i_lico > nb_elts){i_lico = nb_elts;}
      
      for (Int_t i_dim = 0 ; i_dim < limit ; i_dim++)
	{
	  //  switch (i_code){

	  if ( i_code ==  fCodeEv)
	    {
	      fFcout_f << "*TCnaRunEB> expectation values, ";
	    }
	  
	  if ( i_code ==  fCodeVar)
	    {
	      fFcout_f << "*TCnaRunEB> variances, ";
	    }
	  
	  if ( i_code ==  fCodeSvCorrecCovCss)
	    {
	      fFcout_f
		<< "*TCnaRunEB> Sample value correction coefficients, ";
	    }

	  if ( i_code ==  fCodeEvCorCss)
	    {
	      fFcout_f
		<< "*TCnaRunEB> expectation values of the"
		<< " (sample,sample) correlations, ";
	    }
	 	  
	  if ( i_code ==  fCodeSigCorCss)
	    {
	      fFcout_f << "*TCnaRunEB> sigmas of the"
		       << " (sample,sample) correlations, ";
	    }
	 
	  if ( i_code ==  fCodeEvts)
	    {
	      fFcout_f << "*TCnaRunEB> numbers of events, ";
	    }
	  
	    //default:
	    // cout << "*TCnaRunEB::fT1dWriteAscii(const Int_t& i_code)> "
	    //	 << "wrong code , i_code = " << i_code << endl; 
	    // }
	  
	  if(i_code == fCodeEvts)
	    {
	      fFcout_f << g_nam0 << " "
			       << fUserSMEcha << ", "
			       << g_nam1 << " " 
			       << i_dim  << ", "
			       << g_nam2 << " 0 to 99";
	    }
	  if(i_code == fCodeEv  ||
	     i_code == fCodeVar ||
	     i_code == fCodeSvCorrecCovCss)
	    {
	      fFcout_f << g_nam1 << " " << i_dim << ", " << g_nam2
			       << " 0 to " << nb_elts - 1;
	    }
	  if ( i_code == fCodeEvCorCss || i_code == fCodeSigCorCss)
	    {
	      fFcout_f << g_nam1 << " 0 to " << nb_elts - 1;
	    }
	  if(i_code == fCodeEvts)
	    {
	      fFcout_f << ", " << g_nam3 << " "
			       << (fT1d_xmin)[i_dim]
			       << " , " << g_nam4 << " "
			       << (fT1d_xmax)[i_dim];
	    }
	  
	  fFcout_f << " :" << endl;
	  
	  Int_t k_lec = 1;  // indice pour gestion saut de ligne
	  
	  for (Int_t i_elt = 0 ; i_elt < nb_elts ; i_elt++)
	    {
	      fFcout_f.width(8);
	      
	      //  switch (i_code){
	      
	      if ( i_code ==  fCodeEv)
		{
		  switch (i_pass){
		  case 0:
		    fFcout_f << (fjustap_2d_ev)[i_elt][i_dim];
		    break;
		  case 1:
		    fFcout_f << (fjustap_2d_ev)[i_dim][i_elt];
		    break;
		  }
		}
	      
	      if ( i_code ==   fCodeVar || i_code == fCodeSvCorrecCovCss)
		{
		  switch (i_pass){
		  case 0:
		    fFcout_f << fjustap_2d_var[i_elt][i_dim];
		    break;
		  case 1:
		    fFcout_f << fjustap_2d_var[i_dim][i_elt];
		    break;
		  }
		}  
	      
	      if ( i_code ==  fCodeEvCorCss)
		{
		  fFcout_f << fT1d_ev_cor_ss[i_elt];
		}  
	      	      
	      if ( i_code ==  fCodeSigCorCss)
		{
		  fFcout_f << fT1d_sig_cor_ss[i_elt];
		}  
	      
	      if ( i_code ==  fCodeEvts)
		{
		  fFcout_f << fT2d_his_s[i_dim][i_elt];
		}  
	      	      
	      //  default:
	      //  cout << "*TCnaRunEB::fT1dWriteAscii(const Int_t& i_code)> "
	      //     << "wrong code , i_code = " << i_code << endl; 
	      //  }
	      
	      fFcout_f << "  ";	
	      
	      if ( k_lec >= i_lico )
		{ 
		  fFcout_f << endl;	  
		  k_lec = 1;
		}
	      else
		{ 
		  k_lec++;
		}	      
	    }
	}
    }
  fFcout_f << endl;

  //................ closing of the results file      
  fFcout_f.close();

  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB> The results have been writen in the ASCII file: "
	 << fAsciiFileName << endl;}
} 

//----------------------------------------------------------------------
//
//            fT2dWriteAscii: Array 2D of (n_sctx , n_scty) sectors
//                          of size: i_pasx_arg * i_pasy_arg 
//
//                       (private)
//
//----------------------------------------------------------------------

void  TCnaRunEB::fT2dWriteAscii(const Int_t&  i_code,
			      const Int_t&  i_pasx_arg,
			      const Int_t&  i_pasy_arg)
{
//Writing of a matrix by sectors (private)

  Int_t i_pasx = i_pasx_arg;           // taille secteur en x
  Int_t i_pasy = i_pasy_arg;           // taille secteur en y
  
  //------------ formatage des nombres en faisant appel a la classe ios
  
  fFcout_f << setiosflags(ios::showpoint | ios::uppercase);
  fFcout_f.setf(ios::dec, ios::basefield);
  fFcout_f.setf(ios::fixed, ios::floatfield);
  fFcout_f.setf(ios::left, ios::adjustfield);
  fFcout_f.setf(ios::right, ios::adjustfield);
  fFcout_f << setprecision(3) << setw(6);
  
  cout << setiosflags(ios::showpoint | ios::uppercase);
  cout.setf(ios::dec, ios::basefield);
  cout.setf(ios::fixed, ios::floatfield);
  cout.setf(ios::left, ios::adjustfield);
  cout.setf(ios::right, ios::adjustfield);
  cout << setprecision(3) << setw(6);  

  //--------------------- fin du formatage standard C++ -------------------

  //-----------------------------------------------------------------------
  //  Reservation dynamique d'un array Double_t** de dimensions
  //  les multiples de 5 juste au-dessus des dimensions de l'array 2D
  //  a ecrire ( array de dimensions
  //  (fFileHeader->fMaxSampADC,fFileHeader->fMaxSampADC)
  //  (fFileHeader->fMaxCrysInSM,fFileHeader->fMaxCrysInSM) )
  //-----------------------------------------------------------------------
  // Determination des tailles multiples de fSectChanSizeX ou fSectSampSizeX

  //*************** channels (NON UTILISE APPARAMENT. A REVOIR )******
  Int_t justap_chan = 0;

  if ( fFileHeader->fMaxCrysInSM%fSectChanSizeX == 0 ){
      justap_chan = fFileHeader->fMaxCrysInSM;}
  else{
      justap_chan=((fFileHeader->fMaxCrysInSM/fSectChanSizeX)+1)*fSectChanSizeX;}

  //....................... Allocation fjustap_2d_cc

  if ( i_code == fCodeCovScc || i_code == fCodeCorScc||
       i_code == fCodeCovSccMos || i_code == fCodeCorSccMos ){
    if(fjustap_2d_cc == 0)
      {
	//................... Allocation
	fjustap_2d_cc = new Double_t*[justap_chan];              fCnew++;  
	fjustap_1d_cc = new  Double_t[justap_chan*justap_chan];  fCnew++;  
	for(Int_t i = 0 ; i < justap_chan ; i++){
	  fjustap_2d_cc[i] = &fjustap_1d_cc[0] + i*justap_chan;}
      }
  
    //............................... Transfert des valeurs dans fjustap_2d_cc  (=init)
    for(Int_t i = 0; i < fFileHeader->fMaxCrysInSM; i++){
      for(Int_t j = 0; j < fFileHeader->fMaxCrysInSM; j++){
	if ( i_code == fCodeCovScc ){
	  fjustap_2d_cc[i][j] = fT3d_cov_cc[fUserSamp][i][j];}
	if ( i_code == fCodeCorScc ){
	  fjustap_2d_cc[i][j] = fT3d_cor_cc[fUserSamp][i][j];}
	if ( i_code == fCodeCovSccMos ){
	  fjustap_2d_cc[i][j] = fT2d_cov_cc_mos[i][j];}
	if ( i_code == fCodeCorSccMos ){
	  fjustap_2d_cc[i][j] = fT2d_cor_cc_mos[i][j];}
      }
    }
    
    //.......................... mise a zero du reste de la matrice (=init)
    for(Int_t i = fFileHeader->fMaxCrysInSM; i < justap_chan; i++){
      for(Int_t j = fFileHeader->fMaxCrysInSM; j < justap_chan; j++){
	fjustap_2d_cc[i][j] = (Double_t)0.;}}
  }

  //************************************ Samples ***************************
  Int_t justap_samp = 0;
  
  if ( fFileHeader->fMaxSampADC%fSectSampSizeX == 0 ){
    justap_samp = fFileHeader->fMaxSampADC;}
  else{
    justap_samp=((fFileHeader->fMaxSampADC/fSectSampSizeX)+1)*fSectSampSizeX;}

  //....................... allocation fjustap_2d_ss

  if (i_code == fCodeCovCss          || i_code == fCodeCorCss ||
      i_code == fCodeCovCorrecCovCss || i_code == fCodeCorCorrecCovCss){
    if(fjustap_2d_ss == 0)
      {
	//................... Allocation
	fjustap_2d_ss = new Double_t*[justap_samp];              fCnew++;  
	fjustap_1d_ss = new  Double_t[justap_samp*justap_samp];  fCnew++;  
	for(Int_t i = 0 ; i < justap_samp ; i++){
	  fjustap_2d_ss[i] = &fjustap_1d_ss[0] + i*justap_samp;}
      }
  //.............................. Transfert des valeurs dans fjustap_2d_ss (=init)
  for(Int_t i = 0; i < fFileHeader->fMaxSampADC; i++){
    for(Int_t j = 0; j < fFileHeader->fMaxSampADC; j++){
      if( i_code == fCodeCovCss ){
	fjustap_2d_ss[i][j] = fT3d_cov_ss[fUserSMEcha][i][j];}
      if( i_code == fCodeCorCss ){
	fjustap_2d_ss[i][j] = fT3d_cor_ss[fUserSMEcha][i][j];}
      if( i_code == fCodeCovCorrecCovCss ){
	fjustap_2d_ss[i][j] = fT3d_cov_correc_covss_s[fUserSMEcha][i][j];}
      if( i_code == fCodeCorCorrecCovCss ){
	fjustap_2d_ss[i][j] = fT3d_cor_correc_covss_s[fUserSMEcha][i][j];}
    }
  }
  //.......................... mise a zero du reste de la matrice (=init)
  for(Int_t i = fFileHeader->fMaxSampADC; i < justap_samp; i++){
    for(Int_t j = fFileHeader->fMaxSampADC; j < justap_samp; j++){
      fjustap_2d_ss[i][j] = (Double_t)0.;}}
  }
 
  //-------------------------------------------------------------------

  TEBNumbering* MyNumbering = new TEBNumbering();    fCnew++;

  //..................... impressions + initialisations selon i_code
  
  Int_t isx_max = 0;    
  Int_t isy_max = 0;
  
  if(i_code == fCodeCovScc)
    {
      fFcout_f << "*TCnaRunEB> Covariance matrix between SMEchas "
	       << "for sample number " << fUserSamp;
      //    isx_max = fFileHeader->fMaxCrysInSM;
      //    isy_max = fFileHeader->fMaxCrysInSM;
      isx_max = justap_chan;
      isy_max = justap_chan;
    }
  if(i_code == fCodeCorScc)
    {
      fFcout_f << "*TCnaRunEB> Correlation matrix between SMEchas "
	       << "for sample number " << fUserSamp;
      //     isx_max = fFileHeader->fMaxCrysInSM;
      //     isy_max = fFileHeader->fMaxCrysInSM;
      isx_max = justap_chan;
      isy_max = justap_chan;
    }
  
  if(i_code == fCodeCovSccMos)
    {
      fFcout_f << "*TCnaRunEB> Covariance matrix between SMEchas "
	       << "averaged on the samples ";
      //    isx_max = fFileHeader->fMaxCrysInSM;
      //    isy_max = fFileHeader->fMaxCrysInSM;
      isx_max = justap_chan;
      isy_max = justap_chan;
    }
  if(i_code == fCodeCorSccMos)
    {
      fFcout_f << "*TCnaRunEB> Correlation matrix between SMEchas "
	       << "averaged on the samples ";
      //     isx_max = fFileHeader->fMaxCrysInSM;
      //     isy_max = fFileHeader->fMaxCrysInSM;
      isx_max = justap_chan;
      isy_max = justap_chan;
    }

  if(i_code == fCodeCovCss)
    {
      fFcout_f << "*TCnaRunEB> Covariance matrix between samples "
	       << "for SMEcha number " << fUserSMEcha
	       << " (SMTow " << MyNumbering->GetSMTowFromSMEcha(fUserSMEcha)
	       << " , TowEcha " << MyNumbering->GetTowEchaFromSMEcha(fUserSMEcha) << ")";
      //    isx_max = fFileHeader->fMaxSampADC;
      //    isy_max = fFileHeader->fMaxSampADC;
      isx_max = justap_samp;
      isy_max = justap_samp;
    }
  if(i_code == fCodeCorCss)
    { 
      fFcout_f << "*TCnaRunEB> Correlation matrix between samples "
	       << "for SMEcha number " << fUserSMEcha
	       << " (SMTow " << MyNumbering->GetSMTowFromSMEcha(fUserSMEcha) 
	       << " , TowEcha " << MyNumbering->GetTowEchaFromSMEcha(fUserSMEcha) << ")";
      //      isx_max = fFileHeader->fMaxSampADC;
      //      isy_max = fFileHeader->fMaxSampADC;
      isx_max = justap_samp;
      isy_max = justap_samp;
    }
 
  if(i_code == fCodeCovCorrecCovCss)
    { 
      fFcout_f << "*TCnaRunEB> Correction factors to the covariances "
	       << "between samples for SMEcha number " << fUserSMEcha
	       << " (SMTow " << MyNumbering->GetSMTowFromSMEcha(fUserSMEcha)
	       << " , TowEcha " << MyNumbering->GetTowEchaFromSMEcha(fUserSMEcha) << ")";
      //      isx_max = fFileHeader->fMaxSampADC;
      //      isy_max = fFileHeader->fMaxSampADC;
      isx_max = justap_samp;
      isy_max = justap_samp;
    }
  
  if(i_code == fCodeCorCorrecCovCss)
    { 
      fFcout_f << "*TCnaRunEB> Correction factors to the correlations "
	       << "between samples for SMEcha number " << fUserSMEcha
	       << " (SMTow " << MyNumbering->GetSMTowFromSMEcha(fUserSMEcha)
	       << " , TowEcha " << MyNumbering->GetTowEchaFromSMEcha(fUserSMEcha) << ")";
      //      isx_max = fFileHeader->fMaxSampADC;
      //      isy_max = fFileHeader->fMaxSampADC;
      isx_max = justap_samp;
      isy_max = justap_samp;
    }
 
  fFcout_f << endl;

  //............... Calcul des nombres de secteurs selon x
  //                i_pasx  = taille secteur en x 
  //                isx_max = taille de la matrice en x
  //                n_sctx  = nombre de secteurs en x
  //
  if(i_pasx > isx_max){i_pasx = isx_max;}  
  Int_t n_sctx;
  Int_t max_verix; 
  n_sctx = isx_max/i_pasx;
  max_verix = n_sctx*i_pasx; 
  if(max_verix < isx_max){ n_sctx++;}

  //............... Calcul des nombres de secteurs selon y
  //                i_pasy  = taille secteur en y
  //                isy_max = taille de la matrice en y
  //                n_scty  = nombre de secteurs en x
  //
  if(i_pasy > isy_max){i_pasy = isy_max;} 
  Int_t n_scty;
  Int_t max_veriy; 
  n_scty = isy_max/i_pasy;
  max_veriy = n_scty*i_pasy; 
  if(max_veriy < isy_max){ n_scty++;}


  //................ Ecriture de la taille et du nombre des secteurs
  if( i_code == fCodeCovCss || i_code == fCodeCorCss ||
      i_code == fCodeCovCorrecCovCss ||  i_code == fCodeCorCorrecCovCss)
    {
      fFcout_f << "*TCnaRunEB> sector size = " << fSectSampSizeX 
	       << " , number of sectors = " << n_sctx << " x " << n_scty
	       <<endl;
    }
 if( i_code == fCodeCovScc     || i_code == fCodeCorScc ||
     i_code == fCodeCovSccMos  || i_code == fCodeCorSccMos )
    {
      fFcout_f << "*TCnaRunEB> sector size = " << fSectChanSizeX 
	       << " , number of sectors = " << n_sctx << " x " << n_scty
	       << endl;
    }
  fFcout_f << endl;

  //............... impression matrice par secteurs i_pas x i_pas  
  //........................... boucles pour display des secteurs 
  Int_t   ix_inf = -i_pasx;
  
  for(Int_t nsx = 0 ; nsx < n_sctx ; nsx++)
    { 
      //......................... calcul limites secteur
      ix_inf = ix_inf + i_pasx;
      Int_t   ix_sup = ix_inf + i_pasx; 
      
      Int_t   iy_inf = -i_pasy;
      
      for(Int_t nsy = 0 ; nsy < n_scty ; nsy++)
	{       
	  iy_inf = iy_inf + i_pasy;          
	  Int_t   iy_sup = iy_inf + i_pasy;
	  
	  //......................... display du secteur (nsx,nsy)
	  
	  if(i_code == fCodeCovScc || i_code == fCodeCovCss ||
	     i_code == fCodeCovCorrecCovCss || i_code == fCodeCorCorrecCovCss )
	    {fFcout_f << "        ";}
	  if(i_code == fCodeCorScc || i_code == fCodeCorCss)
	    {fFcout_f << "      ";}
	  
	  for (Int_t iy_c = iy_inf ; iy_c < iy_sup ; iy_c++)
	    {
	      if(i_code == fCodeCovScc || i_code == fCodeCovSccMos || i_code == fCodeCovCss ||
		 i_code == fCodeCovCorrecCovCss ||
		 i_code == fCodeCorCorrecCovCss)
		{fFcout_f.width(8);}
	      if(i_code == fCodeCorScc || i_code == fCodeCorSccMos || i_code == fCodeCorCss)
		{fFcout_f.width(6);}
	      fFcout_f << iy_c << "  ";
	    }	  
	  fFcout_f << endl << endl;
	  
	  for (Int_t ix_c = ix_inf ; ix_c < ix_sup ; ix_c++)
	    { 
	      if(i_code == fCodeCovScc|| i_code == fCodeCovSccMos || i_code == fCodeCovCss ||
		 i_code == fCodeCovCorrecCovCss ||
		 i_code == fCodeCorCorrecCovCss)
		{fFcout_f.width(8);}
	      if(i_code == fCodeCorScc || i_code == fCodeCorSccMos || i_code == fCodeCorCss)
		{fFcout_f.width(6);}
	      fFcout_f << ix_c << "   ";
	      
	      for (Int_t iy_c = iy_inf ; iy_c < iy_sup ; iy_c++)
		{
		  if(i_code == fCodeCovScc          ||
		     i_code == fCodeCovSccMos       ||
		     i_code == fCodeCovCss          ||
		     i_code == fCodeCovCorrecCovCss || 
 		     i_code == fCodeCorCorrecCovCss ){
		    fFcout_f.width(8);}

		  if(i_code == fCodeCorScc || i_code == fCodeCorSccMos ||  i_code == fCodeCorCss){
		    fFcout_f.width(6);}
		  
		  if( i_code == fCodeCovScc ||  i_code == fCodeCovSccMos || i_code == fCodeCorScc){
		    fFcout_f << fjustap_2d_cc[ix_c][iy_c] << "  ";}
		  
		  if ( i_code == fCodeCovCss          ||
		       i_code == fCodeCorCss          ||
		       i_code == fCodeCovCorrecCovCss ||
		       i_code == fCodeCorCorrecCovCss )
		    {
		      fFcout_f << fjustap_2d_ss[ix_c][iy_c] << "  ";
		    }
		}
	      fFcout_f << endl;
	    }
	  fFcout_f << endl;
	}
    }

  //........... closing of the results file
  
  fFcout_f.close();
  
  if(fFlagPrint == fCodePrintAllComments){
    cout << "*TCnaRunEB> The results have been writen in the ASCII file: "
	 << fAsciiFileName << endl;}

  delete MyNumbering;                      fCdelete++;
}

//-------------------------------------------------------------------------
//
//    Get methods for different run or file parameters
//
//    W A R N I N G :  some of these methods are called by external code
//
//                D O N ' T    S U P P R E S S !
//
//
//     TString  fFileHeader->fTypAna        = typ_ana
//     Int_t    fFileHeader->fRunNumber     = run_number
//     Int_t    fFileHeader->fFirstEvt      = nfirst
//     Int_t    fFileHeader->fNbOfTakenEvts = nevents
//     Int_t    fFileHeader->fSuperModule   = super_module
//     Int_t    fFileHeader->fNentries      = nentries
//
//-------------------------------------------------------------------------
TString TCnaRunEB::GetRootFileNameShort()  {return fRootFileNameShort;}

TString TCnaRunEB::GetAnalysisName()       {return fFileHeader->fTypAna;}
Int_t   TCnaRunEB::GetRunNumber()          {return fFileHeader->fRunNumber;}
Int_t   TCnaRunEB::GetFirstTakenEvent()    {return fFileHeader->fFirstEvt;}
Int_t   TCnaRunEB::GetNumberOfTakenEvents(){return fFileHeader->fNbOfTakenEvts;}
Int_t   TCnaRunEB::GetSMNumber()           {return fFileHeader->fSuperModule;}
Int_t   TCnaRunEB::GetNentries()           {return fFileHeader->fNentries;}

//------------------------------------------------------------------------
Int_t  TCnaRunEB::PickupNumberOfEvents(const Int_t& SMEcha,
				     const Int_t& sample)
{
  Int_t nb_of_evts = -1;

  if( SMEcha >= 0 && SMEcha < fFileHeader->fMaxCrysInSM )
    {
      if( sample >= 0 && sample < fFileHeader->fMaxSampADC )
	{ 
	  nb_of_evts = fT2d_LastEvtNumber[SMEcha][sample] + 1;
	}
      else
	{
	  cout << "!TCnaRunEB::PickupNumberOfEvents()> sample = "
	   << sample << "OUT OF BOUNDS" << fTTBELL << endl;
	}
    }
  else
    {
      cout << "!TCnaRunEB::PickupNumberOfEvents()> SMEcha = "
	   << SMEcha << "OUT OF BOUNDS" << fTTBELL << endl;
    }

  return nb_of_evts;
}

//=========================== E N D ======================================
