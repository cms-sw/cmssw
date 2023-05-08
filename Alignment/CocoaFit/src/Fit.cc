#include <FWCore/Utilities/interface/Exception.h>
//   COCOA class implementation file
//Id:  Fit.cc
//CAT: Fit
//
//   History: v1.0
//   Pedro Arce

#include <cstdlib>
#include <iomanip>
#include <cmath>  // among others include also floating-point std::abs functions
#include <ctime>
#include <set>

#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaFit/interface/Fit.h"

#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "Alignment/CocoaModel/interface/DeviationsFromFileSensor2D.h"
#include "Alignment/CocoaFit/interface/NtupleManager.h"
#include "Alignment/CocoaFit/interface/FittedEntriesManager.h"
#include "Alignment/CocoaFit/interface/FittedEntriesSet.h"
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif
#include "Alignment/CocoaModel/interface/OpticalObjectMgr.h"
#include "Alignment/CocoaModel/interface/ErrorCorrelationMgr.h"
#include "Alignment/CocoaModel/interface/ErrorCorrelation.h"
#include "Alignment/CocoaModel/interface/FittedEntriesReader.h"
#include "Alignment/CocoaDaq/interface/CocoaDaqReader.h"
#include "Alignment/CocoaFit/interface/CocoaDBMgr.h"

Fit* Fit::instance = nullptr;

ALIMatrix* Fit::AMatrix;
ALIMatrix* Fit::AtMatrix;
ALIMatrix* Fit::WMatrix;
ALIMatrix* Fit::AtWAMatrix;
//op ALIMatrix* Fit::VaMatrix;
ALIMatrix* Fit::DaMatrix;
//op ALIMatrix* Fit::PDMatrix;
//-ALIMatrix* Fit::VyMatrix;
ALIMatrix* Fit::yfMatrix;
//ALIMatrix* Fit::fMatrix;

ALIint Fit::_NoLinesA;
ALIint Fit::_NoColumnsA;
//op ALIMatrix* Fit::thePropagationMatrix;

ALIint Fit::theMinimumEntryQuality;
ALIdouble Fit::thePreviousIterationFitQuality = DBL_MAX;
ALIdouble Fit::theFitQualityCut = -1;
ALIdouble Fit::theRelativeFitQualityCut = -1;
ALIint Fit::theNoFitIterations;
ALIint Fit::MaxNoFitIterations = -1;
ALIdouble Fit::theMinDaFactor = 1.e-8;

ALIuint Fit::nEvent = 1;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Gets the only instance of Model
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Fit& Fit::getInstance() {
  if (!instance) {
    instance = new Fit;
    ALIdouble go;
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();

    gomgr->getGlobalOptionValue("maxDeviDerivative", go);
    ALIUtils::setMaximumDeviationDerivative(go);
    if (ALIUtils::debug >= 3)
      std::cout << " Fit::maximum_deviation_derivative " << ALIUtils::getMaximumDeviationDerivative() << std::endl;

    gomgr->getGlobalOptionValue("maxNoFitIterations", go);
    MaxNoFitIterations = int(go);

    gomgr->getGlobalOptionValue("fitQualityCut", go);
    theFitQualityCut = go;
    if (ALIUtils::debug >= 3)
      std::cout << " theFitQualityCut " << theFitQualityCut << std::endl;

    gomgr->getGlobalOptionValue("RelativeFitQualityCut", go);
    theRelativeFitQualityCut = go;
    if (ALIUtils::debug >= 3)
      std::cout << " theRelativeFitQualityCut " << theRelativeFitQualityCut << std::endl;

    gomgr->getGlobalOptionValue("minDaFactor", go);
    theMinDaFactor = go;
  }

  return *instance;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  startFit: steering method to make the fit
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::startFit() {
  //  Model::setCocoaStatus( COCOA_InitFit );
  NtupleManager* NTmgr = NtupleManager::getInstance();
  if (GlobalOptionMgr::getInstance()->GlobalOptions()["rootResults"] > 0) {
    NTmgr->BookNtuple();
  }

  ALIUtils::setFirstTime(true);

  WriteVisualisationFiles();

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  for (;;) {
    bool bend = fitNextEvent(nEvent);
    if (gomgr->GlobalOptions()["writeDBOptAlign"] > 0 || gomgr->GlobalOptions()["writeDBAlign"] > 0) {
      CocoaDBMgr::getInstance()->DumpCocoaResults();
    }

    if (!bend) {
      if (ALIUtils::debug >= 1)
        std::cout << "@@@ Fit::startFit  ended  n events = " << nEvent << std::endl;
      break;
    }

    //-    if ( ALIUtils::debug >= 0) std::cout << " FIT STATUS " << Model::printCocoaStatus( Model::getCocoaStatus() ) << std::endl;

    nEvent++;
  }

  //---------- Program ended, fill histograms of fitted entries
  if (gomgr->GlobalOptions()["histograms"] > 0) {
    FittedEntriesManager* FEmgr = FittedEntriesManager::getInstance();
    FEmgr->MakeHistos();
  }

  if (GlobalOptionMgr::getInstance()->GlobalOptions()["rootResults"] > 0) {
    NTmgr->WriteNtuple();
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool Fit::fitNextEvent(ALIuint& nEvent) {
  if (Model::getFittedEntriesReader() != nullptr)
    Model::getFittedEntriesReader()->readFittedEntriesFromFile();

  //----- Reset coordinates to those read at the start
  std::vector<OpticalObject*>::iterator voite;
  for (voite = Model::OptOList().begin(); voite != Model::OptOList().end(); ++voite) {
    (*voite)->resetOriginalOriginalCoordinates();
  }

  //----- Reset entries displacements to 0.
  std::vector<Entry*>::iterator veite;
  for (veite = Model::EntryList().begin(); veite != Model::EntryList().end(); ++veite) {
    (*veite)->resetValueDisplacementByFitting();
  }

  ALIbool lastEvent = false;

  //-    DeviationsFromFileSensor2D::setApply( 1 );

  //m  ALIbool moreDataSets = Model::readMeasurementsFromFile( Measurement::only1Date, Measurement::only1Time );

  //----- Check if there are more data sets
  ALIbool moreDataSets = true;
  if (CocoaDaqReader::GetDaqReader() != nullptr)
    moreDataSets = CocoaDaqReader::GetDaqReader()->ReadNextEvent();

  if (ALIUtils::debug >= 5)
    std::cout << CocoaDaqReader::GetDaqReader() << "$$$$$$$$$$$$$$$ More Data Sets to be processed: " << moreDataSets
              << std::endl;

  if (moreDataSets) {
    if (ALIUtils::debug >= 2)
      std::cout << std::endl << "@@@@@@@@@@@@@@@@@@ Starting data set fit : " << nEvent << std::endl;

    //----- Count entries to be fitted, and set their order in theFitPos
    setFittableEntries();

    //----- Dump dimensions of output in 'report.out' file
    ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
    fileout << std::endl << "@@@@@@@ NEW MEASUREMENT SET " << nEvent << std::endl;
    if (ALIUtils::report >= 1)
      ALIUtils::dumpDimensions(fileout);

    //----- reset Number of iterations of non linear fit
    theNoFitIterations = 0;

    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    ALIdouble dumpMat;
    gomgr->getGlobalOptionValue("save_matrices", dumpMat);

    //----- Fit parameters
    double daFactor = 1.;
    Model::setCocoaStatus(COCOA_FirstIterationInEvent);
    for (;;) {
      if (ALIUtils::debug >= 2) {
        std::cout << std::endl << "Fit iteration " << theNoFitIterations << " ..." << std::endl;
      }

      //---------- Calculate the original simulated values of each Measurement (when all entries have their read in values)
      calculateSimulatedMeasurementsWithOriginalValues();  //?? original changed atfer each iteration

      FitQuality fq = fitParameters(daFactor);
      if (dumpMat > 1)
        dumpMatrices();

      //-      evaluateFitQuality( fq, daFactor );

      if (ALIUtils::debug >= 2) {
        std::cout << std::endl << "@@@@ Check fit quality for iteration " << theNoFitIterations << std::endl;
      }

      //----- Check if new iteration must be done
      if (fq == FQsmallDistanceToMinimum) {
        if (ALIUtils::debug >= 2)
          std::cout << std::endl << "@@@@ Fit quality: distance SMALLER than mininum " << std::endl;
        addDaMatrixToEntries();
        if (ALIUtils::report >= 1)
          dumpFittedValues(ALIFileOut::getInstance(Model::ReportFName()), TRUE, TRUE);
        //--- Print entries in all ancestor frames
        ALIdouble go;
        gomgr->getGlobalOptionValue("dumpInAllFrames", go);
        if (go >= 1)
          dumpFittedValuesInAllAncestorFrames(ALIFileOut::getInstance(Model::ReportFName()), FALSE, FALSE);

        break;  // No more iterations
      } else if (fq == FQbigDistanceToMinimum) {
        if (ALIUtils::debug >= 2)
          std::cout << std::endl << "@@@@ Fit quality: distance BIGGER than mininum " << std::endl;
        addDaMatrixToEntries();
        if (ALIUtils::report >= 1)
          dumpFittedValues(ALIFileOut::getInstance(Model::ReportFName()), TRUE, TRUE);

        //----- Next iteration (if not too many already)
        theNoFitIterations++;
        daFactor = 1.;

        //----- Too many iterations: end event here
        if (theNoFitIterations >= MaxNoFitIterations) {
          if (ALIUtils::debug >= 1)
            std::cerr << "!!!! WARNING: Too many iterations " << theNoFitIterations << "  and fit DOES NOT CONVERGE "
                      << std::endl;

          if (ALIUtils::report >= 2) {
            ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
            fileout << "!!!! WARNING: Too many iterations " << theNoFitIterations << "  and fit DOES NOT CONVERGE "
                    << std::endl;
          }
          //	  Model::setCocoaStatus( COCOA_FitCannotImprove );
          break;  // No more iterations
        }

      } else if (fq == FQchiSquareWorsened) {
        if (ALIUtils::debug >= 1) {
          //----- Recalculate fit quality with decreasing values of Da
          std::cerr << "!! WARNING: fit quality has worsened, Recalculate fit quality with decreasing values of Da "
                    << std::endl;
          std::cout << " quality daFactor= " << daFactor << " minimum= " << theMinDaFactor << std::endl;
        }
        daFactor *= 0.5;
        if (daFactor > theMinDaFactor) {
          substractLastDisplacementToEntries(0.5);

          if (ALIUtils::report >= 2) {
            ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
            fileout << " Redoing iteration with Da factor " << daFactor << std::endl;
          }
        } else {
          daFactor *= 2.;
          std::cerr << " !!!ERROR: not possible to get good fit quality even multiplying Da by " << daFactor
                    << std::endl;
          if (ALIUtils::report >= 2) {
            ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
            fileout << " !!!ERROR: not possible to get good fit quality even multiplying Da by " << daFactor
                    << std::endl;
          }
          //	    Model::setCocoaStatus( COCOA_FitCannotImprove );
          //-    std::cout << "fdsaf FIT STATUS " << Model::printCocoaStatus( Model::getCocoaStatus() ) << std::endl;
          break;  // No more iterations
        }
      }
      Model::setCocoaStatus(COCOA_NextIterationInEvent);
    }

    //----- Iteration is finished: dump fitted entries
    if (ALIUtils::debug >= 1)
      calculateSimulatedMeasurementsWithOriginalValues();
    if (gomgr->GlobalOptions()["histograms"] > 0) {
      FittedEntriesManager::getInstance()->AddFittedEntriesSet(new FittedEntriesSet(AtWAMatrix));
    }

    if (GlobalOptionMgr::getInstance()->GlobalOptions()["rootResults"] > 0) {
      NtupleManager* ntupleMgr = NtupleManager::getInstance();
      ntupleMgr->InitNtuple();
      ntupleMgr->FillChi2();
      ntupleMgr->FillOptObjects(AtWAMatrix);
      ntupleMgr->FillMeasurements();
      ntupleMgr->FillFitParameters(AtWAMatrix);
      ntupleMgr->FillNtupleTree();
    }

    //- only if not stopped in worsening quality state        if(ALIUtils::report >= 0) dumpFittedValues( ALIFileOut::getInstance( Model::ReportFName() ));

    /*-      std::vector< OpticalObject* >::iterator voite;
      for( voite = Model::OptOList().begin(); voite !=  Model::OptOList().end(); voite++ ) {
      //-??      	(*voite)->resetOriginalOriginalCoordinates();
	}*/

    //---- If no measurement file, break after looping once
    //-      std::cout << " Measurement::measurementsFileName() " << Measurement::measurementsFileName() << " Measurement::measurementsFileName()" <<std::endl;
    if (CocoaDaqReader::GetDaqReader() == nullptr) {
      //m    if( Measurement::measurementsFileName() == "" ) {
      if (ALIUtils::debug >= 1)
        std::cout << std::endl << "@@@@@@@@@@@@@@@@@@ Fit has ended : only one measurement " << nEvent << std::endl;
      lastEvent = true;
      return !lastEvent;
    }

    //-      std::cout << "  Measurement::only1" <<  Measurement::only1 << std::endl;
    if (Measurement::only1) {
      if (ALIUtils::debug >= 1)
        std::cout << std::endl << "@@@@@@@@@@@@@@@@@@ Fit has ended : 'Measurement::only1'  is set" << std::endl;

      lastEvent = true;
      return !lastEvent;
    }

    if (GlobalOptionMgr::getInstance()->GlobalOptions()["maxEvents"] <= nEvent) {
      if (ALIUtils::debug >= 1)
        std::cout << std::endl
                  << "@@@@@@@@@@@@@@@@@@ Fit has ended : 'Number of events exhausted " << nEvent << std::endl;

      lastEvent = true;
      return !lastEvent;
    }

  } else {
    lastEvent = true;
    if (ALIUtils::debug >= 1)
      std::cout << std::endl << "@@@@@@@@@@@@@@@@@@ Fit has ended : ??no more data sets' " << nEvent << std::endl;
    return !lastEvent;
  }

  if (ALIUtils::debug >= 1)
    std::cout << std::endl << "@@@@@@@@@@@@@@@@@@ Fit has ended : " << nEvent << std::endl;

  return !lastEvent;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::WriteVisualisationFiles() {
#ifdef COCOA_VIS
  if (gomgr->GlobalOptions()["VisOnly"] == 1) {
    calculateSimulatedMeasurementsWithOriginalValues();  //?? original changed atfer each iteration
  }

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if (gomgr->GlobalOptions()["VisWriteVRML"] > 0) {
    if (ALIUtils::getFirstTime())
      ALIVRMLMgr::getInstance().writeFile();
  }
  if (gomgr->GlobalOptions()["VisWriteIguana"] > 0) {
    if (ALIUtils::getFirstTime())
      IgCocoaFileMgr::getInstance().writeFile();
  }

  if (gomgr->GlobalOptions()["VisOnly"] == 1) {
    if (ALIUtils::debug >= 1)
      std::cout << " Visualiation file(s) succesfully written. Ending.... " << std::endl;
    exit(1);
  }
#endif
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Count how many entries are going to be fitted (have quality >=  theMinimumEntryQuality)
//@@ Set for this entries the value of theFitPos
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::setFittableEntries() {
  std::vector<Entry*>::const_iterator vecite;

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  theMinimumEntryQuality = int(gomgr->GlobalOptions()[ALIstring("calcul_type")]) + 1;
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::setFittableEntries: total Entry List size= " << Model::EntryList().size() << std::endl;

  int No_entry_to_fit = 0;
  for (vecite = Model::EntryList().begin(); vecite != Model::EntryList().end(); ++vecite) {
    // Number the parameters that are going to be fitted
    if ((*vecite)->quality() >= theMinimumEntryQuality) {
      (*vecite)->setFitPos(No_entry_to_fit);
      if (ALIUtils::debug >= 4)
        std::cout << " Entry To Fit= " << No_entry_to_fit << " " << (*vecite)->OptOCurrent()->name() << " "
                  << (*vecite)->name() << "   with quality= " << (*vecite)->quality() << std::endl;
      No_entry_to_fit++;
    }
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Main method in class Fit
//@@ fitParameters: get the parameters through the chi square fit
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
FitQuality Fit::fitParameters(const double daFactor) {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::fitParameters: Fit quality daFactor " << daFactor << std::endl;

  redoMatrices();

  //---- Get chi2 of first iteration
  if (Model::getCocoaStatus() == COCOA_FirstIterationInEvent) {
    thePreviousIterationFitQuality = GetSChi2(false);

    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    if (gomgr->GlobalOptions()[ALIstring("stopAfter1stIteration")] == 1) {
      std::cout << "@!! STOPPED by user after 1st iteration " << std::endl;
      exit(1);
    }
  }

  /*  //---------- Open output file
  if( ALIUtils::report >= 1 ) {
    ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
    //t    fileout << " REPORT.OUT " << std::endl;
    //t    ALIUtils::dumpDimensions( fileout );
    fileout << std::endl << "Fit iteration " << theNoFitIterations << " ..." << std::endl;
    }*/

  //-    std::cout << "2 FIT STATUS " << Model::printCocoaStatus( Model::getCocoaStatus() ) << std::endl;

  if (ALIUtils::debug >= 10) {
    std::cout << std::endl << " End fitParameters " << theNoFitIterations << " ..." << std::endl;
  }

  return getFitQuality();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::redoMatrices() {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::redoMatrices" << std::endl;

  deleteMatrices();

  calculateSimulatedMeasurementsWithOriginalValues();

  PropagateErrors();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Propagate the Errors from the entries to the measurements
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//cocoaStatus
void Fit::PropagateErrors() {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::PropagateErrors" << std::endl;

  //----- Create empty matrices of appropiate size
  CreateMatrices();

  //---- count running time
  time_t now;
  now = clock();
  if (ALIUtils::debug >= 2)
    std::cout << "TIME:CREATE_MAT    : " << now << " " << difftime(now, ALIUtils::time_now()) / 1.E6 << std::endl;
  ALIUtils::set_time_now(now);

  //----- Fill the A, W & y matrices with the measurements
  FillMatricesWithMeasurements();

  //---- count running time
  now = clock();
  if (ALIUtils::debug >= 2)
    std::cout << "TIME:MAT_MEAS_FILLED: " << now << " " << difftime(now, ALIUtils::time_now()) / 1.E6 << std::endl;
  ALIUtils::set_time_now(now);

  //----- Fill the A, W & y matrices with the calibrated parameters
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if (gomgr->GlobalOptions()[ALIstring("calcul_type")] == 0) {
    FillMatricesWithCalibratedParameters();

    //---- count running time
    now = clock();
    if (ALIUtils::debug >= 0)
      std::cout << "TIME:MAT_CAL_FILLED : " << now << " " << difftime(now, ALIUtils::time_now()) / 1.E6 << std::endl;
    ALIUtils::set_time_now(now);
  }

  //----- Put by hand some correlations if known previously
  setCorrelationsInWMatrix();

  if (ALIUtils::debug >= 3)
    WMatrix->Dump("WMatrix before inverse");

  //----- Check first that matrix can be inverted
  if (m_norm1(WMatrix->MatNonConst()) == 0) {
    //    Model::setCocoaStatus( COCOA_FitMatrixNonInversable );
    return;  //  Model::getCocoaStatus();
  } else {
    WMatrix->inverse();
  }

  if (ALIUtils::debug >= 3)
    AMatrix->Dump("AMatrix");
  if (ALIUtils::debug >= 3)
    WMatrix->Dump("WMatrix");
  if (ALIUtils::debug >= 3)
    yfMatrix->Dump("yfMatrix");

  if (gomgr->GlobalOptions()["onlyDeriv"] >= 1) {
    std::cout << "ENDING after derivatives are calculated ('onlyDeriv' option set)" << std::endl;
    exit(1);
  }

  multiplyMatrices();

  now = clock();
  if (ALIUtils::debug >= 0)
    std::cout << "TIME:MAT_MULTIPLIED : " << now << " " << difftime(now, ALIUtils::time_now()) / 1.E6 << std::endl;
  ALIUtils::set_time_now(now);

  if (ALIUtils::getFirstTime() == 1)
    ALIUtils::setFirstTime(false);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Calculate the simulated value of each Measurement propagating the LightRay when all the entries have their original values
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::calculateSimulatedMeasurementsWithOriginalValues() {
  //  if( ALIUtils::debug >= 4) OpticalObjectMgr::getInstance()->dumpOptOs();

  //---------- Set DeviationsFromFileSensor2D::apply true
  DeviationsFromFileSensor2D::setApply(true);

  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::calculateSimulatedMeasurementsWithOriginalValues" << std::endl;
  //---------- Loop Measurements
  std::vector<Measurement*>::const_iterator vmcite;
  for (vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); ++vmcite) {
    //----- Calculate Simulated Value Original
    (*vmcite)->calculateOriginalSimulatedValue();
  }

  //---------- Set DeviationsFromFileSensor2D::apply false
  // It cannot be applied when calculating derivatives, because after a displacement the laser could hit another square in matrix and then cause a big step in the derivative
  DeviationsFromFileSensor2D::setApply(false);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::deleteMatrices() {
  //delete matrices created in previous iteration
  delete DaMatrix;
  delete AMatrix;
  delete WMatrix;
  delete yfMatrix;
  //op  delete fMatrix;
  delete AtMatrix;
  delete AtWAMatrix;
  //op  delete VaMatrix;
  //-  delete VyMatrix;
  //op  delete PDMatrix;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Calculate the NoLines & NoColumns and create matrices
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::CreateMatrices() {
  //---------- Count number of measurements
  ALIint NoMeas = 0;
  std::vector<Measurement*>::const_iterator vmcite;
  for (vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); ++vmcite) {
    NoMeas += (*vmcite)->dim();
  }
  if (ALIUtils::debug >= 9)
    std::cout << "NOMEAS" << NoMeas << std::endl;

  //-------- Count number of 'cal'ibrated parameters
  ALIint nEnt_cal = 0;
  ALIint noent = 0;
  //-  std::cout << Model::EntryList().size() << std::endl;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if (gomgr->GlobalOptions()["calcul_type"] == 0) {  // fit also 'cal' parameters
    //-  if( ALIUtils::debug >= 9) std::cout << "NOENTCALLL " << nEnt_cal << std::endl;
    if (ALIUtils::debug >= 5)
      std::cout << " Count number of 'cal'ibrated parameters " << std::endl;
    std::vector<Entry*>::iterator veite;
    for (veite = Model::EntryList().begin(); veite != Model::EntryList().end(); ++veite) {
      if ((*veite)->quality() == 1)
        nEnt_cal++;
      noent++;
      if (ALIUtils::debug >= 6) {
        std::cout << (*veite)->quality() << " " << (*veite)->OptOCurrent()->name() << " " << (*veite)->name()
                  << " # ENT CAL " << nEnt_cal << " # ENT " << noent << std::endl;
      }
    }
  }

  //---------- Count number parameters to be fitted ('cal' + 'unk')
  ALIint NoParamFit = 0;
  std::vector<Entry*>::const_iterator vecite;
  for (vecite = Model::EntryList().begin(); vecite != Model::EntryList().end(); ++vecite) {
    if ((*vecite)->quality() >= theMinimumEntryQuality) {
      NoParamFit++;
      if (ALIUtils::debug >= 99)
        std::cout << (*vecite)->quality() << (*vecite)->OptOCurrent()->name() << (*vecite)->name() << "NoParamFit"
                  << NoParamFit << std::endl;
      //      break;
    }
  }

  //---------- Create Matrices
  ALIint NoLinesA = NoMeas + nEnt_cal;
  ALIint NoColumnsA = NoParamFit;
  AMatrix = new ALIMatrix(NoLinesA, NoColumnsA);

  ALIint NoLinesW = NoLinesA;
  ALIint NoColumnsW = NoLinesA;
  WMatrix = new ALIMatrix(NoLinesW, NoColumnsW);

  ALIint NoLinesY = NoLinesA;
  //op  yMatrix = new ALIMatrix( NoLinesY, 1 );
  yfMatrix = new ALIMatrix(NoLinesY, 1);

  //op  fMatrix = new ALIMatrix( NoLinesY, 1 );

  if (ALIUtils::debug >= 4)
    std::cout << "CreateMatrices: NoLinesA = " << NoLinesA << " NoColumnsA = " << NoColumnsA << std::endl;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Loop Measurements:
//@@    Fill Matrix A with derivatives respect to affecting entries
//@@    Fill Matrix W, y & f with values and sigmas of measurement coordinate
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::FillMatricesWithMeasurements() {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::FillMatricesWithMeasurements" << std::endl;

  int Aline = 0;

  //---------- Loop Measurements
  std::vector<Measurement*>::const_iterator vmcite;
  std::vector<Entry*>::const_iterator vecite;
  for (vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); ++vmcite) {
    if (ALIUtils::debug >= 5)
      std::cout << "FillMatricesWithMeasurements: measurement " << (*vmcite)->name() << " # entries affecting "
                << (*vmcite)->affectingEntryList().size() << std::endl;

    //-------- Array of derivatives with respect to entries
    ALIint measdim = (*vmcite)->dim();
    std::vector<ALIdouble> derivRE;
    //    derivRE = new ALIdouble[measdim];

    //-------- Fill matrix A:
    //------ Loop only Entries affecting this Measurement
    //-std::cout << "number affecting entries: " << (*vmcite)->affectingEntryList().size() << std::endl;
    for (vecite = (*vmcite)->affectingEntryList().begin(); vecite != (*vmcite)->affectingEntryList().end(); ++vecite) {
      //-------- If good quality, get derivative of measurement with respect to this Entry
      if ((*vecite)->quality() >= theMinimumEntryQuality) {
        if (ALIUtils::debug >= 4) {
          //	  std::cout << "FillMatricesWithMeasurements: filling element ( " << Aline << " - " << Aline+measdim-1 << " , " << (*vecite)->fitPos() << std::endl;
          std::cout << "entry affecting: " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << std::endl;
        }
        derivRE = (*vmcite)->DerivativeRespectEntry(*vecite);
        //---------- Fill matrix A with derivatives
        for (ALIuint jj = 0; jj < ALIuint(measdim); jj++) {
          AMatrix->AddData(Aline + jj, (*vecite)->fitPos(), derivRE[jj]);
          if (ALIUtils::debug >= 5)
            std::cout << "FillMatricesWithMeasurements: AMATRIX (" << Aline + jj << "," << (*vecite)->fitPos() << " = "
                      << derivRE[jj] << std::endl;
          //---------- Reset Measurement simulated_value
          (*vmcite)->setValueSimulated(jj, (*vmcite)->valueSimulated_orig(jj));
        }
      }
    }
    //    delete[] derivRE;

    //---------- Fill matrices W, y and f:
    //------ Loop Measurement coordinates
    for (ALIuint jj = 0; jj < ALIuint((*vmcite)->dim()); jj++) {
      ALIdouble sigma = (*vmcite)->sigma()[jj];
      if (sigma == 0.) {
        std::cerr << "EXITING: Measurement number " << vmcite - Model::MeasurementList().begin() << "has 0 error!!"
                  << std::endl;
      } else {
        //----- Fill W Matrix with inverse of sigma squared
        // multiply error by cameraScaleFactor
        ALIdouble sigmanew = sigma * Measurement::cameraScaleFactor;
        //	std::cout << Aline+jj << " WMATRIX FILLING " << sigmanew << " * " << Measurement::cameraScaleFactor << std::endl;
        WMatrix->AddData(Aline + jj, Aline + jj, (sigmanew * sigmanew));
      }
      //op //----- Fill Matrices y with measurement value
      //op yMatrix->AddData( Aline+jj, 0, (*vmcite)->value()[jj] );
      //op //----- Fill f Matrix with simulated_value
      //op fMatrix->AddData( Aline+jj, 0, (*vmcite)->valueSimulated_orig(jj) );
      //----- Fill Matrix y - f with measurement value - simulated value
      yfMatrix->AddData(Aline + jj, 0, (*vmcite)->value()[jj] - (*vmcite)->valueSimulated_orig(jj));
      //      std::cout << " yfMatrix FILLING " << Aline+jj << " + " << (*vmcite)->value()[jj] - (*vmcite)->valueSimulated_orig(jj) << " meas " << (*vmcite)->name() << " val " << (*vmcite)->value()[jj] << " simu val " << (*vmcite)->valueSimulated_orig(jj) << std::endl;
    }
    if (ALIUtils::debug >= 99)
      std::cout << "change line" << Aline << std::endl;
    Aline += measdim;
    if (ALIUtils::debug >= 99)
      std::cout << "change line" << Aline << std::endl;
  }

  if (ALIUtils::debug >= 4)
    AMatrix->Dump("Matrix A with meas");
  if (ALIUtils::debug >= 4)
    WMatrix->Dump("Matrix W with meas");
  if (ALIUtils::debug >= 4)
    yfMatrix->Dump("Matrix y with meas");
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Loop Measurements:
//@@    Fill Matrix A with derivatives respect to affecting entries
//@@    Fill Matrix W, y & f with values and sigmas of measurement coordinate
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::FillMatricesWithCalibratedParameters() {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::FillMatricesWithCalibratedParameters" << std::endl;

  //---------- Count how many measurements
  ALIint NolinMes = 0;
  std::vector<Measurement*>::const_iterator vmcite;
  for (vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); ++vmcite) {
    NolinMes += (*vmcite)->dim();
  }
  if (ALIUtils::debug >= 4)
    std::cout << "@@FillMatricesWithCalibratedParameters" << std::endl;

  std::vector<Entry*>::const_iterator vecite;
  ALIint nEntcal = 0;
  //---------- Loop entries
  for (vecite = Model::EntryList().begin(); vecite != Model::EntryList().end(); ++vecite) {
    //                  (= No parameters to be fitted - No parameters 'unk' )
    //-    std::cout << "entry" << (*veite) << std::endl;
    //----- Take entries of quality = 'cal'
    if ((*vecite)->quality() == 1) {
      //--- Matrix A: fill diagonals with 1. (derivatives of entry w.r.t itself)
      ALIint lineNo = NolinMes + nEntcal;
      ALIint columnNo = (*vecite)->fitPos();  //=? nEntcal
      AMatrix->AddData(lineNo, columnNo, 1.);
      if (ALIUtils::debug >= 4)
        std::cout << "Fit::FillMatricesWithCalibratedParameters:  AMatrix ( " << lineNo << " , " << columnNo
                  << ") = " << 1. << std::endl;

      //--- Matrix W: sigma*sigma
      ALIdouble entsig = (*vecite)->sigma();
      if (ALIUtils::debug >= 4)
        std::cout << "Fit::FillMatricesWithCalibratedParameters:  WMatrix ( " << lineNo << " , " << columnNo
                  << ") = " << entsig * entsig << std::endl;
      WMatrix->AddData(lineNo, lineNo, entsig * entsig);

      //--- Matrix y & f: fill it with 0.
      //op      yMatrix->AddData( lineNo, 0, (*vecite)->value());
      //op      yfMatrix->AddData( lineNo, 0, (*vecite)->value());
      ALIdouble calFit;
      GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
      gomgr->getGlobalOptionValue("calParamInyfMatrix", calFit);
      if (calFit) {
        yfMatrix->AddData(lineNo, 0, -(*vecite)->valueDisplacementByFitting());
        //-	yfMatrix->AddData( lineNo, 0, (*vecite)->value() );
        //-       	yfMatrix->AddData( lineNo, 0, (*vecite)->lastAdditionToValueDisplacementByFitting() );
        //-	ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
        //	fileout << "cal to yf " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << " " << (*vecite)->valueDisplacementByFitting() << endl;
        //	std::cout << "call to yf " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << " " << (*vecite)->valueDisplacementByFitting() << std::endl;

      } else {
        yfMatrix->AddData(lineNo, 0, 0.);
      }
      //t      if(ALIUtils::debug >= 5) std::cout << "Fit::FillMatricesWithCalibratedParameters:  yfMatrix ( " << lineNo << " , " << columnNo  << ") = " << (*yfMatrix)(lineNo)(0) << std::endl;
      nEntcal++;
    }
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Gets the only instance of Model
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::setCorrelationsInWMatrix() {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::setCorrelationsInWMatrix" << std::endl;

  //----- Check if there are correlations to input
  ErrorCorrelationMgr* corrMgr = ErrorCorrelationMgr::getInstance();
  ALIint siz = corrMgr->getNumberOfCorrelations();
  if (siz == 0)
    return;

  //----- Set correlations
  ALIuint ii;
  for (ii = 0; ii < ALIuint(siz); ii++) {
    //t    if(ALIUtils::debug >= 5) std::cout << "globaloption cmslink fit" << Model::GlobalOptions()["cms_link"] << std::endl;
    ErrorCorrelation* corr = corrMgr->getCorrelation(ii);
    setCorrelationFromParamFitted(corr->getEntry1(), corr->getEntry2(), corr->getCorrelation());
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  set correlation between two entries of two OptOs
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::setCorrelationFromParamFitted(const pss& entry1, const pss& entry2, ALIdouble correl) {
  ALIint pmsize = WMatrix->NoLines();
  ALIint fit_pos1 = Model::getEntryByName(entry1.first, entry1.second)->fitPos();
  ALIint fit_pos2 = Model::getEntryByName(entry2.first, entry2.second)->fitPos();
  std::cout << "CHECKsetCorrelatiFPF " << fit_pos1 << " " << fit_pos2 << std::endl;

  if (fit_pos1 >= 0 && fit_pos1 < pmsize && fit_pos2 >= 0 && fit_pos2 < pmsize) {
    setCorrelationFromParamFitted(fit_pos1, fit_pos2, correl);
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::setCorrelationFromParamFitted(const ALIint fit_pos1, const ALIint fit_pos2, ALIdouble correl) {
  //  ALIdouble error1 = sqrt( (*WMatrix)(fit_pos1, fit_pos1) );
  // ALIdouble error2 = sqrt( (*WMatrix)(fit_pos2, fit_pos2) );
  WMatrix->SetCorrelation(fit_pos1, fit_pos2, correl);
  std::cout << "setCorrelatiFPF " << fit_pos1 << " " << fit_pos2 << " " << correl << std::endl;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  multiply matrices needed for fit
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::multiplyMatrices() {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::multiplyMatrices " << std::endl;
  //---------- Calculate transpose of A
  AtMatrix = new ALIMatrix(*AMatrix);
  if (ALIUtils::debug >= 5)
    AtMatrix->Dump("AtMatrix=A");
  //-  std::cout << "call transpose";
  AtMatrix->transpose();
  if (ALIUtils::debug >= 4)
    AtMatrix->Dump("AtMatrix");

  //---------- Calculate At * W * A
  AtWAMatrix = new ALIMatrix(0, 0);
  //  if(ALIUtils::debug >= 5) AtWAMatrix->Dump("AtWAMatrix=0");
  *AtWAMatrix = *AtMatrix * *WMatrix * *AMatrix;
  if (ALIUtils::debug >= 5)
    AtWAMatrix->Dump("AtWAMatrix");

  CheckIfFitPossible();

  //t  AtWAMatrix->EliminateLines(0,48);
  //t AtWAMatrix->EliminateColumns(0,48);
  time_t now;
  now = clock();
  if (ALIUtils::debug >= 0)
    std::cout << "TIME:BEFORE_INVERSE : " << now << " " << difftime(now, ALIUtils::time_now()) / 1.E6 << std::endl;
  ALIUtils::set_time_now(now);

  /*  std::cout << " DETERMINANT W " <<  m_norm1( AtWAMatrix->MatNonConst() ) << std::endl;
  if( m_norm1( AtWAMatrix->MatNonConst() ) == 0 ) {
    std::cout << " DETERMINANT W " <<  m_norm1( AtWAMatrix->MatNonConst() ) << std::endl;
    std::exception();
    } */

  AtWAMatrix->inverse();
  if (ALIUtils::debug >= 4)
    AtWAMatrix->Dump("inverse AtWAmatrix");
  now = clock();
  if (ALIUtils::debug >= 0)
    std::cout << "TIME:AFTER_INVERSE  : " << now << " " << difftime(now, ALIUtils::time_now()) / 1.E6 << std::endl;
  ALIUtils::set_time_now(now);

  //op  thePropagationMatrix = AtWAMatrix;

  //op  VaMatrix = new ALIMatrix( *AtWAMatrix );

  //----- Print out propagated errors of parameters (=AtWA diagonal elements)
  std::vector<Entry*>::const_iterator vecite;

  if (ALIUtils::debug >= 4) {
    std::cout << "PARAM"
              << "        Optical Object "
              << "   entry name "
              << "      Param.Value "
              << " Prog.Error"
              << " Orig.Error" << std::endl;
  }

  ALIint nEnt = 0;
  for (vecite = Model::EntryList().begin(); vecite != Model::EntryList().end(); ++vecite) {
    //------------------ Number of parameters 'cal'
    //                  (= No parameters to be fitted - No parameters 'unk' )
    if ((*vecite)->quality() >= theMinimumEntryQuality) {
      if (ALIUtils::debug >= 4) {
        std::cout << nEnt << "PARAM" << std::setw(26) << (*vecite)->OptOCurrent()->name().c_str() << std::setw(8) << " "
                  << (*vecite)->name().c_str() << " " << std::setw(8) << " " << (*vecite)->value() << " "
                  << std::setw(8) << sqrt(AtWAMatrix->Mat()->me[nEnt][nEnt]) / (*vecite)->OutputSigmaDimensionFactor()
                  << " " << (*vecite)->sigma() / (*vecite)->OutputSigmaDimensionFactor() << " Q" << (*vecite)->quality()
                  << std::endl;
      }
      nEnt++;
    }
  }

  if (ALIUtils::debug >= 5)
    yfMatrix->Dump("PD(y-f)Matrix final");
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ check also that the fit_quality = SMat(0,0) is smaller for each new iteration
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
FitQuality Fit::getFitQuality(const ALIbool canBeGood) {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::getFitQuality" << std::endl;

  double fit_quality = GetSChi2(true);

  //---------- Calculate DS = Variable to recognize convergence (distance to minimum)
  ALIMatrix* DatMatrix = new ALIMatrix(*DaMatrix);
  //  delete DaMatrix; //op
  DatMatrix->transpose();
  if (ALIUtils::debug >= 5)
    DatMatrix->Dump("DatMatrix");
  //op  ALIMatrix* DSMat = new ALIMatrix(*DatMatrix * *AtMatrix * *WMatrix * *PDMatrix);
  ALIMatrix* DSMat = new ALIMatrix(*DatMatrix * *AtMatrix * *WMatrix * *yfMatrix);
  if (ALIUtils::debug >= 5) {
    ALIMatrix* DSMattemp = new ALIMatrix(*DatMatrix * *AtMatrix * *WMatrix);
    DSMattemp->Dump("DSMattempMatrix=Dat*At*W");
    ALIMatrix* DSMattemp2 = new ALIMatrix(*AtMatrix * *WMatrix * *yfMatrix);
    DSMattemp2->Dump("DSMattempMatrix2=At*W*yf");
    ALIMatrix* DSMattemp3 = new ALIMatrix(*AtMatrix * *WMatrix);
    DSMattemp3->Dump("DSMattempMatrix3=At*W");
    AtMatrix->Dump("AtMatrix");
  }

  /*  for( int ii = 0; ii < DatMatrix->NoColumns(); ii++ ){
    std::cout << ii << " DS term " << (*DatMatrix)(0,ii) * (*DSMattemp2)(ii,0) << std::endl;
    }*/
  //  delete AtMatrix; //op
  //  delete WMatrix; //op

  //op  if(ALIUtils::debug >= 5) (*PDMatrix).Dump("PDMatrix");
  if (ALIUtils::debug >= 5)
    (*yfMatrix).Dump("yfMatrix");
  if (ALIUtils::debug >= 5)
    DSMat->Dump("DSMatrix final");
  //  delete yfMatrix; //op

  ALIdouble fit_quality_cut = (*DSMat)(0, 0);
  //-  ALIdouble fit_quality_cut =std::abs( (*DSMat)(0,0) );
  delete DSMat;
  if (ALIUtils::debug >= 0)
    std::cout << theNoFitIterations
              << " Fit quality predicted improvement in distance to minimum is = " << fit_quality_cut << std::endl;
  if (ALIUtils::report >= 2) {
    ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
    fileout << theNoFitIterations
            << " Fit quality predicted improvement in distance to minimum is = " << fit_quality_cut << std::endl;
  }

  //-  double fit_quality_cut = thePreviousIterationFitQuality - fit_quality;
  //- double fit_quality_cut = fit_quality;
  //-  std::cout << "  fit_quality_cut " <<  fit_quality_cut << " fit_quality " << fit_quality << std::endl;

  //----- Check quality
  time_t now;
  now = clock();
  if (ALIUtils::debug >= 0)
    std::cout << "TIME:QUALITY_CHECKED: " << now << " " << difftime(now, ALIUtils::time_now()) / 1.E6 << std::endl;
  ALIUtils::set_time_now(now);

  FitQuality fitQuality;

  //----- Chi2 is bigger, bad
  //    if( theNoFitIterations != 0 && fit_quality_cut > 0. ) {
  if (fit_quality_cut < 0.) {
    fitQuality = FQchiSquareWorsened;
    if (ALIUtils::debug >= 1)
      std::cerr << "!!WARNING: Fit quality has worsened: Fit Quality now = " << fit_quality << " before "
                << thePreviousIterationFitQuality << " diff " << fit_quality - thePreviousIterationFitQuality
                << std::endl;

    //----- Chi2 is smaller, check if we make another iteration
  } else {
    ALIdouble rel_fit_quality = std::abs(thePreviousIterationFitQuality - fit_quality) / fit_quality;
    //----- Small chi2 change: end
    if ((fit_quality_cut < theFitQualityCut || rel_fit_quality < theRelativeFitQualityCut) && canBeGood) {
      if (ALIUtils::debug >= 2)
        std::cout << "$$ Fit::getFitQuality good " << fit_quality_cut << " <? " << theFitQualityCut << " || "
                  << rel_fit_quality << " <? " << theRelativeFitQualityCut << " GOOD " << canBeGood << std::endl;
      fitQuality = FQsmallDistanceToMinimum;
      if (ALIUtils::report >= 1) {
        ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
        fileout << "STOP: SMALL IMPROVEMENT IN ITERATION " << theNoFitIterations << " = " << fit_quality_cut << " < "
                << theFitQualityCut << " OR (RELATIVE) " << rel_fit_quality << " < " << theRelativeFitQualityCut
                << std::endl;
      }
      if (ALIUtils::debug >= 4) {
        std::cout << "STOP: SMALL IMPROVEMENT IN ITERATION " << theNoFitIterations << " = " << fit_quality_cut << " < "
                  << theFitQualityCut << " OR (RELATIVE) " << rel_fit_quality << " < " << theRelativeFitQualityCut
                  << std::endl;
      }

      //----- Big chi2 change: go to next iteration
    } else {
      if (ALIUtils::debug >= 2)
        std::cout << "$$ Fit::getFitQuality bad " << fit_quality_cut << " <? " << theFitQualityCut << " || "
                  << rel_fit_quality << " <? " << theRelativeFitQualityCut << " GOOD " << canBeGood << std::endl;
      fitQuality = FQbigDistanceToMinimum;
      //----- set thePreviousIterationFitQuality for next iteration
      thePreviousIterationFitQuality = fit_quality;

      if (ALIUtils::report >= 2) {
        ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
        fileout << "CONTINUE: BIG IMPROVEMENT IN ITERATION " << theNoFitIterations << " = " << fit_quality_cut
                << " >= " << theFitQualityCut << " AND (RELATIVE) " << rel_fit_quality
                << " >= " << theRelativeFitQualityCut << std::endl;
      }
      if (ALIUtils::debug >= 4) {
        std::cout << "CONTINUE: BIG IMPROVEMENT IN ITERATION " << theNoFitIterations << " = " << fit_quality_cut
                  << " >= " << theFitQualityCut << " AND (RELATIVE) " << rel_fit_quality
                  << " >= " << theRelativeFitQualityCut << std::endl;
      }
    }
  }

  return fitQuality;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble Fit::GetSChi2(ALIbool useDa) {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::GetSChi2  useDa= " << useDa << std::endl;

  ALIMatrix* SMat = nullptr;
  if (useDa) {
    //----- Calculate variables to check quality of this set of parameters

    //----- Calculate Da = (At * W * A)-1 * At * W * (y-f)
    /*t  DaMatrix = new ALIMatrix( *AtWAMatrix );
     *DaMatrix *= *AtMatrix * *WMatrix;
     if(ALIUtils::debug >= 5) DaMatrix->Dump("DaMatrix before yf ");
     *DaMatrix *= *yfMatrix;
     if(ALIUtils::debug >= 5) DaMatrix->Dump("DaMatrix");
    */

    DaMatrix = new ALIMatrix(0, 0);
    //  if(ALIUtils::debug >= 5) AtWAMatrix->Dump("AtWAMatrix=0");
    *DaMatrix = (*AtWAMatrix * *AtMatrix * *WMatrix * *yfMatrix);
    if (ALIUtils::debug >= 5)
      DaMatrix->Dump("DaMatrix");

    //----- Calculate S = chi2 = Fit quality = r^T W r (r = residual = f + A*Da - y )
    //op  ALIMatrix* tmpM = new ALIMatrix( *AMatrix * *DaMatrix + *PDMatrix );
    //  ALIMatrix* tmpM = new ALIMatrix( *AMatrix * *DaMatrix + *yfMatrix );

    ALIMatrix* tmpM = new ALIMatrix(0, 0);
    *tmpM = *AMatrix * *DaMatrix - *yfMatrix;
    if (ALIUtils::debug >= 5)
      tmpM->Dump("A*Da + f - y Matrix ");

    ALIMatrix* tmptM = new ALIMatrix(*tmpM);
    tmptM->transpose();
    if (ALIUtils::debug >= 5)
      tmptM->Dump("tmptM after transpose");
    if (ALIUtils::debug >= 5)
      WMatrix->Dump("WMatrix");

    //  std::cout << "smat " << std::endl;
    //o  ALIMatrix* SMat = new ALIMatrix(*tmptM * *WMatrix * *tmpM);
    ALIMatrix* SMat1 = MatrixByMatrix(*tmptM, *WMatrix);
    //  ALIMatrix* SMat1 = MatrixByMatrix(*AMatrix,*WMatrix);
    if (ALIUtils::debug >= 5)
      SMat1->Dump("(A*Da + f - y)^T * W  Matrix");
    SMat = MatrixByMatrix(*SMat1, *tmpM);
    //  std::cout << "smatc " << std::endl;
    delete tmpM;
    delete tmptM;
    if (ALIUtils::debug >= 5)
      SMat->Dump("SMatrix with Da");
  } else {
    ALIMatrix* yftMat = new ALIMatrix(*yfMatrix);
    yftMat->transpose();
    SMat = new ALIMatrix(*yftMat * *WMatrix * *yfMatrix);
    delete yftMat;
    if (ALIUtils::debug >= 5)
      SMat->Dump("SMatrix no Da");
  }
  ALIdouble fit_quality = (*SMat)(0, 0);
  delete SMat;
  if (ALIUtils::debug >= 5)
    std::cout << " GetSChi2 = " << fit_quality << std::endl;

  PrintChi2(fit_quality, !useDa);

  return fit_quality;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Correct entries with fitted values
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::addDaMatrixToEntries() {
  if (ALIUtils::debug >= 4) {
    std::cout << "@@ Adding correction (DaMatrix) to Entries " << std::endl;
    DaMatrix->Dump("DaMatrix =");
  }

  /*-  Now there are other places where entries are changed
    if( ALIUtils::report >= 3 ) {
    ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
    fileout << std::endl << " CHANGE IN ENTRIES" << std::endl
            << "          Optical Object       Parameter   xxxxxx " << std::endl;
	    }*/

  ALIint nEnt = 0;
  std::vector<Entry*>::const_iterator vecite;
  for (vecite = Model::EntryList().begin(); vecite != Model::EntryList().end(); ++vecite) {
    //------------------ Number of parameters 'cal'
    //                  (= No parameters to be fitted - No parameters 'unk' )
    //-   std::cout << "qual" << (*vecite)->quality() << theMinimumEntryQuality << std::endl;
    if ((*vecite)->quality() >= theMinimumEntryQuality) {
      if (ALIUtils::debug >= 5) {
        std::cout << std::endl
                  << " @@@ PENTRY change " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << " "
                  << " change= " << (*DaMatrix)(nEnt, 0) << " value= " << (*vecite)->valueDisplacementByFitting()
                  << std::endl;
      }
      /*      if( ALIUtils::report >=3 ) {
        ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
        fileout << "dd" << std::setw(30) << (*vecite)->OptOCurrent()->name() 
                << std::setw(8) << " " << (*vecite)->name() << " " 
                << std::setw(8) << " " << (*DaMatrix)(nEnt,0) / (*vecite)->OutputValueDimensionFactor()
	        << " " << (*vecite)->valueDisplacementByFitting() / (*vecite)->OutputValueDimensionFactor() << " fitpos " << (*vecite)->fitPos()
                << std::endl;
		}*/

      //----- Store this displacement
      if (ALIUtils::debug >= 4)
        std::cout << " old valueDisplacementByFitting " << (*vecite)->name() << " "
                  << (*vecite)->valueDisplacementByFitting() << " original value " << (*vecite)->value() << std::endl;

      (*vecite)->addFittedDisplacementToValue((*DaMatrix)(nEnt, 0));

      if (ALIUtils::debug >= 4)
        std::cout << nEnt << " new valueDisplacementByFitting " << (*vecite)->OptOCurrent()->name() << " "
                  << (*vecite)->name() << " = " << (*vecite)->valueDisplacementByFitting() << " "
                  << (*DaMatrix)(nEnt, 0) << std::endl;
      nEnt++;
    }
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Delete the previous addition of fitted values (to try a new one daFactor times smaller )
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::substractLastDisplacementToEntries(const ALIdouble factor) {
  if (ALIUtils::debug >= 4) {
    std::cout << "@@  Fit::substractToHalfDaMatrixToEntries " << std::endl;
  }

  std::vector<Entry*>::const_iterator vecite;
  for (vecite = Model::EntryList().begin(); vecite != Model::EntryList().end(); ++vecite) {
    if ((*vecite)->quality() >= theMinimumEntryQuality) {
      //--     (*vecite)->addFittedDisplacementToValue( -(*DaMatrix)(nEnt,0) );!!! it is not substracting the new value of DaMatrix, but substracting the value that was added last iteration, with which the new value of DaMatrix has been calculated for this iteration

      ALIdouble lastadd = (*vecite)->lastAdditionToValueDisplacementByFitting() * factor;
      //-      if( lastadd < 0 ) lastadd *= -1;
      (*vecite)->addFittedDisplacementToValue(-lastadd);
      (*vecite)->setLastAdditionToValueDisplacementByFitting(-(*vecite)->lastAdditionToValueDisplacementByFitting());
      //      (*vecite)->substractToHalfFittedDisplacementToValue();

      if (ALIUtils::debug >= 4)
        std::cout << " new valueDisplacementByFitting " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name()
                  << " = " << (*vecite)->valueDisplacementByFitting() << " " << std::endl;
    }
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Dump all the entries that have been fitted (those that were 'cal' or 'unk'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::dumpFittedValues(ALIFileOut& fileout, ALIbool printErrors, ALIbool printOrig) {
  //---------- print
  if (ALIUtils::debug >= 0) {
    std::cout << "SRPRPOS "
              << "               Optical Object  "
              << "      Parameter"
              << " Fit.Value "
              << " Orig.Value" << std::endl;
  }
  //---------- Dump header
  if (ALIUtils::debug >= 0)
    std::cout << std::endl << "FITTED VALUES " << std::endl;
  fileout << std::endl
          << "FITTED VALUES " << std::endl
          << "nEnt_unk"
          << "             Optical Object"
          << "        Parameter  ";
  if (printErrors) {
    fileout << " value (+-error)"
            << " orig.val (+-error)";
  } else {
    fileout << " value "
            << " orig.val ";
  }
  fileout << " quality" << std::endl;

  //---------- Iterate over OptO list
  std::vector<Entry*> entries;
  //  const Entry* entry;
  int ii, siz;
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = Model::OptOList().begin(); vocite != Model::OptOList().end(); ++vocite) {
    if ((*vocite)->type() == ALIstring("system"))
      continue;

    fileout << " %%%% Optical Object: " << (*vocite)->longName() << std::endl;

    entries = (*vocite)->CoordinateEntryList();
    siz = entries.size();
    if (siz != 6) {
      std::cerr << "!!! FATAL ERROR: strange number of coordinates = " << siz << std::endl;
      abort();
    }

    //----- Dump entry centre coordinates (centre in current coordinates of parent frame <> summ of displacements, as each displacement is done with a different rotation of parent frame)
    OpticalObject* opto = entries[0]->OptOCurrent();
    const OpticalObject* optoParent = opto->parent();
    printCentreInOptOFrame(opto, optoParent, fileout, printErrors, printOrig);

    //----- Dump entry angles coordinates
    printRotationAnglesInOptOFrame(opto, optoParent, fileout, printErrors, printOrig);

    //----- Dump extra entries
    entries = (*vocite)->ExtraEntryList();
    siz = entries.size();
    for (ii = 0; ii < siz; ii++) {
      double entryvalue = getEntryValue(entries[ii]);
      dumpEntryAfterFit(fileout, entries[ii], entryvalue, printErrors, printOrig);
    }
  }

  dumpEntryCorrelations(fileout);
}
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Dump all the entries that have been fitted in reference frames of all ancestors
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::dumpFittedValuesInAllAncestorFrames(ALIFileOut& fileout, ALIbool printErrors, ALIbool printOrig) {
  //---------- print
  fileout << std::endl
          << "@@@@ FITTED VALUES IN ALL ANCESTORS " << std::endl
          << "nEnt_unk"
          << "             Optical Object"
          << "        Parameter  ";
  if (printErrors) {
    fileout << " value (+-error)";
    if (printOrig) {
      fileout << " orig.val (+-error)";
    }
  } else {
    fileout << " value ";
    if (printOrig) {
      fileout << " orig.val ";
    }
  }
  fileout << " quality" << std::endl;

  //---------- Iterate over OptO list
  std::vector<Entry*> entries;
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = Model::OptOList().begin(); vocite != Model::OptOList().end(); ++vocite) {
    if ((*vocite)->type() == ALIstring("system"))
      continue;

    fileout << " %%%% Optical Object: " << (*vocite)->longName() << std::endl;

    entries = (*vocite)->CoordinateEntryList();

    //----- Dump entry centre coordinates (centre in current coordinates of parent frame <> summ of displacements, as each displacement is done with a different rotation of parent frame)
    OpticalObject* opto = *vocite;
    const OpticalObject* optoParent = opto->parent();
    do {
      fileout << " %% IN FRAME : " << optoParent->longName() << std::endl;
      printCentreInOptOFrame(opto, optoParent, fileout, printErrors, printOrig);

      //----- Dump entry angles coordinates
      printRotationAnglesInOptOFrame(opto, optoParent, fileout, printErrors, printOrig);
      optoParent = optoParent->parent();
    } while (optoParent);
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::printCentreInOptOFrame(const OpticalObject* opto,
                                 const OpticalObject* optoAncestor,
                                 ALIFileOut& fileout,
                                 ALIbool printErrors,
                                 ALIbool printOrig) {
  CLHEP::Hep3Vector centreLocal;
  if (optoAncestor->type() == "system") {
    centreLocal = opto->centreGlob();
  } else {
    centreLocal = opto->centreGlob() - optoAncestor->centreGlob();
    CLHEP::HepRotation parentRmGlobInv = inverseOf(optoAncestor->rmGlob());
    centreLocal = parentRmGlobInv * centreLocal;
  }
  if (ALIUtils::debug >= 2) {
    std::cout << "CENTRE LOCAL " << opto->name() << " " << centreLocal << " GLOBL " << opto->centreGlob()
              << " parent GLOB " << optoAncestor->centreGlob() << std::endl;
    ALIUtils::dumprm(optoAncestor->rmGlob(), " parent rm ");
  }
  std::vector<Entry*> entries = opto->CoordinateEntryList();
  for (ALIuint ii = 0; ii < 3; ii++) {
    /* double entryvalue = getEntryValue( entries[ii] );
       ALIdouble entryvalue;
       if( ii == 0 ) {
       entryvalue = centreLocal.x();
       }else if( ii == 1 ) {
       entryvalue = centreLocal.y();
       }else if( ii == 2 ) {
       entryvalue = centreLocal.z();
       }*/
    dumpEntryAfterFit(
        fileout, entries[ii], centreLocal[ii] / entries[ii]->OutputValueDimensionFactor(), printErrors, printOrig);
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::printRotationAnglesInOptOFrame(const OpticalObject* opto,
                                         const OpticalObject* optoAncestor,
                                         ALIFileOut& fileout,
                                         ALIbool printErrors,
                                         ALIbool printOrig) {
  std::vector<Entry*> entries = opto->CoordinateEntryList();
  std::vector<double> entryvalues = opto->getRotationAnglesInOptOFrame(optoAncestor, entries);
  //-    std::cout << " after return entryvalues[0] " << entryvalues[0] << " entryvalues[1] " << entryvalues[1] << " entryvalues[2] " << entryvalues[2] << std::endl;
  for (ALIuint ii = 3; ii < entries.size(); ii++) {
    dumpEntryAfterFit(
        fileout, entries[ii], entryvalues[ii - 3] / entries[ii]->OutputValueDimensionFactor(), printErrors, printOrig);
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
double Fit::getEntryValue(const Entry* entry) {
  double entryvalue;
  if (entry->type() == "angles") {
    if (ALIUtils::debug >= 2)
      std::cout << "WARNING valueDisplacementByFitting has no sense for angles " << std::endl;

    // commenting out the following line as it is a dead assignment due to the
    // subsequent assignment below
    // -> silence static analyzer warnings, but leaving the commented line in
    //    case someone wants to actively use this code again

    // entryvalue = -999;
  }
  entryvalue = (entry->value() + entry->valueDisplacementByFitting()) / entry->OutputValueDimensionFactor();
  return entryvalue;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::dumpEntryAfterFit(
    ALIFileOut& fileout, const Entry* entry, double entryvalue, ALIbool printErrors, ALIbool printOrig) {
  //-  std::cout << " Fit::dumpEntryAfterFit " << entryvalue << std::endl;
  ALIdouble dimv = entry->OutputValueDimensionFactor();
  ALIdouble dims = entry->OutputSigmaDimensionFactor();
  //----- Dump to std::cout
  if (ALIUtils::debug >= 3) {
    std::cout << "ENTRY: " << std::setw(30) << entry->OptOCurrent()->name() << std::setw(8) << " " << entry->name()
              << " " << std::setw(8) << (entry->value() + entry->valueDisplacementByFitting()) << " " << entry->value()
              << " Q" << entry->quality() << std::endl;
  }

  if (entry->quality() == 2) {
    fileout << "UNK: " << entry->fitPos() << " ";
  } else if (entry->quality() == 1) {
    fileout << "CAL: " << entry->fitPos() << " ";
    //    fileout << "CAL: -1 ";
  } else {
    fileout << "FIX: -1 ";
  }

  fileout << std::setw(30) << entry->OptOCurrent()->name() << std::setw(8) << " " << entry->name() << " "
          << std::setw(8) << std::setprecision(8) << entryvalue;
  if (entry->quality() >= theMinimumEntryQuality) {
    if (printErrors)
      fileout << " +- " << std::setw(8) << sqrt(AtWAMatrix->Mat()->me[entry->fitPos()][entry->fitPos()]) / dims;
  } else {
    if (printErrors)
      fileout << " +- " << std::setw(8) << 0.;
  }
  if (printOrig) {
    fileout << std::setw(8) << " " << entry->value() / dimv;
    if (printErrors)
      fileout << " +- " << std::setw(8) << entry->sigma() / dims << " Q" << entry->quality();

    if (ALIUtils::report >= 2) {
      float dif = (entry->value() + entry->valueDisplacementByFitting()) / dimv - entry->value() / dimv;
      if (std::abs(dif) < 1.E-9)
        dif = 0.;
      fileout << " DIFF= " << dif;
      // << " == " << ( entry->value() + entry->valueDisplacementByFitting() )  / dimv - entryvalue << " @@ " << ( entry->value() + entry->valueDisplacementByFitting() ) / dimv << " @@ " <<  entryvalue;
    } else {
      //	fileout << std::endl;
    }
  }

  fileout << std::endl;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::dumpEntryCorrelations(ALIFileOut& fileout) {
  //----- Only dump correlations bigger than a factor
  ALIdouble minCorrel = 1.E-6;
  //----- Dump correlations
  fileout << std::endl
          << "CORRELATION BETWEEN 'unk' ENTRIES: (>= " << minCorrel << " )" << std::endl
          << "No_1  No_2   correlation " << std::endl;

  ALIuint i1, i2;
  std::vector<Entry*>::iterator veite1, veite2;
  std::string E1, E2;
  for (veite1 = Model::EntryList().begin(); veite1 != Model::EntryList().end(); ++veite1) {
    if ((*veite1)->quality() == 0) {
      continue;
    } else if ((*veite1)->quality() == 1) {
      E1 = "C";
    } else if ((*veite1)->quality() == 2) {
      E1 = "U";
    }
    i1 = (*veite1)->fitPos();

    for (veite2 = veite1 + 1; veite2 != Model::EntryList().end(); ++veite2) {
      i2 = (*veite2)->fitPos();
      if ((*veite2)->quality() == 0) {
        continue;
      } else if ((*veite2)->quality() == 1) {
        E2 = "C";
      } else if ((*veite2)->quality() == 2) {
        E2 = "U";
      }
      ALIdouble corr = AtWAMatrix->Mat()->me[i1][i2];
      ALIdouble corrf = corr / sqrt(AtWAMatrix->Mat()->me[i1][i1]) / sqrt(AtWAMatrix->Mat()->me[i2][i2]);
      if (std::abs(corrf) >= minCorrel) {
        if (ALIUtils::debug >= 0) {
          std::cout << "CORR:" << E1 << "" << E2 << " (" << i1 << ")"
                    << " (" << i2 << ")"
                    << " " << corrf << std::endl;
        }
        fileout << "CORR:" << E1 << "" << E2 << " (" << i1 << ")"
                << " (" << i2 << ")"
                << " " << corrf << std::endl;
      }
    }
  }
  //------- Dump optical object list
  if (ALIUtils::debug >= 2)
    OpticalObjectMgr::getInstance()->dumpOptOs();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Dump matrices used for the fit
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::dumpMatrices() {
  //----- Dump matrices for this iteration
  ALIFileOut& matout = ALIFileOut::getInstance(Model::MatricesFName());
  //  ofstream matout("matrices.out");
  matout << std::endl << " @@@@@@@@@@@@@@@  Iteration No : " << theNoFitIterations << std::endl;
  AMatrix->ostrDump(matout, "Matrix A");
  AtMatrix->ostrDump(matout, "Matrix At");
  WMatrix->ostrDump(matout, "Matrix W");
  AtWAMatrix->ostrDump(matout, "Matrix AtWA");
  //op  VaMatrix->ostrDump( matout, "Matrix Va" );
  DaMatrix->ostrDump(matout, "Matrix Da");
  yfMatrix->ostrDump(matout, "Matrix y");
  //op  fMatrix->ostrDump( matout, "Matrix f" );
  //op  thePropagationMatrix->ostrDump( matout, "propagation Matrix " );
  matout.close();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ findEntryFitPosition
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIint Fit::findEntryFitPosition(const ALIstring& opto_name, const ALIstring& entry_name) {
  ALIint fitposi = -99;

  OpticalObject* opto = Model::getOptOByName(opto_name);
  //-  std::cout << "OPTO = " << opto->name() << std::endl;
  std::vector<Entry*>::const_iterator vecite;
  for (vecite = opto->CoordinateEntryList().begin(); vecite < opto->CoordinateEntryList().end(); ++vecite) {
    //-    std::cout << "ENTRYLIST" << (*vecite)->name() << std::endl;
    if ((*vecite)->name() == entry_name) {
      //-  std::cout << "FOUND " << std::endl;
      fitposi = (*vecite)->fitPos();
    }
  }
  for (vecite = opto->ExtraEntryList().begin(); vecite < opto->ExtraEntryList().end(); ++vecite) {
    //-    std::cout << "ENTRYLIST" << (*vecite)->name() << std::endl;
    if ((*vecite)->name() == entry_name) {
      //- std::cout << "FOUND " << std::endl;
      fitposi = (*vecite)->fitPos();
    }
  }

  if (fitposi == -99) {
    std::cerr << "!!EXITING: entry name not found: " << entry_name << std::endl;
    exit(2);
  } else {
    return fitposi;
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::PrintChi2(ALIdouble fit_quality, ALIbool isFirst) {
  double chi2meas = 0;
  double chi2cal = 0;
  ALIint nMeas = 0, nUnk = 0;

  //----- Calculate the chi2 of measurements
  std::vector<Measurement*>::const_iterator vmcite;
  for (vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); ++vmcite) {
    //--- Calculate Simulated Value Original
    for (ALIuint ii = 0; ii < ALIuint((*vmcite)->dim()); ii++) {
      nMeas++;
      double c2 = ((*vmcite)->value(ii) - (*vmcite)->valueSimulated(ii)) / (*vmcite)->sigma(ii);
      chi2meas += c2 * c2;
      if (ALIUtils::debug >= 2) {
        std::cout << c2 << " adding chi2meas " << chi2meas << " " << (*vmcite)->name() << ": " << ii
                  << " (mm)R: " << (*vmcite)->value(ii) * 1000. << " S: " << (*vmcite)->valueSimulated(ii) * 1000.
                  << " Diff= " << ((*vmcite)->value(ii) - (*vmcite)->valueSimulated(ii)) * 1000. << std::endl;
      }
    }
  }

  //----- Calculate the chi2 of calibrated parameters
  std::vector<Entry*>::iterator veite;
  for (veite = Model::EntryList().begin(); veite != Model::EntryList().end(); ++veite) {
    if ((*veite)->quality() == 2)
      nUnk++;
    if ((*veite)->quality() == 1) {
      double c2 = (*veite)->valueDisplacementByFitting() / (*veite)->sigma();
      //double c2 = (*veite)->value() / (*veite)->sigma();
      chi2cal += c2 * c2;
      if (ALIUtils::debug >= 2)
        std::cout << c2 << " adding chi2cal " << chi2cal << " " << (*veite)->OptOCurrent()->name() << " "
                  << (*veite)->name() << std::endl;
      //-	std::cout << " valueDisplacementByFitting " << (*veite)->valueDisplacementByFitting() << " sigma " << (*veite)->sigma() << std::endl;
    }
  }

  if (ALIUtils::report >= 1) {
    ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
    fileout << " Chi2= " << chi2meas + chi2cal << " / " << nMeas - nUnk << " dof "
            << "  From measurements= " << chi2meas << " from calibrated parameters= " << chi2cal << std::endl;
  }
  if (ALIUtils::debug >= 3)
    std::cout << " quality Chi2 (no correlations) " << chi2meas + chi2cal << " " << chi2meas << " " << chi2cal
              << std::endl;

  if (!isFirst) {
    //    double fit_quality_change = thePreviousIterationFitQuality - fit_quality;

    if (ALIUtils::debug >= 0) {
      std::cout << std::endl << "@@@@ Fit iteration " << theNoFitIterations << " ..." << std::endl;
      //      std::cout << theNoFitIterations << " Chi2 improvement in this iteration = " << fit_quality_change << std::endl;
    }
    if (ALIUtils::report >= 1) {
      ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
      fileout << std::endl << "Fit iteration " << theNoFitIterations << " ..." << std::endl;
      //      fileout << theNoFitIterations << " Chi2 improvement in this iteration = " << fit_quality_change << std::endl;
    }
  }

  //---- Print chi2
  if (ALIUtils::debug >= 0)
    std::cout << theNoFitIterations << " Chi2 after iteration = " << fit_quality << std::endl;
  if (ALIUtils::report >= 1) {
    //--------- Get report file handler
    ALIFileOut& fileout = ALIFileOut::getInstance(Model::ReportFName());
    fileout << theNoFitIterations << " Chi2 after iteration = " << fit_quality << std::endl;
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::CheckIfFitPossible() {
  if (ALIUtils::debug >= 3)
    std::cout << "@@@ Fit::CheckIfFitPossible" << std::endl;

  //----- Check if there is an unknown parameter that is not affecting any measurement
  ALIint NolinMes = 0;
  std::vector<Measurement*>::const_iterator vmcite;
  for (vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); ++vmcite) {
    NolinMes += (*vmcite)->dim();
  }

  std::vector<Entry*>::const_iterator vecite;
  for (vecite = Model::EntryList().begin(); vecite != Model::EntryList().end(); ++vecite) {
    if (ALIUtils::debug >= 4)
      std::cout << "Fit::CheckIfFitPossible looping for entry " << (*vecite)->longName() << std::endl;
    if ((*vecite)->quality() == 2) {
      ALIint nCol = (*vecite)->fitPos();
      //--- Check all measurements
      ALIbool noDepend = TRUE;
      if (ALIUtils::debug >= 4)
        std::cout << "Fit::CheckIfFitPossible looping for entry " << nCol << std::endl;
      for (ALIint ii = 0; ii < NolinMes; ii++) {
        if (ALIUtils::debug >= 5)
          std::cout << " Derivative= (" << ii << "," << nCol << ") = " << (*AMatrix)(ii, nCol) << std::endl;

        if (std::abs((*AMatrix)(ii, nCol)) > ALI_DBL_MIN) {
          if (ALIUtils::debug >= 5)
            std::cout << "Fit::CheckIfFitIsPossible " << nCol << " " << ii << " = " << (*AMatrix)(ii, nCol)
                      << std::endl;
          noDepend = FALSE;
          break;
        }
      }
      if (noDepend) {
        throw cms::Exception("SDFError")
            << "!!!FATAL ERROR: Fit::CheckIfFitPossible() no measurement depends on unknown entry "
            << (*vecite)->OptOCurrent()->name() << "/" << (*vecite)->name() << std::endl
            << "!!! Fit will not be possible! " << std::endl;
      }
    }
  }

  //------ Check if there are two unknown entries that have the derivatives of all measurements w.r.t. then equal (or 100% proportional). In this case any value of the first entry can be fully compensated by another value in the second entry without change in any measurement ---> the two entries cannot be fitted!

  std::vector<Entry*>::const_iterator vecite1, vecite2;
  ALIint nLin = AMatrix->NoLines();
  ALIdouble derivPrec = ALI_DBL_MIN;
  //---------- Loop entries
  for (vecite1 = Model::EntryList().begin(); vecite1 != Model::EntryList().end(); ++vecite1) {
    if ((*vecite1)->quality() == 2) {
      vecite2 = vecite1;
      ++vecite2;
      for (; vecite2 != Model::EntryList().end(); ++vecite2) {
        if ((*vecite2)->quality() == 2) {
          ALIint fitpos1 = (*vecite1)->fitPos();
          ALIint fitpos2 = (*vecite2)->fitPos();
          if (ALIUtils::debug >= 5)
            std::cout << "Fit::CheckIfFitIsPossible checking " << (*vecite1)->longName() << " ( " << fitpos1 << " ) & "
                      << (*vecite2)->longName() << " ( " << fitpos2 << " ) " << std::endl;
          ALIdouble prop = DBL_MAX;
          ALIbool isProp = TRUE;
          for (ALIint ii = 0; ii < nLin; ii++) {
            if (ALIUtils::debug >= 5)
              std::cout << "Fit::CheckIfFitIsPossible " << ii << " : " << (*AMatrix)(ii, fitpos1)
                        << " ?= " << (*AMatrix)(ii, fitpos2) << std::endl;
            if (std::abs((*AMatrix)(ii, fitpos1)) < derivPrec) {
              if (std::abs((*AMatrix)(ii, fitpos2)) > derivPrec) {
                isProp = FALSE;
                break;
              }
            } else {
              ALIdouble propn = (*AMatrix)(ii, fitpos2) / (*AMatrix)(ii, fitpos1);
              if (prop != DBL_MAX && prop != propn) {
                isProp = FALSE;
                break;
              }
              prop = propn;
            }
          }
          if (isProp) {
            std::cerr << "!!!FATAL ERROR: Fit::CheckIfFitPossible() two entries for which the measurements have the "
                         "same dependency (or 100% proportional) "
                      << (*vecite1)->longName() << " & " << (*vecite2)->longName() << std::endl
                      << "!!! Fit will not be possible! " << std::endl;
            throw cms::Exception("SDFError");
          }
        }
      }
    }
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
int Fit::CheckIfMeasIsProportionalToAnother(ALIuint measNo) {
  int measProp = -1;

  std::set<ALIuint> columnsEqual;
  std::set<ALIuint> columnsEqualSave;
  ALIuint biggestColumn = 0;
  ALIdouble biggest = 0.;
  for (int ii = 0; ii < AMatrix->NoColumns(); ii++) {
    if (std::abs((*AMatrix)(measNo, ii)) > biggest) {
      biggest = std::abs((*AMatrix)(measNo, ii));
      biggestColumn = ii;
    }
    columnsEqualSave.insert(ii);
  }

  ALIdouble div;

  for (int jj = 0; jj < AMatrix->NoLines(); jj++) {
    if (jj == int(measNo))
      continue;
    columnsEqual = columnsEqualSave;
    // check if ratio of each column to 'biggestColumn' is the same as for the N measurement
    for (int ii = 0; ii < AMatrix->NoColumns(); ii++) {
      div = (*AMatrix)(measNo, ii) / (*AMatrix)(measNo, biggestColumn);
      if (std::abs((*AMatrix)(jj, ii)) > ALI_DBL_MIN &&
          std::abs(div - (*AMatrix)(jj, ii) / (*AMatrix)(jj, biggestColumn)) > ALI_DBL_MIN) {
        if (ALIUtils::debug >= 3)
          std::cout << "CheckIfMeasIsProportionalToAnother 2 columns = " << ii << " in " << measNo << " & " << jj
                    << std::endl;
      } else {
        if (ALIUtils::debug >= 3)
          std::cout << "CheckIfMeasIsProportionalToAnother 2 columns != " << ii << " in " << measNo << " & " << jj
                    << std::endl;
        // if it is not equal delete this column
        std::set<ALIuint>::iterator ite = columnsEqual.find(ii);
        if (ite != columnsEqual.end()) {
          columnsEqual.erase(ite);
        }
      }
    }
    // check if not all columns have been deleted from columnsEqual
    if (!columnsEqual.empty()) {
      if (ALIUtils::debug >= 3)
        std::cout << "CheckIfMeasIsProportionalToAnother " << measNo << " = " << jj << std::endl;
      measProp = jj;
      break;
    }
  }

  return measProp;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::string Fit::GetMeasurementName(int imeas) {
  std::string measname = " ";

  std::cout << " imeas " << imeas << std::endl;
  int Aline = 0;
  std::vector<Measurement*>::const_iterator vmcite;
  for (vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); ++vmcite) {
    for (ALIuint jj = 0; jj < ALIuint((*vmcite)->dim()); jj++) {
      if (Aline == imeas) {
        char ctmp[20];
        gcvt(jj, 10, ctmp);
        return ((*vmcite)->name()) + ":" + std::string(ctmp);
      }
      Aline++;
    }
  }

  std::cout << " return measname " << measname << std::endl;
  return measname;
}
