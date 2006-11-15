//   COCOA class implementation file
//Id:  Fit.cc
//CAT: Fit
//
//   History: v1.0 
//   Pedro Arce

#include <tree.h>

#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaFit/interface/Fit.h"

#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "Alignment/CocoaModel/interface/DeviationsFromFileSensor2D.h"
#include "Alignment/CocoaAnalysis/interface/FittedEntriesManager.h"
#include "Alignment/CocoaAnalysis/interface/FittedEntriesSet.h"
#ifdef COCOA_VIS
#include "Alignment/CocoaVisMgr/interface/ALIVRMLMgr.h"
#include "Alignment/IgCocoaFileWriter/interface/IgCocoaFileMgr.h"
#endif
#include "Alignment/CocoaModel/interface/OpticalObjectMgr.h"
#include "Alignment/CocoaModel/interface/ErrorCorrelationMgr.h"
#include "Alignment/CocoaModel/interface/ErrorCorrelation.h"
#include "Alignment/CocoaModel/interface/FittedEntriesReader.h"

#include <stdlib.h>
#include <iomanip>
#include <math.h>
#include <time.h>
 
Fit* Fit::instance = 0;

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
ALIint Fit::theNoFitIterations;
ALIint Fit::MaxNoFitIterations = -1;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Gets the only instance of Model
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Fit& Fit::getInstance()
{
  if(!instance) {
    instance = new Fit;
    ALIdouble go;
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
    gomgr->getGlobalOptionValue("MaxDeviDerivative", go );
    ALIUtils::setMaximumDeviationDerivative( go );
    if( ALIUtils::debug >= 3 ) std::cout << " Fit::maximum_deviation_derivative " << ALIUtils::getMaximumDeviationDerivative() << std::endl;
    gomgr->getGlobalOptionValue("MaxNoFitIterations", go );
    MaxNoFitIterations = int(go);
    gomgr->getGlobalOptionValue("FitQualityCut", go );
    theFitQualityCut = go;
      if( ALIUtils::debug >= 3 ) std::cout << " theFitQualityCut " << theFitQualityCut << std::endl;
  }

  return *instance;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  startFit: steering method to make the fit
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::startFit()
{ 
  Model::setCocoaStatus( COCOA_InitFit );

  ALIuint nEvent = 0;
  ALIUtils::setFirstTime( 1 );
  for(;;) {

    if( fitNextEvent( nEvent ) ) break;

    if ( ALIUtils::debug >= 0) std::cout << " FIT STATUS " << Model::printCocoaStatus( Model::getCocoaStatus() ) << std::endl;

  }

  //---------- Program ended, fill histograms of fitted entries
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if(gomgr->GlobalOptions()["histograms"] > 0) {
    FittedEntriesManager* FEmgr = FittedEntriesManager::getInstance();
    FEmgr->MakeHistos();
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool Fit::fitNextEvent( ALIuint& nEvent )
{
  if( Model::getFittedEntriesReader() != 0 ) Model::getFittedEntriesReader()->readFittedEntriesFromFile();

  std::vector< OpticalObject* >::iterator voite;
  for( voite = Model::OptOList().begin(); voite !=  Model::OptOList().end(); voite++ ) {
    (*voite)->resetOriginalOriginalCoordinates();
  }

  std::vector< Entry* >::iterator veite;
  for( veite = Model::EntryList().begin(); veite !=  Model::EntryList().end(); veite++ ) {    
    (*veite)->resetValueDisplacementByFitting();
  }


  ALIbool lastEvent = 0;

  //-    DeviationsFromFileSensor2D::setApply( 1 );
  ALIbool moreDataSets = Model::readMeasurementsFromFile( Measurement::only1Date, Measurement::only1Time );

  //-    if(ALIUtils::debug >= 3)  std::cout << "$$$$$$$$$$$$$$$ moreData Sets " << moreDataSets << std::endl;
  if( moreDataSets ) {
    if( ALIUtils::debug >= 2 ) std::cout << "@@@@@@@@@@@@@@@@@@ Starting fit ..." << std::endl;

    //----- Count entries to be fitted, and set their order in theFitPos
    setFittableEntries();
    
    //----- Dump dimensions of output in 'report.out' file
    ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
    fileout << std::endl << "@@@@@@@ NEW MEASUREMENT SET " << nEvent << std::endl;
    if( ALIUtils::report >= 1 ) ALIUtils::dumpDimensions( fileout );
    
    //----- reset no of iterations of non linear fit
    theNoFitIterations = 0;
    
    //---------- Calculate the original simulated values of each Measurement (when all entries have their read in values)
    calculateSimulatedMeasurementsWithOriginalValues(); //?? original changed atfer each iteration
   
    GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
#ifdef COCOA_VIS
    if(gomgr->GlobalOptions()["VisWriteVRML"] > 0) {
      if(ALIUtils::getFirstTime()) ALIVRMLMgr::getInstance().writeFile();
    }
    if(gomgr->GlobalOptions()["VisWriteIguana"] > 0) {
      if(ALIUtils::getFirstTime()) IgCocoaFileMgr::getInstance().writeFile();
    }
    
    if(gomgr->GlobalOptions()["VisOnly"] == 1) {
      if(ALIUtils::debug >= 1 )std::cout << " Visualiation file(s) succesfully written. Ending.... " << std::endl;
      lastEvent = 1;
      return lastEvent;
    }
#endif    

    //----- fitParameters
    if( fitParameters( 1. ) == COCOA_FitMatrixNonInversable ) {
      lastEvent = 1;
      return lastEvent;
    }
    
    //----- Iteration is finished: dump fitted entries
    if(ALIUtils::debug >= 1) calculateSimulatedMeasurementsWithOriginalValues();
    if(gomgr->GlobalOptions()["histograms"] > 0) {
       FittedEntriesManager::getInstance()->AddFittedEntriesSet( new FittedEntriesSet( AtWAMatrix ) );
    }
    
    
    //- only if not stopped in worsening quality state        if(ALIUtils::report >= 0) dumpFittedValues( ALIFileOut::getInstance( Model::ReportFName() ));
    ALIdouble dumpMat;
    gomgr->getGlobalOptionValue("save_matrices", dumpMat );
    //t matrices are deleted!!!!!!!!!      if( dumpMat != 0 ) dumpMatrices();
    
    /*-      std::vector< OpticalObject* >::iterator voite;
      for( voite = Model::OptOList().begin(); voite !=  Model::OptOList().end(); voite++ ) {
	//-??      	(*voite)->resetOriginalOriginalCoordinates();
	}*/
    
    //---- If no measurement file, break after looping once
    //-      std::cout << " Measurement::measurementsFileName() " << Measurement::measurementsFileName() << " Measurement::measurementsFileName()" <<std::endl;
    if( Measurement::measurementsFileName() == "" ) {
      lastEvent = 1;
      return lastEvent;
    }
    
    //-      std::cout << "  Measurement::only1" <<  Measurement::only1 << std::endl;
    if( Measurement::only1 ) {
      lastEvent = 1;
      return lastEvent;
    }
    nEvent++;
  } else {
    lastEvent = 1;
    return lastEvent;
  }

  return lastEvent;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Count how many entries are going to be fitted (have quality >=  theMinimumEntryQuality)
//@@ Set for this entries the value of theFitPos
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::setFittableEntries() 
{

  std::vector< Entry* >::const_iterator vecite;

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  theMinimumEntryQuality = int(gomgr->GlobalOptions()[ALIstring("calcul_type")]) + 1;
  if ( ALIUtils::debug >= 4) std::cout << "Fit::setFittableEntries: total Entry List size= " << Model::EntryList().size() << std::endl;

  int No_entry_to_fit = 0;
  for ( vecite = Model::EntryList().begin();
       vecite != Model::EntryList().end(); vecite++ ) {  

    // Number the parameters that are going to be fitted
      if ( (*vecite)->quality() >= theMinimumEntryQuality ) {
          (*vecite)->setFitPos( No_entry_to_fit );
          if( ALIUtils::debug >= 4 ) std::cout << " Entry To Fit= " << No_entry_to_fit << " " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << " quality " << (*vecite)->quality() << std::endl; 
          No_entry_to_fit++;
      }
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Main method in class Fit 
//@@ fitParameters: get the parameters through the chi square fit
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
cocoaStatus Fit::fitParameters( const double daFactor )
{
  redoMatrices();

  if(ALIUtils::debug >= 2) std::cout << " Fit quality daFactor " << daFactor << std::endl;

  if(ALIUtils::debug >= 0) {
    std::cout << std::endl << "Fit iteration " << theNoFitIterations << " ..." << std::endl;
  }

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if (gomgr->GlobalOptions()[ ALIstring("stopAfter1stIteration") ] == 1) {
    if(theNoFitIterations > 0 ) {
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
 
  FitQuality fq = getFitQuality( 1 );

  evaluateFitQuality( fq, daFactor );

  //-    std::cout << "2 FIT STATUS " << Model::printCocoaStatus( Model::getCocoaStatus() ) << std::endl;

  if(ALIUtils::debug >= 10) {
    std::cout << std::endl << " End fitParameters " << theNoFitIterations << " ..." << std::endl;
  }

  return Model::getCocoaStatus(); // PropagateErrors() may have changed it to COCOA_FitMatrixNonInversable

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::redoMatrices()
{
  deleteMatrices(); 

  calculateSimulatedMeasurementsWithOriginalValues();

  calculateChi2();

  PropagateErrors();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::evaluateFitQuality( const FitQuality fq, double daFactor )
{
  ALIdouble dumpMat;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("save_matrices", dumpMat );

  //----- Good quality: end  
  if( fq == FQsmallDistanceToMinimum ) {

    //------ Correct entries with fitted values  
    //-    if(ALIUtils::report >= 1) dumpFittedValues( ALIFileOut::getInstance( Model::ReportFName() ));
    addDaMatrixToEntries();
    if(ALIUtils::report >= 1) dumpFittedValues( ALIFileOut::getInstance( Model::ReportFName() ));
    Model::setCocoaStatus( COCOA_FitOK );

  //--------- Bad quality: go to next iteration
  } else if( fq == FQbigDistanceToMinimum ) {

    //----- Correct entries with fitted values 
    //-    if(ALIUtils::report >= 1) dumpFittedValues( ALIFileOut::getInstance( Model::ReportFName() ));
    addDaMatrixToEntries();
    if(ALIUtils::report >= 1) dumpFittedValues( ALIFileOut::getInstance( Model::ReportFName() ));

    //----- Next iteration (if not too many already)
    if( theNoFitIterations < MaxNoFitIterations-1 ) {
      Model::setCocoaStatus( COCOA_FitImproving );

      theNoFitIterations++;

      //      if(ALIUtils::report >= 1) dumpFittedValues( ALIFileOut::getInstance( Model::ReportFName() ));

      //----- Reset the original value of entries
      std::vector< Measurement* >::const_iterator vmcite;
      for ( vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); vmcite++) {
        //--- Calculate Simulated Value Original
        (*vmcite)->calculateOriginalSimulatedValue();

        if ( ALIUtils::debug >= 5 ) std::cout << "BuildMeasLinks: Meas " << (*vmcite) << std::endl;
      }
      if( dumpMat > 1 ) dumpMatrices();

      //----- Make next iteration 
      if( fitParameters( 1. ) == COCOA_FitMatrixNonInversable ) {
	return;
      }

    //----- Too many iterations: end here
    } else {
      if(ALIUtils::report >= 1) dumpFittedValues( ALIFileOut::getInstance( Model::ReportFName() ));
      if( dumpMat > 1 ) dumpMatrices();
      std::cerr << "Too many iterations " << theNoFitIterations << "  and fit DOES NOT CONVERGE " << std::endl;

      Model::setCocoaStatus( COCOA_FitCannotImprove );
      //            return;
      return;
    }
  } else if( fq == FQchiSquareWorsened ) {

    Model::setCocoaStatus( COCOA_FitChi2Worsened );

    //----- Recalculate fit quality with decreasing values of Da
    double minDaFactor = 1.e-8;
    //-    std::cout << " quality daFactor " << daFactor << " " << minDaFactor << std::endl;
    if( daFactor > minDaFactor ){
      substractLastDisplacementToEntries( 0.5 );
      if( fitParameters( daFactor/2. ) == COCOA_FitMatrixNonInversable ) {
	return;
      }
      
    } else {
      std::cerr << " ds!!!ERROR: not possible to get good fit quality even multiplying Da by " << daFactor << std::endl;
      Model::setCocoaStatus( COCOA_FitCannotImprove );
      //-    std::cout << "fdsaf FIT STATUS " << Model::printCocoaStatus( Model::getCocoaStatus() ) << std::endl;
      return;
      //      abort();
    }

  } 

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Propagate the Errors from the entries to the measurements
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//cocoaStatus
void Fit::PropagateErrors()
{

  //---------- Create empty matrices of appropiate size
  CreateMatrices();

  //------- count running time
  time_t now;
  now = clock();
  if(ALIUtils::debug >= 2) std::cout << "TIME:CREATE_MAT    : " << now << " " << difftime(now, ALIUtils::time_now())/1.E6 << std::endl;
  ALIUtils::set_time_now(now); 

  //---------- Fill the A, W & y matrices with the measurements
  FillMatricesWithMeasurements();

  //------- count running time
  now = clock();
  if(ALIUtils::debug >= 0) std::cout << "TIME:MAT_MEAS_FILLED: " << now << " " << difftime(now, ALIUtils::time_now())/1.E6 << std::endl;
  ALIUtils::set_time_now(now); 

  //---------- Fill the A, W & y matrices with the calibrated parameters
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if (gomgr->GlobalOptions()[ ALIstring("calcul_type") ] == 0) {
    FillMatricesWithCalibratedParameters();

    //------- count running time
    now = clock();
    if(ALIUtils::debug >= 0) std::cout << "TIME:MAT_CAL_FILLED : " << now << " " << difftime(now, ALIUtils::time_now())/1.E6 << std::endl;
    ALIUtils::set_time_now(now); 

  }

  //put by hand some correlations if known previously
  setCorrelationsInWMatrix();

  if(ALIUtils::debug >= 3) WMatrix->Dump("WMatrix before inverse");

  //----- Check first that matrix can be inverted
  if( m_norm1( WMatrix->MatNonConst() ) == 0 ) {
    Model::setCocoaStatus( COCOA_FitMatrixNonInversable );
    return;
  //  return Model::getCocoaStatus();
  } else {
    WMatrix->inverse();
  }
  
  if(ALIUtils::debug >= 3) AMatrix->Dump("AMatrix");
  if(ALIUtils::debug >= 3) WMatrix->Dump("WMatrix");
  if(ALIUtils::debug >= 3) yfMatrix->Dump("yfMatrix");

  if(gomgr->GlobalOptions()["onlyDeriv"] >= 1) {
    std::cout << "ENDING after derivatives are calculated ('onlyDeriv' option set)" << std::endl;
    exit(1);
  }

  multiplyMatrices();

  now = clock();
  if(ALIUtils::debug >= 0) std::cout << "TIME:MAT_MULTIPLIED : " << now << " " << difftime(now, ALIUtils::time_now())/1.E6 << std::endl;
  ALIUtils::set_time_now(now); 

  if( ALIUtils::getFirstTime() == 1) ALIUtils::setFirstTime( 0 );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Calculate the simulated value of each Measurement propagating the LightRay when all the entries have their original values
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::calculateSimulatedMeasurementsWithOriginalValues()
{
  //  if( ALIUtils::debug >= 4) OpticalObjectMgr::getInstance()->dumpOptOs();

  //---------- Set DeviationsFromFileSensor2D::apply true
  DeviationsFromFileSensor2D::setApply( 1 );

  if(ALIUtils::debug >= 2) std::cout << "Fit::calculateSimulatedMeasurementsWithOriginalValues" <<std::endl;
  //---------- Loop Measurements
  std::vector< Measurement* >::const_iterator vmcite;
  for ( vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); vmcite++) {
    //----- Calculate Simulated Value Original
    (*vmcite)->calculateOriginalSimulatedValue();
  }

  //---------- Set DeviationsFromFileSensor2D::apply false
  // It cannot be applied when calculating derivatives, because after a displacement the laser could hit another square in matrix and then cause a big step in the derivative
  DeviationsFromFileSensor2D::setApply( 0 );

  //  calculateChi2();
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::deleteMatrices()
{
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
void Fit::CreateMatrices()
{

  //---------- Count number of measurements
  ALIint NoMeas = 0;
  std::vector< Measurement* >::const_iterator vmcite;
  for ( vmcite = Model::MeasurementList().begin();
    vmcite != Model::MeasurementList().end(); vmcite++ ) {
    NoMeas += (*vmcite)->dim();
  }
   if( ALIUtils::debug >= 99) std::cout << "NOMEAS" << NoMeas << std::endl;

   //-------- Count number of 'cal'ibrated parameters
  ALIint NoEnt_cal = 0;
  ALIint noent = 0;
  //-  std::cout << Model::EntryList().size() << std::endl;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  if ( gomgr->GlobalOptions()[ "calcul_type" ] == 0) { // fit also 'cal' parameters
    //-  if( ALIUtils::debug >= 9) std::cout << "NOENTCALLL " << NoEnt_cal << std::endl;
    if( ALIUtils::debug >= 5 ) std::cout << " Count number of 'cal'ibrated parameters " << std::endl;
    std::vector< Entry* >::iterator veite;
    for ( veite = Model::EntryList().begin();
          veite != Model::EntryList().end(); veite++ ) {
      if ( (*veite)->quality() == 1 ) NoEnt_cal++; 
      noent++;
      if( ALIUtils::debug >= 6) {
	std::cout <<(*veite)->quality() << " " << (*veite)->OptOCurrent()->name() << " "
	     << (*veite)->name() << " # ENT CAL " << NoEnt_cal << " # ENT " << noent << std::endl; 
      }
    }
  }

  //---------- Count number parameters to be fitted ('cal' + 'unk')
  ALIint NoParamFit = 0;
  std::vector<Entry*>::const_iterator vecite;    
  for ( vecite = Model::EntryList().begin();  
	vecite != Model::EntryList().end(); vecite++) { 
    if ( (*vecite)->quality() >= theMinimumEntryQuality ) {
      NoParamFit++;
      if( ALIUtils::debug >= 99) std::cout <<(*vecite)->quality() << (*vecite)->OptOCurrent()->name() << (*vecite)->name() << "NoParamFit" << NoParamFit << std::endl;
      //      break;
    }
  }  
 
  //---------- Create Matrices
  ALIint NoLinesA = NoMeas + NoEnt_cal;
  ALIint NoColumnsA = NoParamFit;
  AMatrix = new ALIMatrix( NoLinesA, NoColumnsA );

  ALIint NoLinesW = NoLinesA;
  ALIint NoColumnsW = NoLinesA;
  WMatrix = new ALIMatrix( NoLinesW, NoColumnsW );

  ALIint NoLinesY = NoLinesA;
  //op  yMatrix = new ALIMatrix( NoLinesY, 1 );
  yfMatrix = new ALIMatrix( NoLinesY, 1 );

  //op  fMatrix = new ALIMatrix( NoLinesY, 1 );

  if ( ALIUtils::debug >= 4 ) std::cout << "CreateMatrices: NoLinesA = " << NoLinesA <<
  " NoColumnsA = " << NoColumnsA << std::endl;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Loop Measurements:
//@@    Fill Matrix A with derivatives respect to affecting entries 
//@@    Fill Matrix W, y & f with values and sigmas of measurement coordinate
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::FillMatricesWithMeasurements() 
{

  int Aline = 0; 

  //---------- Loop Measurements
  std::vector<Measurement*>::const_iterator vmcite;
  std::vector<Entry*>::const_iterator vecite;
  for ( vmcite = Model::MeasurementList().begin();
	vmcite != Model::MeasurementList().end(); vmcite++) {
    if( ALIUtils::debug >= 5 ) std::cout << "FillMatricesWithMeasurements: measurement " << (*vmcite)->name() << " # entries affecting " <<(*vmcite)->affectingEntryList().size() << std::endl;

    //-------- Array of derivatives with respect to entries
    ALIint measdim = (*vmcite)->dim(); 
    ALIdouble* derivRE;
    derivRE = new ALIdouble[measdim];

    //-------- Fill matrix A:
    //------ Loop only Entries affecting this Measurement
    //-std::cout << "number affecting entries: " << (*vmcite)->affectingEntryList().size() << std::endl;
    for ( vecite = (*vmcite)->affectingEntryList().begin();  
	  vecite != (*vmcite)->affectingEntryList().end(); vecite++) { 
      //-------- If good quality, get derivative of measurement with respect to this Entry
      if ( (*vecite)->quality() >= theMinimumEntryQuality ) {
	if ( ALIUtils::debug >= 4) {
	  std::cout << "FillMatricesWithMeasurements: filling element ( " << Aline << " - " << Aline+measdim-1 << " , " << (*vecite)->fitPos() << std::endl;
	  std::cout <<"entry affecting: " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() <<std::endl;
	}
	derivRE = (*vmcite)->DerivativeRespectEntry(*vecite);
	//---------- Fill matrix A with derivatives
	for ( ALIuint jj = 0; jj < ALIuint(measdim); jj++) {
	  AMatrix->AddData( Aline+jj, (*vecite)->fitPos(), derivRE[jj] );
	  if( ALIUtils::debug >= 6) std::cout << "AMATRIX (" << Aline+jj << "," << (*vecite)->fitPos() << " = " << derivRE[jj] << std::endl;
	  //---------- Reset Measurement simulated_value
	  (*vmcite)->setValueSimulated( jj, (*vmcite)->valueSimulated_orig(jj) );	  
	}
      }
    }
    delete[] derivRE;
    
    //---------- Fill matrices W, y and f:
    //------ Loop Measurement coordinates
    for ( ALIuint jj=0; jj < ALIuint((*vmcite)->dim()); jj++) {
      ALIdouble sigma = (*vmcite)->sigma()[jj];
      if ( sigma == 0. ) {
	std::cerr << "EXITING: Measurement number " << 
	  vmcite - Model::MeasurementList().begin() << 
	  "has 0 error!!" << std::endl;
      } else {
	//----- Fill W Matrix with inverse of sigma squared
	// multiply error by cameraScaleFactor
	ALIdouble sigmanew = sigma * Measurement::cameraScaleFactor;
	//-	std::cout << Aline+jj << " WMATRIX FILLING " << sigmanew << Measurement::cameraScaleFactor << std::endl;
	WMatrix->AddData( Aline+jj, Aline+jj, (sigmanew*sigmanew) );
      }
      //op //----- Fill Matrices y with measurement value 
      //op yMatrix->AddData( Aline+jj, 0, (*vmcite)->value()[jj] );
      //op //----- Fill f Matrix with simulated_value
      //op fMatrix->AddData( Aline+jj, 0, (*vmcite)->valueSimulated_orig(jj) );
      //----- Fill Matrix y - f with measurement value - simulated value
      yfMatrix->AddData( Aline+jj, 0, (*vmcite)->value()[jj] - (*vmcite)->valueSimulated_orig(jj) );
    }
    if ( ALIUtils::debug >= 99) std::cout << "change line" << Aline << std::endl;
    Aline += measdim;
    if ( ALIUtils::debug >= 99) std::cout << "change line" << Aline << std::endl;
    
  }

  if ( ALIUtils::debug >= 4) AMatrix->Dump("Matrix A with meas");
  if ( ALIUtils::debug >= 4) WMatrix->Dump("Matrix W with meas");
  if ( ALIUtils::debug >= 4) yfMatrix->Dump("Matrix y with meas");

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Loop Measurements:
//@@    Fill Matrix A with derivatives respect to affecting entries 
//@@    Fill Matrix W, y & f with values and sigmas of measurement coordinate
//@@ 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::FillMatricesWithCalibratedParameters() 
{
  //---------- Count how many measurements 
  ALIint NolinMes = 0;
  std::vector<Measurement*>::const_iterator vmcite;
  for ( vmcite = Model::MeasurementList().begin();
       vmcite != Model::MeasurementList().end(); vmcite++) {
      NolinMes += (*vmcite)->dim();
  }
  if(ALIUtils::debug>=4) std::cout << "@@FillMatricesWithCalibratedParameters" << std::endl;

  std::vector< Entry* >::const_iterator vecite;
  ALIint NoEntcal = 0;
  //---------- Loop entries 
  for ( vecite = Model::EntryList().begin();
	vecite != Model::EntryList().end(); vecite++ ) {
//                  (= No parameters to be fitted - No parameters 'unk' )
    //-    std::cout << "entry" << (*veite) << std::endl;
    //----- Take entries of quality = 'cal' 
    if ( (*vecite)->quality() == 1 ){
      //--- Matrix A: fill diagonals with 1. (derivatives of entry w.r.t itself)
      ALIint lineNo = NolinMes + NoEntcal;  
      ALIint columnNo = (*vecite)->fitPos();  //=? NoEntcal
      AMatrix->AddData( lineNo, columnNo, 1. );
      if(ALIUtils::debug >= 4) std::cout << "Fit::FillMatricesWithCalibratedParameters:  AMatrix ( " << lineNo << " , " << columnNo  << ") = " << 1. << std::endl;

      //--- Matrix W: sigma*sigma
      ALIdouble entsig = (*vecite)->sigma();
      if(ALIUtils::debug >= 4) std::cout << "Fit::FillMatricesWithCalibratedParameters:  WMatrix ( " << lineNo << " , " << columnNo  << ") = " << entsig*entsig << std::endl;
      WMatrix->AddData( lineNo, lineNo, entsig*entsig );

      //--- Matrix y & f: fill it with 0. 
      //op      yMatrix->AddData( lineNo, 0, (*vecite)->value());
      //op      yfMatrix->AddData( lineNo, 0, (*vecite)->value());
      ALIdouble calFit;
      GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
      gomgr->getGlobalOptionValue("calParamInyfMatrix", calFit );
      if( calFit ) {
       	yfMatrix->AddData( lineNo, 0, -(*vecite)->valueDisplacementByFitting() );
	//-	yfMatrix->AddData( lineNo, 0, (*vecite)->value() );
	//-       	yfMatrix->AddData( lineNo, 0, (*vecite)->lastAdditionToValueDisplacementByFitting() );
	//-	ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
	//	fileout << "cal to yf " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << " " << (*vecite)->valueDisplacementByFitting() << endl;
	//	cout << "cal to yf " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << " " << (*vecite)->valueDisplacementByFitting() << endl;

      } else {
        yfMatrix->AddData( lineNo, 0, 0. );
      }
      //t      if(ALIUtils::debug >= 5) std::cout << "Fit::FillMatricesWithCalibratedParameters:  yfMatrix ( " << lineNo << " , " << columnNo  << ") = " << (*yfMatrix)(lineNo)(0) << std::endl;
      NoEntcal++;
    }
  }
  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Gets the only instance of Model
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::setCorrelationsInWMatrix()
{  
  //----- Check if there are correlations to input
  ErrorCorrelationMgr* corrMgr = ErrorCorrelationMgr::getInstance();
  ALIint siz = corrMgr->getNumberOfCorrelations();
  if( siz == 0 ) return;

  //---------- Set correlations
  ALIuint ii;
  for( ii = 0; ii < ALIuint(siz); ii++ ){
  //t    if(ALIUtils::debug >= 5) std::cout << "globaloption cmslink fit" << Model::GlobalOptions()["cms_link"] << std::endl;
    ErrorCorrelation* corr = corrMgr->getCorrelation( ii );
    setCorrelationFromParamFitted( corr->entry1(), corr->entry2(), corr->correlation() );
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  set correlation between two entries of two OptOs
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::setCorrelationFromParamFitted( const pss& entry1, const pss& entry2,
 ALIdouble correl ) 
{

  ALIint pmsize = WMatrix->NoLines();
  ALIint fit_pos1 = Model::getEntryByName(entry1.first, entry1.second)->fitPos();
  ALIint fit_pos2 = Model::getEntryByName(entry2.first, entry2.second)->fitPos();
  std::cout << "CHECKsetCorrelatiFPF " << fit_pos1 << " " << fit_pos2 << std::endl;

  if( fit_pos1 >= 0 && fit_pos1 < pmsize && fit_pos2 >= 0 && fit_pos2 < pmsize ) {
    setCorrelationFromParamFitted( fit_pos1, fit_pos2, correl );
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::setCorrelationFromParamFitted( const ALIint fit_pos1, const ALIint fit_pos2, ALIdouble correl ) 
{
  //  ALIdouble error1 = sqrt( (*WMatrix)(fit_pos1, fit_pos1) );
  // ALIdouble error2 = sqrt( (*WMatrix)(fit_pos2, fit_pos2) );
  WMatrix->SetCorrelation( fit_pos1, fit_pos2, correl );
  std::cout << "setCorrelatiFPF " << fit_pos1 << " " << fit_pos2 << " " << correl << std::endl;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  multiply matrices needed for fit
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::multiplyMatrices() 
{
  if(ALIUtils::debug >= 4) std::cout << "@@Multiplying matrices " << std::endl;
  //---------- Calculate transpose of A
  AtMatrix = new ALIMatrix( *AMatrix );
  if(ALIUtils::debug >= 5) AtMatrix->Dump("AtMatrix=A");
  //-  std::cout << "call transpose";
  AtMatrix->transpose();
  if(ALIUtils::debug >= 4) AtMatrix->Dump("AtMatrix");

  //---------- Calculate At * W * A
  AtWAMatrix = new ALIMatrix(0, 0);   
  //  if(ALIUtils::debug >= 5) AtWAMatrix->Dump("AtWAMatrix=0");
  *AtWAMatrix = *AtMatrix * *WMatrix * *AMatrix;   
  if(ALIUtils::debug >= 5) AtWAMatrix->Dump("AtWAMatrix");
  //t  AtWAMatrix->EliminateLines(0,48);
  //t AtWAMatrix->EliminateColumns(0,48);
  time_t now;
  now = clock();
  if(ALIUtils::debug >= 0) std::cout << "TIME:BEFORE_INVERSE : " << now << " " << difftime(now, ALIUtils::time_now())/1.E6 << std::endl;
  ALIUtils::set_time_now(now); 

  AtWAMatrix->inverse();
  if(ALIUtils::debug >= 4) AtWAMatrix->Dump("inverse AtWAmatrix");
  now = clock();
  if(ALIUtils::debug >= 0) std::cout << "TIME:AFTER_INVERSE  : " << now << " " << difftime(now, ALIUtils::time_now())/1.E6 << std::endl;
  ALIUtils::set_time_now(now); 

  //op  thePropagationMatrix = AtWAMatrix;

  //op  VaMatrix = new ALIMatrix( *AtWAMatrix );
  
  //---------- Print out propagated errors of parameters (=AtWA diagonal elements)
  std::vector< Entry* >::const_iterator vecite;
 
  if( ALIUtils::debug >= 4 ) {
    std::cout << "PARAM" << "        Optical Object " << " entry name " << "    Param.Value " 
         << " Prog.Error" << " Orig.Error" << std::endl;
  }

  ALIint NoEnt = 0;
  ALIint NoEntUnk = 0;
  for ( vecite = Model::EntryList().begin();
    vecite != Model::EntryList().end(); vecite++ ) {
//------------------ Number of parameters 'cal' 
//                  (= No parameters to be fitted - No parameters 'unk' )
    if( (*vecite)->quality() >= theMinimumEntryQuality ){
      if( ALIUtils::debug >= 4) {
        std::cout << NoEnt << "PARAM" << std::setw(26) 
	     << (*vecite)->OptOCurrent()->name().c_str() 
	     << std::setw(8) << " " << (*vecite)->name().c_str() << " " 
	     << std::setw(8) << " " << (*vecite)->value() << " " 
	     << std::setw(8) << sqrt(AtWAMatrix->Mat()->me[NoEnt][NoEnt]) / 
                (*vecite)->SigmaDimensionFactor() 
	     << " " << (*vecite)->sigma() / (*vecite)->SigmaDimensionFactor() 
             << " Q" << (*vecite)->quality() << std::endl;
      }
      NoEnt++;
    }
    if ( (*vecite)->quality() == 2 ) NoEntUnk++;
  }

  if(ALIUtils::debug >= 5) yfMatrix->Dump("PD(y-f)Matrix final");  

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ fit_quality_cut = DSMat(0,0) has to be smaller than theFitQualityCut
//@@ check also that the fit_quality = SMat(0,0) is smaller for each new iteration
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
FitQuality Fit::getFitQuality( const ALIbool canBeGood ) 
{
  //---------- Calculate variable to check quality of this set of parameters

  //---------- Calculate Da = (At * W * A)-1 * At * W * (y-f)
  /*t  DaMatrix = new ALIMatrix( *AtWAMatrix );
  *DaMatrix *= *AtMatrix * *WMatrix;
  if(ALIUtils::debug >= 5) DaMatrix->Dump("DaMatrix before yf ");
  *DaMatrix *= *yfMatrix;
  if(ALIUtils::debug >= 5) DaMatrix->Dump("DaMatrix");
  */

  DaMatrix = new ALIMatrix(0, 0);   
  //  if(ALIUtils::debug >= 5) AtWAMatrix->Dump("AtWAMatrix=0");
  *DaMatrix = ( *AtWAMatrix * *AtMatrix * *WMatrix * *yfMatrix);
  if(ALIUtils::debug >= 5) DaMatrix->Dump("DaMatrix");

  ALIMatrix* DaMatrix2 = new ALIMatrix( DaMatrix->NoLines(), DaMatrix->NoColumns() );
  *DaMatrix2 = ( *AtWAMatrix * *AtMatrix * *WMatrix * *yfMatrix);
  if(ALIUtils::debug >= 5) DaMatrix2->Dump("DaMatrix2"); 

  ALIMatrix* DaMatrix3 = MatrixByMatrix( *AtWAMatrix, *AtMatrix );
  ALIMatrix* DaMatrix4 = MatrixByMatrix( *DaMatrix3, *WMatrix );
  ALIMatrix* DaMatrix5 = MatrixByMatrix( *DaMatrix4, *yfMatrix );
  if(ALIUtils::debug >= 5) DaMatrix5->Dump("DaMatrix5"); 
  
  //----- Calculate S = Fit quality = Distance to minimum
  //op  ALIMatrix* tmpM = new ALIMatrix( *AMatrix * *DaMatrix + *PDMatrix );
  //  ALIMatrix* tmpM = new ALIMatrix( *AMatrix * *DaMatrix + *yfMatrix );

  ALIMatrix* tmpM = new ALIMatrix( 0,0 );
  *tmpM  = *AMatrix * *DaMatrix5;
  /*  ALIMatrix tmpMnn = ( *AMatrix * *DaMatrix + *yfMatrix );
  ALIMatrix* tmpM = new ALIMatrix();
  *tmpM = tmpMnn;
  */

  if(ALIUtils::debug >= 5) tmpM->Dump("A*Da + (y-f) Matrix ");
  ALIMatrix* tmptM = new ALIMatrix( *tmpM );
  //  if(ALIUtils::debug >= 5) tmptM->Dump("tmptM before transpose");
  //  tmpM->transpose();
  //  if(ALIUtils::debug >= 5) tmptM->Dump("X after transpose");
  tmptM->transpose();
  if(ALIUtils::debug >= 5) tmptM->Dump("tmptM after transpose");
  //  std::cout << "smat " << std::endl;
  //o  ALIMatrix* SMat = new ALIMatrix(*tmptM * *WMatrix * *tmpM);
  ALIMatrix* SMat1 = MatrixByMatrix(*tmptM,*WMatrix);
  //  ALIMatrix* SMat1 = MatrixByMatrix(*AMatrix,*WMatrix);
  if(ALIUtils::debug >= 5) SMat1->Dump("SMat1");
  ALIMatrix* SMat = MatrixByMatrix(*SMat1,*tmpM);
  //  std::cout << "smatc " << std::endl;
  delete tmpM;
  delete tmptM;
  if(ALIUtils::debug >= 5) SMat->Dump("SMatrixfinal");
  ALIdouble fit_quality = (*SMat)(0,0);
  delete SMat;
  if(ALIUtils::debug >= 0) std::cout << theNoFitIterations << " Fit quality is = " << fit_quality << std::endl;
  if( ALIUtils::report >= 1 ) {
  //--------- Get report file handler
    ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
    fileout << std::endl << "Fit iteration " << theNoFitIterations << " ..." << std::endl;
    fileout << theNoFitIterations << " Fit quality is = " << fit_quality << std::endl;
  }

  //---------- Calculate DS = Variable to recognize convergence (distance to minimum)
  ALIMatrix* DatMatrix = new ALIMatrix( *DaMatrix );
  //  delete DaMatrix; //op
  DatMatrix->transpose();
  if(ALIUtils::debug >= 5) DatMatrix->Dump("DatMatrix");
  //op  ALIMatrix* DSMat = new ALIMatrix(*DatMatrix * *AtMatrix * *WMatrix * *PDMatrix);
  ALIMatrix* DSMat = new ALIMatrix(*DatMatrix * *AtMatrix * *WMatrix * *yfMatrix);
  ALIMatrix* DSMattemp = new ALIMatrix(*DatMatrix * *AtMatrix * *WMatrix);
  if(ALIUtils::debug >= 5) DSMattemp->Dump("DSMattempMatrix=Dat*At*W");
  ALIMatrix* DSMattemp2 = new ALIMatrix(*AtMatrix * *WMatrix * *yfMatrix);
  if(ALIUtils::debug >= 5) DSMattemp2->Dump("DSMattempMatrix2=At*W*yf");
  ALIMatrix* DSMattemp3 = new ALIMatrix(*AtMatrix * *WMatrix);
  if(ALIUtils::debug >= 5) DSMattemp3->Dump("DSMattempMatrix3=At*W");
  if(ALIUtils::debug >= 5) AtMatrix->Dump("AtMatrix");
  /*  for( int ii = 0; ii < DatMatrix->NoColumns(); ii++ ){
    std::cout << ii << " DS term " << (*DatMatrix)(0,ii) * (*DSMattemp2)(ii,0) << std::endl;
    }*/
  //  delete AtMatrix; //op
  //  delete WMatrix; //op

  //op  if(ALIUtils::debug >= 5) (*PDMatrix).Dump("PDMatrix");
  if(ALIUtils::debug >= 5) (*yfMatrix).Dump("yfMatrix");
  if(ALIUtils::debug >= 5) DSMat->Dump("DSMatrix final");
  //  delete yfMatrix; //op

  ALIdouble fit_quality_cut = (*DSMat)(0,0);  
  //-  ALIdouble fit_quality_cut =fabs( (*DSMat)(0,0) );  
  delete DSMat;
  if(ALIUtils::debug >= 0) std::cout << theNoFitIterations << " Fit quality predicted improvement in distance to minimum is = " << fit_quality_cut << std::endl;
  if( ALIUtils::report >= 2 ) {
    ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
    fileout << theNoFitIterations << " Fit quality cut is = " << fit_quality_cut << std::endl;
  }

  //---------- Derivative of S at 0
  ALIMatrix* Sp0Mat = new ALIMatrix(*DatMatrix * *AtMatrix * *WMatrix * *AMatrix * *DaMatrix);
  if(ALIUtils::debug >= 5) Sp0Mat->Dump("Sp0Matrixfinal");
  if(ALIUtils::debug >= 0) std::cout << theNoFitIterations << " Fit quality derivative at 0 = " << -2. * (*Sp0Mat)(0,0) << std::endl;

  delete DatMatrix; //op
  delete Sp0Mat; //op

  //---------- Check quality 
  time_t now;
  now = clock();
  if(ALIUtils::debug >= 0) std::cout << "TIME:QUALITY_CHECKED: " << now << " " << difftime(now, ALIUtils::time_now())/1.E6 << std::endl;
  ALIUtils::set_time_now(now); 

  FitQuality fitQuality;
  //----- quality good enough: end
  if( fit_quality_cut < theFitQualityCut && canBeGood ) {
    fitQuality = FQsmallDistanceToMinimum;
    if(ALIUtils::report >= 1) {
      ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
      fileout << "GOOD QUALITY OF THE FIT FOR ITERATION " << theNoFitIterations << " = " << fit_quality_cut << " < " << theFitQualityCut << std::endl;
    }
    if(ALIUtils::debug >= 4) {
      std::cout << "GOOD QUALITY OF THE FIT FOR ITERATION " << theNoFitIterations << " = " << fit_quality_cut << " < " << theFitQualityCut << std::endl;
    }

  //--------- Bad quality: go to next iteration
  } else {
    //--------- Check that quality in this iteration is not worse than in previous one
    //-    if( theNoFitIterations != 0 && (fit_quality - thePreviousIterationFitQuality) > 1.e-9 ) {
    if( theNoFitIterations != 0 && (fit_quality - thePreviousIterationFitQuality) > 0. ) {
      //t   if( theNoFitIterations != 0 && fit_quality > thePreviousIterationFitQuality ) {
      fitQuality = FQchiSquareWorsened;
      std::cerr << "!!! Fit quality has worsened: Fit Quality now = " << fit_quality
		<< " before " << thePreviousIterationFitQuality << " diff " << fit_quality - thePreviousIterationFitQuality << std::endl;

    } else {
      fitQuality = FQbigDistanceToMinimum;
      //----- set thePreviousIterationFitQuality for next iteration   
      thePreviousIterationFitQuality = fit_quality;     

      if(ALIUtils::report >= 2) {
	ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
	fileout << "BAD QUALITY OF THE FIT FOR ITERATION " << theNoFitIterations << " = " << fit_quality_cut << " >= " << theFitQualityCut << std::endl;
      }
      if(ALIUtils::debug >= 4) {
	std::cout << "BAD QUALITY OF THE FIT FOR ITERATION " << theNoFitIterations << " = " << fit_quality_cut << " >= " << theFitQualityCut << std::endl;
      } 
    }

  } 

  return fitQuality;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Correct entries with fitted values  
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::addDaMatrixToEntries() 
{

  if(ALIUtils::debug >= 4) {
    std::cout << "@@ Adding correction (DaMatrix) to Entries " << std::endl;
    DaMatrix->Dump("DaMatrix =");
  }

  /*-  Now there are other places where entries are changed
    if( ALIUtils::report >= 3 ) {
    ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
    fileout << std::endl << " CHANGE IN ENTRIES" << std::endl
            << "          Optical Object       Parameter   xxxxxx " << std::endl;
	    }*/

  ALIint NoEnt = 0;
  std::vector<Entry*>::const_iterator vecite; 
  for ( vecite = Model::EntryList().begin();
    vecite != Model::EntryList().end(); vecite++ ) {
//------------------ Number of parameters 'cal' 
//                  (= No parameters to be fitted - No parameters 'unk' )
    //-   std::cout << "qual" << (*vecite)->quality() << theMinimumEntryQuality << std::endl;
    if ( (*vecite)->quality() >= theMinimumEntryQuality ){
      if ( ALIUtils::debug >= 5) {
        std::cout << std::endl << " @@@ PENTRY change " 
		  << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << " " 
		  << " change= " << (*DaMatrix)(NoEnt,0)
		  << " value= " << (*vecite)->valueDisplacementByFitting()
		  << std::endl;
      }
      /*      if( ALIUtils::report >=3 ) {
        ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
        fileout << "dd" << std::setw(30) << (*vecite)->OptOCurrent()->name() 
                << std::setw(8) << " " << (*vecite)->name() << " " 
                << std::setw(8) << " " << (*DaMatrix)(NoEnt,0) / (*vecite)->ValueDimensionFactor()
	        << " " << (*vecite)->valueDisplacementByFitting() / (*vecite)->ValueDimensionFactor() << " fitpos " << (*vecite)->fitPos()
                << std::endl;
		}*/

      //----- Store this displacement 
      if(ALIUtils::debug >= 4) std::cout << " old valueDisplacementByFitting " << (*vecite)->name() << " " << (*vecite)->valueDisplacementByFitting() << " original value " <<  (*vecite)->value() <<std::endl; 

      (*vecite)->addFittedDisplacementToValue( (*DaMatrix)(NoEnt,0) );

      if(ALIUtils::debug >= 4) std::cout << NoEnt << " new valueDisplacementByFitting " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << " = " << (*vecite)->valueDisplacementByFitting() << " " << (*DaMatrix)(NoEnt,0) << std::endl ; 
      NoEnt++;
    }
  }

}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Delete the previous addition of fitted values (to try a new one daFactor times smaller )
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::substractLastDisplacementToEntries( const ALIdouble factor ) 
{

  if(ALIUtils::debug >= 4) {
    std::cout << "@@  Fit::substractToHalfDaMatrixToEntries " << std::endl;
  }

  std::vector<Entry*>::const_iterator vecite; 
  for ( vecite = Model::EntryList().begin(); vecite != Model::EntryList().end(); vecite++ ) {
    if ( (*vecite)->quality() >= theMinimumEntryQuality ){
      //--     (*vecite)->addFittedDisplacementToValue( -(*DaMatrix)(NoEnt,0) );!!! it is not substracting the new value of DaMatrix, but substracting the value that was added last iteration, with which the new value of DaMatrix has been calculated for this iteration

      ALIdouble lastadd = (*vecite)->lastAdditionToValueDisplacementByFitting() * factor;
      //-      if( lastadd < 0 ) lastadd *= -1;
      (*vecite)->addFittedDisplacementToValue( -lastadd );
      (*vecite)->setLastAdditionToValueDisplacementByFitting( - (*vecite)->lastAdditionToValueDisplacementByFitting() );
      //      (*vecite)->substractToHalfFittedDisplacementToValue();

	if(ALIUtils::debug >= 4) std::cout << " new valueDisplacementByFitting " << (*vecite)->OptOCurrent()->name() << " " << (*vecite)->name() << " = " << (*vecite)->valueDisplacementByFitting() << " " << std::endl ; 
    }
  }

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Dump all the entries that have been fitted (those that were 'cal' or 'unk'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::dumpFittedValues( ALIFileOut& fileout, ALIbool printErrors )
{
  //  calculateChi2(); here it is calculated when values have already changed after fit, but measurement simulation values is still with old values

  //---------- print
  if(ALIUtils::debug >= 0) {
    std::cout << "SRPARPOS " << "               Optical Object  " 
         << "      Parameter" <<  " Fit.Value " << " Orig.Value" << std::endl;
  }
  //---------- Dump header
  if(ALIUtils::debug >= 0) std::cout << std::endl << "FITTED VALUES " << std::endl; 
  fileout << std::endl << "FITTED VALUES " << std::endl 
	  << "NoEnt_unk"
	  << "             Optical Object" 
	  << "        Parameter  ";
  if( printErrors ) {
    fileout << " value (+-error)" 
	    << " orig.val (+-error)";
  } else {
    fileout << " value " 
	    << " orig.val ";
  }
  fileout << " quality" 
	  << std::endl; 

  ALIint NoEntUnk = 0;
  //---------- Iterate over OptO list
  std::vector< Entry* > entries;
  //  const Entry* entry;
  int ii, siz;
  std::vector< OpticalObject* >::const_iterator vocite;
  for( vocite = Model::OptOList().begin(); vocite != Model::OptOList().end(); vocite++ ) {
    if( (*vocite)->type() == ALIstring("system") ) continue;

    fileout << " %%%% Optical Object: " << (*vocite)->longName() << std::endl;

    entries = (*vocite)->CoordinateEntryList();
    siz = entries.size();
    if( siz != 6 ) {
      std::cerr << "!!! FATAL ERROR: strange number of coordinates = " << siz << std::endl;
      abort();
    }

    //----- Dump entry centre coordinates (centre in current coordinates of parent frame <> summ of displacements, as each displacement is done with a different rotation of parent frame)
    OpticalObject* opto = entries[0]->OptOCurrent();
    const OpticalObject* optoParent = opto->parent();
    Hep3Vector centreLocal;
    if( optoParent->type() == "system" ) {
      centreLocal = opto->centreGlob();
    } else {
      centreLocal = opto->centreGlob() - optoParent->centreGlob();
      CLHEP::HepRotation parentRmGlobInv = inverseOf( optoParent->rmGlob() );
      centreLocal = parentRmGlobInv * centreLocal;
    }
    if(ALIUtils::debug >= 2 ) {
      std::cout << "CENTRE LOCAL "<< opto->name() << " " << centreLocal << " GLOBL " << opto->centreGlob() << " parent GLOB " << optoParent->centreGlob() << std::endl;
      ALIUtils::dumprm( optoParent->rmGlob(), " parent rm " );
    }

    for( ii = 0; ii < 3; ii++ ){
      /*    double entryvalue = getEntryValue( entries[ii] );
      ALIdouble entryvalue;
      if( ii == 0 ) {
	entryvalue = centreLocal.x();
      }else if( ii == 1 ) {
	entryvalue = centreLocal.y();
      }else if( ii == 2 ) {
	entryvalue = centreLocal.z();
	}*/
      dumpEntryAfterFit( fileout, entries[ii], NoEntUnk, centreLocal[ii] / entries[ii]->OutputValueDimensionFactor() );
    }

    //----- Dump entry angles coordinates
    std::vector<double> entryvalues = entries[5]->OptOCurrent()->GetLocalRotationAngles( entries );
    //-    std::cout << " after return entryvalues[0] " << entryvalues[0] << " entryvalues[1] " << entryvalues[1] << " entryvalues[2] " << entryvalues[2] << std::endl;
    for( ii = 3; ii < siz; ii++ ){
      dumpEntryAfterFit( fileout, entries[ii], NoEntUnk, entryvalues[ii-3]/ entries[ii]->OutputValueDimensionFactor() );
    }
    entries = (*vocite)->ExtraEntryList();
    siz = entries.size();
    for( ii = 0; ii < siz; ii++ ){
      double entryvalue = getEntryValue( entries[ii] );
      dumpEntryAfterFit( fileout, entries[ii], NoEntUnk, entryvalue );
    }
  }

  dumpEntryCorrelations( fileout, NoEntUnk );

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
double Fit::getEntryValue( const Entry* entry )
{
  double entryvalue;
  if( entry->type() == "angles") {
    if(ALIUtils::debug >= 2 ) std::cout << "WARNING valueDisplacementByFitting has no sense for angles "  << std::endl;
    entryvalue = -999;
  }
  entryvalue = ( entry->value() + entry->valueDisplacementByFitting() ) / entry->OutputValueDimensionFactor();
  return entryvalue;
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::dumpEntryAfterFit( ALIFileOut& fileout, const Entry* entry, int& NoEntUnk, double entryvalue, ALIbool printErrors )
{
  //-  std::cout << " Fit::dumpEntryAfterFit " << entryvalue << std::endl;
  ALIdouble dimv = entry->OutputValueDimensionFactor();
  ALIdouble dims = entry->OutputSigmaDimensionFactor();
  //----- Dump to std::cout
  if(ALIUtils::debug >= 3) {
    std::cout << "ENTRY: " 
	 << std::setw(30) << entry->OptOCurrent()->name()
	 << std::setw(8) << " " << entry->name() << " " 
	 << std::setw(8) << ( entry->value() + entry->valueDisplacementByFitting() ) 
	 <<" " << entry->value() 
	 << " Q" << entry->quality() << std::endl;
  }
  
  if ( entry->quality() == 2 ) {
    fileout << "UNK: " << entry->fitPos() << " "; 
  } else if ( entry->quality() == 1 ) {
    fileout << "CAL: -1 "; 
  } else {
    fileout << "FIX: -1 ";
  }
  
  fileout << std::setw(30)  << entry->OptOCurrent()->name()
	  << std::setw(8) << " " << entry->name() << " " 
	  << std::setw(8) << std::setprecision(8) << entryvalue;
  if ( entry->quality() >= theMinimumEntryQuality ) {
    if( printErrors ) fileout << " +- " << std::setw(8) << sqrt(AtWAMatrix->Mat()->me[entry->fitPos()][entry->fitPos()]) / dims;
  } else { 
    if( printErrors ) fileout << " +- " << std::setw(8) << 0.;
  }
  fileout << std::setw(8) << " " << entry->value() / dimv;
  if( printErrors ) fileout << " +- " << std::setw(8) << entry->sigma() /dims << " Q" << entry->quality();
  if( ALIUtils::report >= 2) {
    float dif = ( entry->value() + entry->valueDisplacementByFitting() ) / dimv - entry->value() / dimv;
    if( fabs(dif) < 1.E-9 ) dif = 0.;
    fileout << " DIFF= " << dif;
    // << " == " << ( entry->value() + entry->valueDisplacementByFitting() )  / dimv - entryvalue << " @@ " << ( entry->value() + entry->valueDisplacementByFitting() ) / dimv << " @@ " <<  entryvalue;
  } else {
    //	fileout << std::endl;
  }
  fileout << std::endl;

  if ( entry->quality() == 2 ) NoEntUnk++;
  
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::dumpEntryCorrelations( ALIFileOut& fileout, const int NoEntUnk )
{
  //----- Only dump correlations bigger than a factor
  ALIdouble minCorrel = 1.E-6;
  //----- Dump correlations
  fileout << std::endl << "CORRELATION BETWEEN 'unk' ENTRIES: (>= " << minCorrel<< " )" << std::endl
	  << "No_1  No_2   correlation" << NoEntUnk << std::endl;

  ALIuint ii;
  for( ii = 0; ii < ALIuint(NoEntUnk); ii++) {
    for(ALIuint jj = ii+1; jj < ALIuint(NoEntUnk); jj++) {
      ALIdouble corr = AtWAMatrix->Mat()->me[ii][jj];
      if (corr >= minCorrel ) {
	ALIdouble corrf = corr / sqrt(AtWAMatrix->Mat()->me[ii][ii])
	  / sqrt(AtWAMatrix->Mat()->me[jj][jj]);
        if(ALIUtils::debug >= 0) {
          std::cout << "PARACORRlas (" << ii << ")(" << jj << ") " << corrf << std::endl;
	}
	fileout << "(" << ii << ")     (" << jj << ") " << corrf << std::endl;
      }
    }
  }

  //------- Dump optical object list 
  if( ALIUtils::debug >= 2) OpticalObjectMgr::getInstance()->dumpOptOs();
 
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Dump matrices used for the fit
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Fit::dumpMatrices() 
{
  //----- Fill matrices for this iteration
  ALIFileOut& matout = ALIFileOut::getInstance( Model::MatricesFName() );
  //  ofstream matout("matrices.out");
  matout << std::endl << " @@@@@@@@@@@@@@@  Iteration No : " << theNoFitIterations << std::endl;
  AMatrix->ostrDump( matout, "Matrix A" );
  AtMatrix->ostrDump( matout, "Matrix At" );
  WMatrix->ostrDump( matout, "Matrix W" );
  AtWAMatrix->ostrDump( matout, "Matrix AtWA" );
  //op  VaMatrix->ostrDump( matout, "Matrix Va" );
  DaMatrix->ostrDump( matout, "Matrix Da" );
  yfMatrix->ostrDump( matout, "Matrix y" );
  //op  fMatrix->ostrDump( matout, "Matrix f" );
  //op  thePropagationMatrix->ostrDump( matout, "propagation Matrix " );
  matout.close();

}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ findEntryFitPosition 
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIint Fit::findEntryFitPosition( const ALIstring& opto_name, const ALIstring& entry_name ) 
{
  ALIint fitposi = -99;

  OpticalObject* opto = Model::getOptOByName( opto_name );
  //-  std::cout << "OPTO = " << opto->name() << std::endl;
  std::vector<Entry*>::const_iterator vecite;
  for (vecite = opto->CoordinateEntryList().begin(); 
       vecite < opto->CoordinateEntryList().end(); vecite++) {
    //-    std::cout << "ENTRYLIST" << (*vecite)->name() << std::endl;
    if ((*vecite)->name() == entry_name ) {
      //-  std::cout << "FOUND " << std::endl;
      fitposi = (*vecite)->fitPos();
    }
  }
  for (vecite = opto->ExtraEntryList().begin(); 
       vecite < opto->ExtraEntryList().end(); vecite++) {
    //-    std::cout << "ENTRYLIST" << (*vecite)->name() << std::endl;
    if ((*vecite)->name() == entry_name ) {
      //- std::cout << "FOUND " << std::endl;
      fitposi = (*vecite)->fitPos();
    }
  }

  if(fitposi == -99) {
    std::cerr << "!!EXITING: entry name not found: " << entry_name << std::endl;
    exit(2);
  } else {
    return fitposi;
  }
}


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::pair<double,double> Fit::calculateChi2( )
{
  double chi2meas = 0; 
  double chi2cal = 0;
  ALIint nMeas = 0, nUnk = 0;

  //----- Calculate the chi2 of measurements
  std::vector< Measurement* >::const_iterator vmcite;
  for ( vmcite = Model::MeasurementList().begin(); vmcite != Model::MeasurementList().end(); vmcite++) {
    //--- Calculate Simulated Value Original
    for ( ALIuint ii = 0; ii < ALIuint((*vmcite)->dim()); ii++ ){
      nMeas++;
      double c2 = ( (*vmcite)->value(ii) - (*vmcite)->valueSimulated(ii) ) / (*vmcite)->sigma(ii);
      chi2meas += c2*c2; 
      if( ALIUtils::debug >= -3) {
	std::cout << c2 << " adding chi2meas "  << chi2meas << " " << (*vmcite)->name() << ": " << ii << " (mm)R: " << (*vmcite)->value(ii)*1000. << " S: " << (*vmcite)->valueSimulated(ii)*1000. << " Diff= " << ((*vmcite)->value(ii) - (*vmcite)->valueSimulated(ii))*1000. << std::endl;
      }
    }
  }

  //----- Calculate the chi2 of calibrated parameters
  std::vector< Entry* >::iterator veite;
  for ( veite = Model::EntryList().begin();
	veite != Model::EntryList().end(); veite++ ) {
    if ( (*veite)->quality() == 2 ) nUnk++;
    if ( (*veite)->quality() == 1 ) {
      double c2 = (*veite)->valueDisplacementByFitting() / (*veite)->sigma();
      //double c2 = (*veite)->value() / (*veite)->sigma();
      chi2cal += c2*c2;
      if( ALIUtils::debug >= 3) std::cout << c2 << " adding chi2cal "  << chi2cal << " " << (*veite)->OptOCurrent()->name() << " " << (*veite)->name() << std::endl;
      //-	std::cout << " valueDisplacementByFitting " << (*veite)->valueDisplacementByFitting() << " sigma " << (*veite)->sigma() << std::endl;
    }
  }
  
  if( ALIUtils::report >= 1) {
    ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
    fileout << " Chi2= " << chi2meas+chi2cal << " / " << nMeas-nUnk << " dof " << "  From measurements= " << chi2meas << " from calibrated parameters= " << chi2cal << std::endl;
  }
  if( ALIUtils::debug >= 3) std::cout << " quality calculateChi2 " << chi2meas+chi2cal << " " << chi2meas << " " << chi2cal << std::endl;
  std::pair<double, double > chi2(chi2meas, chi2cal);
  return chi2;
}

