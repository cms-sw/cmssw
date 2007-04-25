
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include <assert.h>
#include "Alignment/CocoaModel/interface/Model.h"
#include "Alignment/CocoaFit/interface/Fit.h"
#include "Alignment/CocoaModel/interface/Entry.h"
#include "Alignment/CocoaUtilities/interface/ALIFileOut.h"
//#include "Analysis/FittedEntriesRoot/interface/FERootDump.h"
#include "Alignment/CocoaToDDL/interface/CocoaToDDLMgr.h"

#include <time.h>
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/SystemOfUnits.h"

int main( int argc, char** argv ) 
{
  char* nam = getenv("COCOA_SDF_FILENAME"); 
  if(nam != 0) Model::setSDFName( nam );
  nam = getenv("COCOA_REPORT_FILENAME"); 
  if(nam != 0) Model::setReportFName( nam );
  nam = getenv("COCOA_MATRICES_FILENAME"); 
  if(nam != 0) Model::setMatricesFName( nam );

  ALIstring COCOA_ver = "COCOA_3_2_4";
  //---------- Read the input arguments to set file names
  switch( argc ){
  case 1:
    break;
  case 4:
    if( ALIstring(argv[3]) != "!" ) Model::setMatricesFName( argv[3] );
  case 3:
    if( ALIstring(argv[2]) != "!" ) Model::setReportFName( argv[2] );
  case 2:
    if( ALIstring(argv[1]) != "!" ) Model::setSDFName( argv[1] );
    if( ALIstring(argv[1]) == "-v" ) {
      std::cerr << "COCOA version = " << COCOA_ver << std::endl;
      exit(0);
    }
    break;
  default:
    std::cerr << "WARNING: more than two arguments, from third on will not be taken into account " << std::endl;
    break;
  }

  //---------- Build the Model out of the system description text file
  Model& model = Model::getInstance();

  time_t now, time_start;
  now = clock();
  time_start = now;
  if(ALIUtils::debug >= 0) std::cout << "TIME:START_READING  : " << now << " " << difftime(now, ALIUtils::time_now())/1.E6  << "   " << ALIUtils::debug <<std::endl;
  ALIUtils::set_time_now(now); 

  Model::readSystemDescription();
  now = clock();
  if(ALIUtils::debug >= 0) std::cout << "TIME:ENDED_READING  : " << now << " " << difftime(now, ALIUtils::time_now())/1.E6  << "   " << ALIUtils::debug << std::endl;
  ALIUtils::set_time_now(now); 

  ALIdouble go;
  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  gomgr->getGlobalOptionValue("writeXML", go );

  if( ALIint(go) == 1 ){
    ALIstring xmlfname = Model::SDFName();
    xmlfname += ALIstring(".xml");
    CocoaToDDLMgr::getInstance()->writeDDDFile( xmlfname );
  }

  Fit::getInstance();

  Fit::startFit();
  // scan entries
  //t    std::cout << " ScanMgr::getInstance()->scanEntries " << std::endl;
  //t    ScanMgr::getInstance()->scanEntries();
  //    ScanMgr::getInstance()->dumpResultMeas( std::cout );

  //analysis code

//  FERootDump fehistos;
//  std::cout << " call  fehistos.MakeHistos " << std::endl;
//  fehistos.MakeHistos();

  if(ALIUtils::debug >= 0) std::cout << "............ program ended OK" << std::endl;
  if( ALIUtils::report >=1 ) {
    ALIFileOut& fileout = ALIFileOut::getInstance( Model::ReportFName() );
    fileout << "............ program ended OK" << std::endl;
  }
  now = clock();
  if(ALIUtils::debug >= 0) std::cout << "TIME:PROGRAM ENDED  : "<< now << " " << difftime(now, ALIUtils::time_now())/1.E6  << std::endl;
  if(ALIUtils::debug >= 0) std::cout << "TIME:TOTAL PROGRAM  : "<< now << " " << difftime(now, time_start)/1.E6  << std::endl;

  exit(0);
}
