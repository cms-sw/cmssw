//   COCOA class implementation file
//Id:  Model.cc
//CAT: Model
//
//   History: v1.0
//   Pedro Arce

#include "Alignment/CocoaModel/interface/Model.h"

#include "Alignment/CocoaUtilities/interface/ALIFileIn.h"
#include "Alignment/CocoaUtilities/interface/GlobalOptionMgr.h"
#include "Alignment/CocoaModel/interface/OpticalObject.h"
#include "Alignment/CocoaModel/interface/Measurement.h"
#include "Alignment/CocoaModel/interface/MeasurementSensor2D.h"
#include "Alignment/CocoaModel/interface/MeasurementDistancemeter.h"
#include "Alignment/CocoaModel/interface/MeasurementDistancemeter3dim.h"
#include "Alignment/CocoaModel/interface/MeasurementTiltmeter.h"
#include "Alignment/CocoaModel/interface/MeasurementCOPS.h"
#include "Alignment/CocoaModel/interface/MeasurementDiffEntry.h"
#include "Alignment/CocoaModel/interface/CocoaDaqReaderText.h"
#include "Alignment/CocoaModel/interface/CocoaDaqReaderRoot.h"
#include "Alignment/CocoaUtilities/interface/ALIUtils.h"
#include "Alignment/CocoaModel/interface/EntryAngle.h"
#include "Alignment/CocoaModel/interface/ParameterMgr.h"
#include "Alignment/CocoaModel/interface/ErrorCorrelationMgr.h"
//#include "Alignment/Scan/interface/ScanMgr.h"
#include "Alignment/CocoaModel/interface/EntryMgr.h"
#include "Alignment/CocoaModel/interface/EntryData.h"
#include "Alignment/CocoaModel/interface/FittedEntriesReader.h"

#include "CondFormats/OptAlignObjects/interface/OpticalAlignments.h"
#include "CondFormats/OptAlignObjects/interface/OpticalAlignMeasurements.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <cstdlib>
#include <cctype>
//#include <algo.h>
#include <cassert>
#include <ctime>

#include <algorithm>

#ifdef OS_OSPACE_STD_NAMESPACE
using namespace os_std;
#endif

//using namespace os_std;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Model* Model::theInstance = nullptr;
//map< ALIstring, ALIdouble, std::less<ALIstring> > Model::theParameters;
std::vector<std::vector<ALIstring> > Model::theOptODictionary;
//-map< ALIstring, int, std::less<ALIstring> > Model::theStandardMeasurerTypes;
//-multimap< ALIstring, OpticalObject*, std::less<ALIstring> > Model::_OptOtree;
//map< ALIstring, OpticalObject*, std::less<ALIstring> > Model::theOptOlist;
std::vector<OpticalObject*> Model::theOptOList;
std::vector<Entry*> Model::theEntryVector;
std::vector<Measurement*> Model::theMeasurementVector;
std::vector<ALIdouble> Model::theParamFittedSigmaVector;
std::map<ALIstring, ALIdouble, std::less<ALIstring> > Model::theParamFittedValueDisplacementMap;
std::vector<OpticalObject*> Model::theOptOsToCopyList;
std::vector<OpticalObject*>::const_iterator Model::theOptOsToCopyListIterator;
ALIint Model::CMSLinkIteration = 0;
ALIint Model::Ncmslinkrange = 0;
std::vector<ALIdouble> Model::CMSLinkRangeDetValue;
ALIstring Model::theSDFName = "SystemDescription.txt";
ALIstring Model::theMeasFName = "Measurements.txt";
ALIstring Model::theReportFName = "report.out";
ALIstring Model::theMatricesFName = "matrices.out";
//struct tm Model::theMeasurementsTime = struct tm();
// struct tm Model::theMeasurementsTime;
cocoaStatus Model::theCocoaStatus = COCOA_Init;
FittedEntriesReader* Model::theFittedEntriesReader = nullptr;
std::vector<OpticalAlignInfo> Model::theOpticalAlignments;

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Gets the only instance of Model
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Model& Model::getInstance() {
  if (!theInstance) {
    theInstance = new Model;
  }
  return *theInstance;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Constructor
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Model::Model() {
  //  theMeasurementsTime = clock();
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Reads the System Description file section by section and acts accordingly
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::readSystemDescription() {
  Model::setCocoaStatus(COCOA_Init);

  ALIint data_reading = 0;  // variable to identify the first line of section SYSTEM_TREE_DATA

  //---------- Open the system description file
  ALIFileIn& filein = ALIFileIn::getInstance(Model::SDFName());

  //----------- Set section titles
  std::vector<ALIstring> SectionTitle;
  SectionTitle.push_back(ALIstring("GLOBAL_OPTIONS"));
  SectionTitle.push_back(ALIstring("PARAMETERS"));
  SectionTitle.push_back(ALIstring("SYSTEM_TREE_DESCRIPTION"));
  SectionTitle.push_back(ALIstring("SYSTEM_TREE_DATA"));
  SectionTitle.push_back(ALIstring("MEASUREMENTS"));
  SectionTitle.push_back(ALIstring("REPORT.OUT"));
  std::vector<ALIstring>::iterator SectionTitleIterator;

  //---------------------------------------- Loops lines in SDF file
  std::vector<ALIstring> wordlist;
  ALIint InSectionNo = -1;
  ALIint currentSectionNo = -1;
  while (!filein.eof()) {
    if (!filein.getWordsInLine(wordlist))
      break;  //----- Read line
    assert(!wordlist.empty());

    //----- checking
    if (ALIUtils::debug > 99) {
      ALIUtils::dumpVS(wordlist, " ", std::cout);
    }

    //---------- Get in which section the current line is and act accordingly
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    //@@@ ---------- Check if word is start of section
    SectionTitleIterator = find(SectionTitle.begin(), SectionTitle.end(), *wordlist.begin());
    if (SectionTitleIterator != SectionTitle.end()) {
      //---------- Check that previous sections are correct
      currentSectionNo = SectionTitleIterator - SectionTitle.begin();
      if (currentSectionNo != InSectionNo + 1) {
        if (currentSectionNo != sectReportOut) {
          ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
          std::cerr << "BAD ORDER OF SECTIONS, reading section " << *SectionTitleIterator << std::endl
                    << " currentSectionNo = " << currentSectionNo << " InSectionNo = " << InSectionNo << std::endl
                    << " ---------  Please see documentation  ---------- " << std::endl;
          exit(1);
        }
      } else {
        if (currentSectionNo != sectReportOut) {
          InSectionNo++;
        }
      }
      if (currentSectionNo == sectMeasurements) {
        SetValueDisplacementsFromReportOut();
      }

      if (ALIUtils::debug >= 4)
        std::cout << std::endl << "START OF SECTION: " << currentSectionNo << " " << *SectionTitleIterator << std::endl;

      //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      //@@@ ---------- Reading GLOBAL_OPTIONS section
    } else if (currentSectionNo == sectGlobalOptions) {
      //-       std::cout << "utilsdebug " << ALIUtils::debug << std::endl;
      //-------- Check format of line read
      //----- Two words
      if (wordlist.size() == 2) {
        //----- Second word is number
        int isnumber = ALIUtils::IsNumber(wordlist[1]);
        if (!isnumber && wordlist[0] != ALIstring("external_meas")) {
          ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
          std::cerr << ": EXPECTING A NUMBER, FOUND: " << wordlist[1] << std::endl;
          exit(2);
        }

        GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
        gomgr->setGlobalOption(wordlist[0], ALIUtils::getFloat(wordlist[1]), ALIFileIn::getInstance(Model::SDFName()));

        //-or    GlobalOptions.insert(std::pair<const ALIstring, ALIdouble>(wordlist[0] , atof(wordlist[1].c_str()) ) );

        if (ALIUtils::debug >= 1) {
          ALIUtils::dumpVS(wordlist, "GLOBAL_OPTION:  ", std::cout);
        }

      } else {
        std::cout << "error < 1" << std::endl;
        ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
        std::cerr << ": IN GLOBAL_OPTIONS section TWO-WORD LINES ARE EXPECTED " << std::endl;
        exit(2);
      }

      //------- Set dimension factors for lengths and angles
      ALIUtils::SetLengthDimensionFactors();
      ALIUtils::SetAngleDimensionFactors();
      ALIUtils::SetOutputLengthDimensionFactors();
      ALIUtils::SetOutputAngleDimensionFactors();

      //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      //@@@ ---------- Reading PARAMETERS section
    } else if (currentSectionNo == sectParameters ||
               currentSectionNo ==
                   -1) {  // Allow parameters in first lines (for easy study of different parameter values)

      //---------- Check format of line read
      //----- Two words
      if (wordlist.size() == 2) {
        /* now checked in ParameterMgr
        //----- Second is number
        int isnumber = ALIUtils::IsNumber( wordlist[1] );
        if( !isnumber ) {
          ALIFileIn::getInstance( Model::SDFName() ).ErrorInLine();
          std::cerr << ": EXPECTING A NUMBER, FOUND: " << wordlist[1] << std::endl;
          exit(2);
          } */

        //old---------- Create parameter with value read (or overwrite existing value)
        //---------- Create parameter with value read if parameter does not exist yet
        ParameterMgr* parmgr = ParameterMgr::getInstance();
        parmgr->addParameter(wordlist[0], wordlist[1]);

      } else if (wordlist.size() == 3) {
        if (wordlist[1] != "seed") {
          ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
          std::cerr << ": For a three-word parameter line, second has to be 'seed', it is  " << wordlist[1]
                    << std::endl;
          exit(1);
        }

        if (wordlist[0] == "gauss" || wordlist[0] == "flat") {
          ParameterMgr::getInstance()->setRandomSeed(ALIUtils::getInt(wordlist[2]));
        } else {
          ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
          std::cerr << ": For a three-word parameter line, first has to be 'gauss' or 'flat', it is  " << wordlist[0]
                    << std::endl;
          exit(1);
        }
      } else if (wordlist.size() == 4) {
        if (wordlist[0] == "gauss") {
          ParameterMgr::getInstance()->addRandomGaussParameter(wordlist[1], wordlist[2], wordlist[3]);
        } else if (wordlist[0] == "flat") {
          ParameterMgr::getInstance()->addRandomFlatParameter(wordlist[1], wordlist[2], wordlist[3]);
        } else {
          ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
          std::cerr << ": For a four-word parameter line, first has to be 'gauss' or 'flat', it is  " << wordlist[0]
                    << std::endl;
          exit(1);
        }
      } else {
        ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
        std::cerr << ": IN PARAMETERS section TWO-WORD-LINES ARE EXPECTED " << std::endl;
        ALIUtils::dumpVS(wordlist, " ");
        exit(2);
      }

      //print it out
      if (ALIUtils::debug >= 1) {
        ALIUtils::dumpVS(wordlist, "PARAMETERS:  ", std::cout);
      }

      //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      //@@@ ---------- Reading SYSTEM_TREE_DESCRIPTION section
    } else if (currentSectionNo == sectSystemTreeDescription) {
      //---------- Convert the numbers found in list of components (e.g. 2 laser -> laser laser )
      //----- Backup wordlist and erase it
      std::vector<ALIstring> wordlist2 = wordlist;
      std::vector<ALIstring>::iterator vsite;
      ALIint wsiz = wordlist.size();
      for (ALIint ii = 0; ii < wsiz; ii++) {
        wordlist.pop_back();
      }
      //----- Loop objects looking for numbers to convert
      for (vsite = wordlist2.begin(); vsite != wordlist2.end(); ++vsite) {
        if (ALIUtils::IsNumber(*vsite)) {
          int nOpticalObjects = atoi((*vsite).c_str());
          //----- If number is 1 it is not necessary
          if (nOpticalObjects == 1) {
            if (ALIUtils::debug >= 0)
              std::cerr << "WARNING: in line " << ALIFileIn::getInstance(Model::SDFName()).nline()
                        << " number of repeated OpticalObjects = 1. Please avoid the numbering " << std::endl;
            //-    wordlist.erase( wordlist.begin() + (vsite-wordlist2.begin()) );
          } else {
            //----- The number cannot be the last in the list and you cannot put two numbers together
            if (vsite + 1 == wordlist.end() || ALIUtils::IsNumber(*(vsite + 1))) {
              ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
              std::cerr << "NUMBER SHOULD BE FOLLOWED BY an OpticalObject type" << std::endl;
              exit(2);
            }
            //----- Right format: convert
            //--- Substitute the number by the object type in wordlist
            //-   *( wordlist.begin() + (vsite-wordlist2.begin()) ) = *(vsite+1);
            //--- Add n-1 object types to wordlist (the nth object will be added as the object taht goes after the number)
            for (ALIint ii = 0; ii < nOpticalObjects - 1; ii++) {
              //-std::cout << ii << "inserting in wordlist " << *(vsite+1) << std::endl;
              wordlist.push_back(*(vsite + 1));
            }
          }
        } else {
          //----- Not number, add it to wordlist
          wordlist.push_back(*vsite);
        }
      }

      //---------- Dump system structure
      if (ALIUtils::debug >= 1) {
        ALIUtils::dumpVS(wordlist, "SYSTEM TREE DESCRIPTION: before ordering, OBJECT: ", std::cout);
      }

      //---------- Fill the list of Optical Object with components (theOptODictionary)
      //----- First word is 'object': new OptO
      if (wordlist[0] == ALIstring("object")) {
        //----- Check out repeated objects
        std::vector<std::vector<ALIstring> >::iterator vvsite;
        for (vvsite = theOptODictionary.begin(); vvsite != theOptODictionary.end(); ++vvsite) {
          //-     std::cout << " system" << vvsite << std::endl;

          if (*((*vvsite).begin()) == wordlist[1]) {
            ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
            std::cerr << "SYSTEM_TREE_DESCRIPTION: REPEATED object " << *((*vvsite).begin())
                      << " ( NOT ALLOWED NEITHER WITH EQUAL NOR WITH DIFFERENT COMPONENTS)" << std::endl;
            exit(1);
          }
        }
        //------- Add an item to theOptODictionary
        std::vector<ALIstring> vstemp;
        copy(wordlist.begin() + 1, wordlist.end(), back_inserter(vstemp));
        Model::OptODictionary().push_back(vstemp);
      } else {
        //----- First word is not 'object': add to previous OptO
        if (OptODictionary().empty()) {
          ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
          std::cerr << "SYSTEM_TREE_DESCRIPTION section: FIRST LINE SHOULD START WITH 'object'" << std::endl;
          exit(2);
        }
        copy(wordlist.begin(), wordlist.end(), back_inserter(*(OptODictionary().end() - 1)));
      }

      //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      //---------------------------------- Reading SYSTEM_TREE_DATA section
    } else if (currentSectionNo == sectSystemTreeData) {
      //---------- First line of data:
      if (!data_reading) {
        //        ALIint dictsize = theOptODictionary.size();
        //------- Reorder theOptODictionary
        std::vector<std::vector<ALIstring> > OptODictionary2;
        reorderOptODictionary("system", OptODictionary2);
        if (OptODictionary2.empty()) {
          std::cerr << "SYSTEM_TREE_DESCRIPTION section: no object 'system' found " << std::endl;
          exit(9);
        }
        //------- We start reordering with 'system', therefore if 'system' is not the top most object, the objects not hanging from it would not be considered
        //---- Check if all the objects are here
        std::vector<std::vector<ALIstring> >::const_iterator vvscite, vvscite2;
        //      ALIint dictsizen = 0;
        for (vvscite = theOptODictionary.begin(); vvscite != theOptODictionary.end(); ++vvscite) {
          ALIbool ofound = false;
          for (vvscite2 = OptODictionary2.begin(); vvscite2 != OptODictionary2.end(); ++vvscite2) {
            if (*((*vvscite).begin()) == *((*vvscite2).begin())) {
              ofound = true;
              break;
            }
          }
          if (!ofound) {
            std::cerr << "!!!SYSTEM_TREE_DESCRIPTION section: object " << *((*vvscite).begin())
                      << " is not hanging from object 'system' " << std::endl;
            for (vvscite = OptODictionary().begin(); vvscite != OptODictionary().end(); ++vvscite) {
              const std::vector<ALIstring>& ptemp = *vvscite;
              ALIUtils::dumpVS(ptemp, "OBJECT ", std::cerr);
            }
            exit(9);
          }
        }
        theOptODictionary = OptODictionary2;

        data_reading = 1;

        //------- Dump ordered OptOs
        if (ALIUtils::debug >= 3) {
          std::vector<std::vector<ALIstring> >::iterator itevs;
          for (itevs = OptODictionary().begin(); itevs != OptODictionary().end(); ++itevs) {
            std::vector<ALIstring> ptemp = *itevs;
            ALIUtils::dumpVS(ptemp, " SYSTEM TREE DESCRIPTION: after ordering: OBJECT ", std::cout);
          }
        }

        //---------- Create OpticalObject 'system' (first OpticalObject object):
        //---------- it will create its components and recursively all the System Tree of Optical Objects
        if (wordlist[0] != "system") {
          std::cerr << "SYSTEM_TREE_DATA section: object 'system' is not the first one " << std::endl;
          exit(9);
        }

        OpticalObject* OptOsystem = new OpticalObject(nullptr, "system", wordlist[1], false);
        OptOsystem->construct();
        //-              Model::_OptOtree.insert( std::multimap< ALIstring, OpticalObject*, std::less<ALIstring> >::value_type(OptOsystem->type(), OptOsystem) );
        //              theOptOlist[OptOsystem->name()] = OptOsystem;
        theOptOList.push_back(OptOsystem);

      } else {
        //----------- All system is read by the Optical Objects, it should not reach here
        ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
        std::cerr << " STILL SOME LINES AFTER ALL SYSTEM TREE IS READ!!!" << std::endl;
        exit(9);
      }

      //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      //----------------------------------- Reading MEASUREMENTS section
    } else if (currentSectionNo == sectMeasurements) {
      //---------- Create Measurement with appropiate dimension
      Measurement* meastemp = nullptr;
      ALIstring measType = wordlist[0];
      ALIstring measName;
      if (wordlist.size() == 2) {
        measName = wordlist[1];
        wordlist.pop_back();
      } else {
        measName = "";
      }
      if (measType == ALIstring("SENSOR2D")) {
        meastemp = new MeasurementSensor2D(2, measType, measName);
        meastemp->setConversionFactor(wordlist);
        meastemp->construct();
      } else if (measType == ALIstring("DISTANCEMETER3DIM")) {
        meastemp = new MeasurementDistancemeter3dim(1, measType, measName);
        meastemp->setConversionFactor(wordlist);
        meastemp->construct();
      } else if (measType == ALIstring("DISTANCEMETER") || measType == ALIstring("DISTANCEMETER1DIM")) {
        meastemp = new MeasurementDistancemeter(1, measType, measName);
        meastemp->setConversionFactor(wordlist);
        meastemp->construct();
      } else if (measType == ALIstring("TILTMETER")) {
        meastemp = new MeasurementTiltmeter(1, measType, measName);
        meastemp->setConversionFactor(wordlist);
        meastemp->construct();
      } else if (measType == ALIstring("COPS")) {
        meastemp = new MeasurementCOPS(4, measType, measName);
        meastemp->setConversionFactor(wordlist);
        meastemp->construct();
      } else if (measType == ALIstring("DIFFENTRY")) {
        meastemp = new MeasurementDiffEntry(1, measType, measName);
        meastemp->construct();
      } else if (measType == ALIstring("measurements_from_file") || measType == ALIstring("@measurements_from_file")) {
        new CocoaDaqReaderText(wordlist[1]);
        //m Measurement::setMeasurementsFileName( wordlist[1] );
        //m if ( ALIUtils::debug >= 2) std::cout << " setting measurements_from_file " << measType << " == " << Measurement::measurementsFileName() << std::endl;
        if (wordlist.size() == 4) {
          Measurement::only1 = true;
          Measurement::only1Date = wordlist[2];
          Measurement::only1Time = wordlist[3];
          //-      std::cout << " setting Measurement::only1" <<  Measurement::only1 << std::endl;
        }
      } else if (measType == ALIstring("measurements_from_file_ROOT") ||
                 measType == ALIstring("@measurements_from_file")) {
        new CocoaDaqReaderRoot(wordlist[1]);
      } else if (wordlist[0] == ALIstring("correlations_from_file") ||
                 wordlist[0] == ALIstring("@correlations_from_file")) {
        ErrorCorrelationMgr::getInstance()->readFromReportFile(wordlist[1]);
      } else if (wordlist[0] == ALIstring("copy_measurements") || wordlist[0] == ALIstring("@copy_measurements")) {
        copyMeasurements(wordlist);
        //      } else if( wordlist[0] == "scan" || wordlist[0] == "@scan" ) {
        //      ScanMgr::getInstance()->addOptOEntry( wordlist );
      } else if (wordlist[0] == ALIstring("fittedEntries_from_file")) {
        theFittedEntriesReader = new FittedEntriesReader(wordlist[1]);
        if (ALIUtils::debug >= 2)
          std::cout << " setting fittedEntries_from_file " << wordlist[0] << " == " << wordlist[1] << std::endl;
      } else {
        std::cerr << "Measurement:" << std::endl;
        ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
        std::cerr << "!!! type of measurement not allowed: " << wordlist[0] << std::endl;
        std::cerr << " Allowed types: SENSOR2D, DISTANCEMETER, DISTANCEMETER1DIM, TILTMETER, COPS, DIFFENTRY "
                  << std::endl;
        exit(2);
      }
      //-      meastemp->setGlobalName( wordlist[0] );
      //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      //@@@ ---------- Reading REPORT OUT  section
    } else if (currentSectionNo == sectReportOut) {
      //----- It must be after global options section
      if (InSectionNo + 1 != sectParameters) {
        ALIFileIn::getInstance(Model::SDFName()).ErrorInLine();
        std::cerr << "BAD ORDER OF SECTIONS, reading section " << *SectionTitleIterator << std::endl
                  << " currentSectionNo = " << currentSectionNo << " InSectionNo = " << InSectionNo << std::endl
                  << " ---------  Please see documentation  ---------- " << std::endl;
        exit(1);
      }

      EntryMgr::getInstance()->readEntryFromReportOut(wordlist);
    }
  }

  //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  //@@@ All system read
  //---------- make Measurements links to their OptOs
  if (ALIUtils::debug >= 2)
    std::cout << std::endl << "@@@@ Building Measurements links to OptOs" << std::endl;
  Model::buildMeasurementsLinksToOptOs();

  if (ALIUtils::debug >= 1) {
    std::cout << "----------  SYSTEM SUCCESFULLY READ ----------" << std::endl << std::endl;
  }
  filein.close();

  return;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  reorderOptODictionary: REBUILDS THE SYSTEM_TREE_DESCRIPTION SECTION 'objects'
//@@ (_OptODictionary) IN A HIERARCHICAL (TREE LIKE) ORDER
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::reorderOptODictionary(const ALIstring& ssearch, std::vector<std::vector<ALIstring> >& OptODictionary2) {
  //  int ALIstring_found = 0;
  std::vector<std::vector<ALIstring> >::iterator vvsite;
  std::vector<ALIstring>::iterator vsite;

  //---------------------------------------- Look for 'ssearch' as the first ALIstring of an OptODictionary item
  for (vvsite = OptODictionary().begin(); vvsite != OptODictionary().end(); ++vvsite) {
    if (*((*vvsite).begin()) == ssearch) {
      //     ALIstring_found = 1;
      OptODictionary2.push_back(*vvsite);

      //-    std::cout << "VVSITE";
      //-   ostream_iterator<ALIstring> outs(std::cout,"&");
      //-   copy( (*vvsite).begin(), (*vvsite).end(), outs);

      //---------------------------------- look for components of this _OptODictionary item
      for (vsite = (*vvsite).begin() + 1; vsite != (*vvsite).end(); ++vsite) {
        reorderOptODictionary(*vsite, OptODictionary2);
      }
      break;
    }
  }

  /*  //------- object 'system' should exist
  if(!ALIstring_found && ssearch == "system") {
    std::cerr << "SYSTEM_TREE_DATA section: no 'object system' found " << std::endl;
    exit(9);
    } */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ BuildMeasurementLinksToOptOs
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::buildMeasurementsLinksToOptOs() {
  //---------- Loop Measurements
  std::vector<Measurement*>::const_iterator vmcite;
  for (vmcite = MeasurementList().begin(); vmcite != MeasurementList().end(); ++vmcite) {
    //---------- Transform for each Measurement the Measured OptO names to Measured OptO pointers
    //     (*vmcite)->buildOptOList();

    //---------- Build list of Entries that affect a Measurement
    // (*vmcite)->buildAffectingEntryList();
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Get the value of a parameter in theParameters std::vector
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIint Model::getParameterValue(const ALIstring& sstr, ALIdouble& val) {
  ParameterMgr* parmgr = ParameterMgr::getInstance();
  ALIint iret = parmgr->getParameterValue(sstr, val);

  return iret;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ getOptOByName: Find an OptO name in _OptOlist and return a pointer to it
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OpticalObject* Model::getOptOByName(const ALIstring& opto_name) {
  //---------- Look for Optical Object name in OptOList
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = OptOList().begin(); vocite != OptOList().end(); ++vocite) {
    if ((*vocite)->name() == opto_name)
      break;
  }

  if (vocite == OptOList().end()) {
    //---------- If opto_name not found, exit
    std::cerr << " LIST OF OpticalObjects " << std::endl;
    for (vocite = OptOList().begin(); vocite != OptOList().end(); ++vocite) {
      std::cerr << (*vocite)->name() << std::endl;
    }
    std::cerr << "!!EXITING at getOptOByName: Optical Object " << opto_name << " doesn't exist!!" << std::endl;
    exit(4);
    //       return (OpticalObject*)0;
  } else {
    //---------- If opto_name found, return pointer to it
    if (ALIUtils::debug > 999) {
      std::cout << opto_name.c_str() << "SSOptOitem" << (*vocite) << (*vocite)->name() << "len" << OptOList().size()
                << std::endl;
    }
    return (*vocite);
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ getOptOByType: Find an OptO type in _OptOList (the first one with this type)
//@@ and return a pointer to it
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OpticalObject* Model::getOptOByType(const ALIstring& opto_type) {
  //---------- Look for Optical Object type in OptOList
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = OptOList().begin(); vocite != OptOList().end(); ++vocite) {
    //   std::cout << "OPTOList" << (*msocite).first << std::endl;
    if ((*vocite)->type() == opto_type)
      break;
  }

  if (vocite == OptOList().end()) {
    //---------- If opto_type not found, exit
    std::cerr << "!!EXITING at getOptOByType: Optical Object " << opto_type << " doesn't exist!!" << std::endl;
    exit(4);
  } else {
    //---------- If opto_type found, return pointer to it
    return (*vocite);
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Find an Entry name in theEntryVector and return a pointer to it
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Entry* Model::getEntryByName(const ALIstring& opto_name, const ALIstring& entry_name) {
  //---------- Look for Entry name type in EntryList
  std::vector<Entry*>::const_iterator vecite;
  for (vecite = EntryList().begin(); vecite != EntryList().end(); ++vecite) {
    if (ALIUtils::debug >= 4)
      std::cout << "getEntryByName: " << (*vecite)->OptOCurrent()->name() << " E " << (*vecite)->name()
                << " Searching: " << opto_name << " E " << entry_name << std::endl;
    //-    std::cout << " optoName " << (*vecite)->OptOCurrent()->name()<< " " << (*vecite)->name() << std::endl;
    if ((*vecite)->OptOCurrent()->name() == opto_name && (*vecite)->name() == entry_name) {
      return *vecite;
    }
  }
  //---------- Entry not found!
  std::cerr << "!!!EXITING at getEntryByName: Entry name not found:" << opto_name << "  " << entry_name << std::endl;
  exit(1);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Measurement* Model::getMeasurementByName(const ALIstring& meas_name, ALIbool exists) {
  //---------- Look for Optical Object name in OptOList
  std::vector<Measurement*>::const_iterator vmcite;
  for (vmcite = theMeasurementVector.begin(); vmcite != theMeasurementVector.end(); ++vmcite) {
    if ((*vmcite)->name() == meas_name)
      break;
  }

  if (vmcite != theMeasurementVector.end()) {
    //---------- If opto_name found, return pointer to it
    return (*vmcite);
  } else {
    if (exists) {
      //---------- If opto_name not found, exit
      std::cerr << " LIST OF Measurements " << std::endl;
      for (vmcite = theMeasurementVector.begin(); vmcite != theMeasurementVector.end(); ++vmcite) {
        std::cerr << (*vmcite)->name() << std::endl;
      }
      std::cerr << "!!EXITING at getMeasurementByName: Measurement " << meas_name << " doesn't exist!!" << std::endl;
      abort();
      //       return (OpticalObject*)0;
    } else {
      return nullptr;
    }
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Get an OptO list of components
//@@  Looks the theOptODictionary item that has 'opto_type' as the first ALIstring,
//@@  copies this item to 'vcomponents', substracting the first ALIstring, that is the opto_type itself,
//@@  Returns 1 if item found, 0 if not
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool Model::getComponentOptOTypes(const ALIstring& opto_type, std::vector<ALIstring>& vcomponents) {
  //---------- clean std::vector in which you are going to store opto types
  std::vector<ALIstring>::iterator vsite;
  for (vsite = vcomponents.begin(); vsite != vcomponents.end(); ++vsite) {
    vcomponents.pop_back();
  }

  //---------- Looks the theOptODictionary item that has 'opto_type' as the first ALIstring,
  ALIint ALIstring_found = 0;
  std::vector<std::vector<ALIstring> >::iterator vvsite;
  for (vvsite = OptODictionary().begin(); vvsite != OptODictionary().end(); ++vvsite) {
    if (*((*vvsite).begin()) == opto_type) {
      ALIstring_found = 1;
      //tt  copies this item to 'vcomponents', substracting the first ALIstring, that is the opto_type itself,
      vcomponents = *vvsite;
      vcomponents.erase(vcomponents.begin());
      break;
    }
  }

  if (ALIstring_found) {
    return true;
  } else {
    return false;
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ Get from _OptOList the list of OptOs pointers that has as parent 'opto_name'
//@@ and store it in vcomponents
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool Model::getComponentOptOs(const ALIstring& opto_name, std::vector<OpticalObject*>& vcomponents) {
  //---------- clean std::vector in which you are going to store opto pointers
  std::vector<OpticalObject*>::iterator voite;
  for (voite = vcomponents.begin(); voite != vcomponents.end(); ++voite) {
    vcomponents.pop_back();
  }

  //---------- Get OptO corresponding to name 'opto_name'
  OpticalObject* opto = getOptOByName(opto_name);

  if (ALIUtils::debug >= 99)
    std::cout << opto_name << "getComponentOptOs: opto " << opto << opto->name() << std::endl;
  std::vector<OpticalObject*>::const_iterator vocite;

  if (ALIUtils::debug >= 99)
    std::cout << "optolist size " << OptOList().size() << std::endl;
  ALIbool opto_found = false;
  for (vocite = OptOList().begin(); vocite != OptOList().end(); ++vocite) {
    if ((*vocite)->parent() != nullptr) {
      //        std::cout << "looping OptOlist" << (*vocite)->name() << " parent " <<(*vocite)->parent()->name() << std::endl;
      if ((*vocite)->parent()->name() == opto_name) {
        opto_found = true;
        vcomponents.push_back((*vocite));
      }
    }
  }

  /*  std::pair<multimap< ALIstring, OpticalObject*, std::less<ALIstring> >::iterator,
       std::multimap< ALIstring, OpticalObject*, std::less<ALIstring> >::iterator>
  pmmao =  _OptOtree.equal_range(opto_name);

  if( pmmao.first != _OptOtree.end()) {
    std::multimap< ALIstring, OpticalObject*, std::less<ALIstring> >::const_iterator socite;
    for (socite = pmmao.first; socite != (pmmao.second); socite++) {
         vcomponents.push_back( (*socite).second );
    }
  }
  */
  return opto_found;
}

///**************** FOR COPYING AN OPTO
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ createCopyComponentList:
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool Model::createCopyComponentList(const ALIstring& typ) {
  //---------- Find an OptO with the same type (YOU HAVE TO BE SURE THAT ALL EXISTING OPTOs OF THIS TYPE HAVE SAME COMPONENTS, IF NOT COPYING MAY GIVE YOU UNPREDICTABLE RESULTS)
  if (ALIUtils::debug >= 3)
    std::cout << "createCopyComponentList " << typ << std::endl;
  OpticalObject* start_opto = getOptOByType(typ);

  //---------- clean list of OptOs to copy
  theOptOsToCopyList.erase(theOptOsToCopyList.begin(), theOptOsToCopyList.end());

  //---------- Fill list of OptOs to copy
  fillCopyComponentList(start_opto);
  //- if(ALIUtils::debug >= 9) std::cout << "createCopyComponentList " << typ << theOptOsToCopyList.size() << std::endl;

  theOptOsToCopyListIterator = theOptOsToCopyList.begin();
  return true;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ fillCopyOptOList: Fill list of objects to copy with the components of 'opto'
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool Model::fillCopyComponentList(const OpticalObject* opto) {
  if (ALIUtils::debug >= 3)
    std::cout << "entering fillCopyComponentList(): OptO" << opto->name() << std::endl;
  //---------- Get list of components of 'opto'
  std::vector<OpticalObject*> vopto;
  ALIbool opto_found = getComponentOptOs(opto->name(), vopto);
  if (!opto_found) {
    if (ALIUtils::debug >= 5)
      std::cout << "fillCopyComponentList: NO COMPONENTS TO COPY IN THIS OptO" << opto->name() << std::endl;
  }

  //---------- Loop list of components of 'opto'
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vocite = vopto.begin(); vocite != vopto.end(); ++vocite) {
    theOptOsToCopyList.push_back(*vocite);
    if (ALIUtils::debug >= 5)
      std::cout << "fillCopyOptOList " << (*vocite)->type() << " " << (*vocite)->name() << std::endl;
    //---------- Add components of this component
    fillCopyComponentList(*vocite);
  }
  return opto_found;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ nextOptOToCopy: return next object to copy from theOptOsToCopyListIterator
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OpticalObject* Model::nextOptOToCopy() {
  if (ALIUtils::debug >= 5)
    std::cout << "entering nextOptOToCopy() " << std::endl;
  ++theOptOsToCopyListIterator;
  //  if(ALIUtils::debug >= 5) std::cout <<" nextOptOToCopy " << (*(theOptOsToCopyListIterator-1))->name() << std::endl;
  return *(theOptOsToCopyListIterator - 1);
}

///*************** FOR CMS LINK SYSTEM (to fit it part by part)
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ If fitting CMS link, it has to be fitted part by part, in several iterations
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::CMSLinkFit(ALIint cmslink) {
  /*
  //---------- Get number of fits
  ALIint cmslinkNoFits = 0;
  if( cmslink == 1) {
    cmslinkNoFits = 3;
  } else if( cmslink == 2) {
    cmslinkNoFits = 2;
  }
  if(ALIUtils::debug >= 4) std::cout << " Start CMS link fit with method " << cmslink << " (number of fits = " << cmslinkNoFits << ")" << std::endl;

  //----------- Loop number of cms_link fits
  for(ALIint ilink = ALIint(Model::GlobalOptions()["cms_link"]); ilink <= cmslinkNoFits; ilink++) {

  //----- Iteration 2 of method 2 resembles iteration 3 of method 1
     ALIdouble cmslink_method;
     assert(Model::getGlobalOptionValue("cms_link_method", cmslink_method));
     if( cmslink_method == 2 && ilink == 2) ilink = 3;

//---- Set variable CMSLinkIteration, Checked in Fit.C and other     //- std::cout << "ilink" << ilink << std::endl;
    Model::setGlobalOption("cms_link", ilink);
    Model::CMSLinkIteration = ilink;

    if(ilink > 1)Model::readSystemDescription(); //already read once to fill value Model::GlobalOptions()["cms_link"]

    //---------- Delete the OptO not fitted in this iteration
    //    Model::CMSLinkDeleteOptOs();
    // cannot be here because you may recover a parameter saved in previous iteration that now it is deleted (!!MODIFY THIS)

    //---------- Recover parameters fitted in previous iteration
    Model::CMSLinkRecoverParamFittedSigma( ilink );

    Model::CMSLinkRecoverParamFittedValueDisplacement( ilink );

    //---------- Delete the OptO not fitted in this iteration
    Model::CMSLinkDeleteOptOs();

    //---------- Start fit
    Fit::startFit();

    //---------- Save parameters fitted in this iteration (to be used in next one)
    Model::CMSLinkSaveParamFittedSigma( ilink );

    Model::CMSLinkSaveParamFittedValueDisplacement( ilink );

    //---------- Delete whole system to start anew in next iteration
    Model::CMSLinkCleanModel();

  }
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ CMSLinkCleanModel: clean Model for new iteration while fitting CMS link
//@@ part by part
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::CMSLinkCleanModel() {
  deleteOptO("s");
  //---------- Clean OptOdicitionary (in case this is not first reading)
  ALIuint odsize = theOptODictionary.size();
  for (ALIuint ii = 0; ii < odsize; ii++) {
    theOptODictionary.pop_back();
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ CMSLinkDeleteOptOs
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::CMSLinkDeleteOptOs() {
  ALIint cmslink_iter = Model::CMSLinkIteration;
  ALIdouble cmslink_method;

  GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
  assert(gomgr->getGlobalOptionValue("cms_link_method", cmslink_method));
  ALIdouble cmslink_halfplanes;
  assert(gomgr->getGlobalOptionValue("cms_link_halfplanes", cmslink_halfplanes));
  if (ALIUtils::debug >= 2)
    std::cout << "CMSLinkDeleteOptOs: cms_link_halfplanes " << cmslink_halfplanes << cmslink_iter << std::endl;

  if (cmslink_iter == 1) {
    //---------- First fit: delete everything but laser1 and det_tkU
    //    deleteOptO("s/laserboxL/laser2");
    //-    std::cout << "delete mabs" << std::endl;
    deleteOptO("s/mabsL");
    //------- Method 1: detectors at tracker down
    if (cmslink_method == 1) {
      deleteOptO("s/tracker/det_trkDL");
      deleteOptO("s/tracker/det_trkDR");
    }

    if (cmslink_halfplanes == 2) {
      //      deleteOptO("s/laserboxR/laser2");
      deleteOptO("s/mabsR");
    }

  } else if (cmslink_iter == 2) {
    //---------- Second fit (method 1): delete everything but laser1 and det3
    //    deleteOptO("s/laserboxL/laser2");
    deleteOptO("s/mabsL");
    deleteOptO("s/tracker/CST/wheel_trkL/peri/mirror");  //??
    deleteOptO("s/tracker/CST/wheel_trkL/det_trkU");
    //------- Method 1: detectors on CST, Method 2: detectors on tracker
    //not necessary    deleteOptO("s/tracker/CST/det6");

    if (cmslink_halfplanes <= 1) {
      deleteOptO("s/tracker/CST/wheel_trkR");
    } else if (cmslink_halfplanes == 2) {
      //      deleteOptO("s/laserboxR/laser2");
      deleteOptO("s/mabsR");
      deleteOptO("s/tracker/CST/wheel_trkR/peri/mirror");  //??
      deleteOptO("s/tracker/CST/wheel_trkR/det_trkU");
    }

  } else if (cmslink_iter == 3) {
    //---------- Third fit: delete everything but laser2 and mabs
    //    deleteOptO("s/laserboxL/laser1");
    deleteOptO("s/tracker");

    if (cmslink_halfplanes == 2) {
      //      deleteOptO("s/laserboxR/laser1");
    }
    //---------- Do nothing
  } else {
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ CMSLinkSaveParamFittedSigma
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::CMSLinkSaveParamFittedSigma(ALIint cms_link) {
  /*
  ALIdouble cms_link_halfplanes = (GlobalOptions()["cms_link_halfplanes"]);

  cleanParamFittedSigmaVector();

  //---------- First fit: Save laserbox angles fitted at tracker
  if( cms_link == 1) {
    //?    if (GlobalOptions()["cms_link_method"] < 10){
    saveParamFittedSigma("s/laserboxL","Angles X");
    saveParamFittedSigma("s/laserboxL","Angles Y");

    saveParamFittedCorrelation("s/laserboxL","Angles X",
     "s/tracker/CST","Centre Y");
    saveParamFittedCorrelation("s/laserboxL","Angles Y",
     "s/tracker/CST","Centre X");
    if( cms_link_halfplanes == 2) {
      saveParamFittedSigma("s/laserboxR","Angles X");
      saveParamFittedSigma("s/laserboxR","Angles Y");

      saveParamFittedCorrelation("s/laserboxR","Angles X",
       "s/tracker/CST","Centre Y");
      saveParamFittedCorrelation("s/laserboxR","Angles Y",
       "s/tracker/CST","Centre X");
    }

  } else if( cms_link == 2) {
  //---------- Second fit: Save laserbox angles and position
    saveParamFittedSigma("s/laserboxL","Centre X");
    saveParamFittedSigma("s/laserboxL","Centre Y");
    // Make quality unk to cal
    saveParamFittedSigma("s/laserboxL","Centre Z");
    saveParamFittedSigma("s/laserboxL","Angles X");
    saveParamFittedSigma("s/laserboxL","Angles Y");

    saveParamFittedCorrelation("s/laserboxL","Centre X",
        "s/laserboxL","Angles Y");
    saveParamFittedCorrelation("s/laserboxL","Centre Y",
        "s/laserboxL","Angles X");

   if( cms_link_halfplanes == 2) {
    saveParamFittedSigma("s/laserboxR","Centre X");
    saveParamFittedSigma("s/laserboxR","Centre Y");
    // Make quality unk to cal
    saveParamFittedSigma("s/laserboxR","Centre Z");
    saveParamFittedSigma("s/laserboxR","Angles X");
    saveParamFittedSigma("s/laserboxR","Angles Y");

    saveParamFittedCorrelation("s/laserboxR","Centre X",
        "s/laserboxR","Angles Y");
    saveParamFittedCorrelation("s/laserboxR","Centre Y",
        "s/laserboxR","Angles X");
   }
  } else {
 //---------- Do nothing

  }
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ CMSLinkSaveParamFittedValueDisplacement:
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::CMSLinkSaveParamFittedValueDisplacement(ALIint cms_link) {
  /*
  cleanParamFittedValueDisplacementMap();

  //----------
  if( cms_link == 1 || cms_link == 2 ) {
    std::vector<Entry*>::const_iterator vecite;
    for( vecite = EntryList().begin(); vecite != EntryList().end(); vecite++) {
      if( (*vecite)->valueDisplacementByFitting() != 0 ) {
        ALIstring names = (*vecite)->OptOCurrent()->name() + "/" + (*vecite)->name();
        std::cout << "saeParamFittedValueDisplacementMap" << names << (*vecite)->valueDisplacementByFitting() << std::endl;
        theParamFittedValueDisplacementMap[ names ] = (*vecite)->valueDisplacementByFitting();
      }
    }

  //---------- Do nothing
  } else {

  }
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ CMSLinkRecoverParamFittedSigma:
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::CMSLinkRecoverParamFittedSigma(ALIint cms_link) {
  /*
  ALIdouble cms_link_halfplanes = (GlobalOptions()["cms_link_halfplanes"]);

  if( cms_link == 2) {
  //---------- Second fit: recover laserbox angles
    recoverParamFittedSigma("s/laserboxL","Angles X",0);
    recoverParamFittedSigma("s/laserboxL","Angles Y",1);

    if( cms_link_halfplanes == 2) {
      recoverParamFittedSigma("s/laserboxR","Angles X",0);
      recoverParamFittedSigma("s/laserboxR","Angles Y",1);
    }

  } else if( cms_link == 3) {
  //---------- Third fit: recover laserbox angles and position and rotate angles to mabs
    recoverParamFittedSigma("s/laserboxL","Centre X",0);
    recoverParamFittedSigma("s/laserboxL","Centre Y",1);
    recoverParamFittedSigma("s/laserboxL","Centre Z",2);
    recoverParamFittedSigma("s/laserboxL","Angles X",3);

    //----- Angle around Y is converted to angle around Z when turning 90 deg
    Entry* slaZ = getEntryByName("s/laserboxL","Angles Z");
    //--- prec_level_laser
    Entry* smaZ = getEntryByName("s/mabsL","Angles Z");
    slaZ->setQuality(0);
    slaZ->setValue( smaZ->value() );
    //    smaZ->setQuality(0); //!!???!!?

    Entry* slaY = getEntryByName("s/laserboxL","Angles Y");
    slaY->setQuality(0);

    if( cms_link_halfplanes == 2) {
      recoverParamFittedSigma("s/laserboxR","Centre X",0);
      recoverParamFittedSigma("s/laserboxR","Centre Y",1);
      recoverParamFittedSigma("s/laserboxR","Centre Z",2);
      recoverParamFittedSigma("s/laserboxR","Angles X",3);

      //----- Angle around Y is converted to angle around Z when turning 90 deg
      Entry* slaZ = getEntryByName("s/laserboxR","Angles Z");
      //--- prec_level_laser
      Entry* smaZ = getEntryByName("s/mabsR","Angles Z");
      slaZ->setQuality(0);
      slaZ->setValue( smaZ->value() );
      //    smaZ->setQuality(0); //!!???!!?

      Entry* slaY = getEntryByName("s/laserboxR","Angles Y");
      slaY->setQuality(0);
    }
  } else {
  //---------- Do nothing

  }
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@ CMSLinkRecoverParamFittedValueDisplacement:
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::CMSLinkRecoverParamFittedValueDisplacement(ALIint cms_link) {
  /*
  //---------- Second fit: recover laserbox angles
  if( cms_link == 2 || cms_link == 3 ) {

    std::map<ALIstring, ALIdouble, std::less<ALIstring> >::const_iterator vsdmite;
    if ( ALIUtils::debug >= 99) std::cout << "theParamFittedValueDisplacementMap.size " << theParamFittedValueDisplacementMap.size() << std::endl;
    for( vsdmite =  theParamFittedValueDisplacementMap.begin(); vsdmite !=  theParamFittedValueDisplacementMap.end(); vsdmite++) {
      std::cout << "reoverValueDisp" <<  (*vsdmite).first << "  " << (*vsdmite).second << std::endl;
      Entry* this_entry = getEntryByName( (*vsdmite).first);
      this_entry->displaceOriginal( (*vsdmite).second );
      this_entry->OptOCurrent()->resetGlobalCoordinates();
      this_entry->setValueDisplacementByFitting(  (*vsdmite).second );

    }

  //---------- Do nothing
  } else {

  }
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  Delete an OptO
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::deleteOptO(const ALIstring& opto_name) {
  OpticalObject* opto = getOptOByName(opto_name);
  deleteOptO(opto);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  deleteOptO: delete OptO, its Entries, and the Measurements in which it participates. Then deleteOptO of components
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::deleteOptO(OpticalObject* opto) {
  if (ALIUtils::debug >= 5)
    std::cout << "DELETING OptO" << opto->name() << std::endl;

  //---------- Delete Coordinate Entries of this OptO
  std::vector<Entry*>::const_iterator vecite;
  std::vector<Entry*>::iterator veite2;
  if (ALIUtils::debug >= 9)
    std::cout << "SIZE" << theEntryVector.size() << std::endl;
  for (vecite = opto->CoordinateEntryList().begin(); vecite != opto->CoordinateEntryList().end(); ++vecite) {
    //    ALIuint pos = FindItemInVector( (*veite), opto->CoordinateEntryList() );
    veite2 = find(theEntryVector.begin(), theEntryVector.end(), (*vecite));
    //-  if ( ALIUtils::debug >= 9) std::cout << (*veite2) << "DELETE ENTRY " << (*vecite) <<(*veite2)->OptOCurrent()->name() << (*veite2)->name() << std::endl;
    delete ((*veite2));
    theEntryVector.erase(veite2);
  }

  for (vecite = opto->ExtraEntryList().begin(); vecite != opto->ExtraEntryList().end(); ++vecite) {
    //    ALIuint pos = FindItemInVector( (*veite), opto->CoordinateEntryList() );
    veite2 = find(theEntryVector.begin(), theEntryVector.end(), (*vecite));
    //-    if(ALIUtils::debug >= 9) std::cout << (*veite2) << "DELETE ENTRY " << (*veite2)->OptOCurrent()->name() << (*veite2)->name() << std::endl;
    delete ((*veite2));
    theEntryVector.erase(veite2);
  }

  for (vecite = theEntryVector.begin(); vecite != theEntryVector.end(); ++vecite) {
    //     std::cout << (*vecite) << "ENTReY " << (*vecite)->OptOCurrent()->name() << (*vecite)->name() << std::endl;
  }

  //---------- Delete all Measurement in which opto takes part
  std::vector<Measurement*> MeasToBeDeleted;
  std::vector<Measurement*>::const_iterator vmite;
  std::vector<OpticalObject*>::const_iterator vocite;
  for (vmite = MeasurementList().begin(); vmite != MeasurementList().end(); ++vmite) {
    if (ALIUtils::debug >= 5)
      std::cout << "Deleting Measurement" << (*vmite)->name() << std::endl;
    //----- If any of the OptO Measured is opto, delete this Measurement
    for (vocite = (*vmite)->OptOList().begin(); vocite != (*vmite)->OptOList().end(); ++vocite) {
      if ((*vocite) == opto) {
        //-      std::cout << "MEASTBD" << (*vmite) << std::endl;
        MeasToBeDeleted.push_back(*vmite);
        //?       delete (*vmite);
        break;
      }
    }
  }

  //---------- Delete Measurements from list
  std::vector<Measurement*>::const_iterator vmcite;
  std::vector<Measurement*>::iterator vmite2;
  if (ALIUtils::debug >= 9)
    std::cout << "SIZEMEAS" << MeasToBeDeleted.size() << std::endl;
  for (vmcite = MeasToBeDeleted.begin(); vmcite != MeasToBeDeleted.end(); ++vmcite) {
    vmite2 = find(theMeasurementVector.begin(), theMeasurementVector.end(), (*vmcite));
    //    std::cout << (*vmite2) << "DELETE MSEASU " << (*vmcite) << (*vmite2)->name()[0] << std::endl;
    delete ((*vmite2));
    theMeasurementVector.erase(vmite2);
  }

  //---------- Delete components
  //  std::vector<OpticalObject*>::iterator voite;
  std::vector<OpticalObject*> vopto;
  //  ALIbool opto_found = getComponentOptOs( opto->name(), vopto );
  for (vocite = vopto.begin(); vocite != vopto.end(); ++vocite) {
    deleteOptO(*vocite);
  }

  //---------- Delete this OptO
  //---------- Delete OptO (only from list, to delete it really first delete components)
  /*  map< ALIstring, OpticalObject*, std::less<ALIstring> >::iterator msoite =
find( theOptOList.begin(), theOptOList.end(),
map< ALIstring, OpticalObject*, std::less<ALIstring> >::value_type( opto->name(), opto) );*/
  std::vector<OpticalObject*>::iterator dvoite =
      find(theOptOList.begin(), theOptOList.end(), std::vector<OpticalObject*>::value_type(opto));
  //-  std::cout << (*dvoite) << "DELETE OPTO " << opto <<"WW" << (*dvoite)->name() << std::endl;
  theOptOList.erase(dvoite);
  delete opto;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::saveParamFittedSigma(const ALIstring& opto_name, const ALIstring& entry_name) {
  /*
//---------- Get sigma of param Optical
  Entry* entry = getEntryByName(opto_name, entry_name);
  ALIuint fit_pos = entry->fitPos();
  if( fit_pos < 0 || fit_pos >= Fit::propagationMatrix().NoLines()) {
    std::cerr << "!!EXITING at saveParamFittedSigma: fit position incorrect " <<
      fit_pos << "propagationMatrix size =" << Fit::propagationMatrix().NoLines() << opto_name << std::endl;
    exit(3);
  }
  std::cout << entry_name << "saveParamFittedSigma" << fit_pos << sqrt(Fit::propagationMatrix()( fit_pos, fit_pos)) << std::endl;
  theParamFittedSigmaVector.push_back( sqrt(Fit::propagationMatrix()( fit_pos, fit_pos)) );
  //-    Fit::propagationMatrix().Dump(" the 2PropagationMatrix");
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::saveParamFittedCorrelation(const ALIstring& opto_name1,
                                       const ALIstring& entry_name1,
                                       const ALIstring& opto_name2,
                                       const ALIstring& entry_name2) {
  /*
//---------- Get sigma of param Optical
  Entry* entry1 = getEntryByName(opto_name1, entry_name1);
  Entry* entry2 = getEntryByName(opto_name2, entry_name2);
  ALIuint fit_pos1 = entry1->fitPos();
  ALIuint fit_pos2 = entry2->fitPos();

  //---------- Get correlation if entry has been fitted
  ALIuint pmsize = Fit::propagationMatrix().NoLines();
  if( fit_pos1 >= 0 && fit_pos1 < pmsize && fit_pos2 >= 0 && fit_pos2 < pmsize ) {
    ALIdouble error1 = sqrt( Fit::propagationMatrix()( fit_pos1, fit_pos1) );
    ALIdouble error2 = sqrt( Fit::propagationMatrix()( fit_pos2, fit_pos2) );
    ALIdouble correl = Fit::propagationMatrix()( fit_pos1, fit_pos2) / error1 / error2;
    theParamFittedSigmaVector.push_back( correl );
    if(ALIUtils::debug>=9) {
      std::cout  << "saveParamFittedCorre" << opto_name1 << entry_name1 << fit_pos1 <<
        opto_name2 << entry_name2 << fit_pos2 << "MM " << correl << std::endl;
    }
  } else {
    if(ALIUtils::debug>=9) {
      std::cout  << "NOsaveParamFittedCorre" << opto_name1 << entry_name1 << fit_pos1 <<
        opto_name2 << entry_name2 << fit_pos2 << "MM " << std::endl;
    theParamFittedSigmaVector.push_back( 0. );
    }

  }
  //-    Fit::propagationMatrix().Dump(" the 2PropagationMatrix");
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::recoverParamFittedSigma(const ALIstring& opto_name, const ALIstring& entry_name, const ALIuint position) {
  /*
  if( position >= theParamFittedSigmaVector.size() ) {
    std::cerr << "!!EXITING at recoverParamFittedSigma: position" << position <<
 " bigger than dimension of theParamFittedSigmaVector " << theParamFittedSigmaVector.size() << std::endl;
    exit(3);
  }
  ALIdouble sigma = getParamFittedSigmaVectorItem( position );

  Entry* entry = getEntryByName(opto_name, entry_name);
  entry->setSigma( sigma );
  entry->setQuality( 1 );
  std::cout << "recover " << opto_name << entry_name << entry->sigma() <<std::endl;
  */
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  getParamFittedSigmaVectorItem
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIdouble Model::getParamFittedSigmaVectorItem(const ALIuint position) {
  if (position >= theParamFittedSigmaVector.size()) {
    std::cerr << "!!EXITING at getParamFittedSigma: position" << position
              << " bigger than dimension of theParamFittedSigmaVector " << theParamFittedSigmaVector.size()
              << std::endl;
    exit(3);
  }
  std::vector<ALIdouble>::const_iterator vdcite = theParamFittedSigmaVector.begin() + position;
  return (*vdcite);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@  getParamFittedSigmaVectorItem
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ALIbool Model::readMeasurementsFromFile(ALIstring only1Date, ALIstring only1Time) {
  if (ALIUtils::debug >= 5)
    std::cout << " readMeasurementsFromFile " << Measurement::measurementsFileName() << std::endl;
  if (Measurement::measurementsFileName().empty())
    return true;

  ALIFileIn& filein = ALIFileIn::getInstance(Measurement::measurementsFileName());
  std::vector<ALIstring> wordlist;
  //---------- read date
  //  ALIint retfil = filein.getWordsInLine(wordlist);
  // std::cout << "@@@@@@@@@@@@@@@ RETFIL " << retfil << std::endl;
  //if( retfil == 0 ) {
  if (filein.getWordsInLine(wordlist) == 0) {
    if (ALIUtils::debug >= 4)
      std::cout << "@@@@ No more measurements left" << std::endl;
    return false;
  }

  ////--- Transform to time_t format and save it
  //  struct tm tim;
  //t Model::setMeasurementsTime( tim );

  //if you are looking for only one measurement with a given date and time, loop all measurements until you find it
  if (Measurement::only1) {
    for (;;) {
      if (wordlist[0] == "DATE:" && wordlist[1] == Measurement::only1Date && wordlist[2] == Measurement::only1Time)
        break;
      filein.getWordsInLine(wordlist);
      if (filein.eof()) {
        std::cerr << "!! EXITING date not found in measurements file" << Measurement::only1Date << " "
                  << Measurement::only1Time << std::endl;
        exit(1);
      }
    }
  }

  //set date and time of current measurement
  if (wordlist[0] == "DATE:") {
    Measurement::setCurrentDate(wordlist);
  }

  //---------- loop measurements
  ALIint nMeas = Model::MeasurementList().size();
  if (ALIUtils::debug >= 4) {
    std::cout << " Reading " << nMeas << " measurements from file " << Measurement::measurementsFileName()
              << " DATE: " << wordlist[1] << " " << wordlist[1] << std::endl;
  }
  ALIint ii;
  for (ii = 0; ii < nMeas; ii++) {
    filein.getWordsInLine(wordlist);
    if (wordlist[0] == ALIstring("SENSOR2D") || wordlist[0] == ALIstring("TILTMETER") ||
        wordlist[0] == ALIstring("DISTANCEMETER") || wordlist[0] == ALIstring("DISTANCEMETER1DIM") ||
        wordlist[0] == ALIstring("COPS")) {
      if (wordlist.size() != 2) {
        std::cerr << "!!!EXITING Model::readMeasurementsFromFile. number of words should be 2 instead of "
                  << wordlist.size() << std::endl;
        ALIUtils::dumpVS(wordlist, " ");
        exit(1);
      }
      std::vector<Measurement*>::const_iterator vmcite;
      for (vmcite = MeasurementList().begin(); vmcite != MeasurementList().end(); ++vmcite) {
        //-------- Measurement found, fill data
        /*      ALIint last_slash =  (*vmcite)->name().rfind('/');
        ALIstring oname = (*vmcite)->name();
        if( last_slash != -1 ) {
          oname = oname.substr(last_slash+1, (*vmcite)->name().size()-1);
          }
        */
        ALIint fcolon = (*vmcite)->name().find(':');
        ALIstring oname = (*vmcite)->name();
        oname = oname.substr(fcolon + 1, oname.length());
        //-    std::cout << " measurement name " << (*vmcite)->name() << " short " << oname << std::endl;
        if (oname == wordlist[1]) {
          //-   std::cout << " measurement name found " << oname << std::endl;
          if ((*vmcite)->type() != wordlist[0]) {
            std::cerr << "!!! Reading measurement from file: type in file is " << wordlist[0] << " and should be "
                      << (*vmcite)->type() << std::endl;
            exit(1);
          }
          Measurement* meastemp = *vmcite;

          GlobalOptionMgr* gomgr = GlobalOptionMgr::getInstance();
          ALIbool sigmaFF = gomgr->GlobalOptions()["measurementErrorFromFile"];
          //---------- Read the data
          for (ALIuint ii = 0; ii < meastemp->dim(); ii++) {
            filein.getWordsInLine(wordlist);
            ALIdouble sigma = 0.;
            if (!sigmaFF) {
              // keep the sigma, do not read it from file
              const ALIdouble* sigmav = meastemp->sigma();
              sigma = sigmav[ii];
            }
            //---- Check measurement value type is OK
            if (meastemp->valueType(ii) != wordlist[0]) {
              filein.ErrorInLine();
              std::cerr << "!!!FATAL ERROR: Measurement value type is " << wordlist[0]
                        << " while in setup definition was " << meastemp->valueType(ii) << std::endl;
              exit(1);
            }
            meastemp->fillData(ii, wordlist);
            if (!sigmaFF) {
              meastemp->setSigma(ii, sigma);
            }
          }
          meastemp->correctValueAndSigma();
          break;
        }
      }
      if (vmcite == MeasurementList().end()) {
        for (vmcite = MeasurementList().begin(); vmcite != MeasurementList().end(); ++vmcite) {
          std::cerr << "MEAS: " << (*vmcite)->name() << " " << (*vmcite)->type() << std::endl;
        }
        std::cerr << "!!! Reading measurement from file: measurement not found in list: type in file is " << wordlist[1]
                  << std::endl;
        exit(1);
      }
    } else {
      std::cerr << " wrong type of measurement: " << wordlist[0] << std::endl
                << " Available types are SENSOR2D, TILTMETER, DISTANCEMETER, DISTANCEMETER1DIM, COPS" << std::endl;
      exit(1);
    }
  }
  //-  std::cout << " returning readmeasff" << std::endl;

  return true;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::copyMeasurements(const std::vector<ALIstring>& wl) {
  //----- Check format, e.g.  @copy_measurements _1/_2/*_1

  //----- get list of Measurement's that satisfy the query in their name
  //t  std::string querystr = wl[1].substr( wl[1].find("/")+1, wl[1].length() );
  std::string subsstr1 = wl[1].substr(0, wl[1].find('/'));
  std::string subsstr2 = wl[1].substr(wl[1].find('/') + 1, wl[1].rfind('/') - wl[1].find('/') - 1);
  std::string querystr = wl[1].substr(wl[1].rfind('/') + 1, wl[1].length());

  std::cout << " Model::copyMeasurements "
            << " subsstr1 " << subsstr1 << " subsstr2 " << subsstr2 << " querystr " << querystr << std::endl;

  std::vector<Measurement*> measToCopy;
  std::vector<Measurement*>::iterator mite;
  for (mite = theMeasurementVector.begin(); mite != theMeasurementVector.end(); ++mite) {
    Measurement* meas = (*mite);
    //improve this
    if (meas->name().find(querystr) != std::string::npos) {
      measToCopy.push_back(meas);
    }
  }

  //---- Build new measurements
  Measurement* meastemp = nullptr;
  for (mite = measToCopy.begin(); mite != measToCopy.end(); ++mite) {
    Measurement* meas = (*mite);
    std::vector<ALIstring> wlt;
    wlt.push_back(meas->type());

    //---- create new name
    std::string newName = ALIUtils::changeName(meas->name(), subsstr1, subsstr2);
    std::cout << " newName " << newName << std::endl;
    wlt.push_back(newName);

    ALIstring measType = wlt[0];
    ALIstring measName;
    if (wlt.size() == 2) {
      measName = wlt[1];
    } else {
      measName = "";
    }
    if (meas->type() == ALIstring("SENSOR2D")) {
      meastemp = new MeasurementSensor2D(2, measType, measName);
      //          } else if ( meas->type() == ALIstring("DISTANCEMETER3DIM") ) {
      //            meastemp = new MeasurementDistancemeter3dim( 1, measType, measName );
    } else if (meas->type() == ALIstring("DISTANCEMETER") || meas->type() == ALIstring("DISTANCEMETER1DIM")) {
      meastemp = new MeasurementDistancemeter(1, measType, measName);
    } else if (meas->type() == ALIstring("TILTMETER")) {
      meastemp = new MeasurementTiltmeter(1, measType, measName);
      // } else if ( meas->type() == ALIstring("DIFFCENTRE") ) {
      //   meastemp = new MeasurementDiffCentre( 1, measType, measName );
      // } else if ( meas->type() == ALIstring("DIFFANGLE") ) {
      //   meastemp = new MeasurementDiffAngle( 1, measType, measName );
    } else if (meas->type() == ALIstring("DIFFENTRY")) {
      meastemp = new MeasurementDiffEntry(1, measType, measName);
    } else if (meas->type() == ALIstring("COPS")) {
      meastemp = new MeasurementCOPS(4, measType, measName);
    } else {
      throw cms::Exception("LogicError") << "@SUB=Model::copyMeasurements\n"
                                         << "unknown measurement type: " << meas->type();
    }

    //later        meastemp->copyConversionFactor( wordlist );
    meastemp->copyMeas(meas, subsstr1, subsstr2);

    break;
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::SetValueDisplacementsFromReportOut() {
  if (ALIUtils::debug >= 3)
    std::cout << " Model::SetValueDisplacementsFromReportOut() " << std::endl;

  EntryMgr* entryMgr = EntryMgr::getInstance();

  if (entryMgr->numberOfEntries() != 0) {
    EntryData* entryData;

    std::vector<Entry*>::const_iterator vecite;
    for (vecite = Model::EntryList().begin(); vecite != Model::EntryList().end(); ++vecite) {
      //----- Find the EntryData corresponding to this entry
      entryData = entryMgr->findEntryByLongName((*vecite)->OptOCurrent()->longName(), (*vecite)->name());
      if (ALIUtils::debug >= 3)
        std::cout << "SetValueDisplacementsFromReportOut " << (*vecite)->OptOCurrent()->longName() << " "
                  << (*vecite)->name() << " " << entryData->valueDisplacement() << std::endl;
      (*vecite)->addFittedDisplacementToValue(entryData->valueDisplacement());
    }
  }
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
std::string Model::printCocoaStatus(const cocoaStatus cs) {
  std::string str = "";

  if (cs == COCOA_Init) {
    str = "COCOA_Init ";
  } else if (cs == COCOA_ReadingModel) {
    str = "COCOA_ReadingModel";
  } else if (cs == COCOA_InitFit) {
    str = "COCOA_InitFit";
  } else if (cs == COCOA_FitOK) {
    str = "COCOA_FitOK";
  } else if (cs == COCOA_FitImproving) {
    str = "COCOA_FitImproving";
  } else if (cs == COCOA_FitCannotImprove) {
    str = "COCOA_FitCannotImprove";
  } else if (cs == COCOA_FitChi2Worsened) {
    str = "COCOA_FitChi2Worsened";
  } else if (cs == COCOA_FitMatrixNonInversable) {
    str = "COCOA_FitMatrixNonInversable";
  }

  return str;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::BuildSystemDescriptionFromOA(OpticalAlignments& optAlig) {
  theOpticalAlignments = optAlig.opticalAlignments();

  OpticalAlignInfo oai_system = FindOptAlignInfoByType("system");

  OpticalObject* OptOsystem = new OpticalObject(nullptr, "system", oai_system.name_, false);

  OptOsystem->constructFromOptAligInfo(oai_system);

  //-              Model::_OptOtree.insert( std::multimap< ALIstring, OpticalObject*, std::less<ALIstring> >::value_type(OptOsystem->type(), OptOsystem) );
  //              theOptOlist[OptOsystem->name()] = OptOsystem;
  theOptOList.push_back(OptOsystem);
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
OpticalAlignInfo Model::FindOptAlignInfoByType(const ALIstring& type) {
  OpticalAlignInfo oai;

  ALIbool bFound = false;
  std::vector<OpticalAlignInfo>::iterator ite;
  for (ite = theOpticalAlignments.begin(); ite != theOpticalAlignments.end(); ++ite) {
    //    std::cout << " Model::FindOptAlignInfoByType " <<  (*ite).type_ << " =? " << type << std::endl;
    if ((*ite).type_ == type) {
      if (!bFound) {
        oai = *ite;
        bFound = true;
      } else {
        std::cerr << "!! WARNING: Model::FindOptAlignInfoByType more than one objects of type " << type << std::endl;
        std::cerr << " returning object " << oai.name_ << std::endl << " skipping object " << (*ite).name_ << std::endl;
      }
    }
  }
  if (!bFound) {
    std::cerr << "!! ERROR: Model::FindOptAlignInfoByType object not found, of type " << type << std::endl;
    std::exception();
  }

  return oai;
}

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
void Model::BuildMeasurementsFromOA(OpticalAlignMeasurements& measList) {
  std::vector<OpticalAlignMeasurementInfo>::iterator mite;

  if (ALIUtils::debug >= 5)
    std::cout << " BuildMeasurementsFromOA " << std::endl;
  std::vector<OpticalAlignMeasurementInfo> measInfos = measList.oaMeasurements_;
  for (mite = measInfos.begin(); mite != measInfos.end(); ++mite) {
    std::string measType = (*mite).type_;
    std::string measName = (*mite).name_;
    if (ALIUtils::debug >= 4)
      std::cout << " BuildMeasurementsFromOA measType " << measType << " measName " << measName << std::endl;
    //---------- Create Measurement with appropiate dimension
    Measurement* meastemp = nullptr;
    if (measType == ALIstring("SENSOR2D")) {
      meastemp = new MeasurementSensor2D(2, measType, measName);
    } else if (measType == ALIstring("DISTANCEMETER3DIM")) {
      meastemp = new MeasurementDistancemeter3dim(1, measType, measName);
    } else if (measType == ALIstring("DISTANCEMETER") || measType == ALIstring("DISTANCEMETER1DIM")) {
      meastemp = new MeasurementDistancemeter(1, measType, measName);
    } else if (measType == ALIstring("TILTMETER")) {
      meastemp = new MeasurementTiltmeter(1, measType, measName);
    } else if (measType == ALIstring("COPS")) {
      meastemp = new MeasurementCOPS(4, measType, measName);
      // } else if ( measType == ALIstring("DIFFCENTRE") ) {
      //   meastemp = new MeasurementDiffCentre( 1, measType, measName );
      // } else if ( measType == ALIstring("DIFFANGLE") ) {
      //   meastemp = new MeasurementDiffAngle( 2, measType, measName );
    } else if (measType == ALIstring("DIFFENTRY")) {
      meastemp = new MeasurementDiffEntry(1, measType, measName);
    } else {
      std::cerr << " !!! Model::BuildMeasurementsFromOA : measType not found " << measType << std::endl;
      throw std::exception();
    }
    meastemp->constructFromOA(*mite);
  }
}
