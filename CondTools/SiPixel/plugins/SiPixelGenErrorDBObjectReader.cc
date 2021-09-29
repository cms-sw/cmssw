// This reader still does not work so well.
// The formatting of readout is wrong.
// The comparison only works if the ascii files are available.
// I do not know how to access the internal numbers in a usefull way?
#include <iomanip>
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/SiPixelTransient/interface/SiPixelGenError.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelGenErrorDBObject.h"
#include "CondFormats/DataRecord/interface/SiPixelGenErrorDBObjectRcd.h"
#include "CalibTracker/Records/interface/SiPixelGenErrorDBObjectESProducerRcd.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

class SiPixelGenErrorDBObjectReader : public edm::one::EDAnalyzer<> {
public:
  explicit SiPixelGenErrorDBObjectReader(const edm::ParameterSet&);
  ~SiPixelGenErrorDBObjectReader() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  std::string theGenErrorCalibrationLocation;
  bool theDetailedGenErrorDBErrorOutput;
  bool theFullGenErrorDBOutput;

  edm::ESGetToken<SiPixelGenErrorDBObject, SiPixelGenErrorDBObjectRcd> genErrToken_;
};

SiPixelGenErrorDBObjectReader::SiPixelGenErrorDBObjectReader(const edm::ParameterSet& iConfig)
    : theGenErrorCalibrationLocation(iConfig.getParameter<std::string>("siPixelGenErrorCalibrationLocation")),
      theDetailedGenErrorDBErrorOutput(iConfig.getParameter<bool>("wantDetailedGenErrorDBErrorOutput")),
      theFullGenErrorDBOutput(iConfig.getParameter<bool>("wantFullGenErrorDBOutput")),
      genErrToken_(esConsumes()) {}

SiPixelGenErrorDBObjectReader::~SiPixelGenErrorDBObjectReader() = default;

void SiPixelGenErrorDBObjectReader::beginJob() {}

void SiPixelGenErrorDBObjectReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::LogPrint("SiPixelGenErrorDBObjectReader") << "\nLoading ... " << std::endl;

  SiPixelGenErrorDBObject dbobject = *&iSetup.getData(genErrToken_);
  const SiPixelGenErrorDBObject* db = &iSetup.getData(genErrToken_);

  // these seem to be the only variables I can get directly from the object class
  edm::LogPrint("SiPixelGenErrorDBObjectReader")
      << " DBObject version " << dbobject.version() << " index " << dbobject.index() << " max " << dbobject.maxIndex()
      << " fail " << dbobject.fail() << " numOfTeml " << dbobject.numOfTempl() << std::endl;

  if (theFullGenErrorDBOutput) {
    edm::LogPrint("SiPixelGenErrorDBObjectReader") << "Map info" << std::endl;
    std::vector<short> tempMapId;
    std::map<unsigned int, short> templMap = dbobject.getGenErrorIDs();
    for (std::map<unsigned int, short>::const_iterator it = templMap.begin(); it != templMap.end(); ++it) {
      if (tempMapId.empty())
        tempMapId.push_back(it->second);
      for (unsigned int i = 0; i < tempMapId.size(); ++i) {
        if (tempMapId[i] == it->second)
          continue;
        else if (i == tempMapId.size() - 1) {
          tempMapId.push_back(it->second);
          break;
        }
      }

      edm::LogPrint("SiPixelGenErrorDBObjectReader") << "DetId: " << it->first << " GenErrorID: " << it->second << "\n";
    }

    edm::LogPrint("SiPixelGenErrorDBObjectReader") << "\nMap stores GenError Id(s): ";
    for (unsigned int vindex = 0; vindex < tempMapId.size(); ++vindex)
      edm::LogPrint("SiPixelGenErrorDBObjectReader") << tempMapId[vindex] << " ";
    edm::LogPrint("SiPixelGenErrorDBObjectReader") << std::endl;
  }

  // if the dircetory is an empty string ignore file comparison
  if (theGenErrorCalibrationLocation.empty()) {
    edm::LogPrint("SiPixelGenErrorDBObjectReader")
        << " no file for camparison defined, comparison will be skipped " << std::endl;

  } else {  // do the file comparision

    bool error = false;
    int numOfTempl = dbobject.numOfTempl();
    int index = 0;
    float tempnum = 0, diff = 0;
    float tol = 1.0E-23;
    bool givenErrorMsg = false;

    edm::LogPrint("SiPixelGenErrorDBObjectReader")
        << "\nChecking GenError DB object version " << dbobject.version() << " containing " << numOfTempl
        << " calibration(s) at " << dbobject.sVector()[index + 22] << "T\n";

    for (int i = 0; i < numOfTempl; ++i) {
      //Removes header in db object from diff
      index += 20;

      //Tell the person viewing the output what the GenError ID and version are -- note that version is only valid for >=13
      // Does not work correctly for data
      edm::LogPrint("SiPixelGenErrorDBObjectReader")
          << "Calibration " << i + 1 << " of " << numOfTempl << ", with GenError ID " << dbobject.sVector()[index]
          << "\tand Version " << dbobject.sVector()[index + 1] << "\t--------  " << std::endl;

      //Opening the text-based GenError calibration
      std::ostringstream tout;
      tout << theGenErrorCalibrationLocation.c_str() << "generror_summary_zp" << std::setw(4) << std::setfill('0')
           << std::right << dbobject.sVector()[index] << ".out" << std::ends;

      std::string temp = tout.str();
      std::ifstream in_file(temp.c_str(), std::ios::in);

      edm::LogPrint("SiPixelGenErrorDBObjectReader")
          << " open file " << tout.str() << " " << in_file.is_open() << std::endl;

      if (in_file.is_open()) {
        //Removes header in textfile from diff
        //First read in from the text file -- this will be compared with index = 20
        in_file >> tempnum;

        //Read until the end of the current text file
        while (!in_file.eof()) {
          //Calculate the difference between the text file and the db object
          diff = std::abs(tempnum - dbobject.sVector()[index]);

          //Is there a difference?
          if (diff > tol) {
            //We have the if statement to output the message only once
            if (!givenErrorMsg)
              edm::LogPrint("SiPixelGenErrorDBObjectReader") << "does NOT match\n";
            //If there is an error we want to display a message upon completion
            error = true;
            givenErrorMsg = true;
            //Do we want more detailed output?
            if (theDetailedGenErrorDBErrorOutput) {
              edm::LogPrint("SiPixelGenErrorDBObjectReader")
                  << "from file = " << tempnum << "\t from dbobject = " << dbobject.sVector()[index]
                  << "\tdiff = " << diff << "\t db index = " << index << std::endl;
            }
          }
          //Go to the next entries
          in_file >> tempnum;
          ++index;
        }
        //There were no errors, the two files match.
        if (!givenErrorMsg)
          edm::LogPrint("SiPixelGenErrorDBObjectReader") << "MATCHES\n";
      } else {  //end current file
        edm::LogPrint("SiPixelGenErrorDBObjectReader")
            << " ERROR: cannot open file, comparison will be stopped" << std::endl;
        break;
      }
      in_file.close();
      givenErrorMsg = false;

    }  //end loop over all files

    if (error && !theDetailedGenErrorDBErrorOutput)
      edm::LogPrint("SiPixelGenErrorDBObjectReader")
          << "\nThe were differences found between the files and the database.\n"
          << "If you would like more detailed information please set\n"
          << "wantDetailedOutput = True in the cfg file. If you would like a\n"
          << "full output of the contents of the database file please set\n"
          << "wantFullOutput = True. Make sure that you pipe the output to a\n"
          << "log file. This could take a few minutes.\n\n";

  }  // if compare

  // Try to interpret the object
  std::vector<SiPixelGenErrorStore> thePixelGenError;
  bool status = SiPixelGenError::pushfile(*db, thePixelGenError);
  edm::LogPrint("SiPixelGenErrorDBObjectReader")
      << " status = " << status << " size = " << thePixelGenError.size() << std::endl;

  SiPixelGenError genError(thePixelGenError);
  // these are all 0 because qbin() was not run.
  edm::LogPrint("SiPixelGenErrorDBObjectReader")
      << " some values " << genError.lorxwidth() << " " << genError.lorywidth() << " " << std::endl;

  // Print the full object, I think it does not work, the print is for templates.
  if (theFullGenErrorDBOutput)
    edm::LogPrint("SiPixelGenErrorDBObjectReader") << dbobject << std::endl;
}

void SiPixelGenErrorDBObjectReader::endJob() {}

// I think this was written for templates and not for genErrors
// so it does not work correctly.
std::ostream& operator<<(std::ostream& s, const SiPixelGenErrorDBObject& dbobject) {
  //!-index to keep track of where we are in the object
  int index = 0;
  //!-these are modifiable parameters for the extended GenErrors
  int txsize[4] = {7, 13, 0, 0};
  int tysize[4] = {21, 21, 0, 0};
  //!-entries takes the number of entries in By,Bx,Fy,Fx from the object
  int entries[4] = {0};
  //!-local indicies for loops
  int i, j, k, l, m, n, entry_it;
  //!-changes the size of the GenErrors based on the version
  int sizeSetter = 1, generrorVersion = 0;

  edm::LogPrint("SiPixelGenErrorDBObjectReader") << "\n\nDBobject version: " << dbobject.version() << std::endl;

  for (m = 0; m < dbobject.numOfTempl(); ++m) {
    //To change the size of the output based on which GenError version we are using"
    generrorVersion = (int)dbobject.sVector_[index + 21];

    edm::LogPrint("SiPixelGenErrorDBObjectReader") << " GenError version " << generrorVersion << " " << m << std::endl;

    if (generrorVersion <= 10) {
      edm::LogPrint("SiPixelGenErrorDBObjectReader")
          << "*****WARNING***** This code will not format this GenError version properly *****WARNING*****\n";
      sizeSetter = 0;
    } else if (generrorVersion <= 16)
      sizeSetter = 1;
    else
      edm::LogPrint("SiPixelGenErrorDBObjectReader")
          << "*****WARNING***** This code has not been tested at formatting this version *****WARNING*****\n";

    edm::LogPrint("SiPixelGenErrorDBObjectReader")
        << "\n\n*********************************************************************************************"
        << std::endl;
    edm::LogPrint("SiPixelGenErrorDBObjectReader")
        << "***************                  Reading GenError ID " << dbobject.sVector_[index + 20] << "\t(" << m + 1
        << "/" << dbobject.numOfTempl_ << ")                 ***************" << std::endl;
    edm::LogPrint("SiPixelGenErrorDBObjectReader")
        << "*********************************************************************************************\n\n"
        << std::endl;

    //Header Title
    edm::LogPrint("SiPixelGenErrorDBObjectReader") << " Header Title" << std::endl;
    SiPixelGenErrorDBObject::char2float temp;
    for (n = 0; n < 20; ++n) {
      temp.f = dbobject.sVector_[index];
      s << temp.c[0] << temp.c[1] << temp.c[2] << temp.c[3];
      ++index;
    }

    entries[0] = (int)dbobject.sVector_[index + 3];                                   // Y
    entries[1] = (int)(dbobject.sVector_[index + 4] * dbobject.sVector_[index + 5]);  // X

    //Header
    edm::LogPrint("SiPixelGenErrorDBObjectReader") << " Header " << std::endl;
    s << dbobject.sVector_[index] << "\t" << dbobject.sVector_[index + 1] << "\t" << dbobject.sVector_[index + 2]
      << "\t" << dbobject.sVector_[index + 3] << "\t" << dbobject.sVector_[index + 4] << "\t"
      << dbobject.sVector_[index + 5] << "\t" << dbobject.sVector_[index + 6] << "\t" << dbobject.sVector_[index + 7]
      << "\t" << dbobject.sVector_[index + 8] << "\t" << dbobject.sVector_[index + 9] << "\t"
      << dbobject.sVector_[index + 10] << "\t" << dbobject.sVector_[index + 11] << "\t" << dbobject.sVector_[index + 12]
      << "\t" << dbobject.sVector_[index + 13] << "\t" << dbobject.sVector_[index + 14] << "\t"
      << dbobject.sVector_[index + 15] << "\t" << dbobject.sVector_[index + 16] << std::endl;
    index += 17;

    //Loop over By,Bx,Fy,Fx
    edm::LogPrint("SiPixelGenErrorDBObjectReader") << " ByBxFyFx" << std::endl;
    for (entry_it = 0; entry_it < 4; ++entry_it) {
      //Run,costrk,qavg,...,clslenx
      for (i = 0; i < entries[entry_it]; ++i) {
        s << dbobject.sVector_[index] << "\t" << dbobject.sVector_[index + 1] << "\t" << dbobject.sVector_[index + 2]
          << "\t" << dbobject.sVector_[index + 3] << "\n"
          << dbobject.sVector_[index + 4] << "\t" << dbobject.sVector_[index + 5] << "\t"
          << dbobject.sVector_[index + 6] << "\t" << dbobject.sVector_[index + 7] << "\t"
          << dbobject.sVector_[index + 8] << "\t" << dbobject.sVector_[index + 9] << "\t"
          << dbobject.sVector_[index + 10] << "\t" << dbobject.sVector_[index + 11] << "\n"
          << dbobject.sVector_[index + 12] << "\t" << dbobject.sVector_[index + 13] << "\t"
          << dbobject.sVector_[index + 14] << "\t" << dbobject.sVector_[index + 15] << "\t"
          << dbobject.sVector_[index + 16] << "\t" << dbobject.sVector_[index + 17] << "\t"
          << dbobject.sVector_[index + 18] << std::endl;
        index += 19;
        //YPar
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " YPar" << std::endl;
        for (j = 0; j < 2; ++j) {
          for (k = 0; k < 5; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //YTemp
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " YTemp" << std::endl;
        for (j = 0; j < 9; ++j) {
          for (k = 0; k < tysize[sizeSetter]; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //XPar
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " XPar" << std::endl;
        for (j = 0; j < 2; ++j) {
          for (k = 0; k < 5; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //XTemp
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " Xtemp" << std::endl;
        for (j = 0; j < 9; ++j) {
          for (k = 0; k < txsize[sizeSetter]; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //Y average reco params
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " Y average reco params " << std::endl;
        for (j = 0; j < 4; ++j) {
          for (k = 0; k < 4; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //Yflpar
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " Yflar " << std::endl;
        for (j = 0; j < 4; ++j) {
          for (k = 0; k < 6; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //X average reco params
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " X average reco params" << std::endl;
        for (j = 0; j < 4; ++j) {
          for (k = 0; k < 4; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //Xflpar
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " Xflar" << std::endl;
        for (j = 0; j < 4; ++j) {
          for (k = 0; k < 6; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //Chi2X,Y
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " XY chi2" << std::endl;
        for (j = 0; j < 4; ++j) {
          for (k = 0; k < 2; ++k) {
            for (l = 0; l < 2; ++l) {
              s << dbobject.sVector_[index] << "\t";
              ++index;
            }
          }
          s << std::endl;
        }
        //Y average Chi2 params
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " Y chi2" << std::endl;
        for (j = 0; j < 4; ++j) {
          for (k = 0; k < 4; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //X average Chi2 params
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " X chi2" << std::endl;
        for (j = 0; j < 4; ++j) {
          for (k = 0; k < 4; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //Y average reco params for CPE Generic
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " Y reco params for generic" << std::endl;
        for (j = 0; j < 4; ++j) {
          for (k = 0; k < 4; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //X average reco params for CPE Generic
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " X reco params for generic" << std::endl;
        for (j = 0; j < 4; ++j) {
          for (k = 0; k < 4; ++k) {
            s << dbobject.sVector_[index] << "\t";
            ++index;
          }
          s << std::endl;
        }
        //SpareX,Y
        edm::LogPrint("SiPixelGenErrorDBObjectReader") << " Spare " << std::endl;
        for (j = 0; j < 20; ++j) {
          s << dbobject.sVector_[index] << "\t";
          ++index;
          if (j == 9 || j == 19)
            s << std::endl;
        }
      }
    }
  }
  return s;
}

DEFINE_FWK_MODULE(SiPixelGenErrorDBObjectReader);
