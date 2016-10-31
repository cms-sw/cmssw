/*** Header file ***/
#include "Alignment/MillePedeAlignmentAlgorithm/interface/MillePedeFileReader.h"

/*** system includes ***/
#include <cmath>		// include floating-point std::abs functions
#include <fstream>

/*** core framework functionality ***/
#include "FWCore/MessageLogger/interface/MessageLogger.h"



//=============================================================================
//===   PUBLIC METHOD IMPLEMENTATION                                        ===
//=============================================================================

MillePedeFileReader
::MillePedeFileReader(const edm::ParameterSet& config) :
  millePedeLogFile_(config.getParameter<std::string>("millePedeLogFile")),
  millePedeResFile_(config.getParameter<std::string>("millePedeResFile")),

  sigCut_     (config.getParameter<double>("sigCut")),
  Xcut_       (config.getParameter<double>("Xcut")),
  tXcut_      (config.getParameter<double>("tXcut")),
  Ycut_       (config.getParameter<double>("Ycut")),
  tYcut_      (config.getParameter<double>("tYcut")),
  Zcut_       (config.getParameter<double>("Zcut")),
  tZcut_      (config.getParameter<double>("tZcut")),
  maxMoveCut_ (config.getParameter<double>("maxMoveCut")),
  maxErrorCut_(config.getParameter<double>("maxErrorCut"))
{
}

void MillePedeFileReader
::read() {
  readMillePedeLogFile();
  readMillePedeResultFile();
}

bool MillePedeFileReader
::storeAlignments() {
  return updateDB;
}



//=============================================================================
//===   PRIVATE METHOD IMPLEMENTATION                                       ===
//=============================================================================

void MillePedeFileReader
::readMillePedeLogFile()
{
  std::ifstream logFile;
  logFile.open(millePedeLogFile_.c_str());

  if (logFile.is_open()) {
    edm::LogInfo("MillePedeFileReader") << "Reading millepede log-file";
    std::string line;

    while (getline(logFile, line)) {
      std::string Nrec_string = "NREC =";

      if (line.find(Nrec_string) != std::string::npos) {
        std::istringstream iss(line);
        std::string trash;
        iss >> trash >> trash >> Nrec;

        if (Nrec < 25000) {
          updateDB   = false;
        }
      }
    }

  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede log-file.";

    updateDB   = false;
    Nrec = 0;
  }
}

void MillePedeFileReader
::readMillePedeResultFile()
{
  updateDB = false;	
  std::ifstream resFile;
  resFile.open(millePedeResFile_.c_str());

  if (resFile.is_open()) {
    edm::LogInfo("MillePedeFileReader") << "Reading millepede result-file";
    double Multiplier[6] = {10000.,10000.,10000.,1000000.,1000000.,1000000.};

    std::string line;
    getline(resFile, line); // drop first line

    while (getline(resFile, line)) {
      std::istringstream iss(line);

      std::vector<std::string> tokens;
      std::string token;
      while (iss >> token) {
        tokens.push_back(token);
      }

      if (tokens.size() > 4 /*3*/) {

        int alignable      = std::stoi(tokens[0]);
        int alignableIndex = alignable % 10 - 1;

        double ObsMove = std::stof(tokens[3]) * Multiplier[alignableIndex];
        double ObsErr  = std::stof(tokens[4]) * Multiplier[alignableIndex];

        int det = -1;

        if (alignable >= 60 && alignable <= 69) {
          det = 2; // TPBHalfBarrel (x+)
        } else if (alignable >= 8780 && alignable <= 8789) {
          det = 3; // TPBHalfBarrel (x-)
        } else if (alignable >= 17520 && alignable <= 17529) {
          det = 4; // TPEHalfCylinder (x+,z+)
        } else if (alignable >= 22380 && alignable <= 22389) {
          det = 5; // TPEHalfCylinder (x-,z+)
        } else if (alignable >= 27260 && alignable <= 27269) {
          det = 0; // TPEHalfCylinder (x+,z-)
        } else if (alignable >= 32120 && alignable <= 32129) {
          det = 1; //TPEHalfCylinder (x-,z-)
        } else {
          continue;
        }

        if (alignableIndex == 0 && det >= 0 && det <= 5) {
          Xobs[det] = ObsMove;
          XobsErr[det] = ObsErr;
        } else if (alignableIndex == 1 && det >= 0 && det <= 5) {
          Yobs[det] = ObsMove;
          YobsErr[det] = ObsErr;
        } else if (alignableIndex == 2 && det >= 0 && det <= 5) {
          Zobs[det] = ObsMove;
          ZobsErr[det] = ObsErr;
        } else if (alignableIndex == 3 && det >= 0 && det <= 5) {
          tXobs[det] = ObsMove;
          tXobsErr[det] = ObsErr;
        } else if (alignableIndex == 4 && det >= 0 && det <= 5) {
          tYobs[det] = ObsMove;
          tYobsErr[det] = ObsErr;
        } else if (alignableIndex == 5 && det >= 0 && det <= 5) {
          tZobs[det] = ObsMove;
          tZobsErr[det] = ObsErr;
        }

	if (std::abs(ObsMove) > maxMoveCut_) {
          updateDB    = false;
          break;

        } else if (std::abs(ObsMove) > Cutoffs[alignableIndex]) {
	  
	  if (std::abs(ObsErr) > maxErrorCut_) {
            updateDB    = false;
            break;
          } else {
  	    if (std::abs(ObsMove/ObsErr) < sigCut_) {
	      continue;
            } 
          }
	  updateDB = true;
        }
      }
    }
  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede result-file.";

    updateDB   = false;
    Nrec = 0;
  }
}
