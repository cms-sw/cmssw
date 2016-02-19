#ifndef ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_
#define ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_

/*** system includes ***/
#include <array>
#include <memory>
#include <fstream>
#include <string>

/*** core framework functionality ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"



class MillePedeFileReader {

  //========================== PUBLIC METHODS ==================================
  public: //====================================================================

    explicit MillePedeFileReader(const edm::ParameterSet&);
    ~MillePedeFileReader() {}

    void read();
    bool storeAlignments();

    std::array<double, 6> const& getXobs()     const { return Xobs;     }
    std::array<double, 6> const& getXobsErr()  const { return XobsErr;  }
    std::array<double, 6> const& getTXobs()    const { return tXobs;    }
    std::array<double, 6> const& getTXobsErr() const { return tXobsErr; }

    std::array<double, 6> const& getYobs()     const { return Yobs;     }
    std::array<double, 6> const& getYobsErr()  const { return YobsErr;  }
    std::array<double, 6> const& getTYobs()    const { return tYobs;    }
    std::array<double, 6> const& getTYobsErr() const { return tYobsErr; }

    std::array<double, 6> const& getZobs()     const { return Zobs;     }
    std::array<double, 6> const& getZobsErr()  const { return ZobsErr;  }
    std::array<double, 6> const& getTZobs()    const { return tZobs;    }
    std::array<double, 6> const& getTZobsErr() const { return tZobsErr; }

  //========================= PRIVATE METHODS ==================================
  private: //===================================================================

    void readMillePedeLogFile();
    void readMillePedeResultFile();

  //========================== PRIVATE DATA ====================================
  //============================================================================

    // file-names
    std::string millePedeLogFile_;
    std::string millePedeResFile_;

    // signifiance of movement must be above
    double sigCut_;
    // cutoff in micro-meter & micro-rad
    double Xcut_, tXcut_;
    double Ycut_, tYcut_;
    double Zcut_, tZcut_;
    // maximum movement in micro-meter/rad
    double maxMoveCut_, maxErrorCut_;

    double Cutoffs[6] = {  Xcut_,  Ycut_,  Zcut_,
                          tXcut_, tYcut_, tZcut_};

    bool PedeSuccess = false;
    bool Movements   = false;
    bool Error       = false;
    bool Significant = false;
    bool updateDB    = false;
    bool HitMax      = false;
    bool HitErrorMax = false;

    int Nrec = 0;



    std::array<double, 6> Xobs     = {{0.,0.,0.,0.,0.,0.}};
    std::array<double, 6> XobsErr  = {{0.,0.,0.,0.,0.,0.}};
    std::array<double, 6> tXobs    = {{0.,0.,0.,0.,0.,0.}};
    std::array<double, 6> tXobsErr = {{0.,0.,0.,0.,0.,0.}};

    std::array<double, 6> Yobs     = {{0.,0.,0.,0.,0.,0.}};
    std::array<double, 6> YobsErr  = {{0.,0.,0.,0.,0.,0.}};
    std::array<double, 6> tYobs    = {{0.,0.,0.,0.,0.,0.}};
    std::array<double, 6> tYobsErr = {{0.,0.,0.,0.,0.,0.}};

    std::array<double, 6> Zobs     = {{0.,0.,0.,0.,0.,0.}};
    std::array<double, 6> ZobsErr  = {{0.,0.,0.,0.,0.,0.}};
    std::array<double, 6> tZobs    = {{0.,0.,0.,0.,0.,0.}};
    std::array<double, 6> tZobsErr = {{0.,0.,0.,0.,0.,0.}};

};



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
          PedeSuccess = false;
          Movements   = false;
          Error       = false;
          Significant = false;
          updateDB   = false;
        }
      }
    }

  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede log-file.";

    PedeSuccess = false;
    Movements   = false;
    Error       = false;
    Significant = false;
    updateDB   = false;
    Nrec = 0;
  }
}

void MillePedeFileReader
::readMillePedeResultFile()
{
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
        PedeSuccess = true;

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

        if (abs(ObsMove) > maxMoveCut_) {
          Movements   = false;
          Error       = false;
          Significant = false;
          updateDB   = false;
          HitMax      = false;
          continue;

        } else if (abs(ObsMove) > Cutoffs[alignableIndex]) {
          Movements = true;

          if (abs(ObsErr) > maxErrorCut_) {
            Error       = false;
            Significant = false;
            updateDB   = false;
            HitErrorMax = true;
            continue;
          } else {
            Error = true;
            if (abs(ObsMove/ObsErr) > sigCut_) {
              Significant = true;
            }
          }
        }
        updateDB = true;
      }
    }
  } else {
    edm::LogError("MillePedeFileReader") << "Could not read millepede result-file.";

    PedeSuccess = false;
    Movements   = false;
    Error       = false;
    Significant = false;
    updateDB   = false;
    Nrec = 0;
  }
}



#endif /* ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_ */
