#ifndef ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_
#define ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_

/*** system includes ***/
#include <array>
#include <string>

/*** core framework functionality ***/
#include "FWCore/ParameterSet/interface/ParameterSet.h"


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

    bool updateDB    = false;
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

#endif /* ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_ */
