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

};

#endif /* ALIGNMENT_MILLEPEDEALIGNMENTALGORITHM_INTERFACE_MILLEPEDEFILEREADER_H_ */
