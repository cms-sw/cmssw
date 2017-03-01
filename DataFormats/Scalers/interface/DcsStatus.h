/*X
 *  File: DataFormats/Scalers/interface/DcsStatus.h   (W.Badgett)
 *
 *  The online computed DcsStatus flag values
 *
 */

#ifndef DATAFORMATS_SCALERS_DCSSTATUS_H
#define DATAFORMATS_SCALERS_DCSSTATUS_H

#include "DataFormats/Scalers/interface/TimeSpec.h"

#include <ctime>
#include <iosfwd>
#include <vector>
#include <string>

/*! \file DcsStatus.h
 * \Header file for online DcsStatus value
 * 
 * \author: William Badgett
 *
 */

/// \class DcsStatus.h
/// \brief Persistable copy of online DcsStatus flag values

class DcsStatus
{
 public:

  static const int partitionList[];
  static const char * const partitionName[];

  enum
  {
    EBp         =     0,
    EBm         =     1,
    EEp         =     2,
    EEm         =     3,
    HBHEa       =     5,
    HBHEb       =     6,
    HBHEc       =     7,
    HF          =     8,
    HO          =     9,
    RPC         =    12,
    DT0         =    13,
    DTp         =    14,
    DTm         =    15,
    CSCp        =    16,
    CSCm        =    17,
    CASTOR      =    20,
    ZDC         =    22,
    TIBTID      =    24,
    TOB         =    25,
    TECp        =    26,
    TECm        =    27,
    BPIX        =    28,
    FPIX        =    29,
    ESp         =    30,
    ESm         =    31,
    nPartitions =    25
  };


  DcsStatus();
  DcsStatus(const unsigned char * rawData);
  virtual ~DcsStatus();

  /// name method
  std::string name() const { return "DcsStatus"; }

  /// empty method (= false)
  bool empty() const { return false; }

  unsigned int trigType() const            { return(trigType_);}
  unsigned int eventID() const             { return(eventID_);}
  unsigned int sourceID() const            { return(sourceID_);}
  unsigned int bunchNumber() const         { return(bunchNumber_);}

  int version() const                      { return(version_);}
  timespec collectionTime() const          { return(collectionTime_.get_timespec());}

  unsigned int ready()  const              { return(ready_);}

  bool ready(int partitionNumber) const 
  { return(  (ready_ & ( 1 << partitionNumber )) != 0 );}

  float magnetCurrent() const     { return(magnetCurrent_);}
  float magnetTemperature() const { return(magnetTemperature_);}

  /// equality operator
  int operator==(const DcsStatus& e) const { return false; }

  /// inequality operator
  int operator!=(const DcsStatus& e) const { return false; }

protected:

  unsigned int trigType_;
  unsigned int eventID_;
  unsigned int sourceID_;
  unsigned int bunchNumber_;

  int version_;

  TimeSpec collectionTime_;
  unsigned int ready_;
  float magnetCurrent_;
  float magnetTemperature_;
};

/// Pretty-print operator for DcsStatus
std::ostream& operator<<(std::ostream& s, const DcsStatus& c);

typedef std::vector<DcsStatus> DcsStatusCollection;

#endif
