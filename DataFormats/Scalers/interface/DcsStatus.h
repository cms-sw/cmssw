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

  enum
  {
    EB_p       =     0,
    EB_minus   =     1,
    EE_plus    =     2,
    EE_minus   =     3,
    HBHEa      =     5,
    HBHEb      =     6,
    HBHEc      =     7,
    HF         =     8,
    HO         =     9,
    RPC        =    12,
    DT0        =    13,
    DT_plus    =    14,
    DT_minus   =    15,
    CSC_plus   =    16,
    CSC_minus  =    17,
    DTTF       =    18,
    CSCTF      =    19,
    CASTOR     =    20,
    TIBTID     =    24,
    TOB        =    25,
    TEC_plus   =    26,
    TEC_minus  =    27,
    BPIX       =    28,
    FPIX       =    29,
    ES_plus    =    30,
    ES_minus   =    31
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
  { return((ready_ & ( 1 << partitionNumber ) != 0 ));}

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
};

/// Pretty-print operator for DcsStatus
std::ostream& operator<<(std::ostream& s, const DcsStatus& c);

typedef std::vector<DcsStatus> DcsStatusCollection;

#endif
