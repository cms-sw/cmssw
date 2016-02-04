#ifndef DaqSource_DaqBaseReader_h
#define DaqSource_DaqBaseReader_h

/** \class DaqBaseReader
 *  Base class for a "data reader" for the DaqSource.  
 *
 *  Derived classes must have a constructor accepting a
 *  parameter (const edm::ParameterSet& pset).
 *
 *  $Date: 2010/01/11 16:14:25 $
 *  $Revision: 1.7 $
 *  \author N. Amapane - CERN
 */


#include "DataFormats/Provenance/interface/RunID.h"


class FEDRawDataCollection;
namespace edm {class EventID; class Timestamp; class ParameterSet;}


class DaqBaseReader
{
public:
  //
  // construction/destruction
  //
  DaqBaseReader() {}
  virtual ~DaqBaseReader() {}
  
  //
  // abstract interface
  //
  
  /// set the run number
  virtual void setRunNumber(edm::RunNumber_t runNumber) {}

  /// overload to fill the fed collection to be put in the transient
  /// event store. NOTE: the FEDRawDataCollection data must be created
  /// (with new) within the method; ownership is passed to the caller.
  virtual int fillRawData(edm::EventID& eID,
			   edm::Timestamp& tstamp, 
			   FEDRawDataCollection*& data) = 0;  
  
private:
  //
  // member data
  //

  
};

#endif

