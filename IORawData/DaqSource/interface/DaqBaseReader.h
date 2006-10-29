#ifndef DaqSource_DaqBaseReader_h
#define DaqSource_DaqBaseReader_h

/** \class DaqBaseReader
 *  Base class for a "data reader" for the DaqSource.  
 *
 *  Derived classes must have a constructor accepting a
 *  parameter (const edm::ParameterSet& pset).
 *
 *  $Date: 2006/10/24 15:11:49 $
 *  $Revision: 1.4 $
 *  \author N. Amapane - CERN
 */

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
  
  /// overload to fill the fed collection to be put in the transient
  /// event store. NOTE: the FEDRawDataCollection data must be created
  /// (with new) withing the method; ownership is passed to the caller.
  virtual bool fillRawData(edm::EventID& eID,
			   edm::Timestamp& tstamp, 
			   FEDRawDataCollection*& data) = 0;  
  
private:
  //
  // member data
  //

  
};

#endif

