#ifndef DaqSource_DaqBaseReader_h
#define DaqSource_DaqBaseReader_h

/** \class DaqBaseReader
 *  Base class for a "data reader" for the DaqSource.  
 *
 *  Derived classes must have a constructor accepting a
 *  parameter (const edm::ParameterSet& pset).
 *
 *  $Date: 2005/10/04 18:38:48 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - CERN
 */

class FEDRawDataCollection;
namespace edm {class EventID; class Timestamp;   class ParameterSet;}

class DaqBaseReader {
public:
  /// Constructor
  DaqBaseReader() {}

  /// Destructor
  virtual ~DaqBaseReader() {}
  

  /// Fill in the raw data 
  virtual bool fillRawData(edm::EventID& eID,
			   edm::Timestamp& tstamp, 
			   FEDRawDataCollection& data) = 0;  

private:

};
#endif

