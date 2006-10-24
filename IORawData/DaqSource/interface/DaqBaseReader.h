#ifndef DaqSource_DaqBaseReader_h
#define DaqSource_DaqBaseReader_h

/** \class DaqBaseReader
 *  Base class for a "data reader" for the DaqSource.  
 *
 *  Derived classes must have a constructor accepting a
 *  parameter (const edm::ParameterSet& pset).
 *
 *  $Date: 2005/10/06 18:23:47 $
 *  $Revision: 1.3 $
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
  
  // overload to fill the fed collection to be put in the transient event store 
  virtual bool fillRawData(edm::EventID& eID,
			   edm::Timestamp& tstamp, 
			   FEDRawDataCollection*& data) = 0;  
  
private:
  //
  // member data
  //

  
};

#endif

