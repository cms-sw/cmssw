#ifndef DaqSource_DaqSource_H
#define DaqSource_DaqSource_H

/** \class DaqSource
 *  An input service for raw data. 
 *  The actual source can be the real DAQ, a file, a random generator, etc.
 *
 *  $Date: 2005/10/04 18:38:48 $
 *  $Revision: 1.2 $
 *  \author N. Amapane - S. Argiro'
 */

#include <FWCore/Framework/interface/InputSource.h>
#include <FWCore/Framework/interface/ProductDescription.h>
#include <string>

class FEDRawData;
namespace edm {class ParameterSet; class InputSourceDescription;} 
class DaqBaseReader;


class DaqSource : public edm::InputSource {

 public:
  explicit DaqSource(const edm::ParameterSet& pset, 
		     const edm::InputSourceDescription& desc);

  virtual ~DaqSource();

 private:
  virtual std::auto_ptr<edm::EventPrincipal> read();

  DaqBaseReader * reader_;
  edm::ProductDescription fedrawdataDescription_;
   
  int remainingEvents_;
};

#endif
