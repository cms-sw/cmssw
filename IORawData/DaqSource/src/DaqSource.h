#ifndef DaqSource_H
#define DaqSource_H

/** \class DaqSource
 *  An input service for raw data
 *
 *  $Date: 2005/09/30 08:17:48 $
 *  $Revision: 1.1 $
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
