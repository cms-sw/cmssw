#ifndef DaqSource_H
#define DaqSource_H

/** \class DaqSource
 *  An input service for raw data
 *
 *  $Date: 2005/08/04 15:56:38 $
 *  $Revision: 1.4 $
 *  \author N. Amapane - S. Argiro'
 */

#include <FWCore/Framework/interface/InputSource.h>
#include <FWCore/Framework/interface/ProductDescription.h>
#include <string>

namespace raw {class FEDRawData;}
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
