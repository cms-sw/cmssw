#ifndef CandAlgos_CandSelectorBase_h
#define CandAlgos_CandSelectorBase_h
// Candidate Selector base producer module
// $Id: CandSelector.h,v 1.1 2005/10/24 06:08:18 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "boost/shared_ptr.hpp"
#include <string>

namespace aod {
  class Selector;
}

class CandSelectorBase : public edm::EDProducer {
public:
  explicit CandSelectorBase( const std::string &,
			     const boost::shared_ptr<aod::Selector> & = 
			     boost::shared_ptr<aod::Selector>() );
  ~CandSelectorBase();
  
protected:
  boost::shared_ptr<aod::Selector> select_;

private:
  virtual void produce( edm::Event&, const edm::EventSetup& );
  std::string src_;
};

#endif
