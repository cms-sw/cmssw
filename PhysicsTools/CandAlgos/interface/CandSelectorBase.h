#ifndef CandAlgos_CandSelectorBase_h
#define CandAlgos_CandSelectorBase_h
// Candidate Selector base producer module
// $Id: CandSelectorBase.h,v 1.2 2005/10/25 08:47:05 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/CandUtils/interface/CandSelector.h"
#include "boost/shared_ptr.hpp"
#include <string>

class CandSelectorBase : public edm::EDProducer {
public:
  explicit CandSelectorBase( const std::string &,
			     const boost::shared_ptr<CandSelector> & = 
			     boost::shared_ptr<CandSelector>() );
  ~CandSelectorBase();
  
protected:
  boost::shared_ptr<CandSelector> select_;

private:
  virtual void produce( edm::Event&, const edm::EventSetup& );
  std::string src_;
};

#endif
