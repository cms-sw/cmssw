#ifndef CandAlgos_CandSelectorBase_h
#define CandAlgos_CandSelectorBase_h
// Candidate Selector base producer module
// $Id: CandSelectorBase.h,v 1.1 2005/10/24 09:50:21 llista Exp $
#include "FWCore/Framework/interface/EDProducer.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "boost/shared_ptr.hpp"
#include <string>

class CandSelectorBase : public edm::EDProducer {
public:
  explicit CandSelectorBase( const std::string &,
			     const boost::shared_ptr<aod::Candidate::selector> & = 
			     boost::shared_ptr<aod::Candidate::selector>() );
  ~CandSelectorBase();
  
protected:
  boost::shared_ptr<aod::Candidate::selector> select_;

private:
  virtual void produce( edm::Event&, const edm::EventSetup& );
  std::string src_;
};

#endif
