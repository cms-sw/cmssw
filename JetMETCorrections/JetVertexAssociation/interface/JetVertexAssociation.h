#ifndef JetVertexAssociation_h
#define JetVertexAssociation_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "JetMETCorrections/JetVertexAssociation/interface/JetVertexMain.h"

#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace cms{

  class JetVertexAssociation : public edm::EDProducer{
  
  public:

   JetVertexAssociation (const edm::ParameterSet& ps);

   ~JetVertexAssociation () {}

   void produce(edm::Event& e, const edm::EventSetup& c);

  private:
    typedef std::vector<double> ResultCollection1;
    typedef std::vector<bool> ResultCollection2;

    JetVertexMain m_algo;
    int jet_algo;
  };
}


#endif
