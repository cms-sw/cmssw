#ifndef JetVertexAssociation_h
#define JetVertexAssociation_h

#include "FWCore/Framework/interface/EDProducer.h"

#include "JetMETCorrections/JetVertexAssociation/interface/JetVertexMain.h"

#include <memory>
#include <string>
#include <iostream>
#include <iomanip>
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
    std::string jet_algo;
    std::string track_algo;
    std::string vertex_algo;

  };
}


#endif
