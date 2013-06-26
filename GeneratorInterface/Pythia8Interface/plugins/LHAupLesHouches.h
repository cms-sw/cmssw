#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <memory>
#include <assert.h>

#include <boost/shared_ptr.hpp>

#include <HepMC/GenEvent.h>
#include <HepMC/GenParticle.h>

#include <Pythia.h>
#include <LesHouches.h>
#include <HepMCInterface.h>

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"

class LHAupLesHouches : public Pythia8::LHAup {
  public:
    LHAupLesHouches() {;}

    //void loadRunInfo(const boost::shared_ptr<lhef::LHERunInfo> &runInfo)
    void loadRunInfo(lhef::LHERunInfo* runInfo)
      { this->runInfo = runInfo; }

    //void loadEvent(const boost::shared_ptr<lhef::LHEEvent> &event)
    void loadEvent(lhef::LHEEvent* event)
      { this->event = event; }

  private:

    bool setInit();
    bool setEvent(int idProcIn);

    //boost::shared_ptr<lhef::LHERunInfo> runInfo;
    lhef::LHERunInfo* runInfo;
    //boost::shared_ptr<lhef::LHEEvent>	event;
    lhef::LHEEvent* event;
};
