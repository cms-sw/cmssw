/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"

#include "Alignment/RPTrackBased/interface/StraightTrackAlignment.h"
#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"

/**
 *\brief An EDAnalyzer that runs StraightTrackAlignment.
 **/
class RPStraightTrackAligner : public edm::EDAnalyzer
{
  public:
    RPStraightTrackAligner(const edm::ParameterSet &ps); 
    ~RPStraightTrackAligner() {}

  private:
    unsigned int verbosity;

    bool worker_initialized;
    StraightTrackAlignment worker;

    edm::ESWatcher<VeryForwardRealGeometryRecord> geometryWatcher;

    virtual void beginJob() {}
    virtual void beginRun(edm::Run const&, edm::EventSetup const&);
    virtual void analyze(const edm::Event &e, const edm::EventSetup &es);
    virtual void endJob();
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

RPStraightTrackAligner::RPStraightTrackAligner(const ParameterSet &ps) : 
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
  worker_initialized(false),
  worker(ps)
{
}

//----------------------------------------------------------------------------------------------------

void RPStraightTrackAligner::beginRun(edm::Run const&, edm::EventSetup const& es)
{
}

//----------------------------------------------------------------------------------------------------

void RPStraightTrackAligner::analyze(const edm::Event &e, const edm::EventSetup &es)
{
  if (geometryWatcher.check(es)) {
    if (worker_initialized)
      throw cms::Exception("RPStraightTrackAligner") <<
        "RPStraightTrackAligner can't cope with changing geometry - change in event " << e.id() << endl;
  }

  if (!worker_initialized) {
    worker.Begin(es);
    worker_initialized = true;
  }

  worker.ProcessEvent(e, es);
}

//----------------------------------------------------------------------------------------------------

void RPStraightTrackAligner::endJob()
{
  worker.Finish();
}

DEFINE_FWK_MODULE(RPStraightTrackAligner);

