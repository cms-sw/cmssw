/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors:
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/CTPPSReco/interface/TotemRPLocalTrack.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"

//----------------------------------------------------------------------------------------------------

/**
 * \brief Distills the essential track data from all RPs.
 **/
class CTPPSLocalTrackLiteProducer : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSLocalTrackLiteProducer(const edm::ParameterSet& conf);
  
    virtual ~CTPPSLocalTrackLiteProducer() {}
  
    virtual void produce(edm::Event& e, const edm::EventSetup& c) override;
  
  private:
    edm::EDGetTokenT<edm::DetSetVector<TotemRPLocalTrack>> siStripTrackToken;
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSLocalTrackLiteProducer::CTPPSLocalTrackLiteProducer(edm::ParameterSet const& conf)
{
  siStripTrackToken = consumes<DetSetVector<TotemRPLocalTrack>>(conf.getParameter<edm::InputTag>("tagSiStripTrack"));

  produces<vector<CTPPSLocalTrackLite>>();
}

//----------------------------------------------------------------------------------------------------
 
void CTPPSLocalTrackLiteProducer::produce(edm::Event& e, const edm::EventSetup& es)
{
  // get input from Si strips
  edm::Handle< DetSetVector<TotemRPLocalTrack> > inputSiStripTracks;
  e.getByToken(siStripTrackToken, inputSiStripTracks);

  // prepare output
  vector<CTPPSLocalTrackLite> output;
  
  // process tracks from Si strips
  for (const auto rpv : *inputSiStripTracks)
  {
    const uint32_t rpId = rpv.detId();

    for (const auto t : rpv)
    {
      if (t.isValid())
        output.push_back(CTPPSLocalTrackLite(rpId, t.getX0(), t.getX0Sigma(), t.getY0(), t.getY0Sigma()));
    }
  }

  // save output to event
  e.put(make_unique<vector<CTPPSLocalTrackLite>>(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSLocalTrackLiteProducer);
