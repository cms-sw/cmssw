/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Jan Kašpar
 *   Laurent Forthomme
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"
#include "DataFormats/CTPPSReco/interface/CTPPSLocalTrackLite.h"
#include "DataFormats/ProtonReco/interface/ProtonTrack.h"

#include "RecoCTPPS/ProtonReconstruction/interface/ProtonReconstructionAlgorithm.h"

//----------------------------------------------------------------------------------------------------

class CTPPSProtonReconstruction : public edm::stream::EDProducer<>
{
  public:
    explicit CTPPSProtonReconstruction(const edm::ParameterSet&);
    ~CTPPSProtonReconstruction();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    virtual void produce(edm::Event&, const edm::EventSetup&) override;

    edm::EDGetTokenT< std::vector<CTPPSLocalTrackLite> > tracksToken_;

    edm::ParameterSet beamConditions_;

    bool checkApertures_;

    edm::FileInPath opticsFileBeam1_, opticsFileBeam2_;

    ProtonReconstructionAlgorithm algorithm_;
};

//----------------------------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstruction::CTPPSProtonReconstruction( const edm::ParameterSet& iConfig ) :
  tracksToken_( consumes< std::vector<CTPPSLocalTrackLite> >( iConfig.getParameter<edm::InputTag>( "tagLocalTrackLite" ) ) ),

  beamConditions_             ( iConfig.getParameter<edm::ParameterSet>( "beamConditions" ) ),
  checkApertures_             ( iConfig.getParameter<bool>( "checkApertures" ) ),
  opticsFileBeam1_            ( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam1" ) ),
  opticsFileBeam2_            ( iConfig.getParameter<edm::FileInPath>( "opticsFileBeam2" ) ),

  algorithm_(opticsFileBeam1_.fullPath(), opticsFileBeam2_.fullPath(), beamConditions_)
{
  produces<vector<reco::ProtonTrack>>();
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstruction::fillDescriptions(ConfigurationDescriptions& descriptions)
{
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//----------------------------------------------------------------------------------------------------

CTPPSProtonReconstruction::~CTPPSProtonReconstruction()
{
}

//----------------------------------------------------------------------------------------------------

void CTPPSProtonReconstruction::produce(Event& iEvent, const EventSetup&)
{
  unique_ptr<vector<reco::ProtonTrack>> output( new vector<reco::ProtonTrack> );

  Handle< vector<CTPPSLocalTrackLite> > tracks;
  iEvent.getByToken(tracksToken_, tracks);

  if (tracks->size() > 1)
  {
    printf("===================== %u:%llu =====================\n", iEvent.id().run(), iEvent.id().event());

    for (const auto &tr : *tracks)
    {
      CTPPSDetId rpId(tr.getRPId());
      unsigned int decRPId = rpId.station()*100 + rpId.arm()*10 + rpId.rp();
       printf("%u (%u)\n", tr.getRPId(), decRPId);
    }
  }

  // split input per sector
  vector<const CTPPSLocalTrackLite*> tracks_45, tracks_56;
  for (const auto &tr : *tracks)
  {
    CTPPSDetId rpId(tr.getRPId());
    if (rpId.arm() == 0)
      tracks_45.push_back(&tr);
    else
      tracks_56.push_back(&tr);
  }

  // run reconstruction per sector
  // TODO: remove this condition
  if (tracks_45.size() > 1)
    algorithm_.reconstruct(tracks_45, *output);

  // TODO: remove this condition
  if (tracks_56.size() > 1)
    algorithm_.reconstruct(tracks_56, *output);

  // save output to event
  iEvent.put(move(output));
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(CTPPSProtonReconstruction);
