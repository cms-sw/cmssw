//
// $Id: LeptonVertexSignificance.h,v 1.1 2008/01/07 11:48:26 lowette Exp $
//

#ifndef PhysicsTools_PatUtils_LeptonVertexSignificance_h
#define PhysicsTools_PatUtils_LeptonVertexSignificance_h

/**
  \class    LeptonVertexSignificance LeptonVertexSignificance.h "PhysicsTools/PatUtils/interface/LeptonVertexSignificance.h"
  \brief    Calculates a lepton's vertex association significance

   LeptonVertexSignificance calculates the significance of the association
   of the lepton to a given vertex, as defined in CMS Note 2006/024

  \author   Steven Lowette
  \version  $Id: LeptonVertexSignificance.h,v 1.1 2008/01/07 11:48:26 lowette Exp $
*/


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/TrackReco/interface/Track.h"


class TransientTrackBuilder;


namespace pat {


  class LeptonVertexSignificance {

    public:

      LeptonVertexSignificance();
      LeptonVertexSignificance(const edm::EventSetup & iSetup);
      ~LeptonVertexSignificance();

      float calculate(const Electron & anElectron, const edm::Event & iEvent);
      float calculate(const Muon & aMuon, const edm::Event & iEvent);

    private:

      float calculate(const reco::Track & track, const edm::Event & iEvent);

    private:

      TransientTrackBuilder * theTrackBuilder_;

  };


}

#endif

