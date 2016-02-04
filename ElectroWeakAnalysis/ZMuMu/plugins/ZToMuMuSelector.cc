/* \class ZToMuMuSelector
 *
 * \author Juan Alcaraz, CIEMAT
 *
 */
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Utilities/interface/InputTag.h"

class ZToMuMuSelector : public edm::EDFilter {
public:
  ZToMuMuSelector (const edm::ParameterSet &);
  virtual bool filter(edm::Event&, const edm::EventSetup&);
private:
  edm::InputTag muonTag_;
  edm::InputTag isoTag_;
  double ptCut_;
  double etaCut_;
  double massZMin_;
  double massZMax_;

  bool onlyGlobalMuons_;
  edm::InputTag trackerTag_;
  edm::InputTag isoTrackerTag_;
  int minTrackerHits_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

using namespace edm;
using namespace std;
using namespace reco;

ZToMuMuSelector::ZToMuMuSelector( const ParameterSet & cfg ) :
      muonTag_(cfg.getParameter<edm::InputTag> ("MuonTag")),
      isoTag_(cfg.getParameter<edm::InputTag> ("IsolationTag")),
      ptCut_(cfg.getParameter<double>("PtCut")),
      etaCut_(cfg.getParameter<double>("EtaCut")),
      massZMin_(cfg.getParameter<double>("MassZMin")),
      massZMax_(cfg.getParameter<double>("MassZMax")),

      onlyGlobalMuons_(cfg.getParameter<bool>("OnlyGlobalMuons")),
      trackerTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrackerTag",edm::InputTag("ctfWithMaterialTracks"))),
      isoTrackerTag_(cfg.getUntrackedParameter<edm::InputTag> ("TrackerIsolationTag",edm::InputTag("zMuMuTrackerIsolations"))),
      minTrackerHits_(cfg.getUntrackedParameter<int>("MinTrackerHits",7))
{
}

bool ZToMuMuSelector::filter (Event & ev, const EventSetup &) {

      // Note:    try/catch sequences should disappear in the future
      //          GetByLabel will return a false bool value 
      //          if the collection is not present

      Handle<TrackCollection> muonCollection;

      ev.getByLabel(muonTag_, muonCollection);
      if (!muonCollection.isValid()) {
	LogTrace("") << ">>> Muon collection does not exist !!!";
	return false;
      }

      Handle<edm::ValueMap<bool> > isoMap;
      ev.getByLabel(isoTag_, isoMap);
      if (!isoMap.isValid()) {
	LogTrace("") << ">>> ISO Muon collection does not exist !!!";
	return false;
      }

      Handle<TrackCollection> trackerCollection;
      Handle<edm::ValueMap<bool> > isoTrackerMap;
      if (!onlyGlobalMuons_) {
	ev.getByLabel(trackerTag_, trackerCollection);
	if (!trackerCollection.isValid()) {
	  LogTrace("") << ">>> Tracker collection does not exist !!!";
	  return false;
	}
	
	ev.getByLabel(isoTrackerTag_, isoTrackerMap);
	if (!isoTrackerMap.isValid()) {
	  LogTrace("") << ">>> ISO Tracker collection does not exist !!!";
	  return false;
	}
      }

      unsigned int npairs = 0;
      bool globalCombinationFound = false;
      for (unsigned int i=0; i<muonCollection->size(); i++) {
            TrackRef mu(muonCollection,i);
            LogTrace("") << "> Processing muon number " << i << "...";
            double pt = mu->pt();
            LogTrace("") << "\t... pt= " << pt << " GeV";
            if (pt<ptCut_) continue;
            double eta = mu->eta();
            LogTrace("") << "\t... eta= " << eta;
            if (fabs(eta)>etaCut_) continue;
            bool iso = (*isoMap)[mu];
            LogTrace("") << "\t... isolated? " << iso;
            if (!iso) continue;

            for (unsigned int j=i+1; j<muonCollection->size(); j++) {
                  TrackRef mu2(muonCollection,j);
                  LogTrace("") << "> Processing second muon number " << j << "...";
                  double pt2 = mu2->pt();
                  LogTrace("") << "\t... pt2= " << pt2 << " GeV";
                  if (pt2<ptCut_) continue;
                  double eta2 = mu2->eta();
                  LogTrace("") << "\t... eta2= " << eta2;
                  if (fabs(eta2)>etaCut_) continue;
                  bool iso2 = (*isoMap)[mu2];
                  LogTrace("") << "\t... isolated2? " << iso2;
                  if (!iso2) continue;

                  double z_en = mu->p() + mu2->p();
                  double z_px = mu->px() + mu2->px();
                  double z_py = mu->py() + mu2->py();
                  double z_pz = mu->pz() + mu2->pz();
                  double massZ = z_en*z_en - z_px*z_px - z_py*z_py - z_pz*z_pz;
                  massZ = (massZ>0) ? sqrt(massZ) : 0;
                  LogTrace("") << "\t... Z_en, Z_px, Z_py, Z_pz= " << z_en << ", " << z_px << ", " << z_py << ", " << z_pz << " GeV";
                  LogTrace("") << "\t... (GM-GM) Invariant reconstructed mass= " << massZ << " GeV";
                  if (massZ<massZMin_) continue;
                  if (massZ>massZMax_) continue;
                  globalCombinationFound = true;
                  npairs++;
            }

            if (onlyGlobalMuons_ || globalCombinationFound) continue;

            for (unsigned int j=0; j<trackerCollection->size(); j++) {
                  TrackRef mu2(trackerCollection,j);
                  LogTrace("") << "> Processing track number " << j << "...";
                  double pt2 = mu2->pt();
                  LogTrace("") << "\t... pt3= " << pt2 << " GeV";
                  if (pt2<ptCut_) continue;
                  double eta2 = mu2->eta();
                  LogTrace("") << "\t... eta3= " << eta2;
                  if (fabs(eta2)>etaCut_) continue;
                  int nhits2 = mu2->numberOfValidHits();
                  LogTrace("") << "\t... nhits3= " << nhits2;
                  if (nhits2<minTrackerHits_) continue;
                  bool iso2 = (*isoTrackerMap)[mu2];
                  LogTrace("") << "\t... isolated3? " << iso2;
                  if (!iso2) continue;

                  double z_en = mu->p() + mu2->p();
                  double z_px = mu->px() + mu2->px();
                  double z_py = mu->py() + mu2->py();
                  double z_pz = mu->pz() + mu2->pz();
                  double massZ = z_en*z_en - z_px*z_px - z_py*z_py - z_pz*z_pz;
                  massZ = (massZ>0) ? sqrt(massZ) : 0;
                  LogTrace("") << "\t... Z_en, Z_px, Z_py, Z_pz= " << z_en << ", " << z_px << ", " << z_py << ", " << z_pz << " GeV";
                  LogTrace("") << "\t... (GM-TK) Invariant reconstructed mass= " << massZ << " GeV";
                  if (massZ<massZMin_) continue;
                  if (massZ>massZMax_) continue;
                  npairs++;
            }
      }

      LogTrace("") << "> Number of Z pairs found= " << npairs;
      if (npairs<1) {
            LogTrace("") << ">>>> Event REJECTED";
            return false;
      }
      LogTrace("") << ">>>> Event SELECTED!!!";

      return true;

}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( ZToMuMuSelector );
