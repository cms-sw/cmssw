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
#include "FWCore/Framework/interface/ESHandle.h"

#include "Alignment/RPDataFormats/interface/LocalTrackFit.h"
#include "Alignment/RPTrackBased/interface/LocalTrackFitter.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPTrackCandidateCollection.h"
#include "CondFormats/DataRecord/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"
#include "Alignment/RPTrackBased/interface/AlignmentGeometry.h"
#include "Alignment/RPTrackBased/interface/AlignmentTask.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPFittedTrack.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPFittedTrackCollection.h"

#include "TFile.h"
#include "TTree.h"

/**
 *\brief Fits tracks in all stations and dumps results on a tree (in a ROOT file).
 **/
class RPStraightTrackFitter : public edm::EDAnalyzer
{
  public:
    RPStraightTrackFitter(const edm::ParameterSet &ps); 
    ~RPStraightTrackFitter() {}

  private:
    struct FitData {
      bool valid;
      double x, y, z;
      double ndf, chi2;
  
      FitData() { Reset(); }
      void Reset()
        { valid = false; x = y = z = ndf = chi2 = 0.;}
    };

    unsigned int verbosity;
    std::vector<unsigned int> excludedRPs;
    std::string dumpFileName;
    
    LocalTrackFitter fitter;
    std::map<unsigned int, AlignmentGeometry> alGeometries;

    /// ROOT file with dumpTree
    TFile *dumpFile;

    /// TTree with fit data
    TTree *dumpTree;

    /// data to populate one `row' of the tree
    std::map<unsigned int, FitData> stFitData, rpFitData;


    virtual void beginRun(edm::Run const&, edm::EventSetup const&);
    virtual void analyze(const edm::Event &e, const edm::EventSetup &es);
    virtual void endJob();
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

RPStraightTrackFitter::RPStraightTrackFitter(const ParameterSet &ps) : 
  verbosity(ps.getUntrackedParameter<unsigned int>("verbosity", 0)),
  excludedRPs(ps.getParameter< vector<unsigned int> >("excludedRPs")),
  dumpFileName(ps.getUntrackedParameter<string>("dumpFileName")),
  fitter(ps)
{

}

//----------------------------------------------------------------------------------------------------

void RPStraightTrackFitter::beginRun(edm::Run const&, edm::EventSetup const& es)
{
  ESHandle<TotemRPGeometry> geom;
  es.get<VeryForwardRealGeometryRecord>().get(geom);

  // open dump file and preapre dump tree branches
  dumpFile = new TFile(dumpFileName.c_str(), "recreate");
  dumpTree = new TTree("RPRecoProtonInfo", "RPRecoProtonInfo");

  char buf[100];
  for (unsigned int a = 0; a < 2; a++)
    for (unsigned int s = 0; s < 3; s++) {
      if (s == 1)
        continue;
      
      vector<unsigned int> rps;
      for (unsigned int r = 0; r < 6; r++) {
        unsigned int id = 100*a + 10*s + r;
        
        sprintf(buf, "track_rp_%u", id);
        dumpTree->Branch(buf, &rpFitData[id], "valid/i:x/D:y/D:z/D:ndf/D:chi2/D");
        
        sprintf(buf, "station_fit_rp_%u", id);
        dumpTree->Branch(buf, &stFitData[id], "valid/i:x/D:y/D:z/D:ndf/D:chi2/D");

        rps.push_back(id);
      }

      vector<unsigned int> excludePlanes;
      AlignmentTask::BuildGeometry(rps, excludePlanes, geom.product(), 0., alGeometries[a*10+s]);
    }
}

//----------------------------------------------------------------------------------------------------

void RPStraightTrackFitter::analyze(const edm::Event &event, const edm::EventSetup &es)
{
//  ESHandle<TotemRPGeometry> geom;
//  es.get<VeryForwardRealGeometryRecord>().get(geom);
//
//  Handle< RPTrackCandidateCollection > trackColl;
//  event.getByType(trackColl);
//
//  Handle< RPFittedTrackCollection > fitColl;
//  event.getByType(fitColl);
//
//  // reset fit data
//  for (map<unsigned int, FitData>::iterator it = rpFitData.begin(); it != rpFitData.end(); ++it)
//    it->second.Reset();
//  for (map<unsigned int, FitData>::iterator it = stFitData.begin(); it != stFitData.end(); ++it)
//    it->second.Reset();
//
//  // fill in RP fits
//  for (RPFittedTrackCollection::const_iterator it = fitColl->begin(); it != fitColl->end(); ++it) {
//    FitData &ft = rpFitData[it->first];
//    ft.valid = it->second.IsValid();
//    ft.x = it->second.X0();
//    ft.y = it->second.Y0();
//    ft.z = it->second.Z0();
//    ft.chi2 = it->second.ChiSquared();
//    ft.ndf = it->second.ChiSquared() / it->second.ChiSquaredOverN();
//  }
//
//  // build hit collections per station
//  map<unsigned int, HitCollection> hits;
//  for (RPTrackCandidateCollection::const_iterator it = trackColl->begin(); it != trackColl->end(); ++it) {
//    // skip non fittable candidates
//    if (!it->second.Fittable())
//      continue;
//
//    // skip excluded pots
//    unsigned int rpId = it->first;
//    if (find(excludedRPs.begin(), excludedRPs.end(), rpId) != excludedRPs.end())
//      continue;
//
//    unsigned int stId = rpId/10;
//    const vector<TotemRPRecHit> &cHits = it->second.TrackRecoHits();
//    for (unsigned int i = 0; i < cHits.size(); i++)
//      hits[stId].push_back(cHits[i]);
//  }
//
//  // fill in station fits
//  for (map<unsigned int, HitCollection>::iterator it = hits.begin(); it != hits.end(); ++it) {
//    // fit hits per station
//    LocalTrackFit ltf;
//    if (! fitter.Fit(it->second, alGeometries[it->first], ltf) )
//      continue;
//
//    // get list of active RPs
//    set<unsigned int> selectedRPs;
//    for (HitCollection::iterator hit = it->second.begin(); hit != it->second.end(); ++hit) {
//      unsigned int rpId = hit->id / 10;
//      selectedRPs.insert(rpId);
//    }
//
//    // fill in fit data
//    if (dumpTree) {
//      for (set<unsigned int>::iterator rit = selectedRPs.begin(); rit != selectedRPs.end(); ++rit) {
//      double z = geom->GetRPDevice(*rit)->translation().z();
//      double x = 0., y = 0.;
//      ltf.Eval(z, x, y);
//
//      FitData &ft = stFitData[*rit];
//      ft.valid = true;
//      ft.x = x;
//      ft.y = y;
//      ft.z = z;
//      ft.chi2 = ltf.chi_sq;
//      ft.ndf = ltf.ndf;
//      }
//    }
//  }
//
//  // commit fit data
//  if (dumpTree)
//    dumpTree->Fill();
}

//----------------------------------------------------------------------------------------------------

void RPStraightTrackFitter::endJob()
{
  gDirectory = dumpFile;
  dumpTree->Write();
  delete dumpFile;
}

DEFINE_FWK_MODULE(RPStraightTrackFitter);

