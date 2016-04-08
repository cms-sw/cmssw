/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoTotemRP/RPRecoDataFormats/interface/RPTrackCandidate.h"
#include "RecoTotemRP/RPRecoDataFormats/interface/RPTrackCandidateCollection.h"
#include "CondFormats/DataRecord/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/TotemRPGeometry.h"

#include "Alignment/RPDataFormats/interface/LocalTrackFit.h"
#include "Alignment/RPTrackBased/interface/LocalTrackFitter.h"
#include "Alignment/RPTrackBased/interface/AlignmentGeometry.h"
#include "Alignment/RPTrackBased/interface/AlignmentTask.h"

#include <map>
#include <unordered_set>

/**
 * \brief Filters track candidates useful for track-based alignment (mainly tracks in overlap).
 */
class OverlapTrackFilterFit : public edm::EDFilter
{
  public:

    OverlapTrackFilterFit(const edm::ParameterSet &);

  protected:
    edm::InputTag tagRecognizedPatterns;

    double z0_abs;

    LocalTrackFitter fitter;

    double large_threshold_x, large_threshold_y;

    unsigned int prescale_vvh;
    unsigned int prescale_vvv;
    unsigned int prescale_hh;
    unsigned int prescale_lx;
    unsigned int prescale_ly;

    /// map: arm --> geometry
    std::map<unsigned int, AlignmentGeometry> alGeometries;

    unsigned int counter_tth, counter_bbh;
    unsigned int counter_ttt, counter_bbb;
    unsigned int counter_hh;
    unsigned int counter_lx, counter_ly;

    unsigned int counter_all_events, counter_selected_events;

    void beginRun(edm::Run const&, edm::EventSetup const& es);

    virtual bool filter(edm::Event&, const edm::EventSetup &);

    virtual void endJob();
};


using namespace edm;
using namespace std;

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

OverlapTrackFilterFit::OverlapTrackFilterFit(const ParameterSet &ps) :
  tagRecognizedPatterns(ps.getParameter<edm::InputTag>("tagRecognizedPatterns")),

  z0_abs(ps.getParameter<double>("z0_abs")),

  fitter(ps.getParameterSet("fitter")),

  large_threshold_x(ps.getParameter<double>("large_threshold_x")),
  large_threshold_y(ps.getParameter<double>("large_threshold_y")),

  prescale_vvh(ps.getParameter<unsigned int>("prescale_vvh")),
  prescale_vvv(ps.getParameter<unsigned int>("prescale_vvv")),
  prescale_hh(ps.getParameter<unsigned int>("prescale_hh")),
  prescale_lx(ps.getParameter<unsigned int>("prescale_lx")),
  prescale_ly(ps.getParameter<unsigned int>("prescale_ly")),

  counter_tth(0), counter_bbh(0),
  counter_ttt(0), counter_bbb(0),
  counter_all_events(0), counter_selected_events(0)
{
  printf(">> OverlapTrackFilterFit::OverlapTrackFilterFit\n");
  printf("    prescale_vvh = %u\n", prescale_vvh);
  printf("    prescale_vvv = %u\n", prescale_vvv);
  printf("    prescale_hh = %u\n", prescale_hh);
  printf("    prescale_lx = %u\n", prescale_lx);
  printf("    prescale_ly = %u\n", prescale_ly);
}

//----------------------------------------------------------------------------------------------------

void OverlapTrackFilterFit::beginRun(edm::Run const&, edm::EventSetup const& es)
{
  // get geometry
  ESHandle<TotemRPGeometry> geom;
  es.get<VeryForwardRealGeometryRecord>().get(geom);

  // build alignment geometry for each arm
  vector<unsigned int> rps;
  for (unsigned int a = 0; a < 2; a++)
  {
    for (unsigned int s = 0; s < 3; s++)
    {
      if (s == 1)
        continue;
      
      for (unsigned int r = 0; r < 6; r++)
      {
        unsigned int id = 100*a + 10*s + r;
        
        rps.push_back(id);
      }
    }

    double z0 = (a == 0) ? -z0_abs : +z0_abs;
    vector<unsigned int> excludePlanes;
    AlignmentTask::BuildGeometry(rps, excludePlanes, geom.product(), z0, alGeometries[a]);
  }
}

//----------------------------------------------------------------------------------------------------

bool OverlapTrackFilterFit::filter(edm::Event &event, const EventSetup &es)
{
  // get input
  Handle< RPTrackCandidateCollection > trackColl;
  event.getByLabel(tagRecognizedPatterns, trackColl);

  // flag whether to keep this event
  bool keep = false;

  // make fits for each arm
  for (const auto &gp : alGeometries)
  {
    unsigned int arm = gp.first;

    // select hits
    HitCollection selection;
    for (const auto &tp : *trackColl)
    {
      if (!tp.second.Fittable())
        continue;
    
      unsigned int rp = tp.first;
      if (rp / 100 != arm)
        continue;

      for (const auto &hit : tp.second.TrackRecoHits())
        selection.push_back(hit);
    }

    // make fit
    LocalTrackFit trackFit;
    if (! fitter.Fit(selection, gp.second, trackFit))
      continue;

    // analyze RP structure of the fit
    unordered_set<unsigned int> selectedRPs;
    for (const auto &hit : selection)
      selectedRPs.insert(hit.id/10);

    unsigned int top=0, bot=0, hor=0;
    for (const auto &rp : selectedRPs)
    {
      unsigned int idx = rp % 10;
  
      if (idx == 0 || idx == 4)
        top++;
  
      if (idx == 2 || idx == 3)
        hor++;
  
      if (idx == 1 || idx == 5)
        bot++;
    }

    // update keep flag
    if (top >= 3)
    {
      counter_ttt++;
      if (counter_ttt >= prescale_vvv)
      {
        keep = true;
        counter_ttt = 0;
      }
    }

    if (bot >= 3)
    {
      counter_bbb++;
      if (counter_bbb >= prescale_vvv)
      {
        keep = true;
        counter_bbb = 0;
      }
    }

    if (hor >= 2)
    {
      counter_hh++;
      if (counter_hh >= prescale_hh)
      {
        keep = true;
        counter_hh = 0;
      }
    }

    if (top >= 2 && hor >= 1)
    {
      counter_tth++;
      if (counter_tth >= prescale_vvh)
      {
        keep = true;
        counter_tth = 0;
      }
    }

    if (bot >= 2 && hor >= 1)
    {
      counter_bbh++;
      if (counter_bbh >= prescale_vvh)
      {
        keep = true;
        counter_bbh = 0;
      }
    }

    bool twoUnitsVert = (top >= 2 || bot >= 2);

    if (twoUnitsVert && fabs(trackFit.bx) >= large_threshold_x)
    {
      counter_lx++;
      if (counter_lx >= prescale_lx)
      {
        keep = true;
        counter_lx = 0;
      }
    }

    if (twoUnitsVert && fabs(trackFit.by) >= large_threshold_y)
    {
      counter_ly++;
      if (counter_ly >= prescale_ly)
      {
        keep = true;
        counter_ly = 0;
      }
    }
  }

  counter_all_events++;

  if (keep)
    counter_selected_events++;

  return keep;
}

//----------------------------------------------------------------------------------------------------

void OverlapTrackFilterFit::endJob()
{
  printf(">> OverlapTrackFilterFit::endJob\n");
  printf("    counter_all_events = %u, counter_selected_events = %u ==> selected ratio = %.2E\n",
    counter_all_events, counter_selected_events, double(counter_selected_events) / counter_all_events);
}

DEFINE_FWK_MODULE(OverlapTrackFilterFit);
