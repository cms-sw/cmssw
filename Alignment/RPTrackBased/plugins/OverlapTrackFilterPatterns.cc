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



/**
 * \brief Filters track candidates useful for track-based alignment (mainly tracks in overlap).
 */
class OverlapTrackFilterPatterns : public edm::EDFilter
{
  public:
    edm::InputTag tagRecognizedPatterns;

    unsigned int prescale_vvh;
    unsigned int prescale_vvv;

    OverlapTrackFilterPatterns(const edm::ParameterSet &);

  protected:
    unsigned int counter_tth, counter_bbh;
    unsigned int counter_ttt, counter_bbb;

    unsigned int counter_all_events, counter_selected_events;

    virtual bool filter(edm::Event&, const edm::EventSetup &);

    virtual void endJob();
};


using namespace edm;
using namespace std;

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

OverlapTrackFilterPatterns::OverlapTrackFilterPatterns(const ParameterSet &ps) :
  tagRecognizedPatterns(ps.getParameter<edm::InputTag>("tagRecognizedPatterns")),
  prescale_vvh(ps.getParameter<unsigned int>("prescale_vvh")),
  prescale_vvv(ps.getParameter<unsigned int>("prescale_vvv")),
  counter_tth(0), counter_bbh(0),
  counter_ttt(0), counter_bbb(0),
  counter_all_events(0), counter_selected_events(0)
{
  printf(">> OverlapTrackFilterPatterns::OverlapTrackFilterPatterns\n");
  printf("    prescale_vvh = %u\n", prescale_vvh);
  printf("    prescale_vvv = %u\n", prescale_vvv);
}

//----------------------------------------------------------------------------------------------------

bool OverlapTrackFilterPatterns::filter(edm::Event &event, const EventSetup &es)
{
  //printf("--------------------------------------------------------\n");

  Handle< RPTrackCandidateCollection > trackColl;
  event.getByLabel(tagRecognizedPatterns, trackColl);

  struct RPCount
  {
    unsigned int top, bot, hor;
    RPCount() : top(0), bot(0), hor(0) {}
  };

  // map: arm --> number of active RPs
  map<unsigned int, RPCount> activity;

  for (const auto &p : *trackColl)
  {
    if (!p.second.Fittable())
      continue;

    unsigned int rp = p.first;
    unsigned int arm = rp / 100;
    unsigned int idx = rp % 10;

    //printf("RP %u\n", rp);
    
    if (idx == 0 || idx == 4)
      activity[arm].top++;

    if (idx == 2 || idx == 3)
      activity[arm].hor++;

    if (idx == 1 || idx == 5)
      activity[arm].bot++;
  }

  // keep event?
  bool keep = false;

  for (const auto &ac : activity)
  {
    if (ac.second.top >= 3)
    {
      counter_ttt++;
      //printf("counter_ttt ---> %u\n", counter_ttt);
      if (counter_ttt >= prescale_vvv)
      {
        keep = true;
        //printf("    ==> keep\n");
        counter_ttt = 0;
      }
    }

    if (ac.second.bot >= 3)
    {
      counter_bbb++;
      //printf("counter_bbb ---> %u\n", counter_bbb);
      if (counter_bbb >= prescale_vvv)
      {
        keep = true;
        //printf("    ==> keep\n");
        counter_bbb = 0;
      }
    }

    if (ac.second.top >= 2 && ac.second.hor >= 1)
    {
      counter_tth++;
      //printf("counter_tth ---> %u\n", counter_tth);
      if (counter_tth >= prescale_vvh)
      {
        keep = true;
        //printf("    ==> keep\n");
        counter_tth = 0;
      }
    }

    if (ac.second.bot >= 2 && ac.second.hor >= 1)
    {
      counter_bbh++;
      //printf("counter_bbh ---> %u\n", counter_bbh);
      if (counter_bbh >= prescale_vvh)
      {
        keep = true;
        //printf("    ==> keep\n");
        counter_bbh = 0;
      }
    }
  }

  counter_all_events++;

  if (keep)
    counter_selected_events++;

  return keep;
}

//----------------------------------------------------------------------------------------------------

void OverlapTrackFilterPatterns::endJob()
{
  printf(">> OverlapTrackFilterPatterns::endJob\n");
  printf("    counter_all_events = %u, counter_selected_events = %u ==> selected ratio = %.2E\n",
    counter_all_events, counter_selected_events, double(counter_selected_events) / counter_all_events);
}

DEFINE_FWK_MODULE(OverlapTrackFilterPatterns);
