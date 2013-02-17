#ifndef _BaseMatcher_h_
#define _BaseMatcher_h_

/**\class BaseMatcher

  Base for Sim and Trigger info matchers for SimTrack in CSC & GEM

 Original Author:  "Vadim Khotilovich"
 $Id: BaseMatcher.h,v 1.1 2013/02/11 07:33:06 khotilov Exp $

*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>

class BaseMatcher
{
public:
  
  BaseMatcher(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es)
  : trk_(t), vtx_(v), conf_(ps), ev_(ev), es_(es), verbose_(0) {}

  virtual ~BaseMatcher() {}

  // non-copyable
  BaseMatcher(const BaseMatcher&) = delete;
  BaseMatcher& operator=(const BaseMatcher&) = delete;


  const SimTrack& trk() const {return trk_;}
  const SimVertex& vtx() const {return vtx_;}

  const edm::ParameterSet& conf() const {return conf_;}

  const edm::Event& event() const {return ev_;}
  const edm::EventSetup& eventSetup() const {return es_;}
  
  void setVerbose(int v) { verbose_ = v; }
  int verbose() const { return verbose_; }

private:

  const SimTrack& trk_;
  const SimVertex& vtx_;

  const edm::ParameterSet& conf_;

  const edm::Event& ev_;
  const edm::EventSetup& es_;

  int verbose_;
};

#endif
