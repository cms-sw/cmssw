#ifndef _BaseMatcher_h_
#define _BaseMatcher_h_

/**\class BaseMatcher

  Base for Sim and Trigger info matchers for SimTrack in CSC & GEM

 Original Author:  "Vadim Khotilovich"
 $Id$

*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <SimDataFormats/Track/interface/SimTrackContainer.h>
#include <SimDataFormats/Vertex/interface/SimVertexContainer.h>

class BaseMatcher
{
public:
  
  BaseMatcher(const SimTrack* t, const SimVertex* v,
      const edm::ParameterSet* ps, const edm::Event* ev, const edm::EventSetup* es)
  : trk_(t), vtx_(v), conf_(ps), ev_(ev), es_(es), verbose_(0) {}

  virtual ~BaseMatcher() {}

  const SimTrack* trk() {return trk_;}
  const SimVertex* vtx() {return vtx_;}

  const edm::ParameterSet* conf() {return conf_;}

  const edm::Event* event() {return ev_;}
  const edm::EventSetup* eventSetup() {return es_;}
  
  void setVerbose(int v) { verbose_ = v; }
  int verbose() { return verbose_; }

private:

  const SimTrack* trk_;
  const SimVertex* vtx_;

  const edm::ParameterSet* conf_;

  const edm::Event* ev_;
  const edm::EventSetup* es_;

  int verbose_;
};

#endif
