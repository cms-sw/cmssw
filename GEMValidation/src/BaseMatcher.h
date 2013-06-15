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
  
  /// CSC chamber types, according to CSCDetId::iChamberType()
  enum CSCType {CSC_ALL = 0, CSC_ME1a, CSC_ME1b, CSC_ME12, CSC_ME13,
      CSC_ME21, CSC_ME22, CSC_ME31, CSC_ME32, CSC_ME41, CSC_ME42};


  BaseMatcher(const SimTrack& t, const SimVertex& v,
      const edm::ParameterSet& ps, const edm::Event& ev, const edm::EventSetup& es);

  ~BaseMatcher();

  // non-copyable
  BaseMatcher(const BaseMatcher&) = delete;
  BaseMatcher& operator=(const BaseMatcher&) = delete;


  const SimTrack& trk() const {return trk_;}
  const SimVertex& vtx() const {return vtx_;}

  const edm::ParameterSet& conf() const {return conf_;}

  const edm::Event& event() const {return ev_;}
  const edm::EventSetup& eventSetup() const {return es_;}

  /// check if CSC chamber type is in the used list
  bool useCSCChamberType(int csc_type);
  
  void setVerbose(int v) { verbose_ = v; }
  int verbose() const { return verbose_; }

private:

  const SimTrack& trk_;
  const SimVertex& vtx_;

  const edm::ParameterSet& conf_;

  const edm::Event& ev_;
  const edm::EventSetup& es_;

  int verbose_;

  // list of CSC chamber types to use
  bool useCSCChamberTypes_[11];
};

#endif
