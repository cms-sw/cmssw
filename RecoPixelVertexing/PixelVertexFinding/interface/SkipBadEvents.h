#ifndef RecoPixelVertexing_SkipBadEvents_h
#define RecoPixelVertexing_SkipBadEvents_h
/** \class SkipBadEvents SkipBadEvents.h RecoPixelVertexing/PixelVertexFinding/SkipBadEvents.h 
 * Very simple class that inherits from EDFilter to skip a set of run and events numbers.
 * This won't be needed when PoolSource can do this for us, then I
 * will retire this little guy.
 *
 *  $Date: 2006/06/13 17:33:24 $
 *  $Revision: 1.1 $
 *  \author Aaron Dominguez (UNL)
 */
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <set>

class SkipBadEvents: public edm::EDFilter {
 public: 
  explicit SkipBadEvents(const edm::ParameterSet& config);
  virtual ~SkipBadEvents();
  virtual bool filter(edm::Event& e, const edm::EventSetup& s);
  
 private:
  std::map<int,std::set<int> > skip_; // Skip these run, event pairs
};
#endif
