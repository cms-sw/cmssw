// $Id: RecoMuonCandidate.cc,v 1.3 2006/04/20 14:41:43 llista Exp $
#include "DataFormats/RecoCandidate/interface/RecoMuonCandidate.h"
#include "DataFormats/MuonReco/interface/Muon.h"

using namespace reco;

RecoMuonCandidate::~RecoMuonCandidate() { }

RecoMuonCandidate * RecoMuonCandidate::clone() const { 
  return new RecoMuonCandidate( * this ); 
}

MuonRef RecoMuonCandidate::muon() const {
  return muon_;
}

TrackRef RecoMuonCandidate::track() const {
  return muon_->track();
}

TrackRef RecoMuonCandidate::standAloneMuon() const {
  return muon_->standAlone();
}

TrackRef RecoMuonCandidate::combinedMuon() const {
  return muon_->combined();
}

SuperClusterRef RecoMuonCandidate::superCluster() const {
  return muon_->superCluster();
}

bool RecoMuonCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * dstc = dynamic_cast<const RecoCandidate *>( & c );
  if ( dstc == 0 ) return false;
  TrackRef t1 = track(), t2 = dstc->track();
  if ( ! t1.isNull() && ! t2.isNull() && t1 == t2 ) return true;
  MuonRef m1 = muon(), m2 = dstc->muon();
  if ( ! m1.isNull() && ! m2.isNull() && m1 == m2 ) return true;
  TrackRef st1 = standAloneMuon(), st2 = dstc->standAloneMuon();
  if ( ! st1.isNull() && ! st2.isNull() && st1 == st2 ) return true;
  TrackRef cm1 = combinedMuon(), cm2 = dstc->combinedMuon();
  if ( ! cm1.isNull() && ! cm2.isNull() && cm1 == cm2 ) return true;
  SuperClusterRef s1 = superCluster(), s2 = dstc->superCluster();
  if ( ! s1.isNull() && ! s2.isNull() && s1 == s2 ) return true;
  return false;
}
