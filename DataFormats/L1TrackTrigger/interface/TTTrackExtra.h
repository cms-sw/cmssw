#ifndef DataFormats_L1TrackTrigger_TTrackExtra_h
#define DataFormats_L1TrackTrigger_TTrackExtra_h

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"

template<typename T>
class TTTrackExtra {
public:
  struct StatusFlags{
    enum Status{
      IsGenuine=0x1,
      IsLooselyGenuine=0x2,
      IsCombinatoric=0x4,
      IsUnknown=0x8
    };
  };

private:
  edm::Ref<std::vector<TTTrack<T> > >ttTrkRef_;
  edm::Ref<TrackingParticleCollection> trkPartRef_;
  char flags_;

public:
  TTTrackExtra():flags_(0){}
  TTTrackExtra(const edm::Ref<std::vector<TTTrack<T> > >& ttTrkRef,
	       const edm::Ref<TrackingParticleCollection >& trkPartRef,
	       int flags):
    ttTrkRef_(ttTrkRef),trkPartRef_(trkPartRef),flags_(flags){
  }
   
  edm::Ref<std::vector<TTTrack<T> > > ttTrk()const{return ttTrkRef_;}
  edm::Ref<TrackingParticleCollection> trkPart()const{return trkPartRef_;}
  int flags()const{return flags_;}  
  bool isGenuine()const{return (flags_&StatusFlags::IsGenuine)!=0;}
  bool isLooselyGenuine()const{return (flags_&StatusFlags::IsLooselyGenuine)!=0;}
  bool isCombinatoric()const{return (flags_&StatusFlags::IsCombinatoric)!=0;}
  bool isUnknown()const{return (flags_&StatusFlags::IsUnknown)!=0;}


};

using L1TrackExtra = TTTrackExtra<Ref_Phase2TrackerDigi_>;

#endif
