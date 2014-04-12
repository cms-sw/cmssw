#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

using namespace reco;

IsolatedPixelTrackCandidate::~IsolatedPixelTrackCandidate() { }

IsolatedPixelTrackCandidate * IsolatedPixelTrackCandidate::clone() const { 
  return new IsolatedPixelTrackCandidate( * this ); 
}

TrackRef IsolatedPixelTrackCandidate::track() const {
  return track_;
}

l1extra::L1JetParticleRef IsolatedPixelTrackCandidate::l1tau() const {
  return l1tauJet_;
}

bool IsolatedPixelTrackCandidate::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 &&  checkOverlap( track(), o->track() ) );
}

std::pair<int,int> IsolatedPixelTrackCandidate::towerIndex() const
{
  int ieta=0, iphi=0;
  for (int i=1; i<21; i++)
    {
      if (fabs(track_->eta())<(i*0.087)&&fabs(track_->eta())>(i-1)*0.087) ieta=int(fabs(track_->eta())/track_->eta())*i;
    }
  if (fabs(track_->eta())>1.740&&fabs(track_->eta())<1.830) ieta=int(fabs(track_->eta())/track_->eta())*21;
  if (fabs(track_->eta())>1.830&&fabs(track_->eta())<1.930) ieta=int(fabs(track_->eta())/track_->eta())*22;
  if (fabs(track_->eta())>1.930&&fabs(track_->eta())<2.043) ieta=int(fabs(track_->eta())/track_->eta())*23;

  double delta=track_->phi()+0.174532925;
  if (delta<0) delta=delta+2*acos(-1);
  if (fabs(track_->eta())<1.740) 
    {
      for (int i=0; i<72; i++)
	{
	  if (delta<(i+1)*0.087266462&&delta>i*0.087266462) iphi=i;
	}
    }
  else 
    {
      for (int i=0; i<36; i++)
	{
	  if (delta<2*(i+1)*0.087266462&&delta>2*i*0.087266462) iphi=2*i;
	}
    }

  return std::pair<int,int>(ieta,iphi);

}
