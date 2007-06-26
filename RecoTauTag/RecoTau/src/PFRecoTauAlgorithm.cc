#include "RecoTauTag/RecoTau/interface/PFRecoTauAlgorithm.h"
#include "DataFormats/Math/interface/Point3D.h"
Tau PFRecoTauAlgorithm::tag(const PFIsolatedTauTagInfo& myTagInfo)
{
  //Takes the jet
  const Jet & jet = *(myTagInfo.jet());
  const  PFJet * pfJet = dynamic_cast<const PFJet*>( &jet );

  //Takes the LeadChargedHadron
  float z_PV = 0;
  TrackRef myLeadTk;
  PFCandidateRef myLeadPFChargedHadronCand = (myTagInfo).leadPFChargedHadrCand(MatchingConeSize_, LeadCand_minPt_);
  if (myLeadPFChargedHadronCand->blockRef()->elements().size()!=0){
    for (OwnVector<PFBlockElement>::const_iterator iPFBlock=myLeadPFChargedHadronCand->blockRef()->elements().begin();
	 iPFBlock!=myLeadPFChargedHadronCand->blockRef()->elements().end();iPFBlock++){
      if ((*iPFBlock).type()==1 &&
	  ROOT::Math::VectorUtil::DeltaR(myLeadPFChargedHadronCand->momentum(),(*iPFBlock).trackRef()->momentum())<0.001){
	myLeadTk =(*iPFBlock).trackRef();
	z_PV = myLeadTk->dz();
      }
    }
  }
  
  math::XYZPoint  vtx = math::XYZPoint( 0, 0, z_PV );
  //Qui che ci mettiamo per il vertice
  Tau myTau(pfJet->charge(), pfJet->p4(), vtx );
  myTau.setLeadingTrack(myLeadTk);
  myTau.setLeadingChargedHadron(myTagInfo.leadPFChargedHadrCand(MatchingConeSize_, LeadCand_minPt_));

  return myTau;
  
}
