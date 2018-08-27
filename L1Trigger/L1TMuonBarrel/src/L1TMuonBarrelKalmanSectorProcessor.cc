#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanSectorProcessor.h"



L1TMuonBarrelKalmanSectorProcessor::L1TMuonBarrelKalmanSectorProcessor(const edm::ParameterSet& iConfig,int sector): 
  verbose_(iConfig.getParameter<int>("verbose")),
  sector_(sector)
{
  std::vector<int> wheels = iConfig.getParameter<std::vector<int> >("wheelsToProcess");
    for (const auto wheel : wheels)
      regions_.push_back(L1TMuonBarrelKalmanRegionModule(iConfig.getParameter<edm::ParameterSet>("regionSettings"),wheel,sector));
}



L1TMuonBarrelKalmanSectorProcessor::~L1TMuonBarrelKalmanSectorProcessor() {}

L1MuKBMTrackCollection L1TMuonBarrelKalmanSectorProcessor::process(L1TMuonBarrelKalmanAlgo* trackMaker, const L1MuKBMTCombinedStubRefVector& stubsAll,int bx) {


  L1MuKBMTrackCollection pretracks;
  for (auto& region: regions_) {
    L1MuKBMTrackCollection tmp = region.process(trackMaker,stubsAll,bx);
    if (!tmp.empty())
      pretracks.insert(pretracks.end(),tmp.begin(),tmp.end());
  } 

  L1MuKBMTrackCollection out =cleanAndSort(pretracks,3);
  if (verbose_==1)
    verbose(trackMaker,out);

  return out;

}




L1TMuonBarrelKalmanSectorProcessor::bmtf_out L1TMuonBarrelKalmanSectorProcessor::makeWord(L1TMuonBarrelKalmanAlgo* trackMaker, const L1MuKBMTrackCollection& tracks) {
  L1TMuonBarrelKalmanSectorProcessor::bmtf_out out;
    out.pt_1=0;
    out.qual_1=0;
    out.eta_1=0;
    out.HF_1=0;
    out.phi_1=0;
    out.bx0_1=0;
    out.charge_1=0;
    out.chargeValid_1=0;
    out.dxy_1=0;
    out.addr1_1=0;
    out.addr2_1=0;
    out.addr3_1=0;
    out.addr4_1=0;
    out.reserved_1=0;
    out.wheel_1=0;
    out.ptSTA_1=0;
    out.SE_1=0;

    out.pt_2=0;
    out.qual_2=0;
    out.eta_2=0;
    out.HF_2=0;
    out.phi_2=0;
    out.bx0_2=0;
    out.charge_2=0;
    out.chargeValid_2=0;
    out.dxy_2=0;
    out.addr1_2=0;
    out.addr2_2=0;
    out.addr3_2=0;
    out.addr4_2=0;
    out.reserved_2=0;
    out.wheel_2=0;
    out.ptSTA_2=0;
    out.SE_2=0;

    out.pt_3=0;
    out.qual_3=0;
    out.eta_3=0;
    out.HF_3=0;
    out.phi_3=0;
    out.bx0_3=0;
    out.charge_3=0;
    out.chargeValid_3=0;
    out.dxy_3=0;
    out.addr1_3=0;
    out.addr2_3=0;
    out.addr3_3=0;
    out.addr4_3=0;
    out.reserved_3=0;
    out.wheel_3=0;
    out.ptSTA_3=0;
    out.SE_3=0;

    if (!tracks.empty()) {
      l1t::RegionalMuonCand mu = trackMaker->convertToBMTF(tracks[0]);
      out.pt_1=mu.hwPt();
      out.qual_1=mu.hwQual();
      out.eta_1=mu.hwEta();
      out.HF_1=mu.hwHF();
      out.phi_1=mu.hwPhi();
      out.bx0_1=0;
      out.charge_1=mu.hwSign();
      out.chargeValid_1=mu.hwSignValid();
      out.dxy_1=mu.hwDXY();
      out.addr1_1=mu.trackSubAddress(l1t::RegionalMuonCand::kStat1);
      out.addr2_1=mu.trackSubAddress(l1t::RegionalMuonCand::kStat2);
      out.addr3_1=mu.trackSubAddress(l1t::RegionalMuonCand::kStat3);
      out.addr4_1=mu.trackSubAddress(l1t::RegionalMuonCand::kStat4);
      if (mu.trackSubAddress(l1t::RegionalMuonCand::kWheelSide))
	out.wheel_1=-mu.trackSubAddress(l1t::RegionalMuonCand::kWheelNum);
      else
	out.wheel_1=mu.trackSubAddress(l1t::RegionalMuonCand::kWheelNum);
      out.ptSTA_1=mu.hwPt2();
    }

    if (tracks.size()>1) {
      l1t::RegionalMuonCand mu = trackMaker->convertToBMTF(tracks[1]);
      out.pt_2=mu.hwPt();
      out.qual_2=mu.hwQual();
      out.eta_2=mu.hwEta();
      out.HF_2=mu.hwHF();
      out.phi_2=mu.hwPhi();
      out.bx0_2=0;
      out.charge_2=mu.hwSign();
      out.chargeValid_2=mu.hwSignValid();
      out.dxy_2=mu.hwDXY();
      out.addr1_2=mu.trackSubAddress(l1t::RegionalMuonCand::kStat1);
      out.addr2_2=mu.trackSubAddress(l1t::RegionalMuonCand::kStat2);
      out.addr3_2=mu.trackSubAddress(l1t::RegionalMuonCand::kStat3);
      out.addr4_2=mu.trackSubAddress(l1t::RegionalMuonCand::kStat4);
      if (mu.trackSubAddress(l1t::RegionalMuonCand::kWheelSide))
	out.wheel_2=-mu.trackSubAddress(l1t::RegionalMuonCand::kWheelNum);
      else
	out.wheel_2=mu.trackSubAddress(l1t::RegionalMuonCand::kWheelNum);
      out.ptSTA_2=mu.hwPt2();
    }

    if (tracks.size()>2) {
      l1t::RegionalMuonCand mu = trackMaker->convertToBMTF(tracks[2]);
      out.pt_3=mu.hwPt();
      out.qual_3=mu.hwQual();
      out.eta_3=mu.hwEta();
      out.HF_3=mu.hwHF();
      out.phi_3=mu.hwPhi();
      out.bx0_3=0;
      out.charge_3=mu.hwSign();
      out.chargeValid_3=mu.hwSignValid();
      out.dxy_3=mu.hwDXY();
      out.addr1_3=mu.trackSubAddress(l1t::RegionalMuonCand::kStat1);
      out.addr2_3=mu.trackSubAddress(l1t::RegionalMuonCand::kStat2);
      out.addr3_3=mu.trackSubAddress(l1t::RegionalMuonCand::kStat3);
      out.addr4_3=mu.trackSubAddress(l1t::RegionalMuonCand::kStat4);
      if (mu.trackSubAddress(l1t::RegionalMuonCand::kWheelSide))
	out.wheel_2=-mu.trackSubAddress(l1t::RegionalMuonCand::kWheelNum);
      else
	out.wheel_2=mu.trackSubAddress(l1t::RegionalMuonCand::kWheelNum);
      out.ptSTA_3=mu.hwPt2();
    }
    return out;
  
}





void L1TMuonBarrelKalmanSectorProcessor::verbose(L1TMuonBarrelKalmanAlgo* algo,const L1MuKBMTrackCollection& tracks) {
  L1TMuonBarrelKalmanSectorProcessor::bmtf_out out = makeWord(algo,tracks);
    std::cout << "O "<<sector_<<" "<< out.pt_1 << " " << out.qual_1 << " " << out.eta_1 << " " << out.HF_1 << " " << out.phi_1 << " " << out.charge_1 << " " << out.chargeValid_1 << " " << out.dxy_1 <<" " << out.addr1_1<< " " << out.addr2_1 <<" " << out.addr3_1 << " " << out.addr4_1 << " " << out.wheel_1 << " " << out.ptSTA_1 << " " << out.pt_2 << " " << out.qual_2 << " " << out.eta_2 << " " << out.HF_2 << " " << out.phi_2 << " " << out.charge_2 << " " << out.chargeValid_2 << " " << out.dxy_2 <<" " << out.addr1_2<< " " << out.addr2_2 <<" " << out.addr3_2 << " " << out.addr4_2 << " " << out.wheel_2 << " " << out.ptSTA_2 << " " << out.pt_3 << " " << out.qual_3 << " " << out.eta_3 << " " << out.HF_3 << " " << out.phi_3 << " " << out.charge_3 << " " << out.chargeValid_3 << " " << out.dxy_3 <<" " << out.addr1_3<< " " << out.addr2_3 <<" " << out.addr3_3 << " " << out.addr4_3 << " " << out.wheel_3 << " " << out.ptSTA_3 << std::endl;


}




L1MuKBMTrackCollection L1TMuonBarrelKalmanSectorProcessor::cleanAndSort(const L1MuKBMTrackCollection& pretracks,uint keep) {
  L1MuKBMTrackCollection out;

  if (verbose_) 
    printf(" -----Preselected Kalman Tracks for sector %d----- \n",sector_);


  for(const auto& track1 : pretracks) {
    if (verbose_)
      printf("Pre Track charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d bitmask=%d rank=%d chi=%d pts=%f %f\n",track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.hitPattern(),track1.rank(),track1.approxChi2(),track1.pt(),track1.ptUnconstrained()); 

    bool keep=true;
    for(const auto& track2 : pretracks) {
      if (track1==track2)
	continue;
      if (!track1.overlapTrack(track2))
	continue;

      if (track1.rank()<track2.rank())
	keep=false;

      if ((track1.rank()==track2.rank()) && (fabs(track1.stubs()[0]->whNum())<fabs(track2.stubs()[0]->whNum())))
	keep=false;
    }
    if (keep) 
      out.push_back(track1);
  }

  if (verbose_) {
  printf(" -----Algo Result Kalman Tracks-----\n");
  for (const auto& track1 :out)
    printf("Final Kalman Track charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f\n",track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained()); 
  }



  TrackSorter sorter;
  if (!out.empty())
    std::sort(out.begin(),out.end(),sorter);


  L1MuKBMTrackCollection exported;
  for (uint i=0;i<out.size();++i)
    if (i<=keep)
      exported.push_back(out[i]);
  return exported;
}


