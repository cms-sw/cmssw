
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanStubProcessor.h"
#include "math.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"

L1TMuonBarrelKalmanStubProcessor::L1TMuonBarrelKalmanStubProcessor():
  minPhiQuality_(0),
  minBX_(-3),
  maxBX_(3)
{
 
} 



L1TMuonBarrelKalmanStubProcessor::L1TMuonBarrelKalmanStubProcessor(const edm::ParameterSet& iConfig):
  minPhiQuality_(iConfig.getParameter<int>("minPhiQuality")),
  minBX_(iConfig.getParameter<int>("minBX")),
  maxBX_(iConfig.getParameter<int>("maxBX")),
  eta1_(iConfig.getParameter<std::vector<int> >("cotTheta_1")),
  eta2_(iConfig.getParameter<std::vector<int> >("cotTheta_2")),
  eta3_(iConfig.getParameter<std::vector<int> >("cotTheta_3")),
  disableMasks_(iConfig.getParameter<bool>("disableMasks")),
  verbose_(iConfig.getParameter<int>("verbose"))
{

} 



L1TMuonBarrelKalmanStubProcessor::~L1TMuonBarrelKalmanStubProcessor() {}



bool L1TMuonBarrelKalmanStubProcessor::isGoodPhiStub(const L1MuDTChambPhDigi * stub) {
  if (stub->code()<minPhiQuality_)
    return false;
  return true;
}




L1MuKBMTCombinedStub 
L1TMuonBarrelKalmanStubProcessor::buildStub(const L1MuDTChambPhDigi& phiS,const L1MuDTChambThDigi* etaS) {
  int wheel = phiS.whNum();
  int sector = phiS.scNum();
  int station = phiS.stNum();
  int phi = phiS.phi();
  int phiB = phiS.phiB();
  bool tag = (phiS.Ts2Tag()==1);
  int bx=phiS.bxNum();
  int quality=phiS.code();



  //Now full eta
  int qeta1=0;
  int qeta2=0;
  int eta1=255;
  int eta2=255; 


  bool hasEta=false;
  for (uint i=0;i<7;++i) {
    if (etaS->position(i)==0)
      continue;
    if (!hasEta) {
      eta1=calculateEta(i,etaS->whNum(),etaS->scNum(),etaS->stNum());
      if (etaS->quality(i)==1)
	qeta1=2;
      else
	qeta1=1;
    }
    else {
      eta2=calculateEta(i,etaS->whNum(),etaS->scNum(),etaS->stNum());
      if (etaS->quality(i)==1)
	qeta2=2;
      else
	qeta2=1;
    }
  }
  L1MuKBMTCombinedStub stub(wheel,sector,station,phi,phiB,tag,
			    bx,quality,eta1,eta2,qeta1,qeta2);
  
  return stub;

  }




L1MuKBMTCombinedStub 
L1TMuonBarrelKalmanStubProcessor::buildStubNoEta(const L1MuDTChambPhDigi& phiS) {
  int wheel = phiS.whNum();
  int sector = phiS.scNum();
  int station = phiS.stNum();
  int phi = phiS.phi();
  int phiB = phiS.phiB();
  bool tag = (phiS.Ts2Tag()==1);
  int bx=phiS.bxNum();
  int quality=phiS.code();


  //Now full eta
  int qeta1=0;
  int qeta2=0;
  int eta1=7;
  int eta2=7; 
  L1MuKBMTCombinedStub stub(wheel,sector,station,phi,phiB,tag,
			    bx,quality,eta1,eta2,qeta1,qeta2);

  return stub;

}





L1MuKBMTCombinedStubCollection 
L1TMuonBarrelKalmanStubProcessor::makeStubs(const L1MuDTChambPhContainer* phiContainer,const L1MuDTChambThContainer* etaContainer,const L1TMuonBarrelParams& params) {


  //get the masks from th standard BMTF setup!
  //    const L1TMuonBarrelParamsRcd& bmtfParamsRcd = setup.get<L1TMuonBarrelParamsRcd>();
  //  bmtfParamsRcd.get(bmtfParamsHandle);
  //  const L1TMuonBarrelParams& bmtfParams = *bmtfParamsHandle.product();
  //  masks_ =  bmtfParams.l1mudttfmasks;


  //get the masks!
  L1MuDTTFMasks msks = params.l1mudttfmasks;


  L1MuKBMTCombinedStubCollection  out;
  for (int bx=minBX_;bx<=maxBX_;bx++) {
    for (int wheel=-2;wheel<=2;wheel++) {
      for (uint sector=0;sector<12;sector++) {
	for (uint station=1;station<5;station++) {

	  //Have to cook up something for the fact that KMTF doesnt use 2 SP at whel=0
	  int lwheel1;
	  int lwheel2;
	  if (wheel<0) {
	    lwheel1=wheel-1;
	    lwheel2=wheel-1;
	  }
	  else if (wheel>0) {
	    lwheel1=wheel+1;
	    lwheel2=wheel+1;
	  }
	  else {
	    lwheel1=-1;
	    lwheel2=+1;
	  }
	    
	  bool phiMask=false;
	  bool etaMask=false;
	  if (station==1) {
	    phiMask = msks.get_inrec_chdis_st1(lwheel1, sector) |msks.get_inrec_chdis_st1(lwheel2, sector);  
	    etaMask = msks.get_etsoc_chdis_st1(lwheel1, sector) |msks.get_etsoc_chdis_st1(lwheel2, sector);  	    
	  }
	  if (station==2) {
	    phiMask = msks.get_inrec_chdis_st2(lwheel1, sector) |msks.get_inrec_chdis_st2(lwheel2, sector);  
	    etaMask = msks.get_etsoc_chdis_st2(lwheel1, sector) |msks.get_etsoc_chdis_st2(lwheel2, sector);  

	  }
	  if (station==3) {
	    phiMask = msks.get_inrec_chdis_st3(lwheel1, sector) |msks.get_inrec_chdis_st3(lwheel2, sector);  
	    etaMask = msks.get_etsoc_chdis_st3(lwheel1, sector) |msks.get_etsoc_chdis_st3(lwheel2, sector);  
	  }
	  if (station==4) {
	    phiMask = msks.get_inrec_chdis_st4(lwheel1, sector) |msks.get_inrec_chdis_st4(lwheel2, sector);  
	  }


	  if (disableMasks_)
	    {
	      phiMask=false;
	      etaMask=false;
	    }


	  bool hasEta=false;
	  L1MuDTChambThDigi const*  tseta   = etaContainer->chThetaSegm(wheel,station,sector,bx);
	  if (tseta && (!etaMask)) {
	    hasEta=true;
	  }

	  //	  if (abs(wheel)==2 && station==1)
	  //	    continue;
	    

	  L1MuDTChambPhDigi const* high = phiContainer->chPhiSegm1(wheel,station,sector,bx);
	  if (high && (!phiMask)) {
	    if (high->code()>=minPhiQuality_) {
	      const L1MuDTChambPhDigi& stubPhi = *high;
	      if (hasEta) {
		out.push_back(buildStub(stubPhi,tseta));
	      }
	      else {
		out.push_back(buildStubNoEta(stubPhi));
	      }
	    }
	  }

	  L1MuDTChambPhDigi const* low = phiContainer->chPhiSegm2(wheel,station,sector,bx-1);
	  if (low && ! (phiMask)) {
	    if (low->code()>=minPhiQuality_) {
	      const L1MuDTChambPhDigi& stubPhi = *low;
	      if (hasEta) {
		out.push_back(buildStub(stubPhi,tseta));
	      }
	      else {
		out.push_back(buildStubNoEta(stubPhi));
	      }
	    }
	  }
	}
      }
    }
  }







  return out;
}



int L1TMuonBarrelKalmanStubProcessor::calculateEta(uint i, int wheel,uint sector,uint station) {
  int eta=0;
  if (wheel>0) {
	eta=7*wheel+3-i;
      }
  else if (wheel<0) {
	eta=7*wheel+i-3;
  }
  else {
    if (sector==0 || sector==3 ||sector==4 ||sector==7 ||sector==8 ||sector==11)
      eta=i-3;
    else
      eta=3-i;
  }


  if (station==1)
    eta=-eta1_[eta+17];
  else if (station==2)
    eta=-eta2_[eta+17];
  else 
    eta=-eta3_[eta+17];



  return eta;



}



L1TMuonBarrelKalmanStubProcessor::bmtf_in L1TMuonBarrelKalmanStubProcessor::makePattern(const L1MuDTChambPhContainer* phiContainer,const L1MuDTChambThContainer* etaContainer,int sector, int wheel) {
  L1TMuonBarrelKalmanStubProcessor::bmtf_in out;

    out.ts1_st1_phi=0;
    out.ts1_st1_phib=0;
    out.ts1_st1_q=0;
    out.ts1_st1_rpc=0;
    out.ts1_st1_cal=0;

    const L1MuDTChambPhDigi* seg = phiContainer->chPhiSegm1(wheel,1,sector,0);
    if (seg) {
      out.ts1_st1_phi=seg->phi();
      out.ts1_st1_phib=seg->phiB();
      out.ts1_st1_q=seg->code();
      out.ts1_st1_rpc=0;
      out.ts1_st1_cal=0;
    }
      




    out.ts1_st2_phi=0;
    out.ts1_st2_phib=0;
    out. ts1_st2_q=0;
    out.ts1_st2_rpc=0;
    out.ts1_st2_cal=0;

    seg = phiContainer->chPhiSegm1(wheel,2,sector,0);
    if (seg) {
      out.ts1_st2_phi=seg->phi();
      out.ts1_st2_phib=seg->phiB();
      out.ts1_st2_q=seg->code();
      out.ts1_st2_rpc=0;
      out.ts1_st2_cal=0;
    }


    out.ts1_st3_phi=0;
    out.ts1_st3_phib=0;
    out.ts1_st3_q=0;
    out.ts1_st3_rpc=0;
    out.ts1_st3_cal=0;

    seg = phiContainer->chPhiSegm1(wheel,3,sector,0);
    if (seg) {
      out.ts1_st3_phi=seg->phi();
      out.ts1_st3_phib=seg->phiB();
      out.ts1_st3_q=seg->code();
      out.ts1_st3_rpc=0;
      out.ts1_st3_cal=0;
    }


    out.ts1_st4_phi=0;
    out.ts1_st4_phib=0;
    out.ts1_st4_q=0;
    out.ts1_st4_rpc=0;
    out.ts1_st4_cal=0;

    seg = phiContainer->chPhiSegm1(wheel,4,sector,0);
    if (seg) {
      out.ts1_st4_phi=seg->phi();
      out.ts1_st4_phib=seg->phiB();
      out.ts1_st4_q=seg->code();
      out.ts1_st4_rpc=0;
      out.ts1_st4_cal=0;
    }

    out.eta_hit_st1=0;
    out.eta_hit_st2=0;
    out.eta_hit_st3=0;
    out.eta_qbit_st1=0;
    out.eta_qbit_st2=0;
    out.eta_qbit_st3=0;

    const L1MuDTChambThDigi*  eta   = etaContainer->chThetaSegm(wheel,1,sector,0);
    if (eta) {
      out.eta_hit_st1=eta->position(0)+(eta->position(1)<<1) +(eta->position(2)<<2) +(eta->position(3)<<3) +(eta->position(4)<<4)+(eta->position(5)<<5)+(eta->position(6)<<6);
      out.eta_qbit_st1=eta->quality(0)+(eta->quality(1)<<1) +(eta->quality(2)<<2) +(eta->quality(3)<<3) +(eta->quality(4)<<4)+(eta->quality(5)<<5)+(eta->quality(6)<<6);

    }
    eta   = etaContainer->chThetaSegm(wheel,2,sector,0);
    if (eta) {
      out.eta_hit_st2=eta->position(0)+(eta->position(1)<<1) +(eta->position(2)<<2) +(eta->position(3)<<3) +(eta->position(4)<<4)+(eta->position(5)<<5)+(eta->position(6)<<6);
      out.eta_qbit_st2=eta->quality(0)+(eta->quality(1)<<1) +(eta->quality(2)<<2) +(eta->quality(3)<<3) +(eta->quality(4)<<4)+(eta->quality(5)<<5)+(eta->quality(6)<<6);


    }
    eta   = etaContainer->chThetaSegm(wheel,3,sector,0);
    if (eta) {
      out.eta_hit_st3=eta->position(0)+(eta->position(1)<<1) +(eta->position(2)<<2) +(eta->position(3)<<3) +(eta->position(4)<<4)+(eta->position(5)<<5)+(eta->position(6)<<6);
      out.eta_qbit_st3=eta->quality(0)+(eta->quality(1)<<1) +(eta->quality(2)<<2) +(eta->quality(3)<<3) +(eta->quality(4)<<4)+(eta->quality(5)<<5)+(eta->quality(6)<<6);
    }

    out.bcnt_1a=0;
    out.bcnt_1b=0;
    out.bcnt_1c=0;
    out.bcnt_1d=0;
    out.bcnt_1e=0;
    out.bcnt_1f=0;

    out.bc0_1=0;





    out.ts2_st1_phi=0;
    out.ts2_st1_phib=0;
    out.ts2_st1_q=0;
    out.ts2_st1_rpc=0;
    out.ts2_st1_cal=0;


    seg = phiContainer->chPhiSegm2(wheel,1,sector,-1);
    if (seg) {
      out.ts2_st1_phi=seg->phi();
      out.ts2_st1_phib=seg->phiB();
      out.ts2_st1_q=seg->code();
    }



    out.ts2_st2_phi=0;
    out.ts2_st2_phib=0;
    out.ts2_st2_q=0;
    out.ts2_st2_rpc=0;
    out.ts2_st2_cal=0;

    seg = phiContainer->chPhiSegm2(wheel,2,sector,-1);
    if (seg) {
      out.ts2_st2_phi=seg->phi();
      out.ts2_st2_phib=seg->phiB();
      out.ts2_st2_q=seg->code();
    }


    out.ts2_st3_phi=0;
    out.ts2_st3_phib=0;
    out.ts2_st3_q=0;
    out.ts2_st3_rpc=0;
    out.ts2_st3_cal=0;

    seg = phiContainer->chPhiSegm2(wheel,3,sector,-1);
    if (seg) {
      out.ts2_st3_phi=seg->phi();
      out.ts2_st3_phib=seg->phiB();
      out.ts2_st3_q=seg->code();
    }


    out.ts2_st4_phi=0;
    out.ts2_st4_phib=0;
    out.ts2_st4_q=0;
    out.ts2_st4_rpc=0;
    out.ts2_st4_cal=0;


    seg = phiContainer->chPhiSegm2(wheel,4,sector,-1);
    if (seg) {
      out.ts2_st4_phi=seg->phi();
      out.ts2_st4_phib=seg->phiB();
      out.ts2_st4_q=seg->code();
    }


    out.bcnt_2a=0;
    out.bcnt_2b=0;
    out.bcnt_2c=0;
    out.bcnt_2d=0;
    out.bcnt_2e=0;
    out.bcnt_2f=0;
    out.bc0_2=0;


  return out;
}

void L1TMuonBarrelKalmanStubProcessor::printWord(const L1MuDTChambPhContainer* phiContainer,const L1MuDTChambThContainer* etaContainer,int sector,int wheel) {
  L1TMuonBarrelKalmanStubProcessor::bmtf_in data  = makePattern(phiContainer,etaContainer,sector,wheel);
  std::cout << "I " << sector << " " << wheel << " " << data.ts1_st1_phi << " " << data.ts1_st1_phib << " " << data.ts1_st1_q << " " << data.ts1_st2_phi << " " << data.ts1_st2_phib << " " << data.ts1_st2_q << " " <<  data.ts1_st3_phi << " " << data.ts1_st3_phib << " " << data.ts1_st3_q << " " << data.ts1_st4_phi << " " << data.ts1_st4_phib << " " << data.ts1_st4_q << " " << data.eta_hit_st1 << " " << data.eta_hit_st2 << " " << data.eta_hit_st3 << " " << data.ts2_st1_phi << " " << data.ts2_st1_phib << " " << data.ts2_st1_q << " " << data.ts2_st2_phi << " " << data.ts2_st2_phib << " " << data.ts2_st2_q << " " <<  data.ts2_st3_phi << " " << data.ts2_st3_phib << " " << data.ts2_st3_q << " " << data.ts2_st4_phi << " " << data.ts2_st4_phib << " " << data.ts2_st4_q << " " << data.eta_qbit_st1 << " " << data.eta_qbit_st2 << " " << data.eta_qbit_st3 << std::endl;
}
