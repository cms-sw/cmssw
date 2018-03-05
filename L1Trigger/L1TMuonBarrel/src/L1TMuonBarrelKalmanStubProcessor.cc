#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanStubProcessor.h"
#include "math.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFParameters.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFParametersRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuDTTFMasks.h"
#include "CondFormats/DataRecord/interface/L1MuDTTFMasksRcd.h"

L1TMuonBarrelKalmanStubProcessor::L1TMuonBarrelKalmanStubProcessor():
  minPhiQuality_(0),
  minThetaQuality_(0),
  minBX_(-3),
  maxBX_(3)
{
 
} 



L1TMuonBarrelKalmanStubProcessor::L1TMuonBarrelKalmanStubProcessor(const edm::ParameterSet& iConfig):
  minPhiQuality_(iConfig.getParameter<int>("minPhiQuality")),
  minThetaQuality_(iConfig.getParameter<int>("minThetaQuality")),
  minBX_(iConfig.getParameter<int>("minBX")),
  maxBX_(iConfig.getParameter<int>("maxBX")),
  etaLUT_minus_2_1(iConfig.getParameter<std::vector<int> >("etaLUT_minus_2_1")),
  etaLUT_minus_2_2(iConfig.getParameter<std::vector<int> >("etaLUT_minus_2_2")),
  etaLUT_minus_2_3(iConfig.getParameter<std::vector<int> >("etaLUT_minus_2_3")),
  etaLUT_minus_1_1(iConfig.getParameter<std::vector<int> >("etaLUT_minus_1_1")),
  etaLUT_minus_1_2(iConfig.getParameter<std::vector<int> >("etaLUT_minus_1_2")),
  etaLUT_minus_1_3(iConfig.getParameter<std::vector<int> >("etaLUT_minus_1_3")),
  etaLUT_0_1(iConfig.getParameter<std::vector<int> >("etaLUT_0_1")),
  etaLUT_0_2(iConfig.getParameter<std::vector<int> >("etaLUT_0_2")),
  etaLUT_0_3(iConfig.getParameter<std::vector<int> >("etaLUT_0_3")),
  etaLUT_plus_1_1(iConfig.getParameter<std::vector<int> >("etaLUT_plus_1_1")),
  etaLUT_plus_1_2(iConfig.getParameter<std::vector<int> >("etaLUT_plus_1_2")),
  etaLUT_plus_1_3(iConfig.getParameter<std::vector<int> >("etaLUT_plus_1_3")),
  etaLUT_plus_2_1(iConfig.getParameter<std::vector<int> >("etaLUT_plus_2_1")),
  etaLUT_plus_2_2(iConfig.getParameter<std::vector<int> >("etaLUT_plus_2_2")),
  etaLUT_plus_2_3(iConfig.getParameter<std::vector<int> >("etaLUT_plus_2_3")),
  etaCoarseLUT_minus_2(iConfig.getParameter<std::vector<int> >("etaCoarseLUT_minus_2")),
  etaCoarseLUT_minus_1(iConfig.getParameter<std::vector<int> >("etaCoarseLUT_minus_1")),
  etaCoarseLUT_0(iConfig.getParameter<std::vector<int> >("etaCoarseLUT_0")),
  etaCoarseLUT_plus_1(iConfig.getParameter<std::vector<int> >("etaCoarseLUT_plus_1")),
  etaCoarseLUT_plus_2(iConfig.getParameter<std::vector<int> >("etaCoarseLUT_plus_2")),
  verbose_(iConfig.getParameter<int>("verbose"))
{

} 



L1TMuonBarrelKalmanStubProcessor::~L1TMuonBarrelKalmanStubProcessor() {}



bool L1TMuonBarrelKalmanStubProcessor::isGoodPhiStub(const L1MuDTChambPhDigi * stub) {
  if (stub->code()<minPhiQuality_)
    return false;
  return true;
}

std::pair<bool,bool> L1TMuonBarrelKalmanStubProcessor::isGoodThetaStub(const L1MuDTChambThDigi * stub,uint pos1,uint pos2 ) {

  bool seg1=true;
  bool seg2=true;
  
  if (stub->quality(pos1)<minThetaQuality_)
    seg1=false;
  if (stub->quality(pos2)<minThetaQuality_)
    seg2=false;
  return std::make_pair(seg1,seg2);
}


L1MuKBMTCombinedStub 
L1TMuonBarrelKalmanStubProcessor::buildStub(const L1MuDTChambPhDigi* phiS,const L1MuDTChambThDigi* etaS) {
  int wheel = phiS->whNum();
  int sector = phiS->scNum();
  int station = phiS->stNum();
  int phi = phiS->phi();
  int phiB = phiS->phiB();
  bool tag = (phiS->Ts2Tag()==1);
  int bx=phiS->bxNum();
  int quality=phiS->code();

  //coarse eta
  int coarseEta;
  

  if (wheel==0) {
    coarseEta = etaCoarseLUT_0[station-1];
  }
  else if (wheel==1) {
    coarseEta = etaCoarseLUT_plus_1[station-1];

  }     
  else if (wheel==2) {
    coarseEta = etaCoarseLUT_plus_2[station-1];

  }     
  else if (wheel==-1) {
    coarseEta = etaCoarseLUT_minus_1[station-1];

  }     
  else {
    coarseEta = etaCoarseLUT_minus_2[station-1];
  }     



  //Now full eta
  int qeta1=-1;
  int qeta2=-1;
  int eta1=0;
  int eta2=0; 


  if (etaS) {
    std::vector<int> eposition;
    std::vector<int> equality;
    for (uint i=0;i<7;++i) {
      if (etaS->position(i)==0 || etaS->quality(i)<minThetaQuality_)
	continue;
      int p=0;
      if (wheel==0) {
	if (station==1)
	  p=(etaLUT_0_1[i]);
	if (station==2)
	  p=(etaLUT_0_2[i]);
	if (station==3)
	  p=(etaLUT_0_3[i]);
	if (!(sector==0 || sector==3 || sector==4 || sector==7 ||sector==8 ||sector==11))
	  p=-p;
      }
      if (wheel==1) {
	if (station==1)
	  p=(etaLUT_plus_1_1[i]);
	if (station==2)
	  p=(etaLUT_plus_1_2[i]);
	if (station==3)
	  p=(etaLUT_plus_1_3[i]);
      }

      if (wheel==2) {
	if (station==1)
	  p=(etaLUT_plus_2_1[i]);
	if (station==2)
	  p=(etaLUT_plus_2_2[i]);
	if (station==3)
	  p=(etaLUT_plus_2_3[i]);
      }
      if (wheel==-1) {
	if (station==1)
	  p=(etaLUT_minus_1_1[i]);
	if (station==2)
	  p=(etaLUT_minus_1_2[i]);
	if (station==3)
	  p=(etaLUT_minus_1_3[i]);
      }

      if (wheel==-2) {
	if (station==1)
	  p=(etaLUT_minus_2_1[i]);
	if (station==2)
	  p=(etaLUT_minus_2_2[i]);
	if (station==3)
	  p=(etaLUT_minus_2_3[i]);
      }
      eposition.push_back(p);
      equality.push_back(etaS->quality(i));

    }


    if (eposition.size()>=1) {
      eta1=eposition[0];
      qeta1=equality[0];
    }
    if (eposition.size()>=2) {
      eta2=eposition[1];
      qeta2=equality[1];
    }

  }

  L1MuKBMTCombinedStub stub(wheel,sector,station,phi,phiB,tag,
			    bx,quality,coarseEta,eta1,eta2,qeta1,qeta2);

  return stub;

}






L1MuKBMTCombinedStubCollection 
L1TMuonBarrelKalmanStubProcessor::makeStubs(const L1MuDTChambPhContainer* phiContainer,const L1MuDTChambThContainer* etaContainer) {


  //get the masks from th standard BMTF setup!
  //    const L1TMuonBarrelParamsRcd& bmtfParamsRcd = setup.get<L1TMuonBarrelParamsRcd>();
  //  bmtfParamsRcd.get(bmtfParamsHandle);
  //  const L1TMuonBarrelParams& bmtfParams = *bmtfParamsHandle.product();
  //  masks_ =  bmtfParams.l1mudttfmasks;



  L1MuKBMTCombinedStubCollection  out;
  for (int bx=minBX_;bx<=maxBX_;bx++) {
    for (int wheel=-2;wheel<=2;wheel++) {
      for (uint sector=0;sector<12;sector++) {
	for (uint station=0;station<5;station++) {
	  const L1MuDTChambPhDigi* high = phiContainer->chPhiSegm1(wheel,station,sector,bx);
	  const L1MuDTChambPhDigi* low  = phiContainer->chPhiSegm2(wheel,station,sector,bx);
	  const L1MuDTChambThDigi*  eta   = etaContainer->chThetaSegm(wheel,station,sector,bx);
	  
	  //Temporary mask
	  if (station==1  && abs(wheel)==2)
	    continue;

	  if (high && high->code()>=minPhiQuality_) {
	    out.push_back(buildStub(high,eta));
	    if (verbose_==1) {
	      printf("Original Stub phi: bx=%d wheel=%d sector=%d station=%d tag=%d phi=%d phiB=%d quality=%d\n",high->bxNum(),high->whNum(),high->scNum(),high->stNum(),high->Ts2Tag(),high->phi(),high->phiB(),high->code()); 
	      printf("New Stub phi: bx=%d wheel=%d sector=%d station=%d tag=%d phi=%d phiB=%d quality=%d coarse=%d eta1=%d,eta2=%d,qeta1=%d,qeta2=%d\n",out[out.size()-1].bxNum(),out[out.size()-1].whNum(),out[out.size()-1].scNum(),out[out.size()-1].stNum(),out[out.size()-1].tag(),out[out.size()-1].phi(),out[out.size()-1].phiB(),out[out.size()-1].quality(),out[out.size()-1].coarseEta(),out[out.size()-1].eta1(),out[out.size()-1].eta2(),out[out.size()-1].qeta1(),out[out.size()-1].qeta2()); 
	    }
	  }
	  if (low&&low->code()>=minPhiQuality_) {
	    out.push_back(buildStub(low,eta));
	    if (verbose_==1) {
	      printf("Original Stub phi: bx=%d wheel=%d sector=%d station=%d tag=%d phi=%d phiB=%d quality=%d\n",low->bxNum(),low->whNum(),low->scNum(),low->stNum(),low->Ts2Tag(),low->phi(),low->phiB(),low->code()); 
	      printf("New Stub phi: bx=%d wheel=%d sector=%d station=%d tag=%d phi=%d phiB=%d quality=%d coarse=%d eta1=%d,eta2=%d,qeta1=%d,qeta2=%d\n",out[out.size()-1].bxNum(),out[out.size()-1].whNum(),out[out.size()-1].scNum(),out[out.size()-1].stNum(),out[out.size()-1].tag(),out[out.size()-1].phi(),out[out.size()-1].phiB(),out[out.size()-1].quality(),out[out.size()-1].coarseEta(),out[out.size()-1].eta1(),out[out.size()-1].eta2(),out[out.size()-1].qeta1(),out[out.size()-1].qeta2()); 
	    }
	  }
	}
      }
    }
  }
  return out;
}

