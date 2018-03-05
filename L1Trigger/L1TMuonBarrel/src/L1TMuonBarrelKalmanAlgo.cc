#include <math.h>
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanAlgo.h"


L1TMuonBarrelKalmanAlgo::L1TMuonBarrelKalmanAlgo(const edm::ParameterSet& settings):
  verbose_(settings.getParameter<bool>("verbose")),
  lutService_(new L1TMuonBarrelKalmanLUTs(settings.getParameter<std::string>("lutFile"))),
  initK_(settings.getParameter<std::vector<double> >("initialK")),
  initK2_(settings.getParameter<std::vector<double> >("initialK2")),
  eLoss_(settings.getParameter<std::vector<double> >("eLoss")),
  aPhi_(settings.getParameter<std::vector<double> >("aPhi")),
  aPhiB_(settings.getParameter<std::vector<double> >("aPhiB")),
  aPhiBNLO_(settings.getParameter<std::vector<double> >("aPhiBNLO")),
  bPhi_(settings.getParameter<std::vector<double> >("bPhi")),
  bPhiB_(settings.getParameter<std::vector<double> >("bPhiB")),
  globalChi2Cut_(settings.getParameter<unsigned int>("globalChi2Cut")),
  chiSquare_(settings.getParameter<std::vector<double> >("chiSquare")),
  chiSquareCutPattern_(settings.getParameter<std::vector<int> >("chiSquareCutPattern")),
  chiSquareCutCurv_(settings.getParameter<std::vector<int> >("chiSquareCutCurvMax")),
  chiSquareCut_(settings.getParameter<std::vector<int> >("chiSquareCut")),
  combos4_(settings.getParameter<std::vector<int> >("combos4")),
  combos3_(settings.getParameter<std::vector<int> >("combos3")),
  combos2_(settings.getParameter<std::vector<int> >("combos2")),
  combos1_(settings.getParameter<std::vector<int> >("combos1")),
  useOfflineAlgo_(settings.getParameter<bool>("useOfflineAlgo")),
  mScatteringPhi_(settings.getParameter<std::vector<double> >("mScatteringPhi")),
  mScatteringPhiB_(settings.getParameter<std::vector<double> >("mScatteringPhiB")),
  pointResolutionPhi_(settings.getParameter<double>("pointResolutionPhi")),
  pointResolutionPhiB_(settings.getParameter<double>("pointResolutionPhiB")),
  pointResolutionVertex_(settings.getParameter<double>("pointResolutionVertex"))  

{



}



std::pair<bool,uint> L1TMuonBarrelKalmanAlgo::getByCode(const L1MuKBMTrackCollection& tracks,int mask) {
  for (uint i=0;i<tracks.size();++i) {
    printf("Code=%d, track=%d\n",tracks[i].hitPattern(),mask); 
    if (tracks[i].hitPattern()==mask)
      return std::make_pair(true,i);
  }
  return std::make_pair(false,0);
}
void L1TMuonBarrelKalmanAlgo::addBMTFMuon(int bx,const L1MuKBMTrack& track,  std::unique_ptr<l1t::RegionalMuonCandBxCollection>& out) {

  int  K = abs(track.curvatureAtVertex());
  //calibration
  int sign,signValid;

  if (track.curvatureAtVertex()==0) {
    sign=0;
    signValid=0;
  }
  else if (track.curvatureAtVertex()>0) {
    sign=0;
    signValid=1;
  }
  else  {
    sign=1;
    signValid=1;
  }	  

  if (K<22)
    K=22;

  if (K>2047)
    K=2047;

  float lsb=1.25/float(1<<13);
  int pt = int(2*(1.0/(lsb*float(K))));
  if (pt>511)
    pt=511;

  int  K2 = abs(track.curvatureAtMuon());
  if (K2<22)
    K2=22;

  if (K2>2047)
    K2=2047;
  int pt2 = int(1.0/(lsb*float(K2))); 
  if (pt2>254)
    pt2=254;
  int eta  = track.hasFineEta() ? track.fineEta() : track.coarseEta();
  
  int phi=track.phiAtMuon();
  phi=int((phi*M_PI/(6.0*2048.0)+15.0*M_PI/180.)/(0.625*M_PI/180.0));

  int processor=track.sector();
  int HF = track.hasFineEta();
  
  int quality=rank(track)/64;

  int dxy=abs(track.dxy())>>7;

  int trackAddr;
  std::map<int,int> addr = trackAddress(track,trackAddr);

  l1t::RegionalMuonCand muon(pt,phi,eta,sign,signValid,quality,processor,l1t::bmtf,addr);
  muon.setHwHF(HF);
  muon.setHwPt2(pt2);
  muon.setHwDXY(dxy);


  //nw the words!
  uint32_t word1=pt;
  word1=word1 | quality<<9;
  word1=word1 | (twosCompToBits(eta))<<13;
  word1=word1 | HF<<22;
  word1=word1 | (twosCompToBits(phi))<<23;

  uint32_t word2=sign;
  word2=word2 | signValid<<1;
  word2=word2 | dxy<<2;
  word2=word2 | trackAddr<<4;
  word2=word2 | (twosCompToBits(track.wheel()))<<20;
  word2=word2 | pt2<<23;
  muon.setDataword(word2,word1);
  out->push_back(bx,muon);
}



std::pair<bool,uint> L1TMuonBarrelKalmanAlgo::match(const L1MuKBMTrack& track, const L1MuKBMTCombinedStubRefVector& stubs) {
  L1MuKBMTCombinedStubRefVector selected;

  bool found=false;
  uint best=0;
  int distance=100000;
  uint N=0;
  for (const auto& stub :stubs)  {
    N=N+1;
    if (stub->stNum()!=track.step()) 
      continue;
    if (abs(track.positionAngle()-correctedPhi(stub,track.sector()))<distance) {
      distance = abs(track.positionAngle()-correctedPhi(stub,track.sector()));
      best=N-1;
      found=true;
    }
  }
  return std::make_pair(found,best);
}



int L1TMuonBarrelKalmanAlgo::correctedPhiB(const L1MuKBMTCombinedStubRef& stub) {
  //Promote phiB to 12 bits
  return 8*stub->phiB();

}

int L1TMuonBarrelKalmanAlgo::correctedPhi(const L1MuKBMTCombinedStubRef& stub,int sector) {
  if (stub->scNum()==sector) {
    return stub->phi();
  }
  else if ((stub->scNum()==sector-1) || (stub->scNum()==11 && sector==0)) {
    return stub->phi()-2144;
  }
  else if ((stub->scNum()==sector+1) || (stub->scNum()==0 && sector==11)) {
    return stub->phi()+2144;
  }
  return stub->phi();
} 


int L1TMuonBarrelKalmanAlgo::hitPattern(const L1MuKBMTrack& track)  {
  unsigned int mask = 0;
  for (const auto& stub : track.stubs()) {
    mask = mask+round(pow(2,stub->stNum()-1));
  }
  return mask;
}



int L1TMuonBarrelKalmanAlgo::customBitmask(unsigned int bit1,unsigned int bit2,unsigned int bit3,unsigned int bit4)  {
  return bit1*1+bit2*2+bit3*4+bit4*8;
}

bool L1TMuonBarrelKalmanAlgo::getBit(int bitmask,int pos)  {
  return (bitmask & ( 1 << pos )) >> pos;
}



void L1TMuonBarrelKalmanAlgo::propagate(L1MuKBMTrack& track) {
  int K = track.curvature();
  int phi = track.positionAngle();
  int phiB = track.bendingAngle();
  unsigned int step = track.step();

  int charge=1;
  if (K!=0) 
    charge = K/abs(K);

  //energy loss term only for MU->VERTEX
  //int offset=int(charge*eLoss_[step-1]*K*K);
  //  if (abs(offset)>4096)
  //      offset=4096*offset/abs(offset);
  int KNew =wrapAround(int(K/(1+charge*eLoss_[step-1]*K)),8192);

  //phi propagation
  int phiNew =wrapAround(phi+int(aPhi_[step-1]*K)-int(bPhi_[step-1]*phiB),8192);

  //phiB propagation
  int phiBNew = wrapAround(int(aPhiB_[step-1]*K)  +int(bPhiB_[step-1]*phiB),2048);
  
  //Only for the propagation to vertex we use the LUT for better precision and the full function
  if (step==1)
    phiBNew = wrapAround(int(aPhiB_[step-1]*K/(1+charge*aPhiBNLO_[step-1]*K))+int(bPhiB_[step-1]*phiB),2048);

  ///////////////////////////////////////////////////////
  //Rest of the stuff  is for the offline version only 
  //where we want to check what is happening in the covariaznce matrix 
  
  //Create the transformation matrix
  double a[9];
  a[0] = 1.;
  a[1] = 0.0;
  a[2] = 0.0;
  a[3] = aPhi_[step-1];
  //  a[3] = 0.0;
  a[4] = 1.0;
  a[5] = -bPhi_[step-1];
  //a[6]=0.0;
  a[6] = aPhiB_[step-1];
  a[7] = 0.0;
  a[8] = bPhiB_[step-1];


  ROOT::Math::SMatrix<double,3> P(a,9);

  const std::vector<double>& covLine = track.covariance();
  L1MuKBMTrack::CovarianceMatrix cov(covLine.begin(),covLine.end());
  cov = ROOT::Math::Similarity(P,cov);

  
  //Add the multiple scattering
  double phiRMS = mScatteringPhi_[step-1]*K*K;
  double phiBRMS = mScatteringPhiB_[step-1]*K*K;

  std::vector<double> b(6);
  b[0] = 0;
  b[1] = 0;
  b[2] =phiRMS;
  b[3] =0;
  b[4] = 0;
  b[5] = phiBRMS;

  reco::Candidate::CovarianceMatrix MS(b.begin(),b.end());

  cov = cov+MS;

  if (verbose_) {
    printf("Covariance term for phiB = %f\n",cov(2,2));
    printf("Multiple scattering term for phiB = %f\n",MS(2,2));
  }
 


  track.setCovariance(cov);
  track.setCoordinates(step-1,KNew,phiNew,phiBNew);

}


bool L1TMuonBarrelKalmanAlgo::update(L1MuKBMTrack& track,const L1MuKBMTCombinedStubRef& stub,int mask) {
  updateEta(track,stub);
  if (useOfflineAlgo_) {
    if (mask==3 || mask ==5 || mask==9 ||mask==6|| mask==10 ||mask==12)
          return updateOffline(track,stub);
        else
	  return updateOffline1D(track,stub);

  }
  else
    return updateLUT(track,stub,mask);

}

bool L1TMuonBarrelKalmanAlgo::updateOffline(L1MuKBMTrack& track,const L1MuKBMTCombinedStubRef& stub) {
    int trackK = track.curvature();
    int trackPhi = track.positionAngle();
    int trackPhiB = track.bendingAngle();

    int phi  = correctedPhi(stub,track.sector());
    int phiB = correctedPhiB(stub);



    //Update eta
    track.setCoarseEta(int((track.coarseEta()+stub->coarseEta())/2.0));

    Vector2 residual;
    residual[0] = phi-trackPhi;
    residual[1] = phiB-trackPhiB;

   
   

    //    if (stub->quality()<4)
    //  phiB=trackPhiB;

    Matrix23 H;
    H(0,0)=0.0;
    H(0,1)=1.0;
    H(0,2)=0.0;
    H(1,0)=0.0;
    H(1,1)=0.0;
    H(1,2)=1.0;

    
    CovarianceMatrix2 R;
    R(0,0) = pointResolutionPhi_;
    R(0,1) = 0.0;
    R(1,0) = 0.0;
    R(1,1) = pointResolutionPhiB_;

    const std::vector<double>& covLine = track.covariance();
    L1MuKBMTrack::CovarianceMatrix cov(covLine.begin(),covLine.end());


    CovarianceMatrix2 S = ROOT::Math::Similarity(H,cov)+R;
    if (!S.Invert())
      return false;
    Matrix32 Gain = cov*ROOT::Math::Transpose(H)*S;

    track.setKalmanGain(track.step(),abs(trackK),Gain(0,0),Gain(0,1),Gain(1,0),Gain(1,1),Gain(2,0),Gain(2,1));

    int KNew  = (trackK+int(Gain(0,0)*residual(0)+Gain(0,1)*residual(1)));
    if (abs(KNew)>8192)
      return false;
    
    int phiNew  = wrapAround(trackPhi+residual(0),8192);
    int phiBNew = wrapAround(trackPhiB+int(Gain(2,0)*residual(0)+Gain(2,1)*residual(1)),2048);
    
    track.setResidual(stub->stNum()-1,abs(phi-phiNew)+abs(phiB-phiBNew)/8);


    if (verbose_) {
      printf(" K = %d + %f * %f + %f * %f\n",trackK,Gain(0,0),residual(0),Gain(0,1),residual(1));
      printf(" phiB = %d + %f * %f + %f * %f\n",trackPhiB,Gain(2,0),residual(0),Gain(2,1),residual(1));
    }


    track.setCoordinates(track.step(),KNew,phiNew,phiBNew);
    Matrix33 covNew = cov - Gain*(H*cov);
    L1MuKBMTrack::CovarianceMatrix c;
 
    c(0,0)=covNew(0,0); 
    c(0,1)=covNew(0,1); 
    c(0,2)=covNew(0,2); 
    c(1,0)=covNew(1,0); 
    c(1,1)=covNew(1,1); 
    c(1,2)=covNew(1,2); 
    c(2,0)=covNew(2,0); 
    c(2,1)=covNew(2,1); 
    c(2,2)=covNew(2,2); 
    if (verbose_) {
      printf("Post Fit Covariance Matrix %f %f %f \n",cov(0,0),cov(1,1),cov(2,2));
      
    }

    track.setCovariance(c);
    track.addStub(stub);
    track.setHitPattern(hitPattern(track));

    return true;
}


bool L1TMuonBarrelKalmanAlgo::updateOffline1D(L1MuKBMTrack& track,const L1MuKBMTCombinedStubRef& stub) {
    int trackK = track.curvature();
    int trackPhi = track.positionAngle();
    int trackPhiB = track.bendingAngle();


    int phi  = correctedPhi(stub,track.sector());
    track.setCoarseEta(int((track.coarseEta()+stub->coarseEta())/2.0));

    double residual= phi-trackPhi;

    Matrix13 H;
    H(0,0)=0.0;
    H(0,1)=1.0;
    H(0,2)=0.0;
    

    const std::vector<double>& covLine = track.covariance();
    L1MuKBMTrack::CovarianceMatrix cov(covLine.begin(),covLine.end());

    double S = ROOT::Math::Similarity(H,cov)(0,0)+pointResolutionPhi_;

    if (S==0.0)
      return false;
    Matrix31 Gain = cov*ROOT::Math::Transpose(H)/S;

    track.setKalmanGain(track.step(),abs(trackK),Gain(0,0),0.0,Gain(1,0),0.0,Gain(2,0),0.0);

    int KNew  = wrapAround(trackK+int(Gain(0,0)*residual),8192);
    int phiNew  = wrapAround(trackPhi+residual,8192);
    int phiBNew = wrapAround(trackPhiB+int(Gain(2,0)*residual),2048);
    track.setCoordinates(track.step(),KNew,phiNew,phiBNew);
    Matrix33 covNew = cov - Gain*(H*cov);
    L1MuKBMTrack::CovarianceMatrix c;
 
    c(0,0)=covNew(0,0); 
    c(0,1)=covNew(0,1); 
    c(0,2)=covNew(0,2); 
    c(1,0)=covNew(1,0); 
    c(1,1)=covNew(1,1); 
    c(1,2)=covNew(1,2); 
    c(2,0)=covNew(2,0); 
    c(2,1)=covNew(2,1); 
    c(2,2)=covNew(2,2); 
    track.setCovariance(c);
    track.addStub(stub);
    track.setHitPattern(hitPattern(track));

    return true;
}



bool L1TMuonBarrelKalmanAlgo::updateLUT(L1MuKBMTrack& track,const L1MuKBMTCombinedStubRef& stub,int mask) {
    int trackK = track.curvature();
    int trackPhi = track.positionAngle();
    int trackPhiB = track.bendingAngle();

    //Update eta
    track.setCoarseEta(int((track.coarseEta()+stub->coarseEta())/2.0));

    int phi  = correctedPhi(stub,track.sector());
    int phiB = correctedPhiB(stub);
    //    if (stub->quality()<6)
    //      phiB=trackPhiB;

    Vector2 residual;
    residual[0] = phi-trackPhi;
    residual[1] = phiB-trackPhiB;

    uint absK = abs(trackK);
    if (absK>4095)
      absK = 4095;
    std::vector<float> GAIN = lutService_->trackGain(track.step(),mask,absK/4);
    track.setKalmanGain(track.step(),abs(trackK),GAIN[0],GAIN[1],1,0,GAIN[2],GAIN[3]);

    //For the three stub stuff use only gains 0 and 4
    if (!(mask==3 || mask ==5 || mask==9 ||mask==6|| mask==10 ||mask==12))  {
      GAIN[1]=0.0;
      GAIN[3]=0.0;
    }
      



    int KNew  = wrapAround(trackK+int(GAIN[0]*residual(0)+GAIN[1]*residual(1)),8192);
    int phiNew  = wrapAround(trackPhi+residual(0),8192);
    int phiBNew = wrapAround(trackPhiB+int(GAIN[2]*residual(0)+GAIN[3]*residual(1)),2048);
    track.setCoordinates(track.step(),KNew,phiNew,phiBNew);
    track.addStub(stub);
    track.setHitPattern(hitPattern(track));
    return true;
}




void L1TMuonBarrelKalmanAlgo::updateEta(L1MuKBMTrack& track,const L1MuKBMTCombinedStubRef& stub) {

    if (stub->qeta1()>=0) {
      if (track.hasFineEta()) {
	uint dist2=1000;
	uint dist1 = abs(track.fineEta()-stub->eta1());
	if (stub->qeta2()>=0)
	  dist2 = abs(track.fineEta()-stub->eta2());
	if (dist1<dist2)
	  track.setFineEta((stub->eta1()+track.fineEta())/2);
	else
	  track.setFineEta((stub->eta2()+track.fineEta())/2);
      }else {
	if (stub->qeta2()>=0)
	  track.setFineEta((stub->eta1()+stub->eta2())/2);
	else
	  track.setFineEta(stub->eta1());
      }

    }
}






void L1TMuonBarrelKalmanAlgo::vertexConstraint(L1MuKBMTrack& track) {
  if (useOfflineAlgo_)
    vertexConstraintOffline(track);
  else
    vertexConstraintLUT(track);
 
}


void L1TMuonBarrelKalmanAlgo::vertexConstraintOffline(L1MuKBMTrack& track) {
  double residual = -track.dxy();
  Matrix13 H;
  H(0,0)=0;
  H(0,1)=0;
  H(0,2)=1;

  const std::vector<double>& covLine = track.covariance();
  L1MuKBMTrack::CovarianceMatrix cov(covLine.begin(),covLine.end());
  
  double S = (ROOT::Math::Similarity(H,cov))(0,0)+pointResolutionVertex_;
  S=1.0/S;
  Matrix31 Gain = cov*(ROOT::Math::Transpose(H))*S;
  track.setKalmanGain(track.step(),abs(track.curvature()),Gain(0,0),Gain(1,0),Gain(2,0));

  if (verbose_) {
    printf("sigma3=%f sigma6=%f\n",cov(0,3),cov(3,3));
    printf(" K = %d + %f * %f\n",track.curvature(),Gain(0,0),residual);
  }

  int KNew = wrapAround(int(track.curvature()+Gain(0,0)*residual),8192);
  int phiNew = wrapAround(int(track.positionAngle()+Gain(1,0)*residual),8192);
  int dxyNew = wrapAround(int(track.dxy()+Gain(2,0)*residual),8192);
  if (verbose_)
    printf("Post fit impact parameter=%d\n",dxyNew);
  track.setCoordinatesAtVertex(KNew,phiNew,-residual);
  Matrix33 covNew = cov - Gain*(H*cov);
  L1MuKBMTrack::CovarianceMatrix c;
  c(0,0)=covNew(0,0); 
  c(0,1)=covNew(0,1); 
  c(0,2)=covNew(0,2); 
  c(1,0)=covNew(1,0); 
  c(1,1)=covNew(1,1); 
  c(1,2)=covNew(1,2); 
  c(2,0)=covNew(2,0); 
  c(2,1)=covNew(2,1); 
  c(2,2)=covNew(2,2); 
  track.setCovariance(c);
  //  track.covariance = track.covariance - Gain*H*track.covariance;
}



void L1TMuonBarrelKalmanAlgo::vertexConstraintLUT(L1MuKBMTrack& track) {
  double residual = -track.dxy();
  uint absK = abs(track.curvature());
  if (absK>2047)
    absK = 2047;

std::pair<float,float> GAIN = lutService_->vertexGain(track.hitPattern(),absK/2);
  track.setKalmanGain(track.step(),abs(track.curvature()),GAIN.first,GAIN.second,-1);

  int KNew = wrapAround(int(track.curvature()+GAIN.first*residual),8192);
  int phiNew = wrapAround(int(track.positionAngle()+GAIN.second*residual),8192);
  track.setCoordinatesAtVertex(KNew,phiNew,-residual);
}



void L1TMuonBarrelKalmanAlgo::setFloatingPointValues(L1MuKBMTrack& track,bool vertex) {
  int K,phiINT,etaINT;

  if (track.hasFineEta())
    etaINT=track.fineEta();
  else
    etaINT=track.coarseEta();


  double lsb = 1.25/float(1 << 13);
  double lsbEta = 0.010875;


  if (vertex) {
    K  = track.curvatureAtVertex();
    if (K==0)
      track.setCharge(1);
    else
      track.setCharge(K/abs(K));

    phiINT = track.phiAtVertex();
    double phi= track.sector()*M_PI/6.0+phiINT*M_PI/(6*2048.)-2*M_PI;
    double eta = etaINT*lsbEta;
    if (phi<-M_PI)
      phi=phi+2*M_PI;
    if (K==0)
      K=1;    


    float FK=fabs(K);
    FK = fabs(0.912*FK+(2.557e-5)*FK*FK-6);

    if (FK<51)
      FK=51;
    
    double pt = 1.0/(lsb*(FK));

    track.setPtEtaPhi(pt,eta,phi);
  }
  else {
    K=track.curvatureAtMuon();
    if (K==0)
      K=1;
    if (abs(K)<46)
      K=46*K/abs(K);
    double pt = 1.0/(lsb*abs(K));
    track.setPtUnconstrained(pt);
  }
}



std::pair<bool,L1MuKBMTrack> L1TMuonBarrelKalmanAlgo::chain(const L1MuKBMTCombinedStubRef& seed, const L1MuKBMTCombinedStubRefVector& stubs) {
  L1MuKBMTrackCollection pretracks; 
  std::vector<int> combinatorics;
  switch(seed->stNum()) {
  case 1:
    combinatorics=combos1_;
    break;
  case 2:
    combinatorics=combos2_;
    break;

  case 3:
    combinatorics=combos3_;
    break;

  case 4:
    combinatorics=combos4_;
    break;

  default:
    printf("Something really bad happend\n");
  }

  L1MuKBMTrack nullTrack(seed,correctedPhi(seed,seed->scNum()),correctedPhiB(seed));

  for( const auto& mask : combinatorics) {
    L1MuKBMTrack track(seed,correctedPhi(seed,seed->scNum()),correctedPhiB(seed));
    int phiB = correctedPhiB(seed);
    if (seed->quality()<4)
      phiB=0;
    int charge;
    if (phiB==0)
      charge = 0;
    else
      charge=phiB/abs(phiB);
    int initialK = int(initK_[seed->stNum()-1]*phiB/(1+initK2_[seed->stNum()-1]*charge*phiB));
    if (initialK>8192)
      initialK=8192;
    if (initialK<-8192)
      initialK=-8192;
    
    

    track.setCoordinates(seed->stNum(),initialK,correctedPhi(seed,seed->scNum()),phiB);
    track.setHitPattern(hitPattern(track));
    //Set eta coarse
    track.setCoarseEta(seed->coarseEta());
    //Set fine eta:
    if (seed->qeta1()>=0) {
      if (seed->qeta2()>=0) {
	track.setFineEta((seed->eta1()+seed->eta2())/2);
      }
      else {
	track.setFineEta(seed->eta1());
      }
    }

    //set covariance
    L1MuKBMTrack::CovarianceMatrix covariance;  


    float DK=512*512.;
    covariance(0,0)=DK;
    covariance(0,1)=0;
    covariance(0,2)=0;
    covariance(1,0)=0;
    covariance(1,1)=float(pointResolutionPhi_);
    covariance(1,2)=0;
    covariance(2,0)=0;
    covariance(2,1)=0;
    covariance(2,2)=float(pointResolutionPhiB_);
    track.setCovariance(covariance);
    //
    if (verbose_) {
      printf("New Kalman fit staring at step=%d, phi=%d,phiB=%d with curvature=%d\n",track.step(),track.positionAngle(),track.bendingAngle(),track.curvature());
      printf("BITMASK:");
      for (unsigned int i=0;i<4;++i)
	printf("%d",getBit(mask,i));
      printf("\n");
      printf("------------------------------------------------------\n");
      printf("------------------------------------------------------\n");
      printf("------------------------------------------------------\n");
      printf("stubs:\n");
      for (const auto& stub: stubs) 
	printf("station=%d phi=%d phiB=%d qual=%d \n",stub->stNum(),correctedPhi(stub,seed->scNum()),correctedPhiB(stub),stub->quality()); 
      printf("------------------------------------------------------\n");
      printf("------------------------------------------------------\n");

    }

    while(track.step()>0) {
      // muon station 1 
      if (track.step()==1) {
	track.setCoordinatesAtMuon(track.curvature(),track.positionAngle(),track.bendingAngle());
	setFloatingPointValues(track,false);
	if (verbose_) 
	  printf ("Unconstrained PT  in Muon System: pt=%f\n",track.ptUnconstrained());
      }
      
      propagate(track);
      if (verbose_)
	printf("propagated Coordinates step:%d,phi=%d,phiB=%d,K=%d\n",track.step(),track.positionAngle(),track.bendingAngle(),track.curvature());

      if (track.step()>0) 
	if (getBit(mask,track.step()-1)) {
	  std::pair<bool,uint> bestStub = match(track,stubs);
      	  if ((!bestStub.first) || (!update(track,stubs[bestStub.second],mask)))
	    break;
	  if (verbose_) {
	    printf("updated Coordinates step:%d,phi=%d,phiB=%d,K=%d\n",track.step(),track.positionAngle(),track.bendingAngle(),track.curvature());
	  }
	}
      if (track.step()==0) {
	track.setCoordinatesAtVertex(track.curvature(),track.positionAngle(),track.bendingAngle());
	if (verbose_)
	  printf(" Coordinates before vertex constraint step:%d,phi=%d,dxy=%d,K=%d\n",track.step(),track.phiAtVertex(),track.dxy(),track.curvatureAtVertex());
	estimateChiSquare(track);
	if (abs(track.approxChi2())>globalChi2Cut_)
	  break;
	vertexConstraint(track);
	if (verbose_) {
	  printf(" Coordinates after vertex constraint step:%d,phi=%d,dxy=%d,K=%d  maximum local chi2=%d\n",track.step(),track.phiAtVertex(),track.dxy(),track.curvatureAtVertex(),track.approxChi2());
	  printf("------------------------------------------------------\n");
	  printf("------------------------------------------------------\n");
	}
	setFloatingPointValues(track,true);
	track.setRank(rank(track));
	if (verbose_)
	  printf ("Floating point coordinates at vertex: pt=%f, eta=%f phi=%f\n",track.pt(),track.eta(),track.phi());
	pretracks.push_back(track);
      }
    }
  }

  //Resolve eta
  resolveEtaUnit(pretracks);
  //Now for all the pretracks we need only one 
  L1MuKBMTrackCollection cleaned = cleanAndSort(pretracks,1);

  if (cleaned.size()>0)
    return std::make_pair(true,cleaned[0]);
  return std::make_pair(false,nullTrack);
}       
    





void L1TMuonBarrelKalmanAlgo::estimateChiSquare(L1MuKBMTrack& track) {
  //here we have a simplification of the algorithm for the sake of the emulator - rsult is identical
  // we apply cuts on the firmware as |u -u'|^2 < a+b *K^2 
  int K = track.curvatureAtMuon();

  int chi=0;

  for (const auto& stub: track.stubs()) {
    uint delta=abs(correctedPhi(stub,track.sector())-track.phiAtMuon()+correctedPhiB(stub)-track.phiBAtMuon()-chiSquare_[stub->stNum()-1]*K);
     chi=chi+abs(delta);    
   }
  chi=chi/2;
  if (chi>511)
    chi=511;
   track.setApproxChi2(chi);
}


int L1TMuonBarrelKalmanAlgo::rank(const L1MuKBMTrack& track) {
  //    int offset=0;
    if (hitPattern(track)==customBitmask(0,0,1,1))
      return 65;
    //    return offset+(track.stubs().size()*2+track.quality())*80-track.approxChi2();
    return 700+(track.stubs().size())*80-track.approxChi2();

}



int L1TMuonBarrelKalmanAlgo::wrapAround(int value,int maximum) {
  if (value>maximum-1)
    return value-2*maximum;
  if (value<-maximum)
    return value+2*maximum;
  return value;

}

void L1TMuonBarrelKalmanAlgo::resolveEtaUnit(L1MuKBMTrackCollection& tracks) {
  int bestFineEta=0;
  uint bestSegments=0;
  for (const auto& track : tracks) {
    if (track.stubs().size()>bestSegments && track.hasFineEta()) {
      bestFineEta = track.fineEta();
      bestSegments=track.stubs().size();
    }
  }
  
  for (auto & track :tracks) {
    if (bestSegments!=0 && (!track.hasFineEta())) {
      track.setFineEta(bestFineEta);
      setFloatingPointValues(track,true);
    }
  }



}



L1MuKBMTrackCollection L1TMuonBarrelKalmanAlgo::cleanAndSort(const L1MuKBMTrackCollection& tracks,uint keep) {
  L1MuKBMTrackCollection out;

  if (verbose_) 
    printf(" -----Preselected Kalman Tracks-----\n");


  L1MuKBMTrackCollection pretracks;

  for(const auto& track1 : tracks) {
    if (verbose_)
      printf("Preselected Kalman Track charge=%d pt=%f eta=%f phi=%f curvature=%d curvature STA =%d stubs=%d chi2=%d pts=%f %f\n",track1.charge(),track1.pt(),track1.eta(),track1.phi(),track1.curvatureAtVertex(),track1.curvatureAtMuon(),int(track1.stubs().size()),track1.approxChi2(),track1.pt(),track1.ptUnconstrained()); 

  //Clean up
  bool veto=false;
  for (uint i=0;i<chiSquareCutPattern_.size();++i) {
    if (track1.hitPattern()==chiSquareCutPattern_[i] && abs(track1.curvatureAtVertex())<chiSquareCutCurv_[i] && track1.approxChi2()>chiSquareCut_[i] && track1.curvature()*track1.dxy()<0) {
	veto=true;
	break; 
      }
  }
  if (!veto)
    pretracks.push_back(track1);
  }

  for(const auto& track1 : pretracks) {
    bool keep=true;
    for(const auto& track2 : pretracks) {
      if (track1==track2)
	continue;
      if (!track1.overlap(track2))
	continue;
      if (track1.rank()<track2.rank())
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
  if (out.size()>0)
    std::sort(out.begin(),out.end(),sorter);


  L1MuKBMTrackCollection exported;
  for (uint i=0;i<out.size();++i)
    if (i<=keep)
      exported.push_back(out[i]);
  return exported;
}


int L1TMuonBarrelKalmanAlgo::encode(bool ownwheel,int sector,bool tag) {
  if (ownwheel) {
    if (sector==0) {
      if (tag)
	return 8;
      else 
	return 9;
    }
    else if (sector==1) {
      if (tag)
	return 10;
      else 
	return 11;

    }
    else {
      if (tag)
	return 12;
      else 
	return 13;
    }

  }
  else {
    if (sector==0) {
      if (tag)
	return 0;
      else 
	return 1;
    }
    else if (sector==1) {
      if (tag)
	return 2;
      else 
	return 3;

    }
    else {
      if (tag)
	return 4;
      else 
	return 5;
    }
  }
  return 15;
} 




std::map<int,int> L1TMuonBarrelKalmanAlgo::trackAddress(const L1MuKBMTrack& track,int& word) {
  std::map<int,int> out;
  if (track.wheel()>=0)
    out[l1t::RegionalMuonCand::kWheelSide] = 0;
  else
    out[l1t::RegionalMuonCand::kWheelSide] = 1;

  out[l1t::RegionalMuonCand::kWheelNum] = abs(track.wheel());
  out[l1t::RegionalMuonCand::kStat1]=3;
  out[l1t::RegionalMuonCand::kStat2]=15;
  out[l1t::RegionalMuonCand::kStat3]=15;
  out[l1t::RegionalMuonCand::kStat4]=15;
  out[l1t::RegionalMuonCand::kSegSelStat1]=0;
  out[l1t::RegionalMuonCand::kSegSelStat2]=0;
  out[l1t::RegionalMuonCand::kSegSelStat3]=0;
  out[l1t::RegionalMuonCand::kSegSelStat4]=0;
  out[l1t::RegionalMuonCand::kNumBmtfSubAddr]=0;


  for (const auto stub: track.stubs()) {
    bool ownwheel = stub->whNum() == track.wheel();
    int sector=0;
    if ((stub->scNum()==track.sector()+1) || (stub->scNum()==0 && track.sector()==11))
      sector=+1;
    if ((stub->scNum()==track.sector()-1) || (stub->scNum()==11 && track.sector()==0))
      sector=-1;
    int addr = encode(ownwheel,sector,stub->tag());
   
    if (stub->stNum()==4) {
      addr=addr & 3;
      out[l1t::RegionalMuonCand::kStat1]=addr;
    }      
    if (stub->stNum()==3) {
      out[l1t::RegionalMuonCand::kStat2]=addr;    
    }
    if (stub->stNum()==2) {
      out[l1t::RegionalMuonCand::kStat3]=addr;
    }
    if (stub->stNum()==1) {
      out[l1t::RegionalMuonCand::kStat4]=addr;
    }
  }
    
  word=0;
  word = word | out[l1t::RegionalMuonCand::kStat1]<<12;
  word = word | out[l1t::RegionalMuonCand::kStat2]<<8;
  word = word | out[l1t::RegionalMuonCand::kStat3]<<4;
  word = word | out[l1t::RegionalMuonCand::kStat4];

  

  return out;
}



uint L1TMuonBarrelKalmanAlgo::twosCompToBits(int q) {
  if (q>=0)
    return q;
  else 
    return (~q)+1; 


}
