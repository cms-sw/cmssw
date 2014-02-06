#include "GEMCode/GEMValidation/src/TFTrack.h"

TFTrack::TFTrack(const csc::L1Track *t)
{
  //  init(t);
}

TFTrack::TFTrack(const TFTrack& rhs)
{}

TFTrack::~TFTrack()
{}

void 
TFTrack::init(CSCTFPtLUT* ptLUT,
	      edm::ESHandle< L1MuTriggerScales > &muScales,
	      edm::ESHandle< L1MuTriggerPtScale > &muPtScale)
{
  /*
  //for now, convert using this. LUT in the future
  const unsigned gbl_phi(t->localPhi() + ((t->sector() - 1)*24) + 6);
  if(gbl_phi > 143) gbl_phi -= 143;
  phi_packed_ = gbl_phi & 0xff;
  
  const unsigned eta_sign(t->endcap() == 1 ? 0 : 1);
  const int gbl_eta_(t->eta_packed() | eta_sign << (L1MuRegionalCand::ETA_LENGTH - 1));
  eta_packed_  = gbl_eta & 0x3f;
  
  const unsigned rank(t->rank(), gpt = 0, quality = 0);
  //if (rank != 0 ) {
  //  quality = rank >> L1MuRegionalCand::PT_LENGTH;
  //  gpt = rank & ( (1<<L1MuRegionalCand::PT_LENGTH) - 1);
  //}
  csc::L1Track::decodeRank(rank, gpt, quality);
  q_packed_ = quality & 0x3;
  pt_packed_ = gpt & 0x1f;
  
  //pt = muPtScale->getPtScale()->getLowEdge(pt_packed) + 1.e-6;
  eta_ = muScales->getRegionalEtaScale(2)->getCenter(t->eta_packed());
  phi_ = normalizedPhi( muScales->getPhiScale()->getLowEdge(phi_packed));
  
  //Pt needs some more workaround since it is not in the unpacked data
  //  PtAddress gives an handle on other parameters
  ptadd thePtAddress(t->ptLUTAddress());
  ptdat thePtData  = ptLUT->Pt(thePtAddress);
  // front or rear bit? 
  unsigned trPtBit = (thePtData.rear_rank&0x1f);
  if (thePtAddress.track_fr) trPtBit = (thePtData.front_rank&0x1f);
  // convert the Pt in human readable values (GeV/c)
  pt_ = muPtScale->getPtScale()->getLowEdge(trPtBit); 
  */
  
  //if (trPtBit!=pt_packed) std::cout<<" trPtBit!=pt_packed: "<<trPtBit<<"!="<<pt_packed<<"  pt="<<pt<<" eta="<<eta<<std::endl;
  
//   bool sc_debug = 0;
//   if (sc_debug && deltaOk2){
//     double stpt=-99., steta=-99., stphi=-99.;
//     if (match){
//       stpt = sqrt(match->strk->momentum().perp2());
//       steta = match->strk->momentum().eta();
//       stphi = normalizedPhi( match->strk->momentum().phi() );
//     }
//     //debug = 1;
    
//     double my_phi = normalizedPhi( phi_packed*0.043633231299858237 + 0.0218 ); // M_PI*2.5/180 = 0.0436332312998582370
//     double my_eta = 0.05 * eta_packed + 0.925; //  0.9+0.025 = 0.925
//     //double my_pt = ptscale[pt_packed];
//     //if (fabs(pt - my_pt)>0.005) std::cout<<"scales pt diff: my "<<my_pt<<"  sc: pt "<<pt<<"  eta "<<eta<<" phi "<<phi<<"  mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;
//     if (fabs(eta - my_eta)>0.005) std::cout<<"scales eta diff: my "<<my_eta<<" sc "<<eta<<"  mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;
//     if (fabs(deltaPhi(phi,my_phi))>0.03) std::cout<<"scales phi diff: my "<<my_phi<<" sc "<<phi<<"  mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;
    
//     double old_pt = muPtScale->getPtScale()->getLowEdge(pt_packed) + 1.e-6;
//     if (fabs(pt - old_pt)>0.005) { debug = 1;std::cout<<"lut pt diff: old "<<old_pt<<" lut "<<pt<<"  eta "<<eta<<" phi "<<phi<<"   mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;}
//     double lcl_phi = normalizedPhi( fmod( muScales->getPhiScale()->getLowEdge(t->localPhi()) + 
// 					  (t->sector()-1)*M_PI/3. + //sector 1 starts at 15 degrees 
// 					  M_PI/12. , 2.*M_PI) );
//     if (fabs(deltaPhi(phi,lcl_phi))>0.03) std::cout<<"lcl phi diff: lcl "<<lcl_phi<<" sc "<<phi<<"  mc: pt "<<stpt<<"  eta "<<steta<<" phi "<<stphi<<std::endl;
//   }
}

bool 
TFTrack::hasStubEndcap(int st)
{
  if(st==1 and l1track_->me1ID() > 0) return true;
  if(st==2 and l1track_->me2ID() > 0) return true;
  if(st==3 and l1track_->me3ID() > 0) return true;
  if(st==4 and l1track_->me4ID() > 0) return true;
  return false;
}

bool
TFTrack::hasStubBarrel()
{
  return l1track_->mb1ID() > 0;
}

bool 
TFTrack::hasStubStation(int st)
{
  if(st==0 and hasStubBarrel())       return true;
  if(st==1 and l1track_->me1ID() > 0) return true;
  if(st==2 and l1track_->me2ID() > 0) return true;
  if(st==3 and l1track_->me3ID() > 0) return true;
  if(st==4 and l1track_->me4ID() > 0) return true;
  return false;
}


bool 
TFTrack::hasStubCSCOk(int st)
{
//   if (!hasStubStation(st)) return false;
//   bool cscok = 0;
//   for (size_t s=0; s<ids_.size(); s++) {
//     if (ids_[s].station() == st and mplcts_[s]->deltaOk) { 
//       cscok = 1; 
//       break; 
//     }
//   }
//   return cscok;
  return true;
}


unsigned int 
TFTrack::nStubs(bool mb1, bool me1, bool me2, bool me3, bool me4)
{
  return ( (mb1 and hasStubStation(0)) + (me1 and hasStubStation(1)) + 
	   (me2 and hasStubStation(2)) + (me3 and hasStubStation(3)) + 
	   (me4 and hasStubStation(4)) );
}


unsigned int 
TFTrack::nStubsCSCOk(bool me1, bool me2, bool me3, bool me4)
{
  return ( (me1 and hasStubCSCOk(1)) + (me2 and hasStubCSCOk(2)) + 
	   (me3 and hasStubCSCOk(3)) + (me4 and hasStubCSCOk(4)) );
}


bool 
TFTrack::passStubsMatch(double steta, int minLowHStubs, int minMidHStubs, int minHighHStubs)
{
//    const double steta(match->strk->momentum().eta());
  const int nstubs(nStubs(1,1,1,1,1));
  const int nstubsok(nStubsCSCOk(1,1,1,1));
  if (fabs(steta) <= 1.2)      return nstubsok >=1 and nstubs >= minLowHStubs;
  else if (fabs(steta) <= 2.1) return nstubsok >=2 and nstubs >= minMidHStubs;
  else                         return nstubsok >=2 and nstubs >= minHighHStubs;
}


void 
TFTrack::print()
{
  /*
    std::cout<<"#### TFTRACK PRINT: "<<msg<<" #####"<<std::endl;
    //std::cout<<"## L1MuRegionalCand print: ";
    //l1track_->print();
    //std::cout<<"\n## L1Track Print: ";
    //l1track_->Print();
    //std::cout<<"## TFTRACK:  
    std::cout<<"\tpt_packed: "<<pt_packed<<"  eta_packed: " << eta_packed<<"  phi_packed: " << phi_packed<<"  q_packed: "<< q_packed<<"  bx: "<<l1track_->bx()<<std::endl;
    std::cout<<"\tpt: "<<pt<<"  eta: "<<eta<<"  phi: "<<phi<<"  sector: "<<l1track_->sector()<<"  dr: "<<dr<<"   ok1: "<<deltaOk1<<"  ok2: "<<deltaOk2<<"  okME1: "<<deltaOkME1<<std::endl;
    std::cout<<"\tMB1 ME1 ME2 ME3 ME4 = "<<l1track_->mb1ID()<<" "<<l1track_->me1ID()<<" "<<l1track_->me2ID()<<" "<<l1track_->me3ID()<<" "<<l1track_->me4ID()
        <<" ("<<hasStub(0)<<" "<<hasStub(1)<<" "<<hasStub(2)<<" "<<hasStub(3)<<" "<<hasStub(4)<<")  "
        <<" ("<<hasStubCSCOk(1)<<" "<<hasStubCSCOk(2)<<" "<<hasStubCSCOk(3)<<" "<<hasStubCSCOk(4)<<")"<<std::endl;
    std::cout<<"\tptAddress: 0x"<<std::hex<<l1track_->ptLUTAddress()<<std::dec<<"  mode: "<<mode()<<"  sign: "<<sign()<<"  dphi12: "<<dPhi12()<<"  dphi23: "<<dPhi23()<<std::endl;
    std::cout<<"\thas "<<trgdigis.size()<<" stubs in ";
    for (size_t s=0; s<trgids.size(); s++) 
        std::cout<<trgids[s]<<" w:"<<trgdigis[s]->getKeyWG()<<" s:"<<trgdigis[s]->getStrip()/2 + 1<<" p:"<<trgdigis[s]->getPattern()<<" bx:"<<trgdigis[s]->getBX()<<"; ";
    std::cout<<std::endl;
    std::cout<<"\tstub_etaphis:";
    for (size_t s=0; s<trgids.size(); s++)
        std::cout<<"  "<<trgetaphis[s].first<<" "<<trgetaphis[s].second;
    std::cout<<std::endl;
    std::cout<<"\tstub_petaphis:";
    for (size_t s=0; s<trgstubs.size(); s++)
        std::cout<<"  "<<trgstubs[s].etaPacked()<<" "<<trgstubs[s].phiPacked();
    std::cout<<std::endl;
    std::cout<<"\thas "<<mplcts.size()<<" associated MPCs in ";
    for (size_t s=0; s<ids.size(); s++) 
        std::cout<<ids[s]<<" w:"<<mplcts[s]->trgdigi->getKeyWG()<<" s:"<<mplcts[s]->trgdigi->getStrip()/2 + 1<<" Ok="<<mplcts[s]->deltaOk<<"; ";
    std::cout<<std::endl;
    std::cout<<"\tMPCs meEtap and mePhip: ";
    for (size_t s=0; s<ids.size(); s++) std::cout<<mplcts[s]->meEtap<<", "<<mplcts[s]->mePhip<<";  ";
    std::cout<<std::endl;
    std::cout<<"#### TFTRACK END PRINT #####"<<std::endl;
  */
}
