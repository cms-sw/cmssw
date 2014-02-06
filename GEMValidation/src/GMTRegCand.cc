#include "GEMCode/GEMValidation/src/GMTRegCand.h"

GMTRegCand::GMTRegCand()
{}

GMTRegCand::GMTRegCand(const GMTRegCand& rhs)
{}

GMTRegCand::~GMTRegCand()
{}


void 
GMTRegCand::print()
{
  //   std::string sys="Mu";
//   if (l1reg->type_idx()==2) sys = "CSC";
//   if (l1reg->type_idx()==3) sys = "RPCf";
//   std::cout<<"#### GMTREGCAND ("<<sys<<") PRINT: "<<msg<<" #####"<<std::endl;
//   //l1reg->print();
//   std::cout<<" bx="<<l1reg->bx()<<" values: pt="<<pt<<" eta="<<eta<<" phi="<<phi<<" packed: pt="<<l1reg->pt_packed()<<" eta="<<eta_packed<<" phi="<<phi_packed<<"  q="<<l1reg->quality()<<"  ch="<<l1reg->chargeValue()<<" chOk="<<l1reg->chargeValid()<<std::endl;
//   if (tfcand!=NULL) std::cout<<"has tfcand with "<<ids.size()<<" stubs"<<std::endl;
//   std::cout<<"#### GMTREGCAND END PRINT #####"<<std::endl;
}


// void GMTRegCand::init(const L1MuRegionalCand *t,
// 		      edm::ESHandle< L1MuTriggerScales > &muScales,
// 		      edm::ESHandle< L1MuTriggerPtScale > &muPtScale)
// {
//     l1reg = t;

//     pt = muPtScale->getPtScale()->getLowEdge(t->pt_packed()) + 1.e-6;
//     eta = muScales->getRegionalEtaScale(t->type_idx())->getCenter(t->eta_packed());
//     //std::cout<<"regetac"<<t->type_idx()<<"="<<eta<<std::endl;
//     //std::cout<<"regetalo"<<t->type_idx()<<"="<<muScales->getRegionalEtaScale(t->type_idx())->getLowEdge(t->eta_packed() )<<std::endl;
//     phi = normalizedPhi( muScales->getPhiScale()->getLowEdge(t->phi_packed()));
//     nTFStubs = -1;

//     bool sc_debug = 0;
//     if (sc_debug){
//         double my_phi = normalizedPhi( t->phi_packed()*0.043633231299858237 + 0.0218 ); // M_PI*2.5/180 = 0.0436332312998582370
//         double sign_eta = ( (t->eta_packed() & 0x20) == 0) ? 1.:-1;
//         double my_eta = sign_eta*(0.05 * (t->eta_packed() & 0x1F) + 0.925); //  0.9+0.025 = 0.925
//         double my_pt = ptscale[t->pt_packed()];
//         if (fabs(pt - my_pt)>0.005) std::cout<<"gmtreg scales pt diff: my "<<my_pt<<" sc "<<pt<<std::endl;
//         if (fabs(eta - my_eta)>0.005) std::cout<<"gmtreg scales eta diff: my "<<my_eta<<" sc "<<eta<<std::endl;
//         if (fabs(deltaPhi(phi,my_phi))>0.03) std::cout<<"gmtreg scales phi diff: my "<<my_phi<<" sc "<<phi<<std::endl;
//     }  
// }
