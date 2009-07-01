// ROOT includes
#include <Math/VectorUtil.h>

#include "DataFormats/Math/interface/Vector3D.h"
#include "RecoHI/HiEgammaAlgos/interface/ShapeCalculator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"


#include "TLorentzVector.h"
using namespace edm;
using namespace reco;
using namespace std;
using namespace ROOT::Math::VectorUtil; 

#define PI 3.141592653589793238462643383279502884197169399375105820974945
#define LGNDR2(X) ((3*X*X-1)/2)
#define LGNDR3(X) ((5*X*X*X-3*X)/2)
#define LGNDR4(X) ((35*X*X*X*X-30*X*X+3)/8)
#define LGNDR5(X) ((63*X*X*X*X*X - 70*X*X*X + 15*X)/8)
#define LGNDR6(X) ((231*X*X*X*X*X*X - 315*X*X*X*X + 105*X*X - 5)/16)

SuperFoxWolfram::SuperFoxWolfram(){
        for (int i=0; i<8; i++) sum[i] = 0;
 
}
SuperFoxWolfram::~SuperFoxWolfram(){}
void SuperFoxWolfram::fill( std::vector<TLorentzVector>& plist,std::vector<TLorentzVector>& qlist){
  if(plist.size()*qlist.size()>0){

  for ( std::vector<TLorentzVector>::iterator it1=plist.begin();
        it1!=plist.end(); it1++ ){
  for ( std::vector<TLorentzVector>::iterator it2=qlist.begin();
        it2!=qlist.end(); it2++ ){
    
        TLorentzVector&  p = *it1;
        TLorentzVector&  q = *it2;
        TLorentzVector pvec(p.Px(),p.Py(),p.Pz(),0);
        TLorentzVector qvec(q.Px(),q.Py(),q.Pz(),0);
    
        double mag = pvec.Mag() * qvec.Mag();
        double costh = pvec.Dot(qvec) / mag;
        double cost2 = costh * costh;
        sum[0] += mag;
        sum[1] += mag * costh;
        sum[2] += mag * LGNDR2(costh);
        sum[3] += mag * LGNDR3(costh);
        sum[4] += mag * LGNDR4(costh);
        sum[5] += mag * LGNDR5(costh);
        sum[6] += mag * LGNDR6(costh);
//        cout <<"sfw: "<<mag<<" "<<costh<<endl;
        }
        }
   }
   else for(int i=0;i<8;i++) sum[i]=0.;
}


double SuperFoxWolfram::R(int i){
    if( i < 0 || i > 6 || sum[0] == 0. ) return -7.;
    else {
        double sfwMoment = sum[i]/sum[0];
        if(abs(sfwMoment)>100.) sfwMoment=-7.;
        return sfwMoment;
    }
}


ShapeCalculator::ShapeCalculator (const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
   Handle<BasicClusterCollection> pEBclusters;
   iEvent.getByLabel(InputTag("islandBasicClusters:islandBarrelBasicClusters"), pEBclusters);
   fEBclusters_ = pEBclusters.product(); 

   Handle<BasicClusterCollection> pEEclusters;
   iEvent.getByLabel(InputTag("islandBasicClusters:islandEndcapBasicClusters"), pEEclusters);
   fEEclusters_ = pEEclusters.product(); 

   Handle<EcalRecHitCollection> pEBHit;
   iEvent.getByLabel(InputTag("ecalRecHit:EcalRecHitsEB"), pEBHit);
   fEBHit_ = pEBHit.product(); 

   Handle<EcalRecHitCollection> pEEHit;
   iEvent.getByLabel(InputTag("ecalRecHit:EcalRecHitsEE"), pEEHit);
   fEEHit_ = pEEHit.product(); 


   ESHandle<CaloGeometry> geometryHandle;
   iSetup.get<CaloGeometryRecord>().get(geometryHandle);
   geometry_ = geometryHandle.product();


} 

TLorentzVector ShapeCalculator::thrust()
{ 
  return thrust(recHitPosCollection2_); 
}

int ShapeCalculator::calculate(const reco::SuperCluster* cluster)
{
   using namespace edm; 
   using namespace reco;

   recHitPosCollection_.clear();
   recHitPosCollection2_.clear();

   if(!fEBclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("CxCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   math::XYZVector SCPoint(cluster->x(),cluster->y(),cluster->z());
   double SCEnergy=cluster->energy();
   EcalRecHit testEcalRecHit;

   double maxHit=0;
   DetId maxId=DetId(0);

   if (abs(cluster->eta())<1.479) {

         std::vector<DetId> clusterDetIds = cluster->getHitsByDetId();
         std::vector<DetId>::iterator posCurrent;
         for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
         {
           if ((*posCurrent != DetId(0)) && (fEBHit_->find(*posCurrent) != fEBHit_->end()))
           {
             EcalRecHitCollection::const_iterator itt = fEBHit_->find(*posCurrent);
             testEcalRecHit = *itt; 
             double energy = testEcalRecHit.energy();
             if (energy>maxHit) {
                maxHit=energy;
                maxId=*posCurrent;
             }
             if (energy<0.) continue;
             GlobalPoint positionGP = geometry_->getGeometry(testEcalRecHit.id())->getPosition();
             math::XYZVector position(positionGP.x(),positionGP.y(),positionGP.z());
             math::XYZVector diff = position - SCPoint;
//             diff /= diff.R();
             diff *= energy;
             TLorentzVector recHitPosition(diff.x(),diff.y(),diff.z(),energy);
             recHitPosCollection_.push_back(recHitPosition);
           }
         }
   } else {
 
         std::vector<DetId> clusterDetIds = cluster->getHitsByDetId();
         std::vector<DetId>::iterator posCurrent;
         for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
         {
           if ((*posCurrent != DetId(0)) && (fEEHit_->find(*posCurrent) != fEEHit_->end()))
           {
             EcalRecHitCollection::const_iterator itt = fEEHit_->find(*posCurrent);
             testEcalRecHit = *itt; 
             double energy = testEcalRecHit.energy();
             if (energy>maxHit) {
                maxHit=energy;
                maxId=*posCurrent;
             }
             if (energy<0.) continue;
             GlobalPoint positionGP = geometry_->getGeometry(testEcalRecHit.id())->getPosition();
             math::XYZVector position(positionGP.x(),positionGP.y(),positionGP.z());
             math::XYZVector diff = position - SCPoint;
//             diff /= diff.R();
//             diff *= energy;
             TLorentzVector recHitPosition(diff.x(),diff.y(),diff.z(),energy);
             recHitPosCollection_.push_back(recHitPosition);
           }
         }
   }

   

   if (abs(cluster->eta())<1.479) {

         std::vector<DetId> clusterDetIds = cluster->getHitsByDetId();
         std::vector<DetId>::iterator posCurrent;
         EcalRecHitCollection::const_iterator maxit = fEBHit_->find(maxId);
         testEcalRecHit = *maxit;
         GlobalPoint positionMax = geometry_->getGeometry(testEcalRecHit.id())->getPosition();
         math::XYZVector MaxPoint(positionMax.eta(),positionMax.phi(),0);
         
         for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
//         for(posCurrent = fEBHit_.begin(); posCurrent != fEBHit_.end(); posCurrent++)
         {
//           if ((*posCurrent != DetId(0)) && (fEBHit_->find(*posCurrent) != fEBHit_->end()) && *posCurrent !=maxId)
           if ((*posCurrent != DetId(0)) && (fEBHit_->find(*posCurrent) != fEBHit_->end()) )
           {
             EcalRecHitCollection::const_iterator itt = fEBHit_->find(*posCurrent);
             testEcalRecHit = *itt; 
             double energy = testEcalRecHit.energy();
             if (energy>maxHit) {
                maxHit=energy;
                maxId=*posCurrent;
             }
             if (energy<0.3) continue;
             GlobalPoint positionGP = geometry_->getGeometry(testEcalRecHit.id())->getPosition();
             math::XYZVector position(positionGP.eta(),positionGP.phi(),0);
             math::XYZVector diff = position - MaxPoint;
//             diff /= diff.R();
//             diff *= energy;
             if (diff.X()>0.22||diff.Y()>0.22) continue;
             TLorentzVector recHitPosition(diff.x(),diff.y(),diff.z(),energy);
//             cout <<diff.x()<<" "<<diff.y()<<" "<<diff.z()<<" "<<endl;
             recHitPosCollection2_.push_back(recHitPosition);
           }
         }
   } else {
 
         std::vector<DetId> clusterDetIds = cluster->getHitsByDetId();
         std::vector<DetId>::iterator posCurrent;
         EcalRecHitCollection::const_iterator maxit = fEEHit_->find(maxId);
         testEcalRecHit = *maxit;
         GlobalPoint positionMax = geometry_->getGeometry(testEcalRecHit.id())->getPosition();
         math::XYZVector MaxPoint(positionMax.eta(),positionMax.phi(),0);

         for(posCurrent = clusterDetIds.begin(); posCurrent != clusterDetIds.end(); posCurrent++)
         {
//           if ((*posCurrent != DetId(0)) && (fEEHit_->find(*posCurrent) != fEEHit_->end()) && *posCurrent !=maxId)
           if ((*posCurrent != DetId(0)) && (fEEHit_->find(*posCurrent) != fEEHit_->end()) )
           {
             EcalRecHitCollection::const_iterator itt = fEEHit_->find(*posCurrent);
             testEcalRecHit = *itt; 
             double energy = testEcalRecHit.energy();
             if (energy>maxHit) {
                maxHit=energy;
                maxId=*posCurrent;
             }
             if (energy<0.3) continue;
             GlobalPoint positionGP = geometry_->getGeometry(testEcalRecHit.id())->getPosition();
             math::XYZVector position(positionGP.eta(),positionGP.phi(),0);
             math::XYZVector diff = position - MaxPoint;
//             diff /= diff.R();
//             diff *= energy;
             if (diff.X()>0.22||diff.Y()>0.22) continue;
             TLorentzVector recHitPosition(diff.x(),diff.y(),diff.z(),energy);
             recHitPosCollection2_.push_back(recHitPosition);
           }
         }
   }
   
//   cout <<"Shape: I am good"<<endl;
  cout <<"Ntrks: "<<recHitPosCollection_.size()<<" "<<endl;
   return 1;  
}

TLorentzVector ShapeCalculator::thrust(std::vector<TLorentzVector>& ptl)
{
//   cout <<"Shape: I am thrusting"<<endl;
  int ntrk = ptl.size();
//  cout <<ntrk<<endl;
  double p_trk[10000][3],thr, tvect[3];
  int i_trk[50];

  for(int i=0;i<ntrk;i++){  
//    cout <<"dealing "<<i<<endl;
    p_trk[i][0]=ptl[i].Px();
    p_trk[i][1]=ptl[i].Py();
    p_trk[i][2]=ptl[i].Pz();
  }

//   cout <<"Shape: I am thrusting2"<<endl;
  thrust(ntrk, p_trk, &thr, tvect, i_trk);
  TLorentzVector thr_axis(tvect[0],tvect[1],tvect[2],0);
  cout <<"Thrust: "<<tvect[0]<<" "<<tvect[1]<<" "<<tvect[2]<<endl;

  if(abs(thr_axis.Mag())>1.5||thr_axis.Z()!=0){
//    cout << "size "<<thr_axis.Mag()<<endl;
    TLorentzVector temp(0.,0.,0.,0);
    return temp;
  }else{ 
    return thr_axis;
  }
   
}


double ShapeCalculator::Sper(TLorentzVector& thr)
{
//   cout <<"Shape: I am Sphering"<<endl;
 return Sper(recHitPosCollection2_,thr);
}
double ShapeCalculator::Sper(std::vector<TLorentzVector>& ptl, TLorentzVector& Bthr)
{
//   cout <<"Shape: I am Sphering calculating"<<endl;

 double B_tvec[3]={Bthr.Px(),Bthr.Py(),Bthr.Pz()};
 int ntrk = ptl.size();
 if(ntrk<1) return -7.;
 double p_trk[10000][3];
 double sper=-2.;
 for(int i=0;i<ntrk;i++){
      p_trk[i][0] = ptl[i].Px();
      p_trk[i][1] = ptl[i].Py();
      p_trk[i][2] = ptl[i].Pz();
 }
 spherp(ntrk, p_trk, &sper, B_tvec);
 cout <<"Sper: "<< sper<<endl;
 if(sper >=0. && sper<=1.)
   return sper;
 else
   return -7.;
}

double ShapeCalculator::Moment()
{
 return Moment(recHitPosCollection2_);
}

double ShapeCalculator::Moment(std::vector<TLorentzVector>& ptl)
{
//   cout <<"Shape: I am Sphering calculating"<<endl;

 TLorentzVector thr =  thrust(recHitPosCollection2_);

 int ntrk = ptl.size();
 if(ntrk<1) return -7.;
 double p_trk[10000][3];
 double total=0,totalE=0;
 for(int i=0;i<ntrk;i++){
    total+=(ptl[i].X()*thr.X() +ptl[i].Y()*thr.Y())*(ptl[i].X()*thr.X() +ptl[i].Y()*thr.Y())*ptl[i].E();
    totalE+=ptl[i].E();
 }
 return total/totalE;
}


int ShapeCalculator::thrust(int ntrk, double ptrk[][3], double* thr, double tvec[3], int itrk[]){
//    cout <<"Shape: I am thrusting3"<<endl;

    /* Local variables */
    int nmax=0;
    double pcos=0, pdot=0;
    int npls=0;
    double psqi, psqj, pprp[3], ptst;
    double psqij, p1, p2, p3, pt[10000][3], pjet[3], denmax;
    double rnumer, psqmax, pi1, pi2, pi3, pj1, pj2, pj3, alp, apt, psq;
    double pij1, pij2, pij3;
    int iup1, iup2, iup3;

    /* Function Body */
    *thr = -1.;
    tvec[0] = 0.;
    tvec[1] = 0.;
    tvec[2] = 0.;

    for (int i = 1; i < ntrk; ++i) {
        itrk[i] = 0;
    }

    if (ntrk <= 10000) {
        nmax = ntrk;
        npls = nmax;
        pt[npls][0] = 0.;
        pt[npls][1] = 0.;
        pt[npls][2] = 0.;

        for (int i = 0; i < nmax; ++i) {
            pt[i][0] = ptrk[i][0];
            pt[i][1] = ptrk[i][1];
            pt[i][2] = ptrk[i][2];
            pt[npls][0] += pt[i][0];
            pt[npls][1] += pt[i][1];
            pt[npls][2] += pt[i][2];
        }

        ptst = sqrt( pt[npls][0] * pt[npls][0] + pt[npls][1] * pt[npls][1] + pt[npls][2] * pt[npls][2]); 


        if (ptst >= 1e-7) {
            pt[nmax][0] = -pt[nmax][0] / (double)2.;
            pt[nmax][1] = -pt[nmax][1] / (double)2.;
            pt[nmax][2] = -pt[nmax][2] / (double)2.;
        }
    }

    if (nmax > 2 && nmax <= 50001) {
        psqmax = 0.;

/* ---THE FOLLOWING TWO DO-LOOPS RUN THROUGH ALL COMBINATIONS */
/*   OF PAIRS OF PARTICLES */

        iup1 = 0;
        for (int i = 0; i <= nmax; ++i) {
            if (i == nmax) {
                iup1 = 1;   
            }
             
            iup2 = 0;
            
            for (int j = 0; j <= nmax; ++j) {
                if (i != j) {
                    if (j == nmax) {
                        iup2 = 1;   
                    }
                    p1 = p2 = p3 = 0.; //---GET NORMAL TO PLANE

                    pprp[0] = pt[i][1] * pt[j][2] - pt[i][2] * pt[j][1];
                    pprp[1] = pt[i][2] * pt[j][0] - pt[i][0] * pt[j][2];
                    pprp[2] = pt[i][0] * pt[j][1] - pt[i][1] * pt[j][0];
                            
                    if (iup1 != 1 && iup2 != 1){
                        pdot = pt[nmax][0] * pprp[0] + pt[nmax][1] *
                                 pprp[1] + pt[nmax][2] * pprp[2];   
                    }

                    if (pdot >= 0. || iup1 == 1 || iup2 == 1) {
                        /* ---SELECT THE RIGHT SIDE OF THE PLANE */
                        iup3 = 0;

                        for (int k = 0; k <= nmax; ++k) {
                            if (k != i && k != j) {
                                pdot = pt[k][0] * pprp[0] + pt[k][1]
                                         * pprp[1] + pt[k][2] * pprp[2];
                                         
                                if (pdot >= 0.) {
                                    if (k == nmax) {
                                        iup3 = 1;   
                                    }
                                    p1 += pt[k][0];
                                    p2 += pt[k][1];
                                    p3 += pt[k][2];
                                }
                            }
                        }

                        pi1 = p1 + pt[i][0];
                        pi2 = p2 + pt[i][1];
                        pi3 = p3 + pt[i][2];
                        pj1 = p1 + pt[j][0];
                        pj2 = p2 + pt[j][1];
                        pj3 = p3 + pt[j][2];  
                        pij1 = pi1 + pt[j][0];
                        pij2 = pi2 + pt[j][1];
                        pij3 = pi3 + pt[j][2];

                        psq = p1 * p1 + p2 * p2 + p3 * p3;
                        psqi = pi1 * pi1 + pi2 * pi2 + pi3 * pi3;
                        psqj = pj1 * pj1 + pj2 * pj2 + pj3 * pj3;
                        psqij = pij1 * pij1 + pij2 * pij2 + pij3 * pij3;

                        /* ---GET MAXIMUM MOMENTUM, ASSIGN PLANAR PARTICLES */
                        if (psq > psqmax) {
                            if (iup3 == 1) {
                                psqmax = psq;
                                pjet[0] = p1;
                                pjet[1] = p2;
                                pjet[2] = p3;
                            }
                        }
                        if (psqi > psqmax) {
                            if (iup3 == 1 || iup1 == 1) {
                                psqmax = psqi;
                                pjet[0] = pi1;
                                pjet[1] = pi2;
                                pjet[2] = pi3;
                            }
                        }
                        if (psqj > psqmax) {
                            if (iup3 == 1 || iup2 == 1) {
                                psqmax = psqj;
                                pjet[0] = pi1;
                                pjet[1] = pi2;
                                pjet[2] = pi3;
                            }
                        }
                        if (psqj > psqmax) {  
                            if (iup3 == 1 || iup2 == 1) {
                                psqmax = psqj;
                                pjet[0] = pj1;
                                pjet[1] = pj2;
                                pjet[2] = pj3;
                            }
                        }
                        if (psqij > psqmax) {
                            psqmax = psqij;
                            pjet[0] = pij1;
                            pjet[1] = pij2; 
                            pjet[2] = pij3;  
                        }
                    }
                }
            }
        }

/* ---EFFECT FINAL THRUST RESULTS */
        apt = sqrt(psqmax);
        tvec[0] = pjet[0] / apt;
        tvec[1] = pjet[1] / apt;
        tvec[2] = pjet[2] / apt;
        denmax = 0.;
        rnumer = 0.;
        for (int i= 0; i< ntrk; ++i) {
            alp = sqrt(pt[i][0]*pt[i][0]+pt[i][1]*pt[i][1]+pt[i][2]*pt[i][2]);
            
           pcos = (pjet[0] * pt[i][0] + pjet[1] * pt[i][1] + pjet[2] * pt[i][2]) / apt; 

            if (pcos > 0.) {
                itrk[i] = 1; 
            }
            rnumer += fabs(pcos);
            denmax += alp;
        }
        *thr = rnumer / denmax;

    } else if (nmax == 2) {
        *thr = (double)1.;

        p1 = sqrt( pt[0][0]* pt[0][0] + pt[0][1] * pt[0][1] + pt[0][2] * pt[0][2] * pt[0][2]);
        p2 = sqrt( pt[1][0]* pt[1][0] + pt[1][1] * pt[1][1] + pt[1][2] * pt[1][2] * pt[1][2]);

        if (p1 > p2) {
            tvec[0] = pt[0][0] / p1;
            tvec[1] = pt[0][1] / p1;
            tvec[2] = pt[0][2] / p1;
            itrk[0] = 1;
        } else {
            tvec[0] = pt[1][0] / p2;
            tvec[1] = pt[1][1] / p2;
            tvec[2] = pt[1][2] / p2;  
            itrk[0] = 1;
        }   
        cout <<tvec[0]<<" "<<tvec[1]<<" "<<tvec[2]<<" "<<endl;
    }
    return 0;
} /* thrust_ */

/* Subroutine */
int ShapeCalculator::spherp(int ntrk, double ptrk[][3], double *sper, double jet[3])
{

    /* Local variables */
    double pmod, jetp;
    double spmod, ptmin, pl, pt1, cut, psq, xpt1;


    /* Function Body */
    if (ntrk <= 1) {
        *sper = -1;
        return -1;
    }

    ptmin = 1e5;
    spmod = 0.;
    pt1 = 0.;
    cut = sqrt(2.) / 2.;
    jetp = sqrt(jet[0] * jet[0] + jet[1] * jet[1] + jet[2] * jet[2]);

    for (int i = 0; i < ntrk; ++i) {
        psq = ptrk[i][0] * ptrk[i][0] + ptrk[i][1] * ptrk[i][1] + ptrk[i][2] * ptrk[i][2];
        pmod = sqrt(psq);
        pl = (jet[0] * ptrk[i][0] + jet[1] * ptrk[i][1] + jet[2]
                 * ptrk[i][2]) / (pmod * jetp);

        spmod += pmod;
/* ........................................................... */
/* ....... SELECT PARTOCLE OUTSIDE THE CONE .................. */
/* ........................................................... */
        if (fabs(pl) < cut) {
            xpt1 = psq * (1 - pl * pl);
            pt1 += sqrt((fabs(xpt1)));

        }
    }

//    spmod -= jetp;
    *sper = pt1 / spmod;

} /* spherp_ */



double ShapeCalculator::getCx(const reco::SuperCluster* cluster, double x, double threshold)
{
   using namespace edm;
   using namespace reco;

   if(!fEBclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   math::XYZVector SClusPoint(cluster->position().x(),
                          cluster->position().y(),
                          cluster->position().z());

   double TotalEt = 0;

   TotalEt = - cluster->rawEnergy()/cosh(cluster->eta());

   // Loop over barrel basic clusters 
   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dR = ROOT::Math::VectorUtil::DeltaR(ClusPoint,SClusPoint);
      
      if (dR<x*0.1) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      const GlobalPoint clusPoint(clu->x(),clu->y(),clu->z());
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dR = ROOT::Math::VectorUtil::DeltaR(ClusPoint,SClusPoint);

      if (dR<x*0.1) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   return TotalEt;
}

double ShapeCalculator::getCxRemoveSC(const reco::SuperCluster* cluster, double x, double threshold)
{
   // Calculate Cx and remove the basicClusters used by superCluster

   using namespace edm;
   using namespace reco;

   if(!fEBclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   math::XYZVector SClusPoint(cluster->position().x(),
                          cluster->position().y(),
                          cluster->position().z());

   double TotalEt = 0;

   TotalEt = 0;

   // Loop over barrel basic clusters 
   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dR = ROOT::Math::VectorUtil::DeltaR(ClusPoint,SClusPoint);
      
      // check if this basic cluster is used in the target supercluster
      bool inSuperCluster = checkUsed(cluster,clu);

      if (dR<x*0.1&&inSuperCluster==false) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
     
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      const GlobalPoint clusPoint(clu->x(),clu->y(),clu->z());
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dR = ROOT::Math::VectorUtil::DeltaR(ClusPoint,SClusPoint);

      // check if this basic cluster is used in the target supercluster
      bool inSuperCluster = checkUsed(cluster,clu);

      if (dR<x*0.1&&inSuperCluster==false) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   return TotalEt;
}

double ShapeCalculator::getCCx(const reco::SuperCluster* cluster, double x, double threshold)
{
   using namespace edm;
   using namespace reco;


   if(!fEBclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;

   TotalEt = - cluster->rawEnergy()/cosh(cluster->eta());

   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dEta = fabs(eta-SClusterEta);
 
     if (dEta<x*0.1) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();
      double phi = ClusPoint.phi();

      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;

      if (dEta<x*0.1) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   double Cx = getCx(cluster,x,threshold);
   double CCx = Cx - TotalEt / 40.0 * x;

   return CCx;
}


double ShapeCalculator::getCCxRemoveSC(const reco::SuperCluster* cluster, double x, double threshold)
{
   using namespace edm;
   using namespace reco;


   if(!fEBclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEt = 0;

   TotalEt = 0;

   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();

      double dEta = fabs(eta-SClusterEta);

      // check if this basic cluster is used in the target supercluster
      bool inSuperCluster = checkUsed(cluster,clu);
 
     if (dEta<x*0.1&&inSuperCluster==false) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      math::XYZVector ClusPoint(clu->x(),clu->y(),clu->z());
      double eta = ClusPoint.eta();
      double phi = ClusPoint.phi();

      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;

      // check if this basic cluster is used in the target supercluster
      bool inSuperCluster = checkUsed(cluster,clu);

      if (dEta<x*0.1&&inSuperCluster==false) {
         double et = clu->energy()/cosh(eta);
         if (et<threshold) et=0;
         TotalEt += et;
      } 
   }

   double Cx = getCxRemoveSC(cluster,x,threshold);
   double CCx = Cx - TotalEt / 40.0 * x;

   return CCx;
}


bool ShapeCalculator::checkUsed(const reco::SuperCluster* sc, const reco::BasicCluster* bc)
{
   reco::basicCluster_iterator theEclust = sc->clustersBegin();

   // Loop over the basicClusters inside the target superCluster
   for(;theEclust != sc->clustersEnd(); theEclust++) {
     if ((**theEclust) == (*bc) ) return  true; //matched, so it's used.
   }
   return false;
}

double ShapeCalculator::getBCMax(const reco::SuperCluster* cluster,int i)
{
   reco::basicCluster_iterator theEclust = cluster->clustersBegin();

   double energyMax=0,energySecond=0;
   // Loop over the basicClusters inside the target superCluster
   for(;theEclust != cluster->clustersEnd(); theEclust++) {
     if ((*theEclust)->energy()>energyMax ) {
        energySecond=energyMax;
        energyMax=(*theEclust)->energy();
     } else if ((*theEclust)->energy()>energySecond) {
        energySecond=(*theEclust)->energy();
     }
   }
   if (i==1) return energyMax;
   return energySecond;
}


double ShapeCalculator::getCorrection(const reco::SuperCluster* cluster, double x, double y,double threshold)
{
   using namespace edm;
   using namespace reco;

   // doesn't really work now ^^; (Yen-Jie)
   if(!fEBclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EBclusters for event.";
      return -100;
   }

   if(!fEEclusters_) {       
      LogError("ShapeCalculator") << "Error! Can't get EEclusters for event.";
      return -100;
   }

   double SClusterEta = cluster->eta();
   double SClusterPhi = cluster->phi();
   double TotalEnergy = 0;
   double TotalBC = 0;

   TotalEnergy = 0;

   double Area = PI * (-x*x+y*y) / 100.0;
   double nCrystal = Area / 0.0174 / 0.0174; // ignore the difference between endcap and barrel for the moment....

   for(BasicClusterCollection::const_iterator iclu = fEBclusters_->begin();
       iclu != fEBclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      const GlobalPoint clusPoint(clu->x(),clu->y(),clu->z());
      double eta = clusPoint.eta();
      double phi = clusPoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;
      double dR = sqrt(dEta*dEta+dPhi*dPhi);
 
     if (dR>x*0.1&&dR<y*0.1) {
         double e = clu->energy();
         if (e<threshold) e=0;
         TotalEnergy += e;
         if (e!=0) TotalBC+=clu->getHitsByDetId().size();  // number of crystals
   
      } 
   }

   for(BasicClusterCollection::const_iterator iclu = fEEclusters_->begin();
       iclu != fEEclusters_->end(); ++iclu) {
      const BasicCluster *clu = &(*iclu);
      const GlobalPoint clusPoint(clu->x(),clu->y(),clu->z());
      double eta = clusPoint.eta();
      double phi = clusPoint.phi();
      double dEta = fabs(eta-SClusterEta);
      double dPhi = fabs(phi-SClusterPhi);
      while (dPhi>2*PI) dPhi-=2*PI;
      if (dPhi>PI) dPhi=2*PI-dPhi;
      double dR = sqrt(dEta*dEta+dPhi*dPhi);
 
     if (dR>x*0.1&&dR<y*0.1) {
         double e = clu->energy();
         if (e<threshold) e=0;
         TotalEnergy += e;
         if (e!=0) TotalBC += clu->getHitsByDetId().size(); // number of crystals
      } 
   }


  if (TotalBC==0) return 0;
  return TotalEnergy/nCrystal;
}

