// Original Author:  Rishi Patel
//         Created:  Wed, 01 Aug 2018 14:01:41 GMT
//
// Track jets are clustered in a two-layer process, first by clustering in phi,
// then by clustering in eta
// Introduction to object (p10-13):
// https://indico.cern.ch/event/791517/contributions/3341650/attachments/1818736/2973771/TrackBasedAlgos_L1TMadrid_MacDonald.pdf

// system include files

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1TCorrelator/interface/TkJet.h"
#include "DataFormats/L1TCorrelator/interface/TkJetFwd.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetType.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"


#include <memory>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <cstdlib>
#include "TH1D.h"
#include "TH2D.h"
#include <TMath.h>
#include "L1TrackJetProducer.h"


using namespace std;
using namespace edm;
using namespace l1t;
class L1TrackJetProducer : public stream::EDProducer<> {
public:
  explicit L1TrackJetProducer(const ParameterSet&);
  ~L1TrackJetProducer();
  typedef TTTrack< Ref_Phase2TrackerDigi_ >  L1TTTrackType;
  typedef vector< L1TTTrackType > L1TTTrackCollectionType;

  static void fillDescriptions(ConfigurationDescriptions& descriptions);
  bool TrackQualityCuts(float trk_pt, int trk_nstub, float trk_chi2, float trk_bendchi2, float trk_d0);
  void L2_cluster(vector< Ptr<L1TTTrackType> > L1trk_ptrs, vector<int> ttrk, vector<int> tdtrk,vector<int> ttdtrk, maxzbin &mzb);
  virtual etaphibin *L1_cluster(etaphibin *phislice);

private:
  virtual void beginStream(StreamID) override;
  virtual void produce(Event&, const EventSetup&) override;
  virtual void endStream() override;
  const EDGetTokenT<vector<TTTrack< Ref_Phase2TrackerDigi_ > > > trackToken;
  vector< Ptr< L1TTTrackType > > L1trk_ptrs;
  vector<int> zbincount;
  vector<int> ttrk;
  vector<int> tdtrk;
  vector<int> ttdtrk;
  float trk_zMax;
  float trk_ptMax;
  float trk_ptMin;
  float trk_etaMax;
  float trk_chi2dofMax;
  float trk_bendChi2Max;
  int trk_nPSStubMin;
  int lowpTJetMinTrackMultiplicity;
  int highpTJetMinTrackMultiplicity;
  int zBins;
  int etaBins;
  int phiBins;
  double minTrkJetpT;
  float zStep;
  float etaStep;
  float phiStep;
  bool displaced;
  float d0_cutNStubs4;
  float d0_cutNStubs5;
  float nStubs4DisplacedChi2_Loose;
  float nStubs5DisplacedChi2_Loose;
  float nStubs4Displacedbend_Loose;
  float nStubs5Displacedbend_Loose;
  float nStubs4DisplacedChi2_Tight;
  float nStubs5DisplacedChi2_Tight;
  float nStubs4Displacedbend_Tight;
  float nStubs5Displacedbend_Tight;
};

L1TrackJetProducer::L1TrackJetProducer(const ParameterSet& iConfig):
trackToken(consumes< vector<TTTrack< Ref_Phase2TrackerDigi_> > > (iConfig.getParameter<InputTag>("L1TrackInputTag")))
{
  trk_zMax    = (float)iConfig.getParameter<double>("trk_zMax");
  trk_ptMax   =(float)iConfig.getParameter<double>("trk_ptMax");
  trk_ptMin=(float)iConfig.getParameter<double>("trk_ptMin");
  trk_etaMax=(float)iConfig.getParameter<double>("trk_etaMax");
  trk_chi2dofMax=(float)iConfig.getParameter<double>("trk_chi2dofMax");
  trk_bendChi2Max=(float)iConfig.getParameter<double>("trk_bendChi2Max");
  trk_nPSStubMin=(int)iConfig.getParameter<int>("trk_nPSStubMin");
  minTrkJetpT=iConfig.getParameter<double>("minTrkJetpT");
  etaBins=(int)iConfig.getParameter<int>("etaBins");
  phiBins=(int)iConfig.getParameter<int>("phiBins");
  zBins=(int)iConfig.getParameter<int>("zBins");
  d0_cutNStubs4=(float)iConfig.getParameter<double>("d0_cutNStubs4");
  d0_cutNStubs5=(float)iConfig.getParameter<double>("d0_cutNStubs5");
  lowpTJetMinTrackMultiplicity=(int)iConfig.getParameter<int>("lowpTJetMinTrackMultiplicity");
  highpTJetMinTrackMultiplicity=(int)iConfig.getParameter<int>("highpTJetMinTrackMultiplicity");
  displaced=iConfig.getParameter<bool>("displaced");
  nStubs4DisplacedChi2_Loose=(float)iConfig.getParameter<double>("nStubs4DisplacedChi2_Loose");
  nStubs5DisplacedChi2_Loose=(float)iConfig.getParameter<double>("nStubs5DisplacedChi2_Loose");
  nStubs4Displacedbend_Loose=(float)iConfig.getParameter<double>("nStubs4Displacedbend_Loose");
  nStubs5Displacedbend_Loose=(float)iConfig.getParameter<double>("nStubs5Displacedbend_Loose");
  nStubs4DisplacedChi2_Tight=(float)iConfig.getParameter<double>("nStubs4DisplacedChi2_Tight");
  nStubs5DisplacedChi2_Tight=(float)iConfig.getParameter<double>("nStubs5DisplacedChi2_Tight");
  nStubs4Displacedbend_Tight=(float)iConfig.getParameter<double>("nStubs4Displacedbend_Tight");
  nStubs5Displacedbend_Tight=(float)iConfig.getParameter<double>("nStubs5Displacedbend_Tight");

  zStep = 2.0 * trk_zMax / zBins;
  etaStep = 2.0 * trk_etaMax / etaBins; //etaStep is the width of an etabin
  phiStep = 2 * M_PI / phiBins; ////phiStep is the width of a phibin

  if (displaced) produces<TkJetCollection>("L1TrackJetsExtended");
  else produces<TkJetCollection>("L1TrackJets");
}

L1TrackJetProducer::~L1TrackJetProducer() { }

void L1TrackJetProducer::produce(Event& iEvent, const EventSetup& iSetup) {

  unique_ptr<TkJetCollection> L1L1TrackJetProducer(new TkJetCollection);

  // For TTStubs
  ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  ESHandle<TrackerGeometry> tGeomHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get(tGeomHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  edm::Handle< vector< TTTrack< Ref_Phase2TrackerDigi_ > > > TTTrackHandle;
  iEvent.getByToken(trackToken, TTTrackHandle);
  vector< TTTrack< Ref_Phase2TrackerDigi_ > >::const_iterator iterL1Track;

  L1trk_ptrs.clear();
  zbincount.clear();
  ttrk.clear();
  tdtrk.clear();
  ttdtrk.clear();

  unsigned int this_l1track = 0;
  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    edm::Ptr< L1TTTrackType > trkPtr(TTTrackHandle, this_l1track);
    this_l1track++;
    float trk_pt = trkPtr->momentum().perp();
    int trk_nstubs = (int)trkPtr->getStubRefs().size();
    float trk_chi2dof = trkPtr->chi2Red();
    float trk_d0 = trkPtr->d0();
    float trk_bendchi2 = trkPtr->stubPtConsistency();

    int trk_nPS = 0;
    for (int istub=0; istub<trk_nstubs; istub++) { // loop over the stubs
      DetId detId( trkPtr->getStubRefs().at(istub)->getDetId() );
      if (detId.det() == DetId::Detector::Tracker) {
        if ( (detId.subdetId() == StripSubdetector::TOB && tTopo->tobLayer(detId) <= 3) ||
        (detId.subdetId() == StripSubdetector::TID && tTopo->tidRing(detId) <= 9) ) trk_nPS++;
      }
    }

    if (trk_nPS < trk_nPSStubMin) continue;
    if (!TrackQualityCuts(trk_pt, trk_nstubs, trk_chi2dof, trk_bendchi2, fabs(trk_d0))) continue;
    if (fabs(iterL1Track->z0()) > trk_zMax) continue;
    if (fabs(iterL1Track->momentum().eta()) > trk_etaMax) continue;
    if (trk_pt < trk_ptMin) continue;
    L1trk_ptrs.push_back(trkPtr);
    zbincount.push_back(0);

    if ((fabs(trk_d0)>d0_cutNStubs5 && trk_nstubs>=5)||(trk_nstubs==4 && fabs(trk_d0)>d0_cutNStubs4)) tdtrk.push_back(1);
    else tdtrk.push_back(0); //displaced track
    if ((trk_nstubs>=5 && trk_chi2dof< nStubs5DisplacedChi2_Tight && trk_bendchi2<nStubs5Displacedbend_Tight ) || (trk_nstubs==4 && trk_chi2dof< nStubs4DisplacedChi2_Tight && trk_bendchi2<nStubs4Displacedbend_Tight )) ttrk.push_back(1);
    else ttrk.push_back(0);
    if ((trk_nstubs>=5 && trk_chi2dof< nStubs5DisplacedChi2_Tight && trk_bendchi2<nStubs5Displacedbend_Tight && fabs(trk_d0)>d0_cutNStubs5) || (trk_nstubs==4 && trk_chi2dof< nStubs4DisplacedChi2_Tight && trk_bendchi2<nStubs4Displacedbend_Tight  && fabs(trk_d0)>d0_cutNStubs4)) ttdtrk.push_back(1);
    else ttdtrk.push_back(0);
  }

  if (L1trk_ptrs.size()>0) {
    maxzbin mzb;

    L2_cluster(L1trk_ptrs, ttrk, tdtrk,ttdtrk,mzb);
    edm::Ref< JetBxCollection > jetRef; //null, no Calo Jet Ref
    vector< Ptr< L1TTTrackType > > L1TrackAssocJet;
    if (mzb.clusters!=NULL) {
      for (int j=0; j<mzb.nclust; ++j) {
        //FILL Two Layer Jets for Jet Collection
        if (mzb.clusters[j].pTtot<=trk_ptMin) continue; //protects against reading bad memory
        if (mzb.clusters[j].numtracks<1) continue;
        if (mzb.clusters[j].numtracks>5000) continue;
        float jetEta = mzb.clusters[j].eta;
        float jetPhi = mzb.clusters[j].phi;
        float jetPt = mzb.clusters[j].pTtot;
        float jetPx = jetPt*cos(jetPhi);
        float jetPy = jetPt*sin(jetPhi);
        float jetPz = jetPt*sinh(jetEta);
        float jetP = jetPt*cosh(jetEta);
        int totalTighttrk = mzb.clusters[j].numttrks;
        int totalDisptrk = mzb.clusters[j].numtdtrks;
        int totalTightDisptrk = mzb.clusters[j].numttdtrks;

        math::XYZTLorentzVector jetP4(jetPx,jetPy,jetPz,jetP);
        L1TrackAssocJet.clear();
        for (unsigned int t=0; t<L1trk_ptrs.size(); ++t) {
          if (L1TrackAssocJet.size()==(unsigned int)mzb.clusters[j].numtracks) break;
          float deta = L1trk_ptrs[t]->momentum().eta() - jetEta;
          float dphi = L1trk_ptrs[t]->momentum().phi() - jetPhi;
          float dZ = fabs(mzb.zbincenter - L1trk_ptrs[t]->z0());
          if (dZ<zStep && fabs(deta)<etaStep*2.0 && fabs(dphi)<phiStep*2.0) {
            L1TrackAssocJet.push_back(L1trk_ptrs[t]);
          }
        }
        TkJet trkJet(jetP4, L1TrackAssocJet, mzb.zbincenter, mzb.clusters[j].numtracks, totalTighttrk, totalDisptrk, totalTightDisptrk);
        //trkJet.setDispCounters(DispCounters);
        if (L1TrackAssocJet.size()>0) L1L1TrackJetProducer->push_back(trkJet);
      }
    }
    //free(mzb.clusters);
    if (displaced)  iEvent.put(std::move(L1L1TrackJetProducer), "L1TrackJetsExtended");
    else  iEvent.put(std::move(L1L1TrackJetProducer), "L1TrackJets");
    delete[] mzb.clusters;
  }
}

void L1TrackJetProducer::L2_cluster(vector< Ptr< L1TTTrackType > > L1trk_ptrs, vector<int>ttrk, vector<int>tdtrk, vector<int>ttdtrk, maxzbin &mzb) {
  const int nz = zBins;
  maxzbin all_zBins[nz];
  maxzbin mzbtemp;
  for (int z=0; z<nz; ++z) all_zBins[z] = mzbtemp;

  if (all_zBins==NULL) {
    edm::LogWarning("L1TrackJetProducer")<<" \"all_zBins\" memory not assigned!\n";
    return;
  }

  float zmin = -1.0*trk_zMax;
  float zmax = zmin + 2*zStep;

  etaphibin epbins[phiBins][etaBins]; // create grid of phiBins
  float phi = -1.0 * M_PI;
  float eta;
  float etamin, etamax, phimin, phimax;
  for (int i = 0; i<phiBins; ++i) {
    eta = -1.0 * trk_etaMax;
    for (int j = 0; j<etaBins; ++j) {
      phimin = phi;
      phimax = phi + phiStep;
      etamin = eta;
      eta = eta + etaStep;
      etamax = eta;
      epbins[i][j].phi = (phimin+phimax)/2.0;
      epbins[i][j].eta = (etamin+etamax)/2.0;
    } // for each etabin
    phi = phi+phiStep;
  } // for each phibin (finished creating epbins)
  mzb = all_zBins[0];
  int ntracks=L1trk_ptrs.size();
  // uninitalized arrays
  etaphibin *L1clusters[phiBins];
  etaphibin L2cluster[ntracks];

  for (int zbin=0; zbin<zBins-1; ++zbin) {
    for (int i=0; i<phiBins; ++i) { //First initialize pT, numtracks, used to 0 (or false)
      for (int j=0; j<etaBins; ++j) {
        epbins[i][j].pTtot = 0;
        epbins[i][j].used = false;
        epbins[i][j].numtracks = 0;
        epbins[i][j].numttrks = 0;
        epbins[i][j].numtdtrks = 0;
        epbins[i][j].numttdtrks = 0;
      } //for each etabin
      L1clusters[i] = epbins[i];
    } //for each phibin

    for (unsigned int k=0; k<L1trk_ptrs.size(); ++k) {
      float trkpt = L1trk_ptrs[k]->momentum().perp();
      float trketa = L1trk_ptrs[k]->momentum().eta();
      float trkphi = L1trk_ptrs[k]->momentum().phi();
      float trkZ = L1trk_ptrs[k]->z0();

      for (int i=0; i<phiBins; ++i) {
        for (int j=0; j<etaBins; ++j) {
          L2cluster[k] = epbins[i][j];
          if ((zmin<=trkZ && zmax>=trkZ) &&
          ((epbins[i][j].eta - etaStep/2.0 <= trketa && epbins[i][j].eta + etaStep/2.0 >= trketa)
          && epbins[i][j].phi - phiStep/2.0 <= trkphi && epbins[i][j].phi + phiStep/2.0 >= trkphi
          && (zbincount[k] != 2))) {
            zbincount.at(k) = zbincount.at(k)+1;
            if (trkpt<trk_ptMax) epbins[i][j].pTtot+=trkpt;
            else epbins[i][j].pTtot+=trk_ptMax;
            epbins[i][j].numttrks+=ttrk[k];
            epbins[i][j].numtdtrks+=tdtrk[k];
            epbins[i][j].numttdtrks+=ttdtrk[k];
            ++epbins[i][j].numtracks;
          } // if right bin
        } // for each phibin: j loop
      } // for each phibin: i loop
    } // end loop over tracks

    for (int phislice=0; phislice<phiBins; ++phislice) {
      L1clusters[phislice] = L1_cluster(epbins[phislice]);
      for (int ind=0; L1clusters[phislice][ind].pTtot!=0; ++ind) {
        L1clusters[phislice][ind].used = false;
      }
    }

    //Create clusters array to hold output cluster data for Layer2; can't have more clusters than tracks.
    //Find eta-phibin with maxpT, make center of cluster, add neighbors if not already used.
    float hipT = 0;
    int nclust = 0;
    int phibin = 0;
    int imax = -1;
    int index1; //index of clusters array for each phislice
    float E1 = 0;
    float E0 = 0;
    float E2 = 0;
    int trx1, trx2;
    int ttrk1, ttrk2;
    int tdtrk1, tdtrk2;
    int ttdtrk1, ttdtrk2;
    int used1, used2, used3, used4;

    for (phibin=0; phibin<phiBins; ++phibin) { //Find eta-phibin with highest pT
      while(true) {
        hipT = 0;
        for (index1=0; L1clusters[phibin][index1].pTtot>0; ++index1) {
          if (!L1clusters[phibin][index1].used && L1clusters[phibin][index1].pTtot >= hipT) {
            hipT = L1clusters[phibin][index1].pTtot;
            imax = index1;
          }
        } // for each index within the phibin

        if (hipT == 0)
          break; // If highest pT is 0, all bins are used
        E0 = hipT;   // E0 is pT of first phibin of the cluster
        E1 = 0;
        E2 = 0;
        trx1 = 0;
        trx2 = 0;
        ttrk1 = 0;
        ttrk2 = 0;
        tdtrk1 = 0;
        tdtrk2 = 0;
        ttdtrk1 = 0;
        ttdtrk2 = 0;
        L2cluster[nclust] = L1clusters[phibin][imax];
        L1clusters[phibin][imax].used = true;
        // Add pT of upper neighbor
        // E1 is pT of the middle phibin (should be highest pT)
        if (phibin != phiBins-1) {
          used1 = -1;
          used2 = -1;
          for (index1 = 0; L1clusters[phibin+1][index1].pTtot != 0; ++index1) {
            if (L1clusters[phibin+1][index1].used) continue;
            if (fabs(L1clusters[phibin+1][index1].eta - L1clusters[phibin][imax].eta) <= 1.5*etaStep) {
              E1 += L1clusters[phibin+1][index1].pTtot;
              trx1 += L1clusters[phibin+1][index1].numtracks;
              ttrk1 += L1clusters[phibin+1][index1].numttrks;
              tdtrk1 += L1clusters[phibin+1][index1].numtdtrks;
              ttdtrk1 += L1clusters[phibin+1][index1].numttdtrks;
              if (used1<0)
                used1 = index1;
              else
                used2 = index1;
            } // if cluster is within one phibin
          } // for each cluster in above phibin

          if (E1<E0) { // if E1 isn't higher, E0 and E1 are their own cluster
            L2cluster[nclust].pTtot += E1;
            L2cluster[nclust].numtracks += trx1;
            L2cluster[nclust].numttrks += ttrk1;
            L2cluster[nclust].numtdtrks += tdtrk1;
            L2cluster[nclust].numttdtrks += ttdtrk1;
            if (used1 >= 0)
              L1clusters[phibin+1][used1].used = true;
            if (used2 >= 0)
              L1clusters[phibin+1][used2].used = true;
            nclust++;
            continue;
          }

          if (phibin != phiBins-2) { // E2 will be the pT of the third phibin (should be lower than E1)
            used3 = -1;
            used4 = -1;
            for (index1 = 0; L1clusters[phibin+2][index1].pTtot != 0; ++index1) {
              if (L1clusters[phibin+2][index1].used)
                continue;
              if (fabs(L1clusters[phibin+2][index1].eta - L1clusters[phibin][imax].eta) <= 1.5*etaStep) {
                E2 += L1clusters[phibin+2][index1].pTtot;
                trx2 += L1clusters[phibin+2][index1].numtracks;
                ttrk2 += L1clusters[phibin+2][index1].numttrks;
                tdtrk2 += L1clusters[phibin+2][index1].numtdtrks;
                ttdtrk2 += L1clusters[phibin+2][index1].numttdtrks;
                if (used3<0)
                  used3 = index1;
                else
                  used4 = index1;
              }
            }
            // if indeed E2 < E1, add E1 and E2 to E0, they're all a cluster together
            // otherwise, E0 is its own cluster
            if (E2<E1) {
              L2cluster[nclust].pTtot += E1 + E2;
              L2cluster[nclust].numtracks += trx1 + trx2;
              L2cluster[nclust].numttrks += ttrk1 + ttrk2;
              L2cluster[nclust].numtdtrks += tdtrk1 + tdtrk2;
              L2cluster[nclust].numttdtrks += ttdtrk1 + ttdtrk2;
              L2cluster[nclust].phi = L1clusters[phibin+1][used1].phi;
              if (used1 >= 0)
                L1clusters[phibin+1][used1].used = true;
              if (used2 >= 0)
                L1clusters[phibin+1][used2].used = true;
              if (used3 >= 0)
                L1clusters[phibin+2][used3].used = true;
              if (used4 >= 0)
                L1clusters[phibin+2][used4].used = true;
            }
            nclust++;
            continue;
          } // end Not phiBins-2
          else {
            L2cluster[nclust].pTtot += E1;
            L2cluster[nclust].numtracks += trx1;
            L2cluster[nclust].numttrks += ttrk1;
            L2cluster[nclust].numtdtrks += tdtrk1;
            L2cluster[nclust].numttdtrks += ttdtrk1;
            L2cluster[nclust].phi = L1clusters[phibin+1][used1].phi;
            if (used1 >= 0)
              L1clusters[phibin+1][used1].used = true;
            if (used2 >= 0)
              L1clusters[phibin+1][used2].used = true;
            nclust++;
            continue;
          }
        } //End not last phibin(23)
        else { //if it is phibin 23
          L1clusters[phibin][imax].used = true;
          nclust++;
        }
      } // while hipT not 0
    } // for each phibin

    for (phibin=0; phibin<phiBins; ++phibin)
      delete [] L1clusters[phibin];

    // Now merge clusters, if necessary
    for (int m=0; m<nclust-1; ++m) {
      for (int n=m+1; n<nclust; ++n)
      if (L2cluster[n].eta == L2cluster[m].eta && (fabs(L2cluster[n].phi - L2cluster[m].phi) < 1.5*phiStep || fabs(L2cluster[n].phi - L2cluster[m].phi) > 6.0)) {
        if (L2cluster[n].pTtot > L2cluster[m].pTtot)
          L2cluster[m].phi = L2cluster[n].phi;
        L2cluster[m].pTtot += L2cluster[n].pTtot;
        L2cluster[m].numtracks += L2cluster[n].numtracks;
        L2cluster[m].numttrks += L2cluster[n].numttrks;
        L2cluster[m].numtdtrks += L2cluster[n].numtdtrks;
        L2cluster[m].numttdtrks += L2cluster[n].numttdtrks;
        for (int m1=n; m1<nclust-1; ++m1)
          L2cluster[m1] = L2cluster[m1+1];
        nclust--;
        m = -1;
        break; //?????
      } // end if clusters neighbor in eta
    } // end for (m) loop

    // sum up all pTs in this zbin to find ht
    float ht = 0;
    for (int k=0; k<nclust; ++k) {
      if (L2cluster[k].pTtot>50 && L2cluster[k].numtracks<lowpTJetMinTrackMultiplicity)
        continue;
      if (L2cluster[k].pTtot>100 && L2cluster[k].numtracks<highpTJetMinTrackMultiplicity)
        continue;
      if (L2cluster[k].pTtot>minTrkJetpT)
        ht += L2cluster[k].pTtot;
    }

    // if ht is larger than previous max, this is the new vertex zbin
    all_zBins[zbin].znum = zbin;
    all_zBins[zbin].clusters = new etaphibin[nclust];
    all_zBins[zbin].nclust = nclust;
    all_zBins[zbin].zbincenter = (zmin+zmax)/2.0;
    for (int k=0; k<nclust; ++k) {
      all_zBins[zbin].clusters[k].phi = L2cluster[k].phi;
      all_zBins[zbin].clusters[k].eta = L2cluster[k].eta;
      all_zBins[zbin].clusters[k].pTtot = L2cluster[k].pTtot;
      all_zBins[zbin].clusters[k].numtracks = L2cluster[k].numtracks;
      all_zBins[zbin].clusters[k].numttrks = L2cluster[k].numttrks;
      all_zBins[zbin].clusters[k].numtdtrks = L2cluster[k].numtdtrks;
      all_zBins[zbin].clusters[k].numttdtrks = L2cluster[k].numttdtrks;
    }
    all_zBins[zbin].ht = ht;
    if (ht >= mzb.ht) {
      mzb = all_zBins[zbin];
      mzb.zbincenter = (zmin+zmax)/2.0;
    }
    // Prepare for next zbin!
    zmin = zmin+zStep;
    zmax = zmax+zStep;
  } // for each zbin
  for (int zbin=0; zbin<zBins-1; ++zbin) {
    if (zbin==mzb.znum)
      continue;
    delete[] all_zBins[zbin].clusters;
  }
}

etaphibin *L1TrackJetProducer::L1_cluster(etaphibin *phislice) {
  etaphibin *clusters = new etaphibin[etaBins/2];
  if (clusters==NULL)
    edm::LogWarning("L1TrackJetProducer")<<"Clusters memory not assigned!\n";

  // Find eta-phibin with maxpT, make center of cluster, add neighbors if not already used
  float my_pt, left_pt, right_pt, right2pt;
  int nclust = 0;
  right2pt = 0;
  for (int etabin=0; etabin<etaBins; ++etabin) {
    // assign values for my pT and neighbors' pT
    if (phislice[etabin].used)
      continue;
    my_pt = phislice[etabin].pTtot;
    if (etabin > 0 && !phislice[etabin-1].used) {
      left_pt = phislice[etabin-1].pTtot;
    }
    else
      left_pt = 0;
    if (etabin < etaBins-1 && !phislice[etabin+1].used) {
      right_pt = phislice[etabin+1].pTtot;
      if (etabin < etaBins-2 && !phislice[etabin+2].used) {
        right2pt = phislice[etabin+2].pTtot;
      }
      else
        right2pt = 0;
    }
    else
      right_pt = 0;

    // if I'm not a cluster, move on
    if (my_pt<left_pt || my_pt<=right_pt) {
      // if unused pT in the left neighbor, spit it out as a cluster
      if (left_pt>0) {
        clusters[nclust] = phislice[etabin-1];
        phislice[etabin-1].used = true;
        nclust++;
      }
      continue;
    }

    // I guess I'm a cluster-- should I use my right neighbor?
    // Note: left neighbor will definitely be used because if it
    //       didn't belong to me it would have been used already
    clusters[nclust] = phislice[etabin];
    phislice[etabin].used = true;
    if (left_pt > 0) {
      clusters[nclust].pTtot += left_pt;
      clusters[nclust].numtracks += phislice[etabin-1].numtracks;
      clusters[nclust].numttrks += phislice[etabin-1].numttrks;
      clusters[nclust].numtdtrks += phislice[etabin-1].numtdtrks;
      clusters[nclust].numttdtrks += phislice[etabin-1].numttdtrks;
    }
    if (my_pt >= right2pt && right_pt > 0) {
      clusters[nclust].pTtot += right_pt;
      clusters[nclust].numtracks += phislice[etabin+1].numtracks;
      clusters[nclust].numttrks += phislice[etabin+1].numttrks;
      clusters[nclust].numtdtrks += phislice[etabin+1].numtdtrks;
      clusters[nclust].numttdtrks += phislice[etabin+1].numttdtrks;
      phislice[etabin+1].used = true;
    }
    nclust++;
  } // for each etabin

  // Now merge clusters, if necessary
  for (int m = 0; m<nclust-1; ++m) {
    if (fabs(clusters[m+1].eta - clusters[m].eta) < 1.5*etaStep) {
      if (clusters[m+1].pTtot > clusters[m].pTtot) {
        clusters[m].eta = clusters[m+1].eta;
      }
      clusters[m].pTtot += clusters[m+1].pTtot;
      clusters[m].numtracks += clusters[m+1].numtracks;  // Previous version didn't add tracks when merging
      clusters[m].numttrks += clusters[m+1].numttrks;
      clusters[m].numtdtrks += clusters[m+1].numtdtrks;
      clusters[m].numttdtrks += clusters[m+1].numttdtrks;
      for (int m1 = m+1; m1<nclust-1; ++m1)
        clusters[m1] = clusters[m1+1];
      nclust--;
      m = -1;
    } // end if clusters neighbor in eta
  } // end for (m) loop

  for (int i = nclust; i<etaBins/2; ++i) // zero out remaining unused clusters
    clusters[i].pTtot = 0;
  return clusters;
}

void L1TrackJetProducer::beginStream(StreamID) { }

void L1TrackJetProducer::endStream() { }

bool L1TrackJetProducer::TrackQualityCuts(float trk_pt,int trk_nstub, float trk_chi2,float trk_bendchi2, float trk_d0) {
  bool PassQuality=false;
  if (trk_bendchi2<trk_bendChi2Max && trk_chi2<trk_chi2dofMax && trk_nstub>=4 && !displaced) PassQuality = true;
  if (displaced && trk_bendchi2<nStubs4Displacedbend_Tight && trk_chi2<nStubs4DisplacedChi2_Tight  && trk_nstub==4 && trk_d0<=d0_cutNStubs4) PassQuality = true;
  if (displaced && trk_bendchi2<nStubs4Displacedbend_Loose && trk_chi2<nStubs4DisplacedChi2_Loose  && trk_nstub==4 && trk_d0>d0_cutNStubs4) PassQuality = true;
  if (displaced && trk_bendchi2<nStubs5Displacedbend_Loose && trk_chi2<nStubs5DisplacedChi2_Loose  && trk_nstub>4) PassQuality = true;
  return PassQuality;
}

void L1TrackJetProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TrackJetProducer);
