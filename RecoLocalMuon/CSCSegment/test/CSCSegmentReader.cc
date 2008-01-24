/** \file CSCSegmentReader.cc
 *
 *  $Date: 2007/11/01 19:05:45 $
 *  $Revision: 1.18 $
 *  \author M. Sani
 *
 *  Modified by D. Fortin - UC Riverside
 */

#include <RecoLocalMuon/CSCSegment/test/CSCSegmentReader.h>

#include <FWCore/Framework/interface/MakerMacros.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>

#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>

using namespace std;
using namespace edm;


// Constructor
CSCSegmentReader::CSCSegmentReader(const ParameterSet& pset) {

  simhit                    = 0;
  near_segment              = 0;
  filename                  = pset.getUntrackedParameter<string>("RootFileName");	
  minLayerWithRechitChamber = pset.getUntrackedParameter<int>("minLayerWithRechitPerChamber");
  minLayerWithSimhitChamber = pset.getUntrackedParameter<int>("minLayerWithSimhitPerChamber");
  maxNhits                  = pset.getUntrackedParameter<int>("maxNhits"); 
  minRechitSegment          = pset.getUntrackedParameter<int>("minRechitPerSegment");
  maxPhi                    = pset.getUntrackedParameter<double>("maxPhiSeparation");
  maxTheta                  = pset.getUntrackedParameter<double>("maxThetaSeparation");

  file = new TFile(filename.c_str(), "RECREATE");
    	
  if (file->IsOpen()) cout<<"file open!"<<endl;
  else cout<<"*** Error in opening file ***"<<endl;
   
  hchi2    = new TH1F("h4", "chi2", 500, 0, 5000);    
  hrechit  = new TH1I("h5", "nrechit", 6, 2, 8);  
  hsegment = new TH1I("h6", "segments multiplicity", 20, 0, 20);   
  heta     = new TH1F("h7", "eta sim muons", 50, -2.5, 2.5);  
  hpt      = new TH1F("h8", "pT sim muons", 120, 0, 60);
  hx       = new TH1F("h9", "deltaX", 400, -100, +100);
  hy       = new TH1F("h10", "deltaY",400, -100, +100);
    
  char a[3];
  for (int i=0; i<9; i++) {
    sprintf(a, "h2%d", i);
    hdxOri[i]    = new TH1F(a, "#Delta X", 101, -0.505, 0.505);
    sprintf(a, "h3%d", i);
    hdyOri[i]    = new TH1F(a, "#Delta Y", 101, -5.05, 5.05);
    sprintf(a, "h4%d", i);
    hphiDir[i]   = new TH1F(a, "#Delta #phi", 101, -0.202, 0.202);
    sprintf(a, "h5%d", i);
    hthetaDir[i] = new TH1F(a, "#Delta #theta", 101, -0.404, 0.404);
  }    
}


// Destructor
CSCSegmentReader::~CSCSegmentReader() {

  int ibin = 0;
  heff0    = new TH1F("h0", "efficiency", segMap1.size()*2 + 2, 0, segMap1.size()*2 + 2); 
  heff1    = new TH1F("h1", "efficiency", segMap1.size()*2 + 2, 0, segMap1.size()*2 + 2); 
  heff2    = new TH1F("h2", "efficiency", segMap1.size()*2 + 2, 0, segMap1.size()*2 + 2); 
  heff3    = new TH1F("h3", "efficiency", segMap1.size()*2 + 2, 0, segMap1.size()*2 + 2); 

  cout << "Raw reco efficiency for 6-hit simulated segment (nhits > 3)" << endl;        
  for (map<string,int>::const_iterator it = segMap1.begin(); it != segMap1.end(); it++) {
    ibin++;
    float eff = (float)it->second/(float)chaMap1[it->first]; 
    heff0->SetBinContent(ibin*2, eff);
    heff0->GetXaxis()->SetBinLabel(ibin*2, (it->first).c_str());
    cout << it->first << ": " << it->second << " " << chaMap1[it->first] 
         << "  "      << eff  << endl;
  }
  ibin = 0;
  cout << "Raw reco efficiency for chamber with 6 layers with rechits (nhits > 3)" << endl;        
  for (map<string,int>::const_iterator it = segMap2.begin(); it != segMap2.end(); it++) {
    ibin++;
    float eff = (float)it->second/(float)chaMap2[it->first]; 
    heff1->SetBinContent(ibin*2, eff);
    heff1->GetXaxis()->SetBinLabel(ibin*2, (it->first).c_str());
    cout << it->first << ": " << it->second << " " << chaMap2[it->first] 
         << "  "      << eff << endl;
  }
  ibin = 0;
  cout << "Reco efficiency for building 6-hit segment for 6-hit simulated segment" << endl;        
  for (map<string,int>::const_iterator it = segMap3.begin(); it != segMap3.end(); it++) {
    ibin++;
    float eff = (float)it->second/(float)chaMap1[it->first]; 
    heff2->SetBinContent(ibin*2, eff);
    heff2->GetXaxis()->SetBinLabel(ibin*2, (it->first).c_str());
    cout << it->first << ": " << it->second << " " << chaMap1[it->first] 
         << "  "      << eff  << endl;
  }
  ibin = 0;
  cout << "Reco efficiency for building 6-hit segment for chamber with 6 layers with rechits" << endl;        
  for (map<string,int>::const_iterator it = segMap3.begin(); it != segMap3.end(); it++) {
    ibin++;
    float eff = (float)it->second/(float)chaMap2[it->first]; 
    heff3->SetBinContent(ibin*2, eff);
    heff3->GetXaxis()->SetBinLabel(ibin*2, (it->first).c_str());
    cout << it->first << ": " << it->second << " " << chaMap2[it->first] 
         << "  "      << eff  << endl;
  }

  file->cd();
  heff1->Write();
  heff2->Write();
  heff3->Write();
  hchi2->Write();
  hrechit ->Write();
  hsegment->Write();
  hpt->Write();
  heta->Write();
  hx->Write();
  hy->Write();

  for (int i=0; i<9; i++) {        
    hdxOri[i]->Write();
    hdyOri[i]->Write();
    hphiDir[i]->Write();
    hthetaDir[i]->Write();
  }    
  file->Close();
}


// The real analysis
void CSCSegmentReader::analyze(const Event& event, const EventSetup& eventSetup) {
    
    edm::ESHandle<CSCGeometry> h;
    eventSetup.get<MuonGeometryRecord>().get(h);
    const CSCGeometry* geom = &*h;
    
    Handle<SimTrackContainer> simTracks;
    event.getByLabel("g4SimHits",simTracks);
    
    Handle<PSimHitContainer> simHits; 
    event.getByLabel("g4SimHits","MuonCSCHits",simHits);    
    
    Handle<CSCRecHit2DCollection> recHits; 
    event.getByLabel("csc2DRecHits", recHits);   
    
    Handle<CSCSegmentCollection> cscSegments;
    event.getByLabel("cscSegments", cscSegments);
    
    simInfo(simTracks);
    resolution(simHits, cscSegments, geom);
    recInfo(simHits, recHits, cscSegments, geom);
}


void CSCSegmentReader::recInfo(const edm::Handle<edm::PSimHitContainer> simHits, 
                               const edm::Handle<CSCRecHit2DCollection> recHits, 
                               const edm::Handle<CSCSegmentCollection> cscSegments,
                               const CSCGeometry* geom) {
    
  hsegment->Fill(cscSegments->end() - cscSegments->begin());


  std::vector<CSCDetId> cscChambers;
  for (PSimHitContainer::const_iterator simIt = simHits->begin(); simIt != simHits->end(); simIt++) {
        
    CSCDetId simId = (CSCDetId)(*simIt).detUnitId();

    double g6 = geom->chamber(simId)->layer(6)->surface().toGlobal(LocalPoint(0,0,0)).z();	
    double g1 = geom->chamber(simId)->layer(1)->surface().toGlobal(LocalPoint(0,0,0)).z();	
        
    int firstLayer = 1;
    if (fabs(g1) > fabs(g6)) firstLayer = 6;

    const CSCChamber* chamber = geom->chamber(simId); 

    if (simId.layer() == firstLayer) {

      // Test if have 6 layers with simhits in chamber
      int ith_layer = 0;
      int nLayerWithSimhitsInChamber = 0; 

      for (PSimHitContainer::const_iterator it2 = simHits->begin(); it2 != simHits->end(); it2++) {        
        CSCDetId simId2 = (CSCDetId)(*it2).detUnitId();
        if ((simId2.chamber() == simId.chamber()) &&
            (simId2.station() == simId.station()) &&
            (simId2.ring()    == simId.ring())    &&
            (simId2.endcap()  == simId.endcap())  &&
            (simId2.layer()   != ith_layer     )) {
          nLayerWithSimhitsInChamber++;
          ith_layer = simId2.layer();
        }
      }

      if (nLayerWithSimhitsInChamber < minLayerWithSimhitChamber) continue;

      bool satisfied0 = false;
      ith_layer = 0;
      int nLayerWithRechitsInChamber = 0; 
      int nRecHitChamber = 0;
      // Test if have 6 layers with rechits in chamber
      for (CSCRecHit2DCollection::const_iterator recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
	CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();	
        if ((idrec.endcap()  == simId.endcap())  &&
            (idrec.ring()    == simId.ring())    &&
            (idrec.station() == simId.station()) &&
            (idrec.chamber() == simId.chamber()) &&
            (idrec.layer()   != ith_layer     )) {
          nLayerWithRechitsInChamber++;
          ith_layer = idrec.layer();
        } 
        else if 
           ((idrec.endcap()  == simId.endcap())  &&
            (idrec.ring()    == simId.ring())    &&
            (idrec.station() == simId.station()) &&
            (idrec.chamber() == simId.chamber())) {
          nRecHitChamber++;
        }
      }

      if (simId.ring() < 4) {
        if ( nRecHitChamber > maxNhits ) continue;
      } else {
        if ( nRecHitChamber > 3*maxNhits ) continue;
      }

      string s = chamber->specs()->chamberTypeName();

      chaMap1[s]++;                

      if (nLayerWithRechitsInChamber >= minLayerWithRechitChamber) {
        chaMap2[chamber->specs()->chamberTypeName()]++;
        satisfied0 = true;
      }

      bool satisfied1 = false;
      bool satisfied2 = false;


      int bestMatchIdx= bestMatch( simId, simHits, cscSegments, geom);

      int idx = -1;

      for (CSCSegmentCollection::const_iterator segIt=cscSegments->begin(); segIt!=cscSegments->end(); ++segIt) {
        idx++;

        CSCDetId id = (*segIt).cscDetId();
        if ((simId.endcap()  == id.endcap())  &&
            (simId.ring()    == id.ring())    &&
            (simId.station() == id.station()) &&
            (simId.chamber() == id.chamber())) {              

          satisfied1 = true;

          hrechit->Fill((*segIt).nRecHits());        

          if ( (*segIt).nRecHits() >= minRechitSegment ) {
            satisfied2 = true;

            if (bestMatchIdx == idx ) {         
  	      hchi2->Fill(((*segIt).chi2()/(2*(*segIt).nRecHits()-4)));
//	      hchi2->Fill((*segIt).chi2());
            }
          }
        }    
      }

      if (satisfied1) segMap1[s]++;
      if (satisfied1 && satisfied0) segMap2[s]++;
      if (satisfied2 && satisfied0) segMap3[s]++;

    }   
  }
}


void CSCSegmentReader::simInfo(const edm::Handle<SimTrackContainer> simTracks) {

  for (SimTrackContainer::const_iterator it = simTracks->begin(); it != simTracks->end(); it++) {
        
    if (abs((*it).type()) == 13) {
      hpt->Fill((*it).momentum().pt());
      heta->Fill((*it).momentum().eta());
    }    
  }    
}
  
  
void CSCSegmentReader::resolution(const Handle<PSimHitContainer> simHits, 
                                  const Handle<CSCSegmentCollection> cscSegments, 
                                  const CSCGeometry* geom) {
 
  int idx = -1;

  for (CSCSegmentCollection::const_iterator its = cscSegments->begin(); its != cscSegments->end(); its++) {

    idx++;

    CSCDetId recId = (*its).cscDetId();

    int bestMatchIdx= bestMatch( recId, simHits, cscSegments, geom);

    if (bestMatchIdx != idx) continue;
        
    double segX=-99999., segY=-99999.;
    double simX = 100.,  sim1X = 100., sim2X = 0.;
    double simY = 100.,  sim1Y = 100., sim2Y = 0.;
    double resoPhi = 1., resoTheta = 1.;
    unsigned int simTrack = 0;
        
    const CSCChamber* ch = geom->chamber((*its).cscDetId());

    LocalVector segDir = (*its).localDirection();
    LocalPoint segLoc  = (*its).localPosition();

    segX = segLoc.x();
    segY = segLoc.y();
    
       
    GlobalPoint gpn = ch->layer(6)->surface().toGlobal(LocalPoint(0,0,0)); 	 
    GlobalPoint gpf = ch->layer(1)->surface().toGlobal(LocalPoint(0,0,0));

    int firstLayer = 6;
    int lastLayer = 1;
        
    if (fabs(gpn.z()) > fabs(gpf.z())) {
      firstLayer = 1;
      lastLayer = 6;
    }

    bool check1 = false;
    LocalVector simDir1;
       
    double minDphi = maxPhi;
 
    for (PSimHitContainer::const_iterator ith = simHits->begin(); ith != simHits->end(); ith++) {

      if ( abs((*ith).particleType()) != 13 ) continue;
        
      CSCDetId simId = (CSCDetId)(*ith).detUnitId();

      if ((simId.layer() == firstLayer)) {

        if (ch == geom->chamber(simId)) {

          check1 = true;
       
          LocalVector simDir1_temp = (*ith).momentumAtEntry().unit();

          double simPhi_temp = simDir1_temp.phi();

          double dPhi = simPhi_temp - segDir.phi();
 	  dPhi = fabs(dPhi);

          if ( dPhi < minDphi) {
            minDphi = fabs(dPhi);
            simDir1 = simDir1_temp;
 
            sim1X =(*ith).localPosition().x();
            sim1Y =(*ith).localPosition().y();

            simTrack = (*ith).trackId();
          }
        }
      }    
    }


    minDphi = maxPhi;

    LocalVector simDir6;
    bool check6 = false;
         
    for (PSimHitContainer::const_iterator ith = simHits->begin(); ith != simHits->end(); ith++) {
        
      if ( abs((*ith).particleType()) != 13 ) continue;

      CSCDetId simId = (CSCDetId)(*ith).detUnitId();
            
      if ((simId.layer() == lastLayer) && (simTrack = (*ith).trackId())) {

        check6 = true;

        LocalVector simDir6_temp = (*ith).momentumAtEntry().unit();

        double simPhi_temp = simDir6_temp.phi();

        double dPhi = simPhi_temp - segDir.phi();
 	dPhi = fabs(dPhi);

        if ( dPhi < minDphi) {
          minDphi = fabs(dPhi);
          simDir6 = simDir6_temp;

          simDir6 = (*ith).momentumAtEntry().unit();                    

          sim2X =(*ith).localPosition().x();
          sim2Y =(*ith).localPosition().y();
        }
      }
    }   

    if (check1 && check6) {
      double simTheta = (simDir1.theta() + simDir6.theta())/2.;
      double deltaTheta = segDir.theta()-simTheta;

      double simPhi = (simDir1.phi() + simDir6.phi())/2.;
      double deltaPhi = segDir.phi()-simPhi;

      resoTheta = deltaTheta;
      resoPhi   = deltaPhi;

      simX = (sim1X+sim2X)/2.;
      simY = (sim1Y+sim2Y)/2.;

      float deltaX = segX - simX;
      float deltaY = segY - simY;

      if ( fabs(deltaX) < 4.5 ) {
    
//        string s = chamber->specs()->chamberTypeName();

        int indice = 0;

        hdxOri[indice]->Fill(deltaX);
        hdyOri[indice]->Fill(deltaY);   
        hphiDir[indice]->Fill(resoPhi);
        hthetaDir[indice]->Fill(resoTheta);       
      }       
    }
  }  
}



int CSCSegmentReader::bestMatch( CSCDetId id0,
                                 const Handle<PSimHitContainer> simHits,
                                 const Handle<CSCSegmentCollection> cscSegments,
                                 const CSCGeometry* geom) {

  int bestIndex  = -1;

  const CSCChamber* ch = geom->chamber(id0);

  bool check1 = false;
  bool check6 = false;

  GlobalPoint gpn = ch->layer(6)->surface().toGlobal(LocalPoint(0,0,0));
  GlobalPoint gpf = ch->layer(1)->surface().toGlobal(LocalPoint(0,0,0));

  int firstLayer = 6;
  int lastLayer  = 1;

  if (fabs(gpn.z()) > fabs(gpf.z())) {
    firstLayer = 1;
    lastLayer = 6;
  }

  LocalVector simDir1;

  float sim1X = 0.;
  float sim1Y = 0.;
  float sim1Z = 0.;
  int counter = 0;

  for (PSimHitContainer::const_iterator ith = simHits->begin(); ith != simHits->end(); ith++) {

    if ( abs((*ith).particleType()) != 13 ) continue;

    CSCDetId simId = (CSCDetId)(*ith).detUnitId();

    if ( simId.layer() == firstLayer && ch == geom->chamber(simId) ) {

      check1 = true;

      sim1X +=(*ith).localPosition().x();
      sim1Y +=(*ith).localPosition().y();
      sim1Z  =(*ith).localPosition().z();

      counter++;
    }
  }

  if ( !check1 ) return bestIndex;
  sim1X = sim1X / counter;
  sim1Y = sim1Y / counter;


  float sim2X = 0.;
  float sim2Y = 0.;
  float sim2Z = 0.;
  counter = 0;

  for (PSimHitContainer::const_iterator ith = simHits->begin(); ith != simHits->end(); ith++) {

    if ( abs((*ith).particleType()) != 13 ) continue;

    CSCDetId simId = (CSCDetId)(*ith).detUnitId();
    if ( simId.layer() == lastLayer && ch == geom->chamber(simId) ) {

      check6 = true;

      sim2X +=(*ith).localPosition().x();
      sim2Y +=(*ith).localPosition().y();
      sim2Z  =(*ith).localPosition().z();
      counter++;
    }
  }


  if ( !check6 ) return bestIndex;

  sim2X = sim2X / counter;
  sim2Y = sim2Y / counter;


  float x = (sim2X + sim1X) /2.;
  float y = (sim2Y + sim1Y) /2.;

  float dx = (sim2X - sim1X);
  float dy = (sim2Y - sim1Y);
  float dz = (sim2Z - sim1Z);
  float magSim = sqrt(dx*dx + dy*dy + dz*dz);

  int idxCounter = -1;
  float bestCosTheta = 0.;


  for (CSCSegmentCollection::const_iterator its = cscSegments->begin(); its !=cscSegments->end(); its++) {
    CSCDetId id = (*its).cscDetId();

    if ((id0.endcap()   == id.endcap())  &&
        (id0.ring()     == id.ring())    &&
        (id0.station()  == id.station()) &&
        (id0.chamber() == id.chamber())) {
      idxCounter++;
    } else {
      idxCounter++;
      continue;
    }

    float xreco  = (*its).localPosition().x();
    float yreco  = (*its).localPosition().y();

    float dR = (xreco - x) *  (xreco - x) + (yreco - y) *  (yreco - y);
    dR = sqrt(dR);

    if (dR > 4. ) continue;   // 4 centimeter radius

    LocalVector segDir = (*its).localDirection();

    float xdir = segDir.x();
    float ydir = segDir.y();
    float zdir = segDir.z();
    float magReco = sqrt(xdir*xdir + ydir*ydir + zdir*zdir);

    // Find angular difference between two segments.
    // Use dot product:  cos(theta_12) = v1 . v2 / [ |v1|*|v2| ]

    double costheta = (xdir * dx + ydir * dy + zdir * dz) / (magSim * magReco);
    if (costheta < 0.) costheta = -costheta;

    if (costheta > bestCosTheta) {
      bestCosTheta = costheta;
      bestIndex = idxCounter;
    }

  }

  return bestIndex;
}


DEFINE_FWK_MODULE(CSCSegmentReader);

