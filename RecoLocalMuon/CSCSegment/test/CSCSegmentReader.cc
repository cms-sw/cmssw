/** \file CSCSegmentReader.cc
 *
 *  $Date: 2006/05/02 14:00:28 $
 *  $Revision: 1.1 $
 *  \author M. Sani
 */

#include <RecoLocalMuon/CSCSegment/test/CSCSegmentReader.h>

#include <FWCore/Framework/interface/MakerMacros.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>

#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>

using namespace std;
using namespace edm;

//#include "TAxis.h"

// Constructor
CSCSegmentReader::CSCSegmentReader(const ParameterSet& pset) {

    filename = pset.getUntrackedParameter<string>("RootFileName");	
    minRechitChamber = pset.getUntrackedParameter<int>("minRechitPerChamber");
    minRechitSegment = pset.getUntrackedParameter<int>("minRechitPerSegment");
	//label2 = pset.getUntrackedParameter<string>("label2");
    
    file = new TFile(filename.c_str(), "RECREATE");
    	
	if(file->IsOpen()) 
		cout<<"file open!"<<endl;
    else 
		cout<<"*** Error in opening file ***"<<endl;
   
    hchi2 = new TH1F("h2", "chi2", 120, 0, 60);    
    hrechit = new TH1I("h3", "nrechit", 6, 2, 8);  
    hsegment = new TH1I("h4", "segments multiplicity", 20, 0, 20);   
    heta = new TH1F("h5", "eta sim muons", 50, -2.5, 2.5);  
    hpt = new TH1F("h6", "pT sim muons", 120, 0, 60);
    char a[3];
    for(int i=0; i<4; i++) {

        sprintf(a, "h7%d", i);
        hphi[i] = new TH1F(a, "reso phi", 150, -0.03, 0.03);
        sprintf(a, "h8%d", i);
        htheta[i] = new TH1F(a, "reso theta", 150, -0.15, 0.15);
    }    
}

// Destructor
CSCSegmentReader::~CSCSegmentReader() {

    int ibin = 0;
    heff = new TH1F("h1", "efficiency", segMap.size()*2 + 2, 0, segMap.size()*2 + 2); 

    for(map<string,int>::const_iterator it = segMap.begin(); it != segMap.end(); it++) {
        ibin++;
        float eff = (float)it->second/(float)chaMap[it->first]; 
        heff->SetBinContent(ibin*2, eff);
   		heff->GetXaxis()->SetBinLabel(ibin*2, (it->first).c_str());
        
        cout << it->first << ": " << it->second << " " << chaMap[it->first] 
        << "  " << eff << endl;
    }

    file->cd();
    heff->Write();
    hchi2->Write();
    hrechit->Write();
    hsegment->Write();
    hpt->Write();
    heta->Write();
    for(int i=0; i<4; i++) {
        
        hphi[i]->Write();
        htheta[i]->Write();
    }    
    file->Close();
}

void CSCSegmentReader::recInfo(const edm::Handle<edm::PSimHitContainer> simHits, 
            const edm::Handle<CSCRecHit2DCollection> recHits, const edm::Handle<CSCSegmentCollection> cscSegments,
            const CSCGeometry* geom) {
    
    hsegment->Fill(cscSegments->end() - cscSegments->begin());

    for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
        
        hchi2->Fill(((*it).chi2()/(2*(*it).nRecHits()-4)));
        hrechit->Fill((*it).nRecHits());        
    }        

    std::vector<CSCDetId> cscChambers;
    std::vector<CSCDetId>::const_iterator chIt;
    for(PSimHitContainer::const_iterator simIt = simHits->begin(); simIt != simHits->end(); simIt++) {
    
        if(abs((*simIt).particleType()) == 13) {
            
            bool insert = true;
            CSCDetId id = (CSCDetId)(*simIt).detUnitId();
           
            for(chIt = cscChambers.begin(); chIt != cscChambers.end(); chIt++) {
                
                if ((id.endcap() == (*chIt).endcap()) &&
                    (id.ring() == (*chIt).ring()) &&
                    (id.station() == (*chIt).station()) &&
                    (id.chamber() == (*chIt).chamber()))
                    insert = false;
            }
            
            if (insert)
                cscChambers.push_back(id);        
        }
    }
        
    CSCRangeMapAccessor acc;
    std::vector<CSCSegment> temp;

    for(chIt = cscChambers.begin(); chIt != cscChambers.end(); chIt++) {
        
        const CSCChamber* chamber = geom->chamber(*chIt); 
        CSCRecHit2DCollection::range range = recHits->get(acc.cscChamber(*chIt));
        
        if ((range.second - range.first) >= minRechitChamber) {
        
            chaMap[chamber->specs()->chamberTypeName()]++;
            CSCSegmentCollection::const_iterator segIt;
            
            int pippo = 0;
            for (segIt=cscSegments->begin(); segIt!=cscSegments->end(); ++segIt) {
                
                CSCDetId id = (*segIt).cscDetId();
                if ((id.endcap() == (*chIt).endcap()) &&
                    (id.ring() == (*chIt).ring()) &&
                    (id.station() == (*chIt).station()) &&
                    (id.chamber() == (*chIt).chamber())) {
                    
                    if ((*segIt).nRecHits() >= minRechitSegment) {
                        pippo = 1;
                        break;
                        temp.push_back(*segIt);
                    }
                }   
            }   

            if (pippo)
                segMap[chamber->specs()->chamberTypeName()]++;
        }        
    }
}

void CSCSegmentReader::simInfo(const edm::Handle<EmbdSimTrackContainer> simTracks) {
    for(EmbdSimTrackContainer::const_iterator it = simTracks->begin(); it != simTracks->end(); it++) {
        
        if (abs((*it).type()) == 13) {
            hpt->Fill((*it).momentum().perp());
            heta->Fill((*it).momentum().eta());
        }    
    }    
}
    
void CSCSegmentReader::resolution(const Handle<PSimHitContainer> simHits, 
        const Handle<CSCSegmentCollection> cscSegments, const CSCGeometry* geom) {

	double minPhi = 1000., minTheta = 1000.;
    double resoPhi, resoTheta;

    for(PSimHitContainer::const_iterator ith = simHits->begin(); ith != simHits->end(); ith++) {
    
        LocalVector lv = ith->momentumAtEntry();
        CSCDetId cscDetId(ith->detUnitId());
        const CSCLayer* layer = geom->layer(cscDetId);
        GlobalVector simDir = layer->surface().toGlobal(lv);
        
        for(CSCSegmentCollection::const_iterator its = cscSegments->begin(); its != cscSegments->end(); its++) {
    
            const CSCChamber* chamber = geom->chamber((*its).cscDetId());
            GlobalVector segDir = chamber->surface().toGlobal((*its).localDirection());
            
            double deltaTheta = fabs(segDir.theta()-simDir.theta());
            double deltaPhi = fabs(segDir.phi()-simDir.phi());
            
            if ((deltaPhi < minPhi) && (deltaTheta < minTheta)) {
                
                minPhi = deltaPhi;
                minTheta = deltaTheta;
                resoTheta = (segDir.theta()-simDir.theta());
                resoPhi = (segDir.phi()-simDir.phi());
            }

            string s = chamber->specs()->chamberTypeName();

            int indice = 3;
            if ((s == "ME1/a") || (s == "ME1/b"))        
                indice = 0;
            if ((s == "ME1/2") || (s == "ME1/3"))        
                indice = 1;
            if ((s == "ME2/1") || (s == "ME3/1") || (s == "ME4/1"))        
                indice = 2;
                
            hphi[indice]->Fill(resoPhi);
            htheta[indice]->Fill(resoTheta);
        }
    }       
}

// The real analysis
void CSCSegmentReader::analyze(const Event& event, const EventSetup& eventSetup) {
    
    edm::ESHandle<CSCGeometry> h;
    eventSetup.get<MuonGeometryRecord>().get(h);
    const CSCGeometry* geom = &*h;
    
    Handle<EmbdSimTrackContainer> simTracks;
    event.getByLabel("SimG4Object",simTracks);
    
    Handle<PSimHitContainer> simHits; 
    event.getByLabel("SimG4Object","MuonCSCHits",simHits);    
    
    Handle<CSCRecHit2DCollection> recHits; 
    event.getByLabel("rechitproducer", recHits);   
    
    Handle<CSCSegmentCollection> cscSegments;
    event.getByLabel("segmentproducer", cscSegments);
    
    simInfo(simTracks);
    resolution(simHits, cscSegments, geom);
    recInfo(simHits, recHits, cscSegments, geom);
}


DEFINE_FWK_MODULE(CSCSegmentReader)


