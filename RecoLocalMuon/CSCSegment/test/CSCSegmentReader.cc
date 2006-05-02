#include "CSCSegmentReader.h"

#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h>
#include <SimDataFormats/TrackingHit/interface/PSimHitContainer.h>
#include <DataFormats/CSCRecHit/interface/CSCRangeMapAccessor.h>

#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <Geometry/CSCGeometry/interface/CSCChamber.h>

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include <FWCore/MessageLogger/interface/MessageLogger.h> 

using namespace std;
using namespace edm;

#include "TAxis.h"

// Constructor
CSCSegmentReader::CSCSegmentReader(const ParameterSet& pset) {

    filename = pset.getUntrackedParameter<string>("RootFileName");	
    label1 = pset.getUntrackedParameter<string>("label1");
    minRechitChamber = pset.getUntrackedParameter<int>("minRechitPerChamber");
    minRechitSegment = pset.getUntrackedParameter<int>("minRechitPerSegment");
	//label2 = pset.getUntrackedParameter<string>("label2");
    
    file = new TFile(filename.c_str(), "RECREATE");
    	
	if(file->IsOpen()) 
		cout<<"file open!"<<endl;
    else 
		cout<<"*** Error in opening file ***"<<endl;
    
    h2 = new TH1F("h2", "chi2", 120, 0, 60);    
    h3 = new TH1I("h3", "nrechit", 6, 2, 8);       
}

// Destructor
CSCSegmentReader::~CSCSegmentReader() {

    int ibin = 0;
    h = new TH1F("h1", "efficiency", segMap.size()*2 + 2, 0, segMap.size()*2 + 2); 

    for(map<string,int>::const_iterator it = segMap.begin(); it != segMap.end(); it++) {
        ibin++;
        float eff = (float)it->second/(float)chaMap[it->first]; 
        h->SetBinContent(ibin*2, eff);
   		h->GetXaxis()->SetBinLabel(ibin*2, (it->first).c_str());
        
        cout << it->first << ": " << it->second << " " << chaMap[it->first] 
        << "  " << eff << "   " << recMap[it->first] << endl;
    }

    file->cd();
    h->Write();
    h2->Write();
    h3->Write();
    file->Close();
}

// The real analysis
void CSCSegmentReader::analyze(const Event& event, const EventSetup& eventSetup){
    
    edm::ESHandle<CSCGeometry> h;
    eventSetup.get<MuonGeometryRecord>().get(h);
    const CSCGeometry* geom = &*h;
    
    Handle<CSCSegmentCollection> cscSegments;
    event.getByLabel("segmentproducer", cscSegments);
    Handle<PSimHitContainer> simHits; 
    event.getByLabel("SimG4Object","MuonCSCHits",simHits);    
	Handle<CSCRecHit2DCollection> recHits; 
    event.getByLabel("rechitproducer", recHits);    
	
    cout << cscSegments->end() - cscSegments->begin() << endl;
    for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
        
        h2->Fill(((*it).chi2()/(2*(*it).nRecHits()-4)));
        h3->Fill((*it).nRecHits());        
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
        
        recMap[chamber->specs()->chamberTypeName()] += (range.second - range.first);
        
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

DEFINE_FWK_MODULE(CSCSegmentReader)


