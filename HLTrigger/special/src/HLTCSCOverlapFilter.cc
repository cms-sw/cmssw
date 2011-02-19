#include "HLTrigger/special/interface/HLTCSCOverlapFilter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

HLTCSCOverlapFilter::HLTCSCOverlapFilter(const edm::ParameterSet& iConfig)
   : m_input(iConfig.getParameter<edm::InputTag>("input"))
     , m_minHits(iConfig.getParameter<unsigned int>("minHits"))
     , m_xWindow(iConfig.getParameter<double>("xWindow"))
     , m_yWindow(iConfig.getParameter<double>("yWindow"))
     , m_ring1(iConfig.getParameter<bool>("ring1"))
     , m_ring2(iConfig.getParameter<bool>("ring2"))
     , m_fillHists(iConfig.getParameter<bool>("fillHists"))
{
   if (m_fillHists) {
      edm::Service<TFileService> tfile;
      m_nhitsNoWindowCut = tfile->make<TH1F>("nhitsNoWindowCut", "nhitsNoWindowCut", 16, -0.5, 15.5);
      m_xdiff = tfile->make<TH1F>("xdiff", "xdiff", 200, 0., 10.);
      m_ydiff = tfile->make<TH1F>("ydiff", "ydiff", 200, 0., 10.);
      m_pairsWithWindowCut = tfile->make<TH1F>("pairsWithWindowCut", "pairsWithWindowCut", 226, -0.5, 225.5);
   }
}

HLTCSCOverlapFilter::~HLTCSCOverlapFilter() { }

bool HLTCSCOverlapFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
   edm::Handle<CSCRecHit2DCollection> hits;
   iEvent.getByLabel(m_input, hits);

   edm::ESHandle<CSCGeometry> cscGeometry;
   bool got_cscGeometry = false;

   std::map<int, std::vector<const CSCRecHit2D*> > chamber_tohit;

   for (CSCRecHit2DCollection::const_iterator hit = hits->begin();  hit != hits->end();  ++hit) {
      CSCDetId id(hit->geographicalId());
      int chamber_id = CSCDetId(id.endcap(), id.station(), id.ring(), id.chamber(), 0).rawId();

      if ((m_ring1  &&  (id.ring() == 1  ||  id.ring() == 4))  ||
	  (m_ring2  &&  id.ring() == 2)) {
	 std::map<int, std::vector<const CSCRecHit2D*> >::const_iterator chamber_iter = chamber_tohit.find(chamber_id);
	 if (chamber_iter == chamber_tohit.end()) {
	    std::vector<const CSCRecHit2D*> newlist;
	    newlist.push_back(&(*hit));
	 }
	 chamber_tohit[chamber_id].push_back(&(*hit));
      } // end if this ring is selected
   } // end loop over hits

   bool keep = false;
   unsigned int minHitsSquared = m_minHits * m_minHits;
   for (std::map<int, std::vector<const CSCRecHit2D*> >::const_iterator chamber_iter = chamber_tohit.begin();
	chamber_iter != chamber_tohit.end();
	++chamber_iter) {

      if (m_fillHists) {
	 m_nhitsNoWindowCut->Fill(chamber_iter->second.size());
      }

      if (chamber_iter->second.size() >= m_minHits) {
	 CSCDetId chamber_id(chamber_iter->first);
	 int chamber = chamber_id.chamber();
	 int next = chamber + 1;
      
	 // Some rings have 36 chambers, others have 18.  This will still be valid when ME4/2 is added.
	 if (next == 37  &&  (std::abs(chamber_id.station()) == 1  ||  chamber_id.ring() == 2)) next = 1;
	 if (next == 19  &&  (std::abs(chamber_id.station()) != 1  &&  chamber_id.ring() == 1)) next = 1;
      
	 int next_id = CSCDetId(chamber_id.endcap(), chamber_id.station(), chamber_id.ring(), next, 0).rawId();

	 std::map<int, std::vector<const CSCRecHit2D*> >::const_iterator chamber_next = chamber_tohit.find(next_id);
	 if (chamber_next != chamber_tohit.end()  &&  chamber_next->second.size() >= m_minHits) {
	    if (!got_cscGeometry) {
	       iSetup.get<MuonGeometryRecord>().get(cscGeometry);
	       got_cscGeometry = true;
	    }

	    unsigned int pairs_in_window = 0;
	    for (std::vector<const CSCRecHit2D*>::const_iterator hit1 = chamber_iter->second.begin();  hit1 != chamber_iter->second.end();  ++hit1) {
	       for (std::vector<const CSCRecHit2D*>::const_iterator hit2 = chamber_next->second.begin();  hit2 != chamber_next->second.end();  ++hit2) {
		  GlobalPoint pos1 = cscGeometry->idToDet((*hit1)->geographicalId())->surface().toGlobal((*hit1)->localPosition());
		  GlobalPoint pos2 = cscGeometry->idToDet((*hit2)->geographicalId())->surface().toGlobal((*hit2)->localPosition());

		  if (m_fillHists) {
		     m_xdiff->Fill(pos1.x() - pos2.x());
		     m_ydiff->Fill(pos1.y() - pos2.y());
		  }

		  if (fabs(pos1.x() - pos2.x()) < m_xWindow  &&  fabs(pos1.y() - pos2.y()) < m_yWindow) pairs_in_window++;

		  if (pairs_in_window >= minHitsSquared) {
		     keep = true;
		     if (!m_fillHists) return true;
		  }
	       } // end loop over hits in chamber 2
	    } // end loop over hits in chamber 1

	    if (m_fillHists) {
	       m_pairsWithWindowCut->Fill(pairs_in_window);
	    }

	 } // end if chamber 2 has enough hits
      } // end if chamber 1 has enough hits
   } // end loop over chambers

   return keep;
}
  
