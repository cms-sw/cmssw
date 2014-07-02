#include "DQMOffline/Muon/interface/MuonIdDQM.h"

MuonIdDQM::MuonIdDQM(const edm::ParameterSet& iConfig){
  inputMuonCollection_           = consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("inputMuonCollection"));
  inputDTRecSegment4DCollection_ = consumes<DTRecSegment4DCollection>(iConfig.getParameter<edm::InputTag>("inputDTRecSegment4DCollection"));
  inputCSCSegmentCollection_     = consumes<CSCSegmentCollection>(iConfig.getParameter<edm::InputTag>("inputCSCSegmentCollection"));
  useTrackerMuons_               = iConfig.getUntrackedParameter<bool>("useTrackerMuons");
  useGlobalMuons_                = iConfig.getUntrackedParameter<bool>("useGlobalMuons");
  useTrackerMuonsNotGlobalMuons_ = iConfig.getUntrackedParameter<bool>("useTrackerMuonsNotGlobalMuons");
  useGlobalMuonsNotTrackerMuons_ = iConfig.getUntrackedParameter<bool>("useGlobalMuonsNotTrackerMuons");
  baseFolder_                    = iConfig.getUntrackedParameter<std::string>("baseFolder");
  
}

MuonIdDQM::~MuonIdDQM() {}


void MuonIdDQM::bookHistograms(DQMStore::IBooker & ibooker,
			       edm::Run const & /*iRun*/,
			       edm::EventSetup const & /* iSetup */){

   char name[100], title[200];

   ibooker.cd();
   ibooker.setCurrentFolder(baseFolder_);
   // trackerMuon == 0; globalMuon == 1; trackerMuon && !globalMuon == 2; globalMuon && !trackerMuon == 3
   
   hSegmentIsAssociatedBool = ibooker.book1D("hSegmentIsAssociatedBool", "Segment Is Associated Boolean", 2, -0.5, 1.5);
   
   for (unsigned int i = 0; i < 4; i++) {
      if ((i == 0 && ! useTrackerMuons_) || (i == 1 && ! useGlobalMuons_)) continue;
      if ((i == 2 && ! useTrackerMuonsNotGlobalMuons_) || (i == 3 && ! useGlobalMuonsNotTrackerMuons_)) continue;
      if (i == 0) ibooker.setCurrentFolder(baseFolder_+"/TrackerMuons");
      if (i == 1) ibooker.setCurrentFolder(baseFolder_+"/GlobalMuons");
      if (i == 2) ibooker.setCurrentFolder(baseFolder_+"/TrackerMuonsNotGlobalMuons");
      if (i == 3) ibooker.setCurrentFolder(baseFolder_+"/GlobalMuonsNotTrackerMuons");

      hNumChambers[i] = ibooker.book1D("hNumChambers", "Number of Chambers", 17, -0.5, 16.5);
      hNumMatches[i] = ibooker.book1D("hNumMatches", "Number of Matches", 11, -0.5, 10.5);
      hNumChambersNoRPC[i] = ibooker.book1D("hNumChambersNoRPC", "Number of Chambers No RPC", 11, -0.5, 10.5);

      // by station
      for(int station = 0; station < 4; ++station)
	{
         sprintf(name, "hDT%iNumSegments", station+1);
         sprintf(title, "DT Station %i Number of Segments (No Arbitration)", station+1);
         hDTNumSegments[i][station] = ibooker.book1D(name, title, 11, -0.5, 10.5);

         sprintf(name, "hDT%iDx", station+1);
         sprintf(title, "DT Station %i Delta X", station+1);
         hDTDx[i][station] = ibooker.book1D(name, title, 100, -100., 100.);

         sprintf(name, "hDT%iPullx", station+1);
         sprintf(title, "DT Station %i Pull X", station+1);
         hDTPullx[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

         sprintf(name, "hDT%iDdXdZ", station+1);
         sprintf(title, "DT Station %i Delta DxDz", station+1);
         hDTDdXdZ[i][station] = ibooker.book1D(name, title, 100, -1., 1.);

         sprintf(name, "hDT%iPulldXdZ", station+1);
         sprintf(title, "DT Station %i Pull DxDz", station+1);
         hDTPulldXdZ[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

         if (station < 3) {
            sprintf(name, "hDT%iDy", station+1);
            sprintf(title, "DT Station %i Delta Y", station+1);
            hDTDy[i][station] = ibooker.book1D(name, title, 100, -150., 150.);

            sprintf(name, "hDT%iPully", station+1);
            sprintf(title, "DT Station %i Pull Y", station+1);
            hDTPully[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

            sprintf(name, "hDT%iDdYdZ", station+1);
            sprintf(title, "DT Station %i Delta DyDz", station+1);
            hDTDdYdZ[i][station] = ibooker.book1D(name, title, 100, -2., 2.);

            sprintf(name, "hDT%iPulldYdZ", station+1);
            sprintf(title, "DT Station %i Pull DyDz", station+1);
            hDTPulldYdZ[i][station] = ibooker.book1D(name, title, 100, -20., 20.);
         }

         sprintf(name, "hCSC%iNumSegments", station+1);
         sprintf(title, "CSC Station %i Number of Segments (No Arbitration)", station+1);
         hCSCNumSegments[i][station] = ibooker.book1D(name, title, 11, -0.5, 10.5);

         sprintf(name, "hCSC%iDx", station+1);
         sprintf(title, "CSC Station %i Delta X", station+1);
         hCSCDx[i][station] = ibooker.book1D(name, title, 100, -50., 50.);

         sprintf(name, "hCSC%iPullx", station+1);
         sprintf(title, "CSC Station %i Pull X", station+1);
         hCSCPullx[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

         sprintf(name, "hCSC%iDdXdZ", station+1);
         sprintf(title, "CSC Station %i Delta DxDz", station+1);
         hCSCDdXdZ[i][station] = ibooker.book1D(name, title, 100, -1., 1.);

         sprintf(name, "hCSC%iPulldXdZ", station+1);
         sprintf(title, "CSC Station %i Pull DxDz", station+1);
         hCSCPulldXdZ[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

         sprintf(name, "hCSC%iDy", station+1);
         sprintf(title, "CSC Station %i Delta Y", station+1);
         hCSCDy[i][station] = ibooker.book1D(name, title, 100, -50., 50.);

         sprintf(name, "hCSC%iPully", station+1);
         sprintf(title, "CSC Station %i Pull Y", station+1);
         hCSCPully[i][station] = ibooker.book1D(name, title, 100, -20., 20.);

         sprintf(name, "hCSC%iDdYdZ", station+1);
         sprintf(title, "CSC Station %i Delta DyDz", station+1);
         hCSCDdYdZ[i][station] = ibooker.book1D(name, title, 100, -1., 1.);

         sprintf(name, "hCSC%iPulldYdZ", station+1);
         sprintf(title, "CSC Station %i Pull DyDz", station+1);
         hCSCPulldYdZ[i][station] = ibooker.book1D(name, title, 100, -20., 20.);
      }// station
   }

}

void MuonIdDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
   using namespace edm;
   using namespace reco;

   iEvent.getByToken(inputMuonCollection_, muonCollectionH_);
   iEvent.getByToken(inputDTRecSegment4DCollection_, dtSegmentCollectionH_);
   iEvent.getByToken(inputCSCSegmentCollection_, cscSegmentCollectionH_);
   iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);

   for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
         muon != muonCollectionH_->end(); ++muon)
   {
      // trackerMuon == 0; globalMuon == 1; trackerMuon && !globalMuon == 2; globalMuon && !trackerMuon == 3
      for (unsigned int i = 0; i < 4; i++) {
         if (i == 0 && (! useTrackerMuons_ || ! muon->isTrackerMuon())) continue;
         if (i == 1 && (! useGlobalMuons_ || ! muon->isGlobalMuon())) continue;
         if (i == 2 && (! useTrackerMuonsNotGlobalMuons_ || (! (muon->isTrackerMuon() && ! muon->isGlobalMuon())))) continue;
         if (i == 3 && (! useGlobalMuonsNotTrackerMuons_ || (! (muon->isGlobalMuon() && ! muon->isTrackerMuon())))) continue;

         hNumChambers[i]->Fill(muon->numberOfChambers());
         hNumMatches[i]->Fill(muon->numberOfMatches(Muon::SegmentAndTrackArbitration));
         hNumChambersNoRPC[i]->Fill(muon->numberOfChambersNoRPC());

         // by station
         for(int station = 0; station < 4; ++station)
         {
            // only fill num segments if we crossed (or nearly crossed) a chamber
            if (muon->trackX(station+1, MuonSubdetId::DT, Muon::NoArbitration) < 900000)
               hDTNumSegments[i][station]->Fill(muon->numberOfSegments(station+1, MuonSubdetId::DT, Muon::NoArbitration));
            Fill(hDTDx[i][station], muon->dX(station+1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration));
            Fill(hDTPullx[i][station], muon->pullX(station+1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration, true));
            Fill(hDTDdXdZ[i][station], muon->dDxDz(station+1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration));
            Fill(hDTPulldXdZ[i][station], muon->pullDxDz(station+1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration, true));

            if (station < 3) {
               Fill(hDTDy[i][station], muon->dY(station+1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration));
               Fill(hDTPully[i][station], muon->pullY(station+1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration, true));
               Fill(hDTDdYdZ[i][station], muon->dDyDz(station+1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration));
               Fill(hDTPulldYdZ[i][station], muon->pullDyDz(station+1, MuonSubdetId::DT, Muon::SegmentAndTrackArbitration, true));
            }

            // only fill num segments if we crossed (or nearly crossed) a chamber
            if (muon->trackX(station+1, MuonSubdetId::CSC, Muon::NoArbitration) < 900000)
               hCSCNumSegments[i][station]->Fill(muon->numberOfSegments(station+1, MuonSubdetId::CSC, Muon::NoArbitration));
            Fill(hCSCDx[i][station], muon->dX(station+1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration));
            Fill(hCSCPullx[i][station], muon->pullX(station+1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration, true));
            Fill(hCSCDdXdZ[i][station], muon->dDxDz(station+1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration));
            Fill(hCSCPulldXdZ[i][station], muon->pullDxDz(station+1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration, true));
            Fill(hCSCDy[i][station], muon->dY(station+1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration));
            Fill(hCSCPully[i][station], muon->pullY(station+1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration, true));
            Fill(hCSCDdYdZ[i][station], muon->dDyDz(station+1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration));
            Fill(hCSCPulldYdZ[i][station], muon->pullDyDz(station+1, MuonSubdetId::CSC, Muon::SegmentAndTrackArbitration, true));
         }
      }
   }// muon

   for(DTRecSegment4DCollection::const_iterator segment = dtSegmentCollectionH_->begin();
         segment != dtSegmentCollectionH_->end(); ++segment)
   {
      LocalPoint  segmentLocalPosition       = segment->localPosition();
      LocalVector segmentLocalDirection      = segment->localDirection();
      LocalError  segmentLocalPositionError  = segment->localPositionError();
      LocalError  segmentLocalDirectionError = segment->localDirectionError();
      bool segmentFound = false;

      for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
            muon != muonCollectionH_->end(); ++muon)
      {
         if (! muon->isMatchesValid())
            continue;

         for(std::vector<MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
               chamberMatch != muon->matches().end(); ++chamberMatch) {
            for(std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                  segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch)
            {
               if (fabs(segmentMatch->x       - segmentLocalPosition.x()                           ) < 1E-6 &&
                   fabs(segmentMatch->y       - segmentLocalPosition.y()                           ) < 1E-6 &&
                   fabs(segmentMatch->dXdZ    - segmentLocalDirection.x()/segmentLocalDirection.z()) < 1E-6 &&
                   fabs(segmentMatch->dYdZ    - segmentLocalDirection.y()/segmentLocalDirection.z()) < 1E-6 &&
                   fabs(segmentMatch->xErr    - sqrt(segmentLocalPositionError.xx())               ) < 1E-6 &&
                   fabs(segmentMatch->yErr    - sqrt(segmentLocalPositionError.yy())               ) < 1E-6 &&
                   fabs(segmentMatch->dXdZErr - sqrt(segmentLocalDirectionError.xx())              ) < 1E-6 &&
                   fabs(segmentMatch->dYdZErr - sqrt(segmentLocalDirectionError.yy())              ) < 1E-6)
               {
                  segmentFound = true;
                  break;
               }
            }// segmentMatch
            if (segmentFound) break;
         }// chamberMatch
         if (segmentFound) break;
      }// muon

      if (segmentFound)
         hSegmentIsAssociatedBool->Fill(1.);
      else
         hSegmentIsAssociatedBool->Fill(0.);
   }// dt segment

   for(CSCSegmentCollection::const_iterator segment = cscSegmentCollectionH_->begin();
         segment != cscSegmentCollectionH_->end(); ++segment)
   {
      LocalPoint  segmentLocalPosition       = segment->localPosition();
      LocalVector segmentLocalDirection      = segment->localDirection();
      LocalError  segmentLocalPositionError  = segment->localPositionError();
      LocalError  segmentLocalDirectionError = segment->localDirectionError();
      bool segmentFound = false;

      for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
            muon != muonCollectionH_->end(); ++muon)
      {
         if (! muon->isMatchesValid())
            continue;

         for(std::vector<MuonChamberMatch>::const_iterator chamberMatch = muon->matches().begin();
               chamberMatch != muon->matches().end(); ++chamberMatch) {
            for(std::vector<MuonSegmentMatch>::const_iterator segmentMatch = chamberMatch->segmentMatches.begin();
                  segmentMatch != chamberMatch->segmentMatches.end(); ++segmentMatch)
            {
               if (fabs(segmentMatch->x       - segmentLocalPosition.x()                           ) < 1E-6 &&
                   fabs(segmentMatch->y       - segmentLocalPosition.y()                           ) < 1E-6 &&
                   fabs(segmentMatch->dXdZ    - segmentLocalDirection.x()/segmentLocalDirection.z()) < 1E-6 &&
                   fabs(segmentMatch->dYdZ    - segmentLocalDirection.y()/segmentLocalDirection.z()) < 1E-6 &&
                   fabs(segmentMatch->xErr    - sqrt(segmentLocalPositionError.xx())               ) < 1E-6 &&
                   fabs(segmentMatch->yErr    - sqrt(segmentLocalPositionError.yy())               ) < 1E-6 &&
                   fabs(segmentMatch->dXdZErr - sqrt(segmentLocalDirectionError.xx())              ) < 1E-6 &&
                   fabs(segmentMatch->dYdZErr - sqrt(segmentLocalDirectionError.yy())              ) < 1E-6)
               {
                  segmentFound = true;
                  break;
               }
            }// segmentMatch
            if (segmentFound) break;
         }// chamberMatch
         if (segmentFound) break;
      }// muon

      if (segmentFound)
         hSegmentIsAssociatedBool->Fill(1.);
      else
         hSegmentIsAssociatedBool->Fill(0.);
   }// csc segment
}

void MuonIdDQM::Fill(MonitorElement* me, float f) {
   if (fabs(f) > 900000) return;
   //if (fabs(f) < 1E-8) return;
   me->Fill(f);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIdDQM);
