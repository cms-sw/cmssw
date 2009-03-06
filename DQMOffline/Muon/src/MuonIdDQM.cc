#include "DQMOffline/Muon/interface/MuonIdDQM.h"

MuonIdDQM::MuonIdDQM(const edm::ParameterSet& iConfig)
{
   inputMuonCollection_ = iConfig.getParameter<edm::InputTag>("inputMuonCollection");
   inputDTRecSegment4DCollection_ = iConfig.getParameter<edm::InputTag>("inputDTRecSegment4DCollection");
   inputCSCSegmentCollection_ = iConfig.getParameter<edm::InputTag>("inputCSCSegmentCollection");
   useTrackerMuons_ = iConfig.getUntrackedParameter<bool>("useTrackerMuons");
   useGlobalMuons_ = iConfig.getUntrackedParameter<bool>("useGlobalMuons");
   baseFolder_ = iConfig.getUntrackedParameter<std::string>("baseFolder");

   dbe_ = 0;
   dbe_ = edm::Service<DQMStore>().operator->();
}

MuonIdDQM::~MuonIdDQM() {}

void 
MuonIdDQM::beginJob(const edm::EventSetup&)
{
   // trackerMuon == 0; globalMuon == 1
   for (unsigned int i = 0; i < 2; i++) {
      if ((i == 0 && ! useTrackerMuons_) || (i == 1 && ! useGlobalMuons_)) continue;
      if (i == 0) dbe_->setCurrentFolder(baseFolder_+"/TrackerMuons");
      if (i == 1) dbe_->setCurrentFolder(baseFolder_+"/GlobalMuons");

      hNumChambers[i] = dbe_->book1D("hNumChambers", "Number of Chambers", 11, -0.5, 10.5);
      hNumMatches[i] = dbe_->book1D("hNumMatches", "Number of Matches", 11, -0.5, 10.5);
   }

   if (useTrackerMuons_) {
      dbe_->setCurrentFolder(baseFolder_+"/TrackerMuons");

      char name[100], title[200];

      // by station
      for(int station = 0; station < 4; ++station)
      {
         sprintf(name, "hDT%iNumSegments", station+1);
         sprintf(title, "DT Station %i Number of Segments (No Arbitration)", station+1);
         hDTNumSegments[station] = dbe_->book1D(name, title, 11, -0.5, 10.5);

         sprintf(name, "hDT%iDx", station+1);
         sprintf(title, "DT Station %i Delta X", station+1);
         hDTDx[station] = dbe_->book1D(name, title, 100, -100., 100.);

         sprintf(name, "hDT%iPullx", station+1);
         sprintf(title, "DT Station %i Pull X", station+1);
         hDTPullx[station] = dbe_->book1D(name, title, 100, -20., 20.);

         if (station < 3) {
            sprintf(name, "hDT%iDy", station+1);
            sprintf(title, "DT Station %i Delta Y", station+1);
            hDTDy[station] = dbe_->book1D(name, title, 100, -150., 150.);

            sprintf(name, "hDT%iPully", station+1);
            sprintf(title, "DT Station %i Pull Y", station+1);
            hDTPully[station] = dbe_->book1D(name, title, 100, -20., 20.);
         }

         sprintf(name, "hCSC%iNumSegments", station+1);
         sprintf(title, "CSC Station %i Number of Segments (No Arbitration)", station+1);
         hCSCNumSegments[station] = dbe_->book1D(name, title, 11, -0.5, 10.5);

         sprintf(name, "hCSC%iDx", station+1);
         sprintf(title, "CSC Station %i Delta X", station+1);
         hCSCDx[station] = dbe_->book1D(name, title, 100, -50., 50.);

         sprintf(name, "hCSC%iPullx", station+1);
         sprintf(title, "CSC Station %i Pull X", station+1);
         hCSCPullx[station] = dbe_->book1D(name, title, 100, -20., 20.);

         sprintf(name, "hCSC%iDy", station+1);
         sprintf(title, "CSC Station %i Delta Y", station+1);
         hCSCDy[station] = dbe_->book1D(name, title, 100, -50., 50.);

         sprintf(name, "hCSC%iPully", station+1);
         sprintf(title, "CSC Station %i Pull Y", station+1);
         hCSCPully[station] = dbe_->book1D(name, title, 100, -20., 20.);
      }// station

      hSegmentIsAssociatedBool = dbe_->book1D("hSegmentIsAssociatedBool", "Segment Is Associated Boolean", 2, -0.5, 1.5);
   }
}

void
MuonIdDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   iEvent.getByLabel(inputMuonCollection_, muonCollectionH_);
   iEvent.getByLabel(inputDTRecSegment4DCollection_, dtSegmentCollectionH_);
   iEvent.getByLabel(inputCSCSegmentCollection_, cscSegmentCollectionH_);
   iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);

   for(MuonCollection::const_iterator muon = muonCollectionH_->begin();
         muon != muonCollectionH_->end(); ++muon)
   {
      // trackerMuon == 0; globalMuon == 1
      for (unsigned int i = 0; i < 2; i++) {
         if (i == 0 && (! useTrackerMuons_ || ! muon->isTrackerMuon())) continue;
         if (i == 1 && (! useGlobalMuons_ || ! muon->isGlobalMuon())) continue;

         hNumChambers[i]->Fill(muon->numberOfChambers());
         hNumMatches[i]->Fill(muon->numberOfMatches());
      }

      if (! useTrackerMuons_ || ! muon->isTrackerMuon()) continue;

      // by station
      for(int station = 0; station < 4; ++station)
      {
         hDTNumSegments[station]->Fill(muon->numberOfSegments(station+1, MuonSubdetId::DT, Muon::NoArbitration));
         hDTDx[station]->Fill(muon->dX(station+1, MuonSubdetId::DT));
         hDTPullx[station]->Fill(muon->pullX(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));

         if (station < 3) {
            hDTDy[station]->Fill(muon->dY(station+1, MuonSubdetId::DT));
            hDTPully[station]->Fill(muon->pullY(station+1, MuonSubdetId::DT, Muon::SegmentArbitration, true));
         }

         hCSCNumSegments[station]->Fill(muon->numberOfSegments(station+1, MuonSubdetId::CSC, Muon::NoArbitration));
         hCSCDx[station]->Fill(muon->dX(station+1, MuonSubdetId::CSC));
         hCSCPullx[station]->Fill(muon->pullX(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));
         hCSCDy[station]->Fill(muon->dY(station+1, MuonSubdetId::CSC));
         hCSCPully[station]->Fill(muon->pullY(station+1, MuonSubdetId::CSC, Muon::SegmentArbitration, true));
      }
   }// muon

   if (! useTrackerMuons_) return;

   for(DTRecSegment4DCollection::const_iterator segment = dtSegmentCollectionH_->begin();
         segment != dtSegmentCollectionH_->end(); ++segment)
   {
      LocalPoint  segmentLocalPosition       = segment->localPosition();
      LocalVector segmentLocalDirection      = segment->localDirection();
      LocalError  segmentLocalPositionError  = segment->localPositionError();
      LocalError  segmentLocalDirectionError = segment->localDirectionError();
      const GeomDet* segmentGeomDet = geometry_->idToDet(segment->geographicalId());
      GlobalPoint segmentGlobalPosition = segmentGeomDet->toGlobal(segment->localPosition());
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
      const GeomDet* segmentGeomDet = geometry_->idToDet(segment->geographicalId());
      GlobalPoint segmentGlobalPosition = segmentGeomDet->toGlobal(segment->localPosition());
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

void 
MuonIdDQM::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonIdDQM);
