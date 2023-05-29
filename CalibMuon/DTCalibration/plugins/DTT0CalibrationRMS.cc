/*
 *  See header file for a description of this class.
 *
 *  \author S. Bolognesi - INFN Torino
 */
#include "CalibMuon/DTCalibration/plugins/DTT0CalibrationRMS.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"

#include "CondFormats/DTObjects/interface/DTT0.h"

#include "TH1I.h"
#include "TFile.h"
#include "TKey.h"

using namespace std;
using namespace edm;
// using namespace cond;

// Constructor
DTT0CalibrationRMS::DTT0CalibrationRMS(const edm::ParameterSet& pset) : dtGeomToken_(esConsumes()) {
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  if (debug)
    edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS]Constructor called!";

  // Get the token to retrieve digis from the event
  digiToken = consumes<DTDigiCollection>(pset.getUntrackedParameter<string>("digiLabel"));

  // The root file which contain the histos per layer
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName", "DTT0PerLayer.root");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");

  theCalibWheel =
      pset.getUntrackedParameter<string>("calibWheel", "All");  //FIXME amke a vector of integer instead of a string
  if (theCalibWheel != "All") {
    stringstream linestr;
    int selWheel;
    linestr << theCalibWheel;
    linestr >> selWheel;
    edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMSPerLayer] chosen wheel " << selWheel;
  }

  // Sector/s to calibrate
  theCalibSector =
      pset.getUntrackedParameter<string>("calibSector", "All");  //FIXME amke a vector of integer instead of a string
  if (theCalibSector != "All") {
    stringstream linestr;
    int selSector;
    linestr << theCalibSector;
    linestr >> selSector;
    edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMSPerLayer] chosen sector " << selSector;
  }

  vector<string> defaultCell;
  defaultCell.push_back("None");
  cellsWithHistos = pset.getUntrackedParameter<vector<string> >("cellsWithHisto", defaultCell);
  for (vector<string>::const_iterator cell = cellsWithHistos.begin(); cell != cellsWithHistos.end(); ++cell) {
    if ((*cell) != "None") {
      stringstream linestr;
      int wheel, sector, station, sl, layer, wire;
      linestr << (*cell);
      linestr >> wheel >> sector >> station >> sl >> layer >> wire;
      wireIdWithHistos.push_back(DTWireId(wheel, station, sector, sl, layer, wire));
    }
  }

  hT0SectorHisto = nullptr;

  nevents = 0;
  eventsForLayerT0 = pset.getParameter<unsigned int>("eventsForLayerT0");
  eventsForWireT0 = pset.getParameter<unsigned int>("eventsForWireT0");
  rejectDigiFromPeak = pset.getParameter<unsigned int>("rejectDigiFromPeak");
  tpPeakWidth = pset.getParameter<double>("tpPeakWidth");
  //useReferenceWireInLayer_ = true;
  correctByChamberMean_ = pset.getParameter<bool>("correctByChamberMean");
}

// Destructor
DTT0CalibrationRMS::~DTT0CalibrationRMS() {
  if (debug)
    edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS]Destructor called!";

  theFile->Close();
}

/// Perform the real analysis
void DTT0CalibrationRMS::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  if (debug || event.id().event() % 500 == 0)
    edm::LogVerbatim("DTCalibration") << "--- [DTT0CalibrationRMS] Analysing Event: #Run: " << event.id().run()
                                      << " #Event: " << event.id().event();
  nevents++;

  // Get the digis from the event
  const Handle<DTDigiCollection>& digis = event.getHandle(digiToken);

  // Get the DT Geometry
  dtGeom = eventSetup.getHandle(dtGeomToken_);

  // Iterate through all digi collections ordered by LayerId
  DTDigiCollection::DigiRangeIterator dtLayerIt;
  for (dtLayerIt = digis->begin(); dtLayerIt != digis->end(); ++dtLayerIt) {
    // Get the iterators over the digis associated with this LayerId
    const DTDigiCollection::Range& digiRange = (*dtLayerIt).second;

    // Get the layerId
    const DTLayerId layerId = (*dtLayerIt).first;  //FIXME: check to be in the right sector

    if ((theCalibWheel != "All") && (layerId.superlayerId().chamberId().wheel() != selWheel))
      continue;
    if ((theCalibSector != "All") && (layerId.superlayerId().chamberId().sector() != selSector))
      continue;

    // Loop over all digis in the given layer
    for (DTDigiCollection::const_iterator digi = digiRange.first; digi != digiRange.second; ++digi) {
      double t0 = (*digi).countsTDC();

      //Use first bunch of events to fill t0 per layer
      if (nevents < eventsForLayerT0) {
        //Get the per-layer histo from the map
        TH1I* hT0LayerHisto = theHistoLayerMap[layerId];
        //If it doesn't exist, book it
        if (hT0LayerHisto == nullptr) {
          theFile->cd();
          hT0LayerHisto = new TH1I(getHistoName(layerId).c_str(),
                                   "T0 from pulses by layer (TDC counts, 1 TDC count = 0.781 ns)",
                                   200,
                                   t0 - 100,
                                   t0 + 100);
          if (debug)
            edm::LogVerbatim("DTCalibration") << "  New T0 per Layer Histo: " << hT0LayerHisto->GetName();
          theHistoLayerMap[layerId] = hT0LayerHisto;
        }

        //Fill the histos
        theFile->cd();
        if (hT0LayerHisto != nullptr) {
          hT0LayerHisto->Fill(t0);
        }
      }

      //Use all the remaining events to compute t0 per wire
      if (nevents > eventsForLayerT0) {
        // Get the wireId
        const DTWireId wireId(layerId, (*digi).wire());
        if (debug) {
          edm::LogVerbatim("DTCalibration")
              << "   Wire: " << wireId << "\n       time (TDC counts): " << (*digi).countsTDC();
        }

        //Fill the histos per wire for the chosen cells
        vector<DTWireId>::iterator it_wire = find(wireIdWithHistos.begin(), wireIdWithHistos.end(), wireId);
        if (it_wire != wireIdWithHistos.end()) {
          if (theHistoWireMap.find(wireId) == theHistoWireMap.end()) {
            theHistoWireMap[wireId] = new TH1I(getHistoName(wireId).c_str(),
                                               "T0 from pulses by wire (TDC counts, 1 TDC count = 0.781 ns)",
                                               7000,
                                               0,
                                               7000);
            if (debug)
              edm::LogVerbatim("DTCalibration") << "  New T0 per wire Histo: " << (theHistoWireMap[wireId])->GetName();
          }
          if (theHistoWireMap_ref.find(wireId) == theHistoWireMap_ref.end()) {
            theHistoWireMap_ref[wireId] = new TH1I((getHistoName(wireId) + "_ref").c_str(),
                                                   "T0 from pulses by wire (TDC counts, 1 TDC count = 0.781 ns)",
                                                   7000,
                                                   0,
                                                   7000);
            if (debug)
              edm::LogVerbatim("DTCalibration")
                  << "  New T0 per wire Histo: " << (theHistoWireMap_ref[wireId])->GetName();
          }

          TH1I* hT0WireHisto = theHistoWireMap[wireId];
          //Fill the histos
          theFile->cd();
          if (hT0WireHisto)
            hT0WireHisto->Fill(t0);
        }

        //Check the tzero has reasonable value
        if (abs(hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin()) - t0) > rejectDigiFromPeak) {
          if (debug)
            edm::LogVerbatim("DTCalibration") << "digi skipped because t0 per sector "
                                              << hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin());
          continue;
        }

        //Use second bunch of events to compute a t0 reference per wire
        if (nevents < (eventsForLayerT0 + eventsForWireT0)) {
          //Fill reference wire histos
          if (it_wire != wireIdWithHistos.end()) {
            TH1I* hT0WireHisto_ref = theHistoWireMap_ref[wireId];
            theFile->cd();
            if (hT0WireHisto_ref)
              hT0WireHisto_ref->Fill(t0);
          }
          if (!nDigiPerWire_ref[wireId]) {
            mK_ref[wireId] = 0;
          }
          nDigiPerWire_ref[wireId] = nDigiPerWire_ref[wireId] + 1;
          mK_ref[wireId] = mK_ref[wireId] + (t0 - mK_ref[wireId]) / nDigiPerWire_ref[wireId];
        }
        //Use last all the remaining events to compute the mean and sigma t0 per wire
        else if (nevents > (eventsForLayerT0 + eventsForWireT0)) {
          if (abs(t0 - mK_ref[wireId]) > tpPeakWidth)
            continue;
          if (!nDigiPerWire[wireId]) {
            theAbsoluteT0PerWire[wireId] = 0;
            qK[wireId] = 0;
            mK[wireId] = 0;
          }
          nDigiPerWire[wireId] = nDigiPerWire[wireId] + 1;
          theAbsoluteT0PerWire[wireId] = theAbsoluteT0PerWire[wireId] + t0;
          //theSigmaT0PerWire[wireId] = theSigmaT0PerWire[wireId] + (t0*t0);
          qK[wireId] =
              qK[wireId] + ((nDigiPerWire[wireId] - 1) * (t0 - mK[wireId]) * (t0 - mK[wireId]) / nDigiPerWire[wireId]);
          mK[wireId] = mK[wireId] + (t0 - mK[wireId]) / nDigiPerWire[wireId];
        }
      }  //end if(nevents>1000)
    }    //end loop on digi
  }      //end loop on layer

  //Use the t0 per layer histos to have an indication about the t0 position
  if (nevents == eventsForLayerT0) {
    for (map<DTLayerId, TH1I*>::const_iterator lHisto = theHistoLayerMap.begin(); lHisto != theHistoLayerMap.end();
         ++lHisto) {
      if (debug)
        edm::LogVerbatim("DTCalibration") << "Reading histogram " << (*lHisto).second->GetName() << " with mean "
                                          << (*lHisto).second->GetMean() << " and RMS " << (*lHisto).second->GetRMS();

      //Take the mean as a first t0 estimation
      if ((*lHisto).second->GetRMS() < 5.0) {
        if (hT0SectorHisto == nullptr) {
          hT0SectorHisto = new TH1D("hT0AllLayerOfSector",
                                    "T0 from pulses per layer in sector",
                                    //20, (*lHisto).second->GetMean()-100, (*lHisto).second->GetMean()+100);
                                    700,
                                    0,
                                    7000);
        }
        if (debug)
          edm::LogVerbatim("DTCalibration") << " accepted";
        hT0SectorHisto->Fill((*lHisto).second->GetMean());
      }
      //Take the mean of noise + 400ns as a first t0 estimation
      // if((*lHisto).second->GetRMS()>10.0 && ((*lHisto).second->GetRMS()<15.0)){
      // 	double t0_estim = (*lHisto).second->GetMean() + 400;
      // 	if(hT0SectorHisto == 0){
      // 	  hT0SectorHisto = new TH1D("hT0AllLayerOfSector","T0 from pulses per layer in sector",
      // 				    //20, t0_estim-100, t0_estim+100);
      // 				    700, 0, 7000);
      // 	}
      // 	if(debug)
      // 	  edm::LogVerbatim("DTCalibration")<<" accepted + 400ns";
      // 	hT0SectorHisto->Fill((*lHisto).second->GetMean() + 400);
      //       }
      //if (debug)
      //  edm::LogVerbatim("DTCalibration");

      theT0LayerMap[(*lHisto).second->GetName()] = (*lHisto).second->GetMean();
      theSigmaT0LayerMap[(*lHisto).second->GetName()] = (*lHisto).second->GetRMS();
    }
    if (!hT0SectorHisto) {
      edm::LogVerbatim("DTCalibration")
          << "[DTT0CalibrationRMS]: All the t0 per layer are still uncorrect: trying with greater number of events";
      eventsForLayerT0 = eventsForLayerT0 * 2;
      return;
    }
    if (debug)
      edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS] t0 reference for this sector "
                                        << hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin());
  }
}

void DTT0CalibrationRMS::endJob() {
  DTT0* t0sAbsolute = new DTT0();
  DTT0* t0sRelative = new DTT0();
  DTT0* t0sWRTChamber = new DTT0();

  //if(debug)
  edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMSPerLayer]Writing histos to file!";

  theFile->cd();
  theFile->WriteTObject(hT0SectorHisto);
  //hT0SectorHisto->Write();
  for (map<DTWireId, TH1I*>::const_iterator wHisto = theHistoWireMap.begin(); wHisto != theHistoWireMap.end();
       ++wHisto) {
    (*wHisto).second->Write();
  }
  for (map<DTWireId, TH1I*>::const_iterator wHisto = theHistoWireMap_ref.begin(); wHisto != theHistoWireMap_ref.end();
       ++wHisto) {
    (*wHisto).second->Write();
  }
  for (map<DTLayerId, TH1I*>::const_iterator lHisto = theHistoLayerMap.begin(); lHisto != theHistoLayerMap.end();
       ++lHisto) {
    (*lHisto).second->Write();
  }

  //if(debug)
  edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS] Compute and store t0 and sigma per wire";

  for (map<DTWireId, double>::const_iterator wiret0 = theAbsoluteT0PerWire.begin();
       wiret0 != theAbsoluteT0PerWire.end();
       ++wiret0) {
    if (nDigiPerWire[(*wiret0).first]) {
      double t0 = (*wiret0).second / nDigiPerWire[(*wiret0).first];

      theRelativeT0PerWire[(*wiret0).first] = t0 - hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin());

      //theSigmaT0PerWire[(*wiret0).first] = sqrt((theSigmaT0PerWire[(*wiret0).first] / nDigiPerWire[(*wiret0).first]) - t0*t0);
      theSigmaT0PerWire[(*wiret0).first] = sqrt(qK[(*wiret0).first] / nDigiPerWire[(*wiret0).first]);

      edm::LogVerbatim("DTCalibration") << "Wire " << (*wiret0).first << " has t0 " << t0 << "(absolute) "
                                        << theRelativeT0PerWire[(*wiret0).first] << "(relative)"
                                        << "    sigma " << theSigmaT0PerWire[(*wiret0).first];

      t0sAbsolute->set((*wiret0).first, t0, theSigmaT0PerWire[(*wiret0).first], DTTimeUnits::counts);
    } else {
      edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS] ERROR: no digis in wire " << (*wiret0).first;
      abort();
    }
  }

  if (correctByChamberMean_) {
    ///Loop on superlayer to correct between even-odd layers (2 different test pulse lines!)
    // Get all the sls from the setup
    const vector<const DTSuperLayer*> superLayers = dtGeom->superLayers();
    // Loop over all SLs
    for (vector<const DTSuperLayer*>::const_iterator sl = superLayers.begin(); sl != superLayers.end(); sl++) {
      //Compute mean for odd and even superlayers
      double oddLayersMean = 0;
      double evenLayersMean = 0;
      double oddLayersDen = 0;
      double evenLayersDen = 0;
      for (map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
           wiret0 != theRelativeT0PerWire.end();
           ++wiret0) {
        if ((*wiret0).first.layerId().superlayerId() == (*sl)->id()) {
          if (debug)
            edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS] Superlayer " << (*sl)->id() << "layer "
                                              << (*wiret0).first.layerId().layer() << " with " << (*wiret0).second;
          if (((*wiret0).first.layerId().layer()) % 2) {
            oddLayersMean = oddLayersMean + (*wiret0).second;
            oddLayersDen++;
          } else {
            evenLayersMean = evenLayersMean + (*wiret0).second;
            evenLayersDen++;
          }
        }
      }
      oddLayersMean = oddLayersMean / oddLayersDen;
      evenLayersMean = evenLayersMean / evenLayersDen;
      //if(debug && oddLayersMean)
      edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS] Relative T0 mean for  odd layers " << oddLayersMean
                                        << "  even layers" << evenLayersMean;

      //Compute sigma for odd and even superlayers
      double oddLayersSigma = 0;
      double evenLayersSigma = 0;
      for (map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
           wiret0 != theRelativeT0PerWire.end();
           ++wiret0) {
        if ((*wiret0).first.layerId().superlayerId() == (*sl)->id()) {
          if (((*wiret0).first.layerId().layer()) % 2) {
            oddLayersSigma = oddLayersSigma + ((*wiret0).second - oddLayersMean) * ((*wiret0).second - oddLayersMean);
          } else {
            evenLayersSigma =
                evenLayersSigma + ((*wiret0).second - evenLayersMean) * ((*wiret0).second - evenLayersMean);
          }
        }
      }
      oddLayersSigma = oddLayersSigma / oddLayersDen;
      evenLayersSigma = evenLayersSigma / evenLayersDen;
      oddLayersSigma = sqrt(oddLayersSigma);
      evenLayersSigma = sqrt(evenLayersSigma);

      //if(debug && oddLayersMean)
      edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS] Relative T0 sigma for  odd layers " << oddLayersSigma
                                        << "  even layers" << evenLayersSigma;

      //Recompute the mean for odd and even superlayers discarding fluctations
      double oddLayersFinalMean = 0;
      double evenLayersFinalMean = 0;
      for (map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
           wiret0 != theRelativeT0PerWire.end();
           ++wiret0) {
        if ((*wiret0).first.layerId().superlayerId() == (*sl)->id()) {
          if (((*wiret0).first.layerId().layer()) % 2) {
            if (abs((*wiret0).second - oddLayersMean) < (2 * oddLayersSigma))
              oddLayersFinalMean = oddLayersFinalMean + (*wiret0).second;
          } else {
            if (abs((*wiret0).second - evenLayersMean) < (2 * evenLayersSigma))
              evenLayersFinalMean = evenLayersFinalMean + (*wiret0).second;
          }
        }
      }
      oddLayersFinalMean = oddLayersFinalMean / oddLayersDen;
      evenLayersFinalMean = evenLayersFinalMean / evenLayersDen;
      //if(debug && oddLayersMean)
      edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS] Final relative T0 mean for  odd layers "
                                        << oddLayersFinalMean << "  even layers" << evenLayersFinalMean;

      for (map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
           wiret0 != theRelativeT0PerWire.end();
           ++wiret0) {
        if ((*wiret0).first.layerId().superlayerId() == (*sl)->id()) {
          double t0 = -999;
          if (((*wiret0).first.layerId().layer()) % 2)
            t0 = (*wiret0).second + (evenLayersFinalMean - oddLayersFinalMean);
          else
            t0 = (*wiret0).second;

          edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS] Wire " << (*wiret0).first << " has t0 "
                                            << (*wiret0).second << " (relative, after even-odd layer corrections)  "
                                            << "    sigma " << theSigmaT0PerWire[(*wiret0).first];

          //Store the results into DB
          t0sRelative->set((*wiret0).first, t0, theSigmaT0PerWire[(*wiret0).first], DTTimeUnits::counts);
        }
      }
    }

    ///Change t0 absolute reference -> from sector peak to chamber average
    edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS]Computing relative t0 wrt to chamber average";
    //Compute the reference for each chamber
    map<DTChamberId, double> sumT0ByChamber;
    map<DTChamberId, int> countT0ByChamber;
    for (DTT0::const_iterator tzero = t0sRelative->begin(); tzero != t0sRelative->end(); ++tzero) {
      int channelId = tzero->channelId;
      if (channelId == 0)
        continue;
      DTWireId wireId(channelId);
      DTChamberId chamberId(wireId.chamberId());
      //sumT0ByChamber[chamberId] = sumT0ByChamber[chamberId] + tzero->t0mean;
      // @@@ better DTT0 usage
      float t0mean_f;
      float t0rms_f;
      t0sRelative->get(wireId, t0mean_f, t0rms_f, DTTimeUnits::counts);
      sumT0ByChamber[chamberId] = sumT0ByChamber[chamberId] + t0mean_f;
      // @@@ NEW DTT0 END
      countT0ByChamber[chamberId]++;
    }

    //Change reference for each wire and store the new t0s in the new map
    for (DTT0::const_iterator tzero = t0sRelative->begin(); tzero != t0sRelative->end(); ++tzero) {
      int channelId = tzero->channelId;
      if (channelId == 0)
        continue;
      DTWireId wireId(channelId);
      DTChamberId chamberId(wireId.chamberId());
      //double t0mean = (tzero->t0mean) - (sumT0ByChamber[chamberId]/countT0ByChamber[chamberId]);
      //double t0rms = tzero->t0rms;
      // @@@ better DTT0 usage
      float t0mean_f;
      float t0rms_f;
      t0sRelative->get(wireId, t0mean_f, t0rms_f, DTTimeUnits::counts);
      double t0mean = t0mean_f - (sumT0ByChamber[chamberId] / countT0ByChamber[chamberId]);
      double t0rms = t0rms_f;
      // @@@ NEW DTT0 END
      t0sWRTChamber->set(wireId, t0mean, t0rms, DTTimeUnits::counts);
      edm::LogVerbatim("DTCalibration") << "Changing t0 of wire " << wireId << " from " << t0mean_f << " to " << t0mean;
    }
  }

  ///Write the t0 map into DB
  if (debug)
    edm::LogVerbatim("DTCalibration") << "[DTT0CalibrationRMS]Writing values in DB!";
  // FIXME: to be read from cfg?
  string t0Record = "DTT0Rcd";
  // Write the t0 map to DB
  if (correctByChamberMean_)
    DTCalibDBUtils::writeToDB(t0Record, t0sWRTChamber);
  else
    DTCalibDBUtils::writeToDB(t0Record, t0sAbsolute);
}

string DTT0CalibrationRMS::getHistoName(const DTWireId& wId) const {
  string histoName = "Ch_" + std::to_string(wId.wheel()) + "_" + std::to_string(wId.station()) + "_" +
                     std::to_string(wId.sector()) + "_SL" + std::to_string(wId.superlayer()) + "_L" +
                     std::to_string(wId.layer()) + "_W" + std::to_string(wId.wire()) + "_hT0Histo";
  return histoName;
}

string DTT0CalibrationRMS::getHistoName(const DTLayerId& lId) const {
  string histoName = "Ch_" + std::to_string(lId.wheel()) + "_" + std::to_string(lId.station()) + "_" +
                     std::to_string(lId.sector()) + "_SL" + std::to_string(lId.superlayer()) + "_L" +
                     std::to_string(lId.layer()) + "_hT0Histo";
  return histoName;
}
