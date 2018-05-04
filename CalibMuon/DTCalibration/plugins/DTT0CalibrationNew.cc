/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/05/11 17:17:17 $
 *  $Revision: 1.6 $
 *  \author S. Bolognesi - INFN Torino
 *  06/08/2008 Mofified by Antonio.Vilela.Pereira@cern.ch
 */

#include "CalibMuon/DTCalibration/plugins/DTT0CalibrationNew.h"
#include "CalibMuon/DTCalibration/interface/DTCalibDBUtils.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"

#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "CondFormats/DTObjects/interface/DTT0.h"

#include <CondFormats/DTObjects/interface/DTTtrig.h>
#include <CondFormats/DataRecord/interface/DTTtrigRcd.h>

#include "TKey.h"
#include "TF1.h"

#include <assert.h>

using namespace std;
using namespace edm;

// Constructor
DTT0CalibrationNew::DTT0CalibrationNew(const edm::ParameterSet& pset) :
   debug(pset.getUntrackedParameter<bool>("debug")),
   digiToken(consumes<DTDigiCollection>(pset.getUntrackedParameter<string>("digiLabel"))),
   theFile(pset.getUntrackedParameter<string>("rootFileName", "DTT0PerLayer.root").c_str(), "RECREATE"),
   nevents(0),
   eventsForLayerT0(pset.getParameter<unsigned int>("eventsForLayerT0")),
   eventsForWireT0(pset.getParameter<unsigned int>("eventsForWireT0")),
   tpPeakWidth(pset.getParameter<double>("tpPeakWidth")),
   tpPeakWidthPerLayer(pset.getParameter<double>("tpPeakWidthPerLayer")),
   rejectDigiFromPeak(pset.getParameter<unsigned int>("rejectDigiFromPeak")), 
   hLayerPeaks("hLayerPeaks", "", 3000, 0, 3000),
   spectrum(20)

{
   // Get the debug parameter for verbose output
   if(debug) 
      cout << "[DTT0CalibrationNew]Constructor called!" << endl;

   theCalibWheel =  pset.getUntrackedParameter<string>("calibWheel", "All"); //FIXME amke a vector of integer instead of a string
   if(theCalibWheel != "All") {
      stringstream linestr;
      int selWheel;
      linestr << theCalibWheel;
      linestr >> selWheel;
      cout << "[DTT0CalibrationNewPerLayer] chosen wheel " << selWheel << endl;
   }

   // Sector/s to calibrate
   theCalibSector =  pset.getUntrackedParameter<string>("calibSector", "All"); //FIXME amke a vector of integer instead of a string
   if(theCalibSector != "All") {
      stringstream linestr;
      int selSector;
      linestr << theCalibSector;
      linestr >> selSector;
      cout << "[DTT0CalibrationNewPerLayer] chosen sector " << selSector << endl;
   }

   vector<string> defaultCell;
   const auto& cellsWithHistos = pset.getUntrackedParameter<vector<string> >("cellsWithHisto", defaultCell);
   for(const auto& cell : cellsWithHistos){
      stringstream linestr;
      int wheel, sector, station, sl, layer, wire;
      linestr << cell;
      linestr >> wheel >> sector >> station >> sl >> layer >> wire;
      wireIdWithHistos.push_back(DTWireId(wheel, station, sector, sl, layer, wire));
   }
}

// Destructor
DTT0CalibrationNew::~DTT0CalibrationNew(){
   if(debug) 
      cout << "[DTT0CalibrationNew]Destructor called!" << endl;

   theFile.Close();
}

/// Perform the real analysis
void DTT0CalibrationNew::analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {
   nevents++;

   // Get the digis from the event
   Handle<DTDigiCollection> digis; 
   event.getByToken(digiToken, digis);

   // Get the DT Geometry
   if (nevents == 1)
      eventSetup.get<MuonGeometryRecord>().get(dtGeom);

   //const auto begin = *(digis->begin());
   //const auto end = *(digis->end());
   //std::cout << __LINE__ << " " << &(begin) << " " << &(end) << " " << sizeof(begin) << " " << typeid(begin).name() << std::endl;
   //std::cout << __LINE__ << " " << &(*begin) << " " << &(*end) << std::endl;
   //std::cout << __LINE__ << " " << &(*(digis->begin())) << " " << &(*(digis->end())) << std::endl;
   //for (; begin != end; begin++)
   //   std::cout << __LINE__ << std::endl;
   // Iterate through all digi collections ordered by LayerId   
   for (const auto& digis_per_layer : *digis)
   {
      //std::cout << __LINE__ << std::endl;
      // Get the iterators over the digis associated with this LayerId
      const DTDigiCollection::Range& digiRange = digis_per_layer.second;

      // Get the layerId
      const DTLayerId layerId = digis_per_layer.first;
      //const DTChamberId chamberId = layerId.superlayerId().chamberId();

      if((theCalibWheel != "All") && (layerId.superlayerId().chamberId().wheel() != selWheel))
         continue;
      if((theCalibSector != "All") && (layerId.superlayerId().chamberId().sector() != selSector))
         continue;

      // Loop over all digis in the given layer
      for (DTDigiCollection::const_iterator digi = digiRange.first;
            digi != digiRange.second;
            ++digi)
      {
         const double t0 = digi->countsTDC();
         const DTWireId wireIdtmp(layerId, (*digi).wire());
         all_wires.insert(wireIdtmp.wire());

         // Use first bunch of events to fill t0 per layer
         if(nevents <= eventsForLayerT0){
            // If it doesn't exist, book it
            if(not theHistoLayerMap.count(layerId)){
               theHistoLayerMap[layerId] = TH1I(getHistoName(layerId).c_str(),
                     "T0 from pulses by layer (TDC counts, 1 TDC count = 0.781 ns)",
                     3000,
                     0,
                     3000
                     );
               if(debug)
                  cout << "  New T0 per Layer Histo: " << theHistoLayerMap[layerId].GetName() << endl;
            }
            theHistoLayerMap[layerId].Fill(t0);
         }

         // Use all the remaining events to compute t0 per wire
         if(nevents>eventsForLayerT0){
            // Get the wireId
            const DTWireId wireId(layerId, (*digi).wire());
            if(debug) {
               cout << "   Wire: " << wireId << endl
                  << "       time (TDC counts): " << (*digi).countsTDC()<< endl;
            }   

            //Fill the histos per wire for the chosen cells
            //if (std::find(wireIdWithHistos.begin(),wireIdWithHistos.end(),wireId) != wireIdWithHistos.end())
            if (std::find(layerIdWithWireHistos.begin(), layerIdWithWireHistos.end(), layerId) != layerIdWithWireHistos.end() or
                std::find(wireIdWithHistos.begin(),wireIdWithHistos.end(),wireId) != wireIdWithHistos.end())
            {
               //If it doesn't exist, book it
               if(theHistoWireMap.count(wireId) == 0){
                  theHistoWireMap[wireId] = TH1I(getHistoName(wireId).c_str(),
                        "T0 from pulses by wire (TDC counts, 1 TDC count = 0.781 ns)",
                        7000,
                        0,
                        7000
                        );
                  if(debug)
                     cout << "  New T0 per wire Histo: " << theHistoWireMap[wireId].GetName() << endl;
               }
               theHistoWireMap[wireId].Fill(t0);
            }

            //Select per layer
            if(fabs(theTPPeakMap[layerId] - t0) > rejectDigiFromPeak){
               if(debug)
                  cout<<"digi skipped because t0 too far from peak " << theTPPeakMap[layerId] << endl;
               continue;     
            }

            //Find to ref. per chamber
            /*
               if (not theCountT0ByChamber[chamberId])
               {
               theSumT0ByChamber[chamberId] = 0;
               theCountT0ByChamber[chamberId] = 0;
               theMeanT0ByChamber[chamberId] = 0;
               theSigmaT0ByChamber[chamberId] = 0;
               }
               theCountT0ByChamber[chamberId]++;
               theSumT0ByChamber[chamberId] = theSumT0ByChamber[chamberId] + t0;
               theSigmaT0ByChamber[chamberId] += (theSumT0ByChamber[chamberId] - 1) * pow(t0 -theMeanT0ByChamber[chamberId], 2) / theSumT0ByChamber[chamberId];
               theMeanT0ByChamber[chamberId] += (t0 - theMeanT0ByChamber[chamberId]) / theCountT0ByChamber[chamberId];
               */

            //Use second bunch of events to compute a t0 reference per wire
            if(nevents <= (eventsForLayerT0 + eventsForWireT0)){
               if(!nDigiPerWire_ref[wireId]){
                  mK_ref[wireId] = 0;
               }
               nDigiPerWire_ref[wireId] = nDigiPerWire_ref[wireId] + 1;
               mK_ref[wireId] = mK_ref[wireId] + (t0-mK_ref[wireId])/nDigiPerWire_ref[wireId];
            }
            //Use last all the remaining events to compute the mean and sigma t0 per wire
            else if(nevents > (eventsForLayerT0 + eventsForWireT0)){
               if(abs(t0-mK_ref[wireId]) > tpPeakWidth)
                  continue;
               if(!nDigiPerWire[wireId]){
                  theAbsoluteT0PerWire[wireId] = 0;
                  qK[wireId] = 0;
                  mK[wireId] = 0;
               }
               nDigiPerWire[wireId] = nDigiPerWire[wireId] + 1;
               theAbsoluteT0PerWire[wireId] = theAbsoluteT0PerWire[wireId] + t0;
               //theSigmaT0PerWire[wireId] = theSigmaT0PerWire[wireId] + (t0*t0);
               qK[wireId] = qK[wireId] + ((nDigiPerWire[wireId]-1)*(t0-mK[wireId])*(t0-mK[wireId])/nDigiPerWire[wireId]);
               mK[wireId] = mK[wireId] + (t0-mK[wireId])/nDigiPerWire[wireId];
            }
         }//end if(nevents>1000)
      }//end loop on digi
   }//end loop on layer

   //Use the t0 per layer histos to have an indication about the t0 position 
   if(nevents == eventsForLayerT0){
      for(const auto& lHisto : theHistoLayerMap)
      {
         const auto& layerId = lHisto.first;
         const auto& hist = lHisto.second;
         if(debug)
            cout << "Reading histogram " << hist.GetName() << " with mean " << hist.GetMean() << " and RMS " << hist.GetRMS() << endl;

         //Find peaks
         int npeaks = spectrum.Search(&hist, (tpPeakWidthPerLayer / 2.), "", 0.3);

         double *peaks = spectrum.GetPositionX();
         //Put in a std::vector<float>
         vector<double> peakMeans(peaks, peaks + npeaks);
         //Sort the peaks in ascending order
         sort(peakMeans.begin(), peakMeans.end());

         if (peakMeans.empty())
         {
            theTPPeakMap[layerId] = hist.GetMaximumBin();
            std::cout << "No peaks found by peakfinder in layer " << layerId << ". Taking maximum bin at " << theTPPeakMap[layerId] << ". Please check!" << std::endl;
            layerIdWithWireHistos.push_back(layerId);
         }
         else if (fabs(hist.GetXaxis()->FindBin(peakMeans.front()) - hist.GetXaxis()->FindBin(peakMeans.back())) < rejectDigiFromPeak)
         {
            theTPPeakMap[layerId] = peakMeans[peakMeans.size() / 2];
         }
         else
         {
            bool peak_set = false;
            for (const auto& peak : peakMeans)
            {
               // Skip if at low edge
               if (peak - tpPeakWidthPerLayer <= 0)
                  continue;
               // Get integral of peak
               double sum = 0;
               for (int ibin=peak - tpPeakWidthPerLayer; ibin<peak + tpPeakWidthPerLayer; ibin++)
               {
                  sum += hist.GetBinContent(ibin);
               }
               // Skip if peak too small
               if (sum < hist.GetMaximum() / 2)
                  continue;

               // Passed all cuts
               theTPPeakMap[layerId] = peak;
               peak_set = true;
               break;
            }
            if (peak_set)
            {
               std::cout << "Peaks to far away from each other in layer " << layerId << ". Maybe cross talk? Taking first good peak at " << theTPPeakMap[layerId] << ". Please check!" << std::endl;
               layerIdWithWireHistos.push_back(layerId);
            }
            else
            {
               theTPPeakMap[layerId] = hist.GetMaximumBin();
               std::cout << "Peaks to far away from each other in layer " << layerId << " and no good peak found. Taking maximum bin at " << theTPPeakMap[layerId] << ". Please check!" << std::endl;
               layerIdWithWireHistos.push_back(layerId);
            }
         }
         if (peakMeans.size() > 5)
         {
            std::cout << "Found more than 5 peaks in layer " << layerId << ". Please check!" << std::endl;
            if (std::find(layerIdWithWireHistos.begin(), layerIdWithWireHistos.end(), layerId) == layerIdWithWireHistos.end())
               layerIdWithWireHistos.push_back(layerId);
         }
         // Check for noise
         int nspikes = 0;
         for (int ibin=0; ibin<hist.GetNbinsX(); ibin++)
         {
            if (hist.GetBinContent(ibin + 1) > hist.GetMaximum() * 0.001)
               nspikes++;
         }
         if (nspikes > 50)
         {
            std::cout << "Found a lot of (>50) small spikes in layer " << layerId << ". Please check if all wires are functioning as expected!" << std::endl;
            if (std::find(layerIdWithWireHistos.begin(), layerIdWithWireHistos.end(), layerId) == layerIdWithWireHistos.end())
               layerIdWithWireHistos.push_back(layerId);
         }
         //if (peakMeans.empty())
         //   std::cout << "PEAK IN LAYER (EMPTY) " << layerId << " " << theTPPeakMap[layerId] << std::endl;
         //else if (peakMeans.size() > 5)
         //   std::cout << "PEAK IN LAYER (>5) " << layerId << " " << theTPPeakMap[layerId] << std::endl;
         //else if (peakMeans.back() > 1000)
         //   std::cout << "PEAK IN LAYER (TO HIGH) " << layerId << " " << theTPPeakMap[layerId] << std::endl;

         hLayerPeaks.Fill(theTPPeakMap[layerId]);

      }
   }
}

void DTT0CalibrationNew::endJob() {

   std::cout << "WIRES" << std::endl;
   for (auto& wire : all_wires)
      std::cout << wire << std::endl;
   std::cout << "Analyzed " << nevents << " events" << std::endl;

   DTT0* t0sWRTChamber = new DTT0();

   if(debug) 
      cout << "[DTT0CalibrationNewPerLayer]Writing histos to file!" << endl;

   theFile.cd();
   //hT0SectorHisto->Write();
   hLayerPeaks.Write();
   for(const auto& wHisto : theHistoWireMap){
      wHisto.second.Write(); 
   }
   for(const auto& lHisto : theHistoLayerMap ) {
      lHisto.second.Write(); 
   }  

   if(debug) 
      cout << "[DTT0CalibrationNew] Compute and store t0 and sigma per wire" << endl;

   // Calculate uncertainties per wire (counting experiment)
   for (auto& wiret0 : theAbsoluteT0PerWire)
   {
      auto& wireId = wiret0.first;
      if (nDigiPerWire[wireId] > 1 )
         theSigmaT0PerWire[wireId] = qK[wireId] / (nDigiPerWire[wireId] - 1);
      else
         theSigmaT0PerWire[wireId] = 999.; // Only one measurement: uncertainty -> infinity
      // syst uncert
      //theSigmaT0PerWire[wireId] += pow(0.5, 2));
      // Every time the same measurement. Use Laplace estimator as estimation how propable it is to measure another value due to limited size of sample
      if (theSigmaT0PerWire[wireId] == 0)
      {
         std::cout << "NOERROR " << wireId << " " << nDigiPerWire[wireId] << " " <<  pow(1. / (nDigiPerWire[wireId] + 1), 2) << std::endl;
         theSigmaT0PerWire[wireId] += pow(1. / (nDigiPerWire[wireId] + 1), 2);
      }
      std::cout << "START " << wireId << " " << wiret0.second / nDigiPerWire[wireId] << " " << sqrt(theSigmaT0PerWire[wireId]) << " " << nDigiPerWire[wireId] << std::endl;
   }

   // function to calculate weighted means
   /*auto weighted_mean_function = [] (const std::list<double>& values, const std::list<double>& sigmas)
   {
      auto value_iterator = values.begin();
      auto sigma_iterator = sigmas.begin();
      double weighted_mean = 0;
      double weighted_sigma = 0;
      assert(values.size() == sigmas.size());
      for (; value_iterator != values.end(); value_iterator++, sigma_iterator++)
      {
         double weight = 1. / pow(*sigma_iterator, 2);
         weighted_mean += (*value_iterator * weight);
         weighted_sigma += weight;
      }
      weighted_mean /= weighted_sigma;
      weighted_sigma = 1. / sqrt(weighted_sigma);
      return std::make_pair(weighted_mean, weighted_sigma);
   };*/

   // function to calculate unweighted means
   auto unweighted_mean_function = [] (const std::list<double>& values, const std::list<double>& sigmas)
   {
      double mean = 0;
      for (auto& value : values)
      {
         mean += value;
      }
      mean /= values.size();

      double uncertainty = 0;
      for (auto& value : values)
      {
         uncertainty += pow(value - mean, 2);
      }
      uncertainty /= values.size();
      uncertainty = sqrt(uncertainty);
      return std::make_pair(mean, uncertainty);
   };

   // correct for odd-even effect in each super layer
   std::map<DTSuperLayerId, std::pair<double, double> > weighted_mean_sigma_even;
   std::map<DTSuperLayerId, std::pair<double, double> > weighted_mean_sigma_odd;
   for (const auto& superlayer : dtGeom->superLayers())
   {
      const auto superlayer_id = superlayer->id();
      std::list<double> values_even;
      std::list<double> sigmas_even;
      std::list<double> values_odd;
      std::list<double> sigmas_odd;

      for (const auto& wiret0 : theAbsoluteT0PerWire)
      {
         const auto& wireId = wiret0.first;
         if (wireId.layerId().superlayerId() == superlayer_id)
         {
            const auto& t0 = wiret0.second / nDigiPerWire[wireId];
            if (wireId.layerId().layer() % 2)
            {
               values_odd.push_back(t0);
               sigmas_odd.push_back(sqrt(theSigmaT0PerWire[wireId]));
            }
            else
            {
               values_even.push_back(t0);
               sigmas_even.push_back(sqrt(theSigmaT0PerWire[wireId]));
            }
         }
      }
      // get mean and uncertainty
      weighted_mean_sigma_even.emplace(superlayer_id, unweighted_mean_function(values_even, sigmas_even));
      weighted_mean_sigma_odd.emplace(superlayer_id, unweighted_mean_function(values_odd, sigmas_odd));
      std::cout << "EVEN " << superlayer_id << " " << weighted_mean_sigma_even[superlayer_id].first << " +- " << weighted_mean_sigma_even[superlayer_id].second << std::endl;
      std::cout << "ODD " << superlayer_id << " " << weighted_mean_sigma_odd[superlayer_id].first << " +- " << weighted_mean_sigma_odd[superlayer_id].second << std::endl;
   }

   // filter outliers
   for (const auto& superlayer : dtGeom->superLayers())
   {
      const auto superlayer_id = superlayer->id();
      std::list<double> values_even;
      std::list<double> sigmas_even;
      std::list<double> values_odd;
      std::list<double> sigmas_odd;

      for (const auto& wiret0 : theAbsoluteT0PerWire)
      {
         const auto& wireId = wiret0.first;
         if (wireId.layerId().superlayerId() == superlayer_id)
         {
            const auto& t0 = wiret0.second / nDigiPerWire[wireId];
            if (wireId.layerId().layer() % 2 and abs(t0 - weighted_mean_sigma_odd[superlayer_id].first) < 2*weighted_mean_sigma_odd[superlayer_id].second)
            {
               values_odd.push_back(t0);
               sigmas_odd.push_back(sqrt(theSigmaT0PerWire[wireId]));
            }
            else 
            {
               if (abs(t0 - weighted_mean_sigma_even[superlayer_id].first) < 2*weighted_mean_sigma_even[superlayer_id].second)
               {
                  values_even.push_back(t0);
                  sigmas_even.push_back(sqrt(theSigmaT0PerWire[wireId]));
               }
            }
         }
      }
      // get mean and uncertainty
      weighted_mean_sigma_even[superlayer_id] = unweighted_mean_function(values_even, sigmas_even);
      weighted_mean_sigma_odd[superlayer_id] = unweighted_mean_function(values_odd, sigmas_odd);
   }

   // apply correction
   for (auto& wiret0 : theAbsoluteT0PerWire)
   {
      const auto& wire_id = wiret0.first;
      const auto& superlayer_id = wiret0.first.layerId().superlayerId();
      const auto& layer = wiret0.first.layerId().layer();
      auto& t0 = wiret0.second;
      t0 /= nDigiPerWire[wire_id];
      if (not layer % 2)
         continue;
      // t0 is reference. Changing it changes the map
      t0 += weighted_mean_sigma_even[superlayer_id].first - weighted_mean_sigma_odd[superlayer_id].first;
      theSigmaT0PerWire[wire_id] += pow(weighted_mean_sigma_odd[superlayer_id].second, 2) + pow(weighted_mean_sigma_even[superlayer_id].second, 2);
      std::cout << "ODD-EVEN-CORRECTED " << wire_id << " " << t0 << " " << sqrt(theSigmaT0PerWire[wire_id]) << std::endl;
   }

   // get chamber mean
   std::map<DTChamberId, std::list<double> > values_per_chamber;
   std::map<DTChamberId, std::list<double> > sigmas_per_chamber;
   for (const auto& wire_t0 : theAbsoluteT0PerWire)
   {
      const auto& wire_id = wire_t0.first;
      const auto& chamber_id = wire_id.chamberId();
      const auto& t0 = wire_t0.second;
      values_per_chamber[chamber_id].push_back(t0);
      sigmas_per_chamber[chamber_id].push_back(sqrt(theSigmaT0PerWire[wire_id]));
   }

   std::map<DTChamberId, std::pair<double, double> > weighted_mean_per_chamber;
   for (const auto& chamber_mean : values_per_chamber)
   {
      const auto& chamber_id = chamber_mean.first;
      const auto& means = chamber_mean.second;
      const auto& sigmas = sigmas_per_chamber[chamber_id];
      weighted_mean_per_chamber.emplace(chamber_id, unweighted_mean_function(means, sigmas));
      std::cout << "CHAMBER MEAN " << chamber_id << " " << weighted_mean_per_chamber[chamber_id].first << " +- " << weighted_mean_per_chamber[chamber_id].second << std::endl;
   }

   // calculate relative values
   for (const auto& wire_t0 : theAbsoluteT0PerWire)
   {
      const auto& wire_id = wire_t0.first;
      const auto& chamber_id = wire_id.chamberId();
      const auto& t0 = wire_t0.second;
      theRelativeT0PerWire.emplace(wire_id, t0 - weighted_mean_per_chamber[chamber_id].first);
      //theSigmaT0PerWire[wire_id] += pow(weighted_mean_per_chamber[chamber_id].second, 2);
      cout << "[DTT0CalibrationNew] Wire " << wire_id << " has    t0 "<< theRelativeT0PerWire[wire_id] << " (relative, after even-odd layer corrections)  "
         << "    sigma "<< sqrt(theSigmaT0PerWire[wire_id]) <<endl;
   }

   for(const auto& wire_t0 : theRelativeT0PerWire)
   {
      const auto& wire_id = wire_t0.first;
      const auto& t0 = wire_t0.second;
      t0sWRTChamber->set(wire_id,
            t0,
            sqrt(theSigmaT0PerWire[wire_id]),
            DTTimeUnits::counts
            );
   }

   ///Write the t0 map into DB
   if(debug) 
      cout << "[DTT0CalibrationNew]Writing values in DB!" << endl;
   // FIXME: to be read from cfg?
   string t0Record = "DTT0Rcd";
   // Write the t0 map to DB
   DTCalibDBUtils::writeToDB(t0Record, t0sWRTChamber);
   delete t0sWRTChamber;

   /*
      for(map<DTWireId, double>::const_iterator wiret0 = theAbsoluteT0PerWire.begin();
      wiret0 != theAbsoluteT0PerWire.end();
      ++wiret0){
      DTChamberId chamberId = wiret0->first.chamberId();
      double t0 = wiret0->second/nDigiPerWire[wiret0->first];
   //double t0 = mK[wiret0->first];
   if (nDigiPerWire[wiret0->first] > 1 )
   theSigmaT0PerWire[wiret0->first] = sqrt(qK[wiret0->first]/(nDigiPerWire[wiret0->first] - 1));
   else
   theSigmaT0PerWire[wiret0->first] = 999.; // Only one measurement: uncertainty -> infinity
   values[chamberId].push_back(t0);
   theSigmaT0PerWire[(*wiret0).first] = sqrt(pow(theSigmaT0PerWire[(*wiret0).first], 2) + pow(0.5, 2));
   sigmas[chamberId].push_back(theSigmaT0PerWire[wiret0->first]);
   }

   for (auto chamberId_iterator=values.begin(); chamberId_iterator != values.end(); chamberId_iterator++)
   {
   const auto& chamberId = chamberId_iterator->first;
   double weight = 0;
   auto value_iterator = values[chamberId].begin();
   auto sigma_iterator = sigmas[chamberId].begin();
   double tmp = 0;
   theSigmaT0ByChamber[chamberId] = 0;
   if (values[chamberId].size() != sigmas[chamberId].size())
   std::cout << "FUCK MY LIFE" << std::endl;
   for (; value_iterator != values[chamberId].end(); value_iterator++, sigma_iterator++)
   {
   weight = 1. / pow(*sigma_iterator, 2);
   theRefT0ByChamber[chamberId] += (*value_iterator * weight);
   theSigmaT0ByChamber[chamberId] += weight;
   tmp += *value_iterator;
   std::cout << *value_iterator << " " << *sigma_iterator << std::endl;
   }
   //std::cout << tmp << " " << values[chamberId].size() << " " << sigmas[chamberId].size() << " " << theCountT0ByChamber[chamberId] << " " << theSumT0ByChamber[chamberId] << std::endl;
   theRefT0ByChamber[chamberId] /= theSigmaT0ByChamber[chamberId];
   theSigmaT0ByChamber[chamberId] = 1. / sqrt(theSigmaT0ByChamber[chamberId]);
   std::cout << "PER CHAMBER " << chamberId << " " << theCountT0ByChamber[chamberId] << " " <<  theRefT0ByChamber[chamberId] << " " << theSigmaT0ByChamber[chamberId] << " " << tmp / values[chamberId].size() << std::endl;
   }

   for(map<DTWireId, double>::const_iterator wiret0 = theAbsoluteT0PerWire.begin();
   wiret0 != theAbsoluteT0PerWire.end();
   ++wiret0){
   if(nDigiPerWire[(*wiret0).first]){
   double t0 = (*wiret0).second/nDigiPerWire[(*wiret0).first];
   DTChamberId chamberId = ((*wiret0).first).chamberId();
   //theRelativeT0PerWire[(*wiret0).first] = t0 - hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin());
   theRelativeT0PerWire[(*wiret0).first] = t0 - theRefT0ByChamber[chamberId];        
   cout<<"Wire "<<(*wiret0).first<<" has    t0 "<<t0<<"(absolute) "<<theRelativeT0PerWire[(*wiret0).first]<<"(relative)";

   //theSigmaT0PerWire[(*wiret0).first] = sqrt((theSigmaT0PerWire[(*wiret0).first] / nDigiPerWire[(*wiret0).first]) - t0*t0);
   //if (nDigiPerWire[(*wiret0).first] > 1 )
   //   theSigmaT0PerWire[(*wiret0).first] = sqrt(qK[(*wiret0).first]/(nDigiPerWire[(*wiret0).first] - 1));
   //else
   //   theSigmaT0PerWire[(*wiret0).first] = 999.; // Only one measurement: uncertainty -> infinity
   // Systematic uncertainty: Channel Width
   theSigmaT0PerWire[(*wiret0).first] = sqrt(pow(theSigmaT0PerWire[(*wiret0).first], 2) + pow(theSigmaT0ByChamber[chamberId], 2));
   //theSigmaT0PerWire[(*wiret0).first] = sqrt(pow(theSigmaT0PerWire[(*wiret0).first], 2) + pow(0.5, 2));
   cout<<"    sigma "<<theSigmaT0PerWire[(*wiret0).first]<<endl;
   }
   else{
   cout<<"[DTT0CalibrationNew] ERROR: no digis in wire "<<(*wiret0).first<<endl;
   abort();
   }
   }

   ///Loop on superlayer to correct between even-odd layers (2 different test pulse lines!)
   // Get all the sls from the setup
   const vector<const DTSuperLayer*> superLayers = dtGeom->superLayers();     
   // Loop over all SLs
   double oddLayersFinalMean=0; 
   double evenLayersFinalMean=0;
   for(vector<const DTSuperLayer*>::const_iterator  sl = superLayers.begin();
         sl != superLayers.end(); sl++) {


      //Compute mean for odd and even superlayers
      double oddLayersMean=0;
      double evenLayersMean=0; 
      double oddLayersDen=0;
      double evenLayersDen=0;
      for(map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
            wiret0 != theRelativeT0PerWire.end();
            ++wiret0){
         if((*wiret0).first.layerId().superlayerId() == (*sl)->id()){
            if(debug)
               cout<<"[DTT0CalibrationNew] Superlayer "<<(*sl)->id()
                  <<"layer " <<(*wiret0).first.layerId().layer()<<" with "<<(*wiret0).second<<endl;
            if(((*wiret0).first.layerId().layer()) % 2){
               oddLayersMean = oddLayersMean + (*wiret0).second;
               oddLayersDen++;
            }
            else{
               evenLayersMean = evenLayersMean + (*wiret0).second;
               evenLayersDen++;
            }
         }
      }
      oddLayersMean = oddLayersMean/oddLayersDen;
      evenLayersMean = evenLayersMean/evenLayersDen;
      if(debug && oddLayersMean)
         cout<<"[DTT0CalibrationNew] Relative T0 mean for  odd layers "<<oddLayersMean<<"  even layers"<<evenLayersMean<<endl;

      //Compute sigma for odd and even superlayers
      double oddLayersSigma=0;
      double evenLayersSigma=0;
      for(map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
            wiret0 != theRelativeT0PerWire.end();
            ++wiret0){
         if((*wiret0).first.layerId().superlayerId() == (*sl)->id()){
            if(((*wiret0).first.layerId().layer()) % 2){
               oddLayersSigma = oddLayersSigma + ((*wiret0).second - oddLayersMean) * ((*wiret0).second - oddLayersMean);
            }
            else{
               evenLayersSigma = evenLayersSigma + ((*wiret0).second - evenLayersMean) * ((*wiret0).second - evenLayersMean);
            }
         }
      }
      oddLayersSigma = oddLayersSigma/oddLayersDen;
      evenLayersSigma = evenLayersSigma/evenLayersDen;
      oddLayersSigma = sqrt(oddLayersSigma);
      evenLayersSigma = sqrt(evenLayersSigma);

      if(debug && evenLayersMean)
         cout<<"[DTT0CalibrationNew] Relative T0 sigma for  odd layers "<<oddLayersSigma<<"  even layers"<<evenLayersSigma<<endl;

      //Recompute the mean for odd and even superlayers discarding fluctations
      oddLayersDen=0;
      evenLayersDen=0;
      for(map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
            wiret0 != theRelativeT0PerWire.end();
            ++wiret0){
         if((*wiret0).first.layerId().superlayerId() == (*sl)->id()){
            if(((*wiret0).first.layerId().layer()) % 2){
               if(abs((*wiret0).second - oddLayersMean) < (2*oddLayersSigma))
               {
                  oddLayersFinalMean = oddLayersFinalMean + (*wiret0).second;
                  oddLayersDen++;
               }
            }
            else{
               if(abs((*wiret0).second - evenLayersMean) < (2*evenLayersSigma))
               {
                  evenLayersFinalMean = evenLayersFinalMean + (*wiret0).second;
                  evenLayersDen++;
               }
            }
         }
      }
      oddLayersFinalMean = oddLayersFinalMean/oddLayersDen;
      evenLayersFinalMean = evenLayersFinalMean/evenLayersDen;
   }
   for(vector<const DTSuperLayer*>::const_iterator  sl = superLayers.begin();
         sl != superLayers.end(); sl++) {
      if(debug && oddLayersFinalMean)
         cout<<"[DTT0CalibrationNew] Final relative T0 mean for  odd layers "<<oddLayersFinalMean<<"  even layers"<<evenLayersFinalMean<<endl;

      for(map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
            wiret0 != theRelativeT0PerWire.end();
            ++wiret0){
         if((*wiret0).first.layerId().superlayerId() == (*sl)->id()){
            double t0=-999;
            if(((*wiret0).first.layerId().layer()) % 2)
               t0 = (*wiret0).second + (evenLayersFinalMean - oddLayersFinalMean);
            else
               t0 = (*wiret0).second;

            //cout<<"[DTT0CalibrationNew] Wire "<<(*wiret0).first<<" has    t0 "<<t0<<" (relative, after even-odd layer corrections)  "
            //    <<"    sigma "<<theSigmaT0PerWire[(*wiret0).first]<<endl;
            //Store the results into DB
            t0s->set((*wiret0).first, t0, theSigmaT0PerWire[(*wiret0).first],DTTimeUnits::counts); 
         }
      }
   }

   ///Change t0 absolute reference -> from sector peak to chamber average
   if(debug) 
      cout << "[DTT0CalibrationNew]Computing relative t0 wrt to chamber average" << endl;
   //Compute the reference for each chamber
   map<DTChamberId,double> sumT0ByChamber;
   map<DTChamberId,int> countT0ByChamber;
   for(DTT0::const_iterator tzero = t0s->begin();
         tzero != t0s->end(); ++tzero) {
      // @@@ NEW DTT0 FORMAT
      //    DTChamberId chamberId((*tzero).first.wheelId,
      //                        (*tzero).first.stationId,
      //                        (*tzero).first.sectorId);
      //    sumT0ByChamber[chamberId] = sumT0ByChamber[chamberId] + (*tzero).second.t0mean;
      int channelId = tzero->channelId;
      if ( channelId == 0 ) continue;
      DTWireId wireId(channelId);
      DTChamberId chamberId(wireId.chamberId());
      //sumT0ByChamber[chamberId] = sumT0ByChamber[chamberId] + tzero->t0mean;
      // @@@ better DTT0 usage
      float t0mean_f;
      float t0rms_f;
      t0s->get(wireId,t0mean_f,t0rms_f,DTTimeUnits::counts);
      sumT0ByChamber[chamberId] = sumT0ByChamber[chamberId] + t0mean_f;
      // @@@ NEW DTT0 END
      countT0ByChamber[chamberId]++;
   }
   //Change reference for each wire and store the new t0s in the new map
   for(DTT0::const_iterator tzero = t0s->begin();
         tzero != t0s->end(); ++tzero) {
      // @@@ NEW DTT0 FORMAT
      //    DTChamberId chamberId((*tzero).first.wheelId,
      //                        (*tzero).first.stationId,
      //                        (*tzero).first.sectorId);
      //    double t0mean = ((*tzero).second.t0mean) - (sumT0ByChamber[chamberId]/countT0ByChamber[chamberId]);
      //    double t0rms = (*tzero).second.t0rms;
      //    DTWireId wireId((*tzero).first.wheelId,
      //                  (*tzero).first.stationId,
      //                  (*tzero).first.sectorId,
      //                  (*tzero).first.slId,
      //                  (*tzero).first.layerId,
      //                  (*tzero).first.cellId);
      int channelId = tzero->channelId;
      if ( channelId == 0 ) continue;
      DTWireId wireId( channelId );
      DTChamberId chamberId(wireId.chamberId());
      //double t0mean = (tzero->t0mean) - (sumT0ByChamber[chamberId]/countT0ByChamber[chamberId]);
      //double t0rms = tzero->t0rms;
      // @@@ better DTT0 usage
      float t0mean_f;
      float t0rms_f;
      t0s->get(wireId,t0mean_f,t0rms_f,DTTimeUnits::counts);
      double t0mean = t0mean_f - (sumT0ByChamber[chamberId]/countT0ByChamber[chamberId]);
      double t0rms = t0rms_f;
      cout<<"[DTT0CalibrationNew] Wire "<<wireId<<" has    t0 "<<t0mean<<" (relative, after even-odd layer corrections)  "
         <<"    sigma "<<t0rms<<endl;
      // @@@ NEW DTT0 END
      t0sWRTChamber->set(wireId,
            t0mean,
            t0rms,
            DTTimeUnits::counts);
      if(debug){
         //cout<<"Chamber "<<chamberId<<" has reference "<<(sumT0ByChamber[chamberId]/countT0ByChamber[chamberId]);
         //      cout<<"Changing t0 of wire "<<wireId<<" from "<<(*tzero).second.t0mean<<" to "<<t0mean<<endl;
         cout<<"Changing t0 of wire "<<wireId<<" from "<<tzero->t0mean<<" to "<<t0mean<<endl;
      }
   }

   ///Write the t0 map into DB
   if(debug) 
      cout << "[DTT0CalibrationNew]Writing values in DB!" << endl;
   // FIXME: to be read from cfg?
   string t0Record = "DTT0Rcd";
   // Write the t0 map to DB
   DTCalibDBUtils::writeToDB(t0Record, t0sWRTChamber);
   */
}

string DTT0CalibrationNew::getHistoName(const DTWireId& wId) const {
   string histoName;
   stringstream theStream;
   theStream << "Ch_" << wId.wheel() << "_" << wId.station() << "_" << wId.sector()
      << "_SL" << wId.superlayer() << "_L" << wId.layer() << "_W"<< wId.wire() <<"_hT0Histo";
   theStream >> histoName;
   return histoName;
}

string DTT0CalibrationNew::getHistoName(const DTLayerId& lId) const {
   string histoName;
   stringstream theStream;
   theStream << "Ch_" << lId.wheel() << "_" << lId.station() << "_" << lId.sector()
      << "_SL" << lId.superlayer() << "_L" << lId.layer() <<"_hT0Histo";
   theStream >> histoName;
   return histoName;
}

