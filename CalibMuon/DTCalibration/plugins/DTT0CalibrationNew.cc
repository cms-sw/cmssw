/*
 *  See header file for a description of this class.
 *
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

#include "CondFormats/DTObjects/interface/DTTtrig.h"
#include "CondFormats/DataRecord/interface/DTTtrigRcd.h"

#include "TH1I.h"
#include "TFile.h"
#include "TKey.h"
#include "TSpectrum.h"
#include "TF1.h"

using namespace std;
using namespace edm;
// using namespace cond;

// Constructor
DTT0CalibrationNew::DTT0CalibrationNew(const edm::ParameterSet& pset) {
  // Get the debug parameter for verbose output
  debug = pset.getUntrackedParameter<bool>("debug");
  if(debug) 
    cout << "[DTT0CalibrationNew]Constructor called!" << endl;

  // Get the label to retrieve digis from the event
  digiLabel = pset.getUntrackedParameter<string>("digiLabel");

  dbLabel  = pset.getUntrackedParameter<string>("dbLabel", "");

  // The root file which contain the histos per layer
  string rootFileName = pset.getUntrackedParameter<string>("rootFileName","DTT0PerLayer.root");
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
 
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
  defaultCell.push_back("None");
  cellsWithHistos = pset.getUntrackedParameter<vector<string> >("cellsWithHisto", defaultCell);
  for(vector<string>::const_iterator cell = cellsWithHistos.begin(); cell != cellsWithHistos.end(); cell++){
    if((*cell) != "None"){
      stringstream linestr;
      int wheel,sector,station,sl,layer,wire;
      linestr << (*cell);
      linestr >> wheel >> sector >> station >> sl >> layer >> wire;
      wireIdWithHistos.push_back(DTWireId(wheel,station,sector,sl,layer,wire));
    }
  }

  hT0SectorHisto=0;

  nevents=0;
  eventsForLayerT0 = pset.getParameter<unsigned int>("eventsForLayerT0");
  eventsForWireT0 = pset.getParameter<unsigned int>("eventsForWireT0");
  tpPeakWidth = pset.getParameter<double>("tpPeakWidth");
  tpPeakWidthPerLayer = pset.getParameter<double>("tpPeakWidthPerLayer");
  timeBoxWidth = pset.getParameter<unsigned int>("timeBoxWidth"); 
  rejectDigiFromPeak = pset.getParameter<unsigned int>("rejectDigiFromPeak"); 

  spectrum = new TSpectrum(5);
  retryForLayerT0 = 0;	
}

// Destructor
DTT0CalibrationNew::~DTT0CalibrationNew(){
  if(debug) 
    cout << "[DTT0CalibrationNew]Destructor called!" << endl;

  delete spectrum;
  theFile->Close();
}

 /// Perform the real analysis
void DTT0CalibrationNew::analyze(const edm::Event & event, const edm::EventSetup& eventSetup) {
  if(debug || event.id().event() % 500==0)
    cout << "--- [DTT0CalibrationNew] Analysing Event: #Run: " << event.id().run()
	 << " #Event: " << event.id().event() << endl;
  nevents++;

  // Get the digis from the event
  Handle<DTDigiCollection> digis; 
  event.getByLabel(digiLabel, digis);

  // Get the DT Geometry
  eventSetup.get<MuonGeometryRecord>().get(dtGeom);

  // Get ttrig DB
  edm::ESHandle<DTTtrig> tTrigMap;
  eventSetup.get<DTTtrigRcd>().get(dbLabel,tTrigMap);
  
  // Iterate through all digi collections ordered by LayerId   
  DTDigiCollection::DigiRangeIterator dtLayerIt;
  for (dtLayerIt = digis->begin();
       dtLayerIt != digis->end();
       ++dtLayerIt){
    // Get the iterators over the digis associated with this LayerId
    const DTDigiCollection::Range& digiRange = (*dtLayerIt).second;
  
    // Get the layerId
    const DTLayerId layerId = (*dtLayerIt).first; //FIXME: check to be in the right sector
    const DTChamberId chamberId = layerId.superlayerId().chamberId();

    if((theCalibWheel != "All") && (layerId.superlayerId().chamberId().wheel() != selWheel))
      continue;
    if((theCalibSector != "All") && (layerId.superlayerId().chamberId().sector() != selSector))
      continue;
 
    //if(debug) {
    //  cout << "Layer " << layerId<<" with "<<distance(digiRange.first, digiRange.second)<<" digi"<<endl;
    //}

    float tTrig,tTrigRMS, kFactor;
    tTrigMap->get(layerId.superlayerId(), tTrig, tTrigRMS, kFactor, DTTimeUnits::counts );
    if(debug&&(nevents <= 1)){
	cout << "  Superlayer: " << layerId.superlayerId() << endl 
	     << "            tTrig,tTrigRMS= " << tTrig << ", " << tTrigRMS << endl;
    }	

    // Loop over all digis in the given layer
    for (DTDigiCollection::const_iterator digi = digiRange.first;
	 digi != digiRange.second;
	 digi++) {
      double t0 = (*digi).countsTDC();

      //Use first bunch of events to fill t0 per layer
      if(nevents < eventsForLayerT0){
	//Get the per-layer histo from the map
	TH1I *hT0LayerHisto = theHistoLayerMap[layerId];
	//If it doesn't exist, book it
	if(hT0LayerHisto == 0){
	  theFile->cd();
	  float hT0Min = tTrig - 2*tTrigRMS;
	  float hT0Max = hT0Min + timeBoxWidth;
	  hT0LayerHisto = new TH1I(getHistoName(layerId).c_str(),
				   "T0 from pulses by layer (TDC counts, 1 TDC count = 0.781 ns)",
				   timeBoxWidth,hT0Min,hT0Max);
	  if(debug)
	    cout << "  New T0 per Layer Histo: " << hT0LayerHisto->GetName() << endl;
	  theHistoLayerMap[layerId] = hT0LayerHisto;
	}
    
	//Fill the histos
	theFile->cd();
	if(hT0LayerHisto != 0) {
	  //  if(debug)
	  // cout<<"Filling histo "<<hT0LayerHisto->GetName()<<" with digi "<<t0<<" TDC counts"<<endl;
	  hT0LayerHisto->Fill(t0);
	}
      }

      //Use all the remaining events to compute t0 per wire
      if(nevents>eventsForLayerT0){
	// Get the wireId
	const DTWireId wireId(layerId, (*digi).wire());
	if(debug) {
	  cout << "   Wire: " << wireId << endl
	       << "       time (TDC counts): " << (*digi).countsTDC()<< endl;
	}   

	//Fill the histos per wire for the chosen cells
	vector<DTWireId>::iterator it = find(wireIdWithHistos.begin(),wireIdWithHistos.end(),wireId);
	if (it!=wireIdWithHistos.end()){
 	  //Get the per-wire histo from the map
	  TH1I *hT0WireHisto = theHistoWireMap[wireId];	
	  //If it doesn't exist, book it
	  if(hT0WireHisto == 0){
	    theFile->cd(); 
	    hT0WireHisto = new TH1I(getHistoName(wireId).c_str(),"T0 from pulses by wire (TDC counts, 1 TDC count = 0.781 ns)",7000,0,7000);
	    //hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin())-100,
	    //hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin())+100);
	    if(debug)
	      cout << "  New T0 per wire Histo: " << hT0WireHisto->GetName() << endl;
	    theHistoWireMap[wireId] = hT0WireHisto;
	  }
	  //Fill the histos
	  theFile->cd();
	  if(hT0WireHisto != 0) {
	    //if(debug)
	    // cout<<"Filling histo "<<hT0WireHisto->GetName()<<" with digi "<<t0<<" TDC counts"<<endl;
	    hT0WireHisto->Fill(t0);
	  }
	}

	//Check the tzero has reasonable value
	//float hT0Min = tTrig - 2*tTrigRMS;
	//float hT0Max = hT0Min + timeBoxWidth;
	/*if(abs(hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin()) - t0) > rejectDigiFromPeak){
	  if(debug)
	    cout<<"digi skipped because t0 per sector "<<hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin())<<endl;
	  continue;
	}*/
	/*if((t0 < hT0Min)||(t0 > hT0Max)){
          if(debug)
            cout<<"digi skipped because t0 outside of interval (" << hT0Min << "," << hT0Max << ")" <<endl;
          continue;
        }*/
	//Select per layer
	if(fabs(theTPPeakMap[layerId] - t0) > rejectDigiFromPeak){
	  if(debug)
            cout<<"digi skipped because t0 too far from peak " << theTPPeakMap[layerId] << endl;
          continue;	
	}

	//Find to ref. per chamber
	theSumT0ByChamber[chamberId] = theSumT0ByChamber[chamberId] + t0;
	theCountT0ByChamber[chamberId]++;

	//Use second bunch of events to compute a t0 reference per wire
	if(nevents< (eventsForLayerT0 + eventsForWireT0)){
	  if(!nDigiPerWire_ref[wireId]){
	    mK_ref[wireId] = 0;
	  }
	  nDigiPerWire_ref[wireId] = nDigiPerWire_ref[wireId] + 1;
	  mK_ref[wireId] = mK_ref[wireId] + (t0-mK_ref[wireId])/nDigiPerWire_ref[wireId];
	}
	//Use last all the remaining events to compute the mean and sigma t0 per wire
	else if(nevents>(eventsForLayerT0 + eventsForWireT0)){
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
    bool increaseEvtsForLayerT0 = false;	
    for(map<DTLayerId, TH1I*>::const_iterator lHisto = theHistoLayerMap.begin();
	lHisto != theHistoLayerMap.end();
	lHisto++){
      if(debug)
	cout<<"Reading histogram "<<(*lHisto).second->GetName()<<" with mean "<<(*lHisto).second->GetMean()<<" and RMS "<<(*lHisto).second->GetRMS() << endl;

      //Find peaks
      //int npeaks = spectrum->Search((*lHisto).second,0.5,"goff");
      //int npeaks = spectrum->Search((*lHisto).second,(tpPeakWidthPerLayer/2.),"goff",0.3);
      int npeaks = spectrum->Search((*lHisto).second,(tpPeakWidthPerLayer/2.),"",0.3);

      double *peaks = spectrum->GetPositionX();
      //Put in a std::vector<float>
      vector<float> peakMeans(peaks,peaks + npeaks);
      //Sort the peaks in ascending order
      sort(peakMeans.begin(),peakMeans.end());
				
      //Find best peak -- preliminary criteria: find peak closest to center of time box	
      float tTrig,tTrigRMS, kFactor;
      tTrigMap->get((*lHisto).first.superlayerId(), tTrig, tTrigRMS, kFactor, DTTimeUnits::counts );

      float timeBoxCenter = (2*tTrig + (float)timeBoxWidth)/2.;	
      float hMin = (*lHisto).second->GetXaxis()->GetXmin();
      float hMax = (*lHisto).second->GetXaxis()->GetXmax();		
      vector<float>::const_iterator tpPeak = peakMeans.end();
      for(vector<float>::const_iterator it = peakMeans.begin(); it != peakMeans.end(); ++it){
	float mean = *it;

	int bin = (*lHisto).second->GetXaxis()->FindBin(mean);
	float yp = (*lHisto).second->GetBinContent(bin);
	if(debug) cout << "Peak : (" << mean << "," << yp << ")" << endl; 

	//Find RMS
	float previous_peak = (it == peakMeans.begin())?hMin:*(it - 1);
        float next_peak = (it == (peakMeans.end()-1))?hMax:*(it + 1);

	float rangemin = mean - (mean - previous_peak)/8.;
        float rangemax = mean + (next_peak - mean)/8.;
	int binmin = (*lHisto).second->GetXaxis()->FindBin(rangemin);
	int binmax = (*lHisto).second->GetXaxis()->FindBin(rangemax);
	(*lHisto).second->GetXaxis()->SetRange(binmin,binmax);
	//RMS estimate
	float rms_seed = (*lHisto).second->GetRMS();

	/*rangemin = mean - 2*rms_seed;
	rangemax = mean + 2*rms_seed;
	if(debug) cout << "Seed for RMS, Fit min, Fit max: " << rms_seed << ", " << rangemin << ", " << rangemax << endl;
	//Fit to gaussian
	string funcname("fitFcn_");
	funcname += (*lHisto).second->GetName();
	if(debug) cout << "Fitting function " << funcname << endl; 
	TF1* func = new TF1(funcname.c_str(),"gaus",rangemin,rangemax);
	func->SetParameters(yp,mean,rms_seed);
	(*lHisto).second->Fit(func,"Q","",rangemin,rangemax);

	float fitconst = func->GetParameter(0);
	float fitmean = func->GetParameter(1);
	float fitrms = func->GetParameter(2);
	float chisquare = func->GetChisquare()/func->GetNDF();
	if(debug) cout << "Gaussian fit constant,mean,RMS,chi2= " << fitconst << ", " << fitmean << ", " << fitrms << ", " << chisquare << endl;*/

	//Reject peaks with RMS larger than specified
	//if(fitrms > tpPeakWidth) continue;
	if(rms_seed > tpPeakWidthPerLayer) continue;

	if(fabs(mean - timeBoxCenter) < fabs(*tpPeak - timeBoxCenter)) tpPeak = it;
      }	
      //Didn't find peak	
      /*if(tpPeak == peakMeans.end()){
	if(retryForLayerT0 < 2){
	  increaseEvtsForLayerT0 = true;
	  retryForLayerT0++;
	  break;
	} 
      }*/

      float selPeak = (tpPeak != peakMeans.end())?*tpPeak:(*lHisto).second->GetBinCenter((*lHisto).second->GetMaximumBin());		
      if(debug) cout << "Peak selected at " << selPeak << endl;
	
      theTPPeakMap[(*lHisto).first] = selPeak;
		
      //Take the mean as a first t0 estimation
      /*if((*lHisto).second->GetRMS() < tpPeakWidth){
	if(hT0SectorHisto == 0){
	  hT0SectorHisto = new TH1D("hT0AllLayerOfSector","T0 from pulses per layer in sector", 
				    //20, (*lHisto).second->GetMean()-100, (*lHisto).second->GetMean()+100);
				    700, 0, 7000);
				    //300,3300,3600);	
	}
	if(debug)
	  cout<<" accepted"<<endl;
	//TH1I* aux_histo = (*lHisto).second;
	//aux_histo->GetXaxis()->SetRangeUser(3300,3600);
	hT0SectorHisto->Fill((*lHisto).second->GetMean());
	//hT0SectorHisto->Fill(aux_histo->GetMean());
      }
      //Take the mean of noise + 400ns as a first t0 estimation
      //if((*lHisto).second->GetRMS()>10.0 && ((*lHisto).second->GetRMS()<15.0)){
      //double t0_estim = (*lHisto).second->GetMean() + 400;
      //if(hT0SectorHisto == 0){
      //  hT0SectorHisto = new TH1D("hT0AllLayerOfSector","T0 from pulses per layer in sector", 
      //			    //20, t0_estim-100, t0_estim+100);
      //			    700, 0, 7000);
      //}
      //if(debug)
      //  cout<<" accepted + 400ns"<<endl;
      //hT0SectorHisto->Fill((*lHisto).second->GetMean() + 400);
      //}
      if(debug)
	cout<<endl;

      theT0LayerMap[(*lHisto).second->GetName()] = (*lHisto).second->GetMean();
      theSigmaT0LayerMap[(*lHisto).second->GetName()] = (*lHisto).second->GetRMS();*/
    }
    /*if(!hT0SectorHisto){
      cout<<"[DTT0CalibrationNew]: All the t0 per layer are still uncorrect: trying with greater number of events"<<endl;
      eventsForLayerT0 = eventsForLayerT0*2;
      return;
    }
    if(debug)
      cout<<"[DTT0CalibrationNew] t0 reference for this sector "<<
	hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin())<<endl;*/
    if(increaseEvtsForLayerT0){
	cout<<"[DTT0CalibrationNew]: t0 per layer are still uncorrect: trying with greater number of events"<<endl;
	eventsForLayerT0 = eventsForLayerT0*2;
	return;
    }		
  } 
}

void DTT0CalibrationNew::endJob() {

  DTT0* t0s = new DTT0();
  DTT0* t0sWRTChamber = new DTT0();

  if(debug) 
    cout << "[DTT0CalibrationNewPerLayer]Writing histos to file!" << endl;

  theFile->cd();
  //hT0SectorHisto->Write();
  for(map<DTWireId, TH1I*>::const_iterator wHisto = theHistoWireMap.begin();
      wHisto != theHistoWireMap.end();
      wHisto++) {
    (*wHisto).second->Write(); 
  }
  for(map<DTLayerId, TH1I*>::const_iterator lHisto = theHistoLayerMap.begin();
      lHisto != theHistoLayerMap.end();
      lHisto++) {
    (*lHisto).second->Write(); 
  }  

  if(debug) 
    cout << "[DTT0CalibrationNew] Compute and store t0 and sigma per wire" << endl;

  for(map<DTChamberId,double>::const_iterator chamber = theSumT0ByChamber.begin();
      chamber != theSumT0ByChamber.end();
      ++chamber) theRefT0ByChamber[(*chamber).first] = theSumT0ByChamber[(*chamber).first]/((double)theCountT0ByChamber[(*chamber).first]);
  
  for(map<DTWireId, double>::const_iterator wiret0 = theAbsoluteT0PerWire.begin();
      wiret0 != theAbsoluteT0PerWire.end();
      wiret0++){
    if(nDigiPerWire[(*wiret0).first]){
      double t0 = (*wiret0).second/nDigiPerWire[(*wiret0).first];
      DTChamberId chamberId = ((*wiret0).first).chamberId();
      //theRelativeT0PerWire[(*wiret0).first] = t0 - hT0SectorHisto->GetBinCenter(hT0SectorHisto->GetMaximumBin());
      theRelativeT0PerWire[(*wiret0).first] = t0 - theRefT0ByChamber[chamberId];	
      cout<<"Wire "<<(*wiret0).first<<" has    t0 "<<t0<<"(absolute) "<<theRelativeT0PerWire[(*wiret0).first]<<"(relative)";

      //theSigmaT0PerWire[(*wiret0).first] = sqrt((theSigmaT0PerWire[(*wiret0).first] / nDigiPerWire[(*wiret0).first]) - t0*t0);
      theSigmaT0PerWire[(*wiret0).first] = sqrt(qK[(*wiret0).first]/nDigiPerWire[(*wiret0).first]);
      cout<<"    sigma "<<theSigmaT0PerWire[(*wiret0).first]<<endl;
    }
    else{
      cout<<"[DTT0CalibrationNew] ERROR: no digis in wire "<<(*wiret0).first<<endl;
      abort();
    }
  }

  ///Loop on superlayer to correct between even-odd layers (2 different test pulse lines!)
  // Get all the sls from the setup
  const vector<const DTSuperLayer*>& superLayers = dtGeom->superLayers();     
  // Loop over all SLs
  for(auto  sl = superLayers.begin();
      sl != superLayers.end(); sl++) {

  
    //Compute mean for odd and even superlayers
    double oddLayersMean=0;
    double evenLayersMean=0; 
    double oddLayersDen=0;
    double evenLayersDen=0;
    for(map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
	wiret0 != theRelativeT0PerWire.end();
	wiret0++){
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
	wiret0++){
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

    if(debug && oddLayersMean)
      cout<<"[DTT0CalibrationNew] Relative T0 sigma for  odd layers "<<oddLayersSigma<<"  even layers"<<evenLayersSigma<<endl;

    //Recompute the mean for odd and even superlayers discarding fluctations
    double oddLayersFinalMean=0; 
    double evenLayersFinalMean=0;
    for(map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
	wiret0 != theRelativeT0PerWire.end();
	wiret0++){
      if((*wiret0).first.layerId().superlayerId() == (*sl)->id()){
	if(((*wiret0).first.layerId().layer()) % 2){
	  if(abs((*wiret0).second - oddLayersMean) < (2*oddLayersSigma))
	    oddLayersFinalMean = oddLayersFinalMean + (*wiret0).second;
	}
	else{
	  if(abs((*wiret0).second - evenLayersMean) < (2*evenLayersSigma))
	    evenLayersFinalMean = evenLayersFinalMean + (*wiret0).second;
	}
      }
    }
    oddLayersFinalMean = oddLayersFinalMean/oddLayersDen;
    evenLayersFinalMean = evenLayersFinalMean/evenLayersDen;
    if(debug && oddLayersMean)
      cout<<"[DTT0CalibrationNew] Final relative T0 mean for  odd layers "<<oddLayersFinalMean<<"  even layers"<<evenLayersFinalMean<<endl;

    for(map<DTWireId, double>::const_iterator wiret0 = theRelativeT0PerWire.begin();
	wiret0 != theRelativeT0PerWire.end();
	wiret0++){
      if((*wiret0).first.layerId().superlayerId() == (*sl)->id()){
	double t0=-999;
	if(((*wiret0).first.layerId().layer()) % 2)
	  t0 = (*wiret0).second + (evenLayersFinalMean - oddLayersFinalMean);
	else
	  t0 = (*wiret0).second;

	cout<<"[DTT0CalibrationNew] Wire "<<(*wiret0).first<<" has    t0 "<<(*wiret0).second<<" (relative, after even-odd layer corrections)  "
	    <<"    sigma "<<theSigmaT0PerWire[(*wiret0).first]<<endl;
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
      tzero != t0s->end(); tzero++) {
// @@@ NEW DTT0 FORMAT
//    DTChamberId chamberId((*tzero).first.wheelId,
//			  (*tzero).first.stationId,
//			  (*tzero).first.sectorId);
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
      tzero != t0s->end(); tzero++) {
// @@@ NEW DTT0 FORMAT
//    DTChamberId chamberId((*tzero).first.wheelId,
//			  (*tzero).first.stationId,
//			  (*tzero).first.sectorId);
//    double t0mean = ((*tzero).second.t0mean) - (sumT0ByChamber[chamberId]/countT0ByChamber[chamberId]);
//    double t0rms = (*tzero).second.t0rms;
//    DTWireId wireId((*tzero).first.wheelId,
//		    (*tzero).first.stationId,
//		    (*tzero).first.sectorId,
//		    (*tzero).first.slId,
//		    (*tzero).first.layerId,
//		    (*tzero).first.cellId);
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

