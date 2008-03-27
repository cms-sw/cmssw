/*
 *  simple validation package for CSC DIGIs, RECHITs and SEGMENTs.
 *
 *  Michael Schmitt
 *  Andy Kubik
 *  Northwestern University
 */
#include "RecoLocalMuon/CSCValidation/interface/CSCValidation.h"

using namespace std;
using namespace edm;


///////////////////
//  CONSTRUCTOR  //
///////////////////
CSCValidation::CSCValidation(const ParameterSet& pset){

  // Get the various input parameters
  rootFileName     = pset.getUntrackedParameter<string>("rootFileName");
  isSimulation     = pset.getUntrackedParameter<bool>("isSimulation");

  // set counter to zero
  nEventsAnalyzed = 0;
  
  // Create the root file for the histograms
  theFile = new TFile(rootFileName.c_str(), "RECREATE");
  theFile->cd();

  // Create sub-directories for digis/rechits/segments
  theFile->mkdir("Digis");
  theFile->mkdir("recHits");
  theFile->mkdir("Segments");
  theFile->mkdir("Calib");
  theFile->mkdir("PedestalNoise");
  theFile->cd();

  // Create object of class CSCValidationHistos to manage histograms
  histos = new CSCValHists();
  // book histos
  histos->bookHists();
  // setup trees to hold global position data for rechits and segments
  histos->setupTrees();


}

//////////////////
//  DESTRUCTOR  //
//////////////////
CSCValidation::~CSCValidation(){

  // wirte histos to the specified file
  histos->writeHists(theFile, isSimulation);

}



////////////////
//  Analysis  //
////////////////
void CSCValidation::analyze(const Event & event, const EventSetup& eventSetup){
  
  // increment counter
  nEventsAnalyzed++;

  int iRun   = event.id().run();
  int iEvent = event.id().event();

  LogInfo("EventInfo") << "Run: " << iRun << "    Event: " << iEvent;

  // Variables for occupancy plots
  bool wireo[2][4][4][36];
  bool stripo[2][4][4][36];
  bool rechito[2][4][4][36];
  bool segmento[2][4][4][36];

  for (int e = 0; e < 2; e++){
    for (int s = 0; s < 4; s++){
      for (int r = 0; r < 4; r++){
        for (int c = 0; c < 36; c++){
          wireo[e][s][r][c] = false;
          stripo[e][s][r][c] = false;
          rechito[e][s][r][c] = false;
          segmento[e][s][r][c] = false;
        }
      }
    }
  }

  // ==============================================
  //
  // look at Calibrations
  //
  // ==============================================

  // Only do this for the first event
  if (nEventsAnalyzed == 1){

    LogDebug("Calibrations") << "Loading Calibrations...";

    // get the gains
    edm::ESHandle<CSCDBGains> hGains;
    eventSetup.get<CSCDBGainsRcd>().get( hGains );
    const CSCDBGains* pGains = hGains.product();
    // get the crosstalks
    edm::ESHandle<CSCDBCrosstalk> hCrosstalk;
    eventSetup.get<CSCDBCrosstalkRcd>().get( hCrosstalk );
    const CSCDBCrosstalk* pCrosstalk = hCrosstalk.product();
    // get the noise matrix
    edm::ESHandle<CSCDBNoiseMatrix> hNoiseMatrix;
    eventSetup.get<CSCDBNoiseMatrixRcd>().get( hNoiseMatrix );
    const CSCDBNoiseMatrix* pNoiseMatrix = hNoiseMatrix.product();
    // get pedestals
    edm::ESHandle<CSCDBPedestals> hPedestals;
    eventSetup.get<CSCDBPedestalsRcd>().get( hPedestals );
    const CSCDBPedestals* pPedestals = hPedestals.product();

    LogDebug("Calibrations") << "Calibrations Loaded!";

    for (int i = 0; i < 400; i++){
      int bin = i+1;
      float gain =  pGains->gains[i].gain_slope;
      float xsl  =  pCrosstalk->crosstalk[i].xtalk_slope_left;
      float xsr  =  pCrosstalk->crosstalk[i].xtalk_slope_right;
      float xil  =  pCrosstalk->crosstalk[i].xtalk_intercept_left;
      float xir  =  pCrosstalk->crosstalk[i].xtalk_intercept_right;
      float pedp =  pPedestals->pedestals[i].ped;
      float pedr =  pPedestals->pedestals[i].rms;
      float n33  =  pNoiseMatrix->matrix[i].elem33;
      float n34  =  pNoiseMatrix->matrix[i].elem34;
      float n35  =  pNoiseMatrix->matrix[i].elem35;
      float n44  =  pNoiseMatrix->matrix[i].elem44;
      float n45  =  pNoiseMatrix->matrix[i].elem45;
      float n46  =  pNoiseMatrix->matrix[i].elem46;
      float n55  =  pNoiseMatrix->matrix[i].elem55;
      float n56  =  pNoiseMatrix->matrix[i].elem56;
      float n57  =  pNoiseMatrix->matrix[i].elem57;
      float n66  =  pNoiseMatrix->matrix[i].elem66;
      float n67  =  pNoiseMatrix->matrix[i].elem67;
      float n77  =  pNoiseMatrix->matrix[i].elem77;
      // fill histo
      histos->fillCalibHistos(gain, xsl, xsr, xil, xir, pedp, pedr, n33, n34, n35, n44, n45,
                              n46, n55, n56, n57, n66, n67, n77, bin);
    }

  } // end calib



  // ==============================================
  //
  // look at DIGIs
  //
  // ==============================================


  //
  // These declarations create handles to the types of records that you want
  // to retrieve from event "e".
  //
  edm::Handle<CSCWireDigiCollection> wires;
  edm::Handle<CSCStripDigiCollection> strips;
  /*
  edm::Handle<CSCComparatorDigiCollection> comparators;
  edm::Handle<CSCALCTDigiCollection> alcts;
  edm::Handle<CSCCLCTDigiCollection> clcts;
  edm::Handle<CSCRPCDigiCollection> rpcs;
  edm::Handle<CSCCorrelatedLCTDigiCollection> correlatedlcts;
  */

  
  // Pass the handle to the method "getByType", which is used to retrieve
  // one and only one instance of the type in question out of event "e". If
  // zero or more than one instance exists in the event an exception is thrown.
  //

  event.getByLabel("muonCSCDigis","MuonCSCWireDigi",wires);
  event.getByLabel("muonCSCDigis","MuonCSCStripDigi",strips);
  /*
  event.getByLabel("muonCSCDigis","MuonCSCComparatorDigi",comparators);
  event.getByLabel("muonCSCDigis","MuonCSCALCTDigi",alcts);
  event.getByLabel("muonCSCDigis","MuonCSCCLCTDigi",clcts);
  event.getByLabel("muonCSCDigis","MuonCSCRPCDigi",rpcs);
  event.getByLabel("muonCSCDigis","MuonCSCCorrelatedLCTDigi",correlatedlcts);
  */

  //
  // WIRE GROUPS
  int nWireGroupsTotal = 0;
  for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    if (kEndcap == 2) kEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    int kLayer   = id.layer();
    std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myWire = digiItr->getWireGroup();
      int myTBin = digiItr->getTimeBin();
      nWireGroupsTotal++;
      int kCodeBroad  = kEndcap * ( 4*(kStation-1) + kRing) ;
      int kCodeNarrow = kEndcap * ( 100*(kRing-1) + kChamber) ;
      // fill wire histos
      histos->fillWireHistos(myWire, myTBin, kCodeNarrow, kCodeBroad,
                             kEndcap, kStation, kRing, kChamber, kLayer);
      wireo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;      
    }
  } // end wire loop
  

  //
  // STRIPS
  //
  int nStripsFired = 0;
  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    if (kEndcap == 2) kEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    int kLayer   = id.layer();
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myStrip = digiItr->getStrip();
      std::vector<int> myADCVals = digiItr->getADCCounts();
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float threshold = 13.3 ;
      float diff = 0.;
      for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
	diff = (float)myADCVals[iCount]-thisPedestal;
	if (diff > threshold) { thisStripFired = true; }
      } 
      if (thisStripFired) {
        nStripsFired++;
        int kCodeBroad  = kEndcap * ( 4*(kStation-1) + kRing) ;
        int kCodeNarrow = kEndcap * ( 100*(kRing-1) + kChamber) ;
        // fill strip histos
        histos->fillStripHistos(myStrip, kCodeNarrow, kCodeBroad,
                                kEndcap, kStation, kRing, kChamber, kLayer);
        stripo[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;
      }
    }
  } // end strip loop

  //=======================================================
  //
  // Look at the Pedestal Noise Distributions
  //
  //=======================================================

  for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
    CSCDetId id = (CSCDetId)(*j).first;
    int kEndcap  = id.endcap();
    if (kEndcap == 2) kEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    int kLayer   = id.layer();
    std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
    std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
    for( ; digiItr != last; ++digiItr) {
      int myStrip = digiItr->getStrip();
      std::vector<int> myADCVals = digiItr->getADCCounts();
      float TotalADC = getSignal(*strips, id, myStrip);
      bool thisStripFired = false;
      float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
      float thisSignal = (1./6)*(myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
      float threshold = 13.3;
      if(kStation == 1 && kRing == 4)
	{
	  kRing = 1;
	  if(myStrip <= 16) myStrip += 64; // no trapping for any bizarreness
	}
      int globalStrip = kEndcap*( kStation*1000000 + kRing*100000 + kChamber*1000 + kLayer*100 + myStrip);
      if (TotalADC > threshold) { thisStripFired = true;}
      if (!thisStripFired){
	float ADC = thisSignal - thisPedestal;
        histos->fillNoiseHistos(ADC, globalStrip, kStation, kRing);
      }
    }
  }




  // ==============================================
  //
  // look at RECHITs
  //
  // ==============================================

  // Get the CSC Geometry :
  ESHandle<CSCGeometry> cscGeom;
  eventSetup.get<MuonGeometryRecord>().get(cscGeom);
  
  // Get the RecHits collection :
  Handle<CSCRecHit2DCollection> recHits; 
  event.getByLabel("csc2DRecHits",recHits);  
  int nRecHits = recHits->size();

  // Get the SimHits (if applicable)
  Handle<PSimHitContainer> simHits;
  if (isSimulation) event.getByLabel("g4SimHits", "MuonCSCHits", simHits);
 
 
  // ---------------------
  // Loop over rechits 
  // ---------------------
  int iHit = 0;

  // Build iterator for rechits and loop :
  CSCRecHit2DCollection::const_iterator recIt;
  for (recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    iHit++;

    // Find chamber with rechits in CSC 
    CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    int kEndcap  = idrec.endcap();
    if (kEndcap == 2) kEndcap = -1;
    int kRing    = idrec.ring();
    int kStation = idrec.station();
    int kChamber = idrec.chamber();
    int kLayer   = idrec.layer();

    rechito[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;

    // Store rechit as a Local Point:
    LocalPoint rhitlocal = (*recIt).localPosition();  
    float xreco = rhitlocal.x();
    float yreco = rhitlocal.y();
    float zreco = rhitlocal.z();
    float phireco = rhitlocal.phi();
    LocalError rerrlocal = (*recIt).localPositionError();  
    float xxerr = rerrlocal.xx();
    float yyerr = rerrlocal.yy();
    float xyerr = rerrlocal.xy();

    // Find the strip containing this hit
    CSCRecHit2D::ChannelContainer hitstrips = (*recIt).channels();
    int nStrips     =  hitstrips.size();
    int centerid    =  nStrips/2 + 1;
    int centerStrip =  hitstrips[centerid - 1];

    // Find the charge associated with this hit

    CSCRecHit2D::ADCContainer adcs = (*recIt).adcs();
    int adcsize = adcs.size();
    float rHSumQ = 0;
    float sumsides = 0;
    for (int i = 0; i < adcsize; i++){
      if (i != 3 && i != 7 && i != 11){
        rHSumQ = rHSumQ + adcs[i]; 
      }
      if (adcsize == 12 && (i < 3 || i > 7) && i < 12){
        sumsides = sumsides + adcs[i];
      }
    }
    float rHratioQ = sumsides/rHSumQ;
    if (adcsize != 12) rHratioQ = -99;

    // Get the signal timing of this hit
    //float rHtime = (*recIt).tpeak();
    float rHtime = getTiming(*strips, idrec, centerStrip);

    // Get pointer to the layer:
    const CSCLayer* csclayer = cscGeom->layer( idrec );

    // Transform hit position from local chamber geometry to global CMS geom
    GlobalPoint rhitglobal= csclayer->toGlobal(rhitlocal);
    float grecx   =  rhitglobal.x();
    float grecy   =  rhitglobal.y();
    float grecz   =  rhitglobal.z();
    float grecphi =  rhitglobal.phi();
    float grecr   =  sqrt(grecx*grecx + grecy+grecy);



    float simHitXres = -99;
    float simHitYres = -99;
    float mindiffX   = 99;
    float mindiffY   = 10;
    // If MC, find closest muon simHit to check resolution:
    if (isSimulation){
      PSimHitContainer::const_iterator simIt;
      for (simIt = simHits->begin(); simIt != simHits->end(); simIt++){
        // Get DetID for this simHit:
        CSCDetId sId = (CSCDetId)(*simIt).detUnitId();
        // Check if the simHit detID matches that of current recHit
        // and make sure it is a muon hit:
        if (sId == idrec && abs((*simIt).particleType()) == 13){
          // Get the position of this simHit in local coordinate system:
          LocalPoint sHitlocal = (*simIt).localPosition();
          // Now we need to make reasonably sure that this simHit is
          // responsible for this recHit:
          if ((sHitlocal.x() - xreco) < mindiffX && (sHitlocal.y() - yreco) < mindiffY){
            simHitXres = (sHitlocal.x() - xreco);
            simHitYres = (sHitlocal.y() - yreco);
            mindiffX = (sHitlocal.x() - xreco);
          }
        }
      }
    }

    // Fill the rechit position branch
    histos->fillRechitTree(xreco, yreco, grecx, grecy, kEndcap, kStation, kRing, kChamber, kLayer);
    
    // Simple occupancy variables
    int kCodeBroad  = kEndcap * ( 4*(kStation-1) + kRing) ;
    int kCodeNarrow = kEndcap * ( 100*(kRing-1) + kChamber) ;

    // Fill some histograms
    histos->fillRechitHistos(kCodeNarrow, kCodeBroad, xreco, yreco, grecx, grecy,
                             rHSumQ, rHratioQ, rHtime, simHitXres,
                             kEndcap, kStation, kRing, kChamber, kLayer);

  } //end rechit loop

  // ==============================================
  //
  // look at SEGMENTs
  //
  // ===============================================

  // get CSC segment collection
  Handle<CSCSegmentCollection> cscSegments;
  event.getByLabel("cscSegments", cscSegments);
  int nSegments = cscSegments->size();

  // -----------------------
  // loop over segments
  // -----------------------
  int iSegment = 0;
  for(CSCSegmentCollection::const_iterator it=cscSegments->begin(); it != cscSegments->end(); it++) {
    iSegment++;
    //
    CSCDetId id  = (CSCDetId)(*it).cscDetId();
    int kEndcap  = id.endcap();
    if (kEndcap == 2) kEndcap = -1;
    int kRing    = id.ring();
    int kStation = id.station();
    int kChamber = id.chamber();
    segmento[kEndcap-1][kStation-1][kRing-1][kChamber-1] = true;


    //
    float chisq    = (*it).chi2();
    int nhits      = (*it).nRecHits();
    int nDOF       = 2*nhits-4;
    double chisqProb = ChiSquaredProbability( (double)chisq, nDOF );
    LocalPoint localPos = (*it).localPosition();
    float segX     = localPos.x();
    float segY     = localPos.y();
    float segZ     = localPos.z();
    float segPhi   = localPos.phi();
    LocalVector segDir = (*it).localDirection();
    double theta   = segDir.theta();
    double phi     = segDir.phi();

    //
    // try to get the CSC recHits that contribute to this segment.
    std::vector<CSCRecHit2D> theseRecHits = (*it).specificRecHits();
    int nRH = (*it).nRecHits();
    int jRH = 0;
    HepMatrix sp(6,1);
    HepMatrix se(6,1);
    for ( vector<CSCRecHit2D>::const_iterator iRH = theseRecHits.begin(); iRH != theseRecHits.end(); iRH++) {
      jRH++;
      CSCDetId idRH = (CSCDetId)(*iRH).cscDetId();
      int kEndcap  = idRH.endcap();
      int kRing    = idRH.ring();
      int kStation = idRH.station();
      int kChamber = idRH.chamber();
      int kLayer   = idRH.layer();

      // Find the strip containing this hit
      CSCRecHit2D::ChannelContainer hitstrips = (*iRH).channels();
      int nStrips     =  hitstrips.size();
      int centerid    =  nStrips/2 + 1;
      int centerStrip =  hitstrips[centerid - 1];

      // If this segment has 6 hits, find the position of each hit on the strip in units of stripwidth and store values
      if (nRH == 6){
        float stpos = (*iRH).positionWithinStrip();
        se(kLayer,1) = (*iRH).errorWithinStrip();
        // Take into account half-strip staggering of layers (ME1/1 has no staggering)
        if (kStation == 1 && (kRing == 1 || kRing == 4)) sp(kLayer,1) = stpos + centerStrip;
        else{
          if (kLayer == 1 || kLayer == 3 || kLayer == 5) sp(kLayer,1) = stpos + centerStrip;
          if (kLayer == 2 || kLayer == 4 || kLayer == 6) sp(kLayer,1) = stpos - 0.5 + centerStrip;
        }
      }

    }

    float residual = -99;
    // Fit all points except layer 3, then compare expected value for layer 3 to reconstructed value
    if (nRH == 6){
      float expected = fitX(sp,se);
      residual = expected - sp(3,1);
    }

    // global transformation
    float globX = 0.;
    float globY = 0.;
    float globZ = 0.;
    float globpPhi = 0.;
    float globR = 0.;
    float globTheta = 0.;
    float globPhi   = 0.;
    const CSCChamber* cscchamber = cscGeom->chamber(id);
    if (cscchamber) {
      GlobalPoint globalPosition = cscchamber->toGlobal(localPos);
      globX = globalPosition.x();
      globY = globalPosition.y();
      globZ = globalPosition.z();
      globpPhi =  globalPosition.phi();
      globR   =  sqrt(globX*globX + globY*globY);
      GlobalVector globalDirection = cscchamber->toGlobal(segDir);
      globTheta = globalDirection.theta();
      globPhi   = globalDirection.phi();
    }

    // Simple occupancy variables
    int kCodeBroad  = kEndcap * ( 4*(kStation-1) + kRing) ;
    int kCodeNarrow = kEndcap * ( 100*(kRing-1) + kChamber) ;

    // Fill segment position branch
    histos->fillSegmentTree(segX, segY, globX, globY, kEndcap, kStation, kRing, kChamber);

    // Fill histos
    histos->fillSegmentHistos(kCodeNarrow, kCodeBroad, nhits, theta, globX, globY,
                              residual, chisqProb, globTheta, globPhi,
                              kEndcap, kStation, kRing, kChamber);


  } // end segment loop

  // Fill # per even histos (how many stips/wires/rechits/segments per event)
  histos->fillEventHistos(nWireGroupsTotal,nStripsFired,nRecHits,nSegments);

  // Fill occupancy plots
  histos->fillOccupancyHistos(wireo,stripo,rechito,segmento);


  // do Efficiency
  doEfficiencies(recHits, cscSegments);

}

//-------------------------------------------------------------------------------------
// Fits a straight line to a set of 5 points with errors.  Functions assumes 6 points
// and removes hit in layer 3.  It then returns the expected position value in layer 3
// based on the fit.
//-------------------------------------------------------------------------------------
float CSCValidation::fitX(HepMatrix points, HepMatrix errors){

  float S   = 0;
  float Sx  = 0;
  float Sy  = 0;
  float Sxx = 0;
  float Sxy = 0;
  float sigma2 = 0;

  for (int i=1;i<7;i++){
    if (i != 3){
      sigma2 = errors(i,1)*errors(i,1);
      S = S + (1/sigma2);
      Sy = Sy + (points(i,1)/sigma2);
      Sx = Sx + ((i)/sigma2);
      Sxx = Sxx + (i*i)/sigma2;
      Sxy = Sxy + (((i)*points(i,1))/sigma2);
    }
  }

  float delta = S*Sxx - Sx*Sx;
  float intercept = (Sxx*Sy - Sx*Sxy)/delta;
  float slope = (S*Sxy - Sx*Sy)/delta;

  float chi = 0;
  float chi2 = 0;

  // calculate chi2 (not currently used)
  for (int i=1;i<7;i++){
    chi = (points(i,1) - intercept - slope*i)/(errors(i,1));
    chi2 = chi2 + chi*chi;
  }

  return (intercept + slope*3);

}

//---------------------------------------------------------------------------------------
// Find the signal timing based on a weighted mean of the pulse.
// Function is meant to take the DetId and center strip number of a recHit and return
// the timing in units of time buckets (50ns)
//---------------------------------------------------------------------------------------

float CSCValidation::getTiming(const CSCStripDigiCollection& stripdigis, CSCDetId idRH, int centerStrip){

  float ADC[8];
  float timing = 0;

  // Loop over strip digis responsible for this recHit and sum charge
  CSCStripDigiCollection::DigiRangeIterator sIt;

  for (sIt = stripdigis.begin(); sIt != stripdigis.end(); sIt++){
    CSCDetId id = (CSCDetId)(*sIt).first;
    if (id == idRH){
      vector<CSCStripDigi>::const_iterator digiItr = (*sIt).second.first;
      vector<CSCStripDigi>::const_iterator last = (*sIt).second.second;
      for ( ; digiItr != last; ++digiItr ) {
        int thisStrip = digiItr->getStrip();
        if (thisStrip == (centerStrip)){
          float diff = 0;
          vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
          for (unsigned int iCount = 0; iCount < myADCVals.size(); iCount++) {
            diff = (float)myADCVals[iCount]-thisPedestal;
            ADC[iCount] = diff;
          }
        }
      }

    }

  }

  timing = (ADC[2]*2 + ADC[3]*3 + ADC[4]*4 + ADC[5]*5 + ADC[6]*6)/(ADC[2] + ADC[3] + ADC[4] + ADC[5] + ADC[6]);
  return timing;


}

//----------------------------------------------------------------------------
// Calculate basic efficiencies for recHits and Segments
// Author: S. Stoynev
//----------------------------------------------------------------------------

void CSCValidation::doEfficiencies(edm::Handle<CSCRecHit2DCollection> recHits, edm::Handle<CSCSegmentCollection> cscSegments){

  bool AllRecHits[2][4][4][36][6];
  bool AllSegments[2][4][4][36];
  //bool MultiSegments[2][4][4][36];
  for(int iE = 0;iE<2;iE++){
    for(int iS = 0;iS<4;iS++){
      for(int iR = 0; iR<4;iR++){
        for(int iC =0;iC<36;iC++){
          AllSegments[iE][iS][iR][iC] = false;
          //MultiSegments[iE][iS][iR][iC] = false;
          for(int iL=0;iL<6;iL++){
            AllRecHits[iE][iS][iR][iC][iL] = false;
          }
        }
      }
    }
  }
  
  for (CSCRecHit2DCollection::const_iterator recIt = recHits->begin(); recIt != recHits->end(); recIt++) {
    //CSCDetId idrec = (CSCDetId)(*recIt).cscDetId();
    CSCDetId  idrec = (CSCDetId)(*recIt).cscDetId();
    AllRecHits[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()][idrec.layer() -1] = true;

  }
   

  for(CSCSegmentCollection::const_iterator segIt=cscSegments->begin(); segIt != cscSegments->end(); segIt++) {
    CSCDetId idseg  = (CSCDetId)(*segIt).cscDetId();
    //if(AllSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()]){
    //MultiSegments[idrec.endcap() -1][idrec.station() -1][idrec.ring() -1][idrec.chamber()] = true;
    //}
    AllSegments[idseg.endcap() -1][idseg.station() -1][idseg.ring() -1][idseg.chamber()] = true;
  }

  
  for(int iE = 0;iE<2;iE++){
    for(int iS = 0;iS<4;iS++){
      for(int iR = 0; iR<4;iR++){
        for(int iC =0;iC<36;iC++){
          int NumberOfLayers = 0;
          for(int iL=0;iL<6;iL++){
            if(AllRecHits[iE][iS][iR][iC][iL]){
              NumberOfLayers++;
            }
          }
          int bin = 0;
          if (iS==0) bin = iR+1;
          else bin = (iS+1)*2 + (iR+1);
          if(NumberOfLayers>1){
            //if(!(MultiSegments[iE][iS][iR][iC])){
            if(AllSegments[iE][iS][iR][iC]){
              //---- Efficient segment evenents
              histos->fillEfficiencyHistos(bin,1);
            }
            //---- All segment events (normalization)
            histos->fillEfficiencyHistos(bin+10,1);
            //}
          }
          if(AllSegments[iE][iS][iR][iC]){
            if(NumberOfLayers==6){
              //---- Efficient rechit events
              histos->fillEfficiencyHistos(bin,2);
            }
            //---- All rechit events (normalization)
            histos->fillEfficiencyHistos(bin+10,2);
          }
        }
      }
    }
  }

}


//---------------------------------------------------------------------------------------
// Given a set of digis, the CSCDetId, and the central strip of your choosing, returns
// the avg. Signal-Pedestal for 6 time bin x 5 strip .
//
// Author: P. Jindal
//---------------------------------------------------------------------------------------

float CSCValidation::getSignal(const CSCStripDigiCollection&
 stripdigis, CSCDetId idCS, int centerStrip){

  float SigADC[5];
  float TotalADC = 0;
  SigADC[0] = 0;
  SigADC[1] = 0;
  SigADC[2] = 0;
  SigADC[3] = 0;
  SigADC[4] = 0;

 
  // Loop over strip digis 
  CSCStripDigiCollection::DigiRangeIterator sIt;
  
  for (sIt = stripdigis.begin(); sIt != stripdigis.end(); sIt++){
    CSCDetId id = (CSCDetId)(*sIt).first;
    if (id == idCS){

      // First, find the Signal-Pedestal for center strip
      vector<CSCStripDigi>::const_iterator digiItr = (*sIt).second.first;
      vector<CSCStripDigi>::const_iterator last = (*sIt).second.second;
      for ( ; digiItr != last; ++digiItr ) {
        int thisStrip = digiItr->getStrip();
        if (thisStrip == (centerStrip)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[0] = thisSignal - 6*thisPedestal;
	}
     // Now,find the Signal-Pedestal for neighbouring 4 strips
        if (thisStrip == (centerStrip+1)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[1] = thisSignal - 6*thisPedestal;
	}
        if (thisStrip == (centerStrip+2)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[2] = thisSignal - 6*thisPedestal;
	}
        if (thisStrip == (centerStrip-1)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[3] = thisSignal - 6*thisPedestal;
	}
        if (thisStrip == (centerStrip-2)){
	  std::vector<int> myADCVals = digiItr->getADCCounts();
          float thisPedestal = 0.5*(float)(myADCVals[0]+myADCVals[1]);
	  float thisSignal = (myADCVals[2]+myADCVals[3]+myADCVals[4]+myADCVals[5]+myADCVals[6]+myADCVals[7]);
	  SigADC[4] = thisSignal - 6*thisPedestal;
	}
      }
      TotalADC = 0.2*(SigADC[0]+SigADC[1]+SigADC[2]+SigADC[3]+SigADC[4]);
    }
  }
  return TotalADC;
}

DEFINE_FWK_MODULE(CSCValidation);

