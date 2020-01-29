{
  //=== Draw rapidity sector boundaries on picture of tracker in r-z view.

  // WARNING: this script only works on machines where the line:
  // OpenGL.CanvasPreferGL: 1
  // appears in $ROOTSYS/etc/system.rootrc  (not at RAL).
  // Required for semi-transparent plotting.

  // In unnamed scripts, variables not forgotten at end, so must delete them before rerunning script, so ...
  gROOT->Reset();

  // Adjust these to line up histogram hisTracker with underlying image of CMS Tracker.
  // Then when aligned, comment out line trackerBorder.Draw() below.
  const float leftMargin   = 23;
  const float rightMargin  = 33;
  const float topMargin    = 13;
  const float bottomMargin = 14;

  gStyle->SetOptTitle(0);
  gStyle->SetOptStat("");
  gStyle->SetPadGridX(false);
  gStyle->SetPadGridY(false);

  const float trkInRad  = 20;
  const float trkOutRad = 110;
  const float trkLength = 270;
  const float beamLen   = 15;

  const unsigned int nSec = 9;
  const float chosenR   = 50;
  const unsigned int nSecEdges = nSec + 1;
  // Standard eta sectors
  const float eta[nSecEdges] = {0,0.31,0.61,0.89,1.16,1.43,1.7,1.95,2.16,2.4};
  // Sectors optimised by Ben Gayther to balance GP output data rate (at expense of increased HT tracks).
  //const float eta[nSecEdges] = {0.0, 0.19, 0.38, 0.57, 0.77, 1.01, 1.31, 1.66, 2.03, 2.40};
  const unsigned int nSubSec = 2;
  const unsigned int nSubSecEdges = nSubSec + 1;

  // Optionally draw a stub with the given digitized (r,z) coordinates.
  const bool drawStub = true;
  const int iDigi_RT = 492;
  const int iDigi_Z  = -1403; 

  TCanvas d1("d1","d1",1000,800);

  // Open picture of CMS tracker
  // http://ghugo.web.cern.ch/ghugo/layouts/cabling/OT614_200_IT404_layer2_10G/layout.html
  // Adjust the range of trackerBorder below to correspond to the coordinate range shown in this picture.
  TImage *img = TImage::Open("TrackerLayout.png");
  img->Draw("x");
  d1.Update();

  //Create a transparent pad filling the full canvas  
  TPad p("p","p",0,0,1,1);
  p.Range(-beamLen-leftMargin, -1-bottomMargin, trkLength+30+rightMargin, trkOutRad+15+topMargin);
  p.SetFillStyle(4000);
  p.SetFrameFillStyle(4000);
  p.Draw();
  p.cd();
 
  TPolyLine trackerBorder;

  trackerBorder.SetNextPoint(0.,0.);  
  trackerBorder.SetNextPoint(295.,0.);  
  trackerBorder.SetNextPoint(295.,123.);  
  trackerBorder.SetNextPoint(0.,123.);  
  trackerBorder.SetNextPoint(0.,0.);  
  //trackerBorder.Draw();

  /*
  TPolyLine subsecBoundary[nSec][nSubSecEdges];

  // Draw sub-sector boundaries.
  for (unsigned int i = 0; i < nSec; i++) {
    float subsecWidth = (eta[i+1] - eta[i])/float(nSubSec);
    for (unsigned int j = 0; j < nSubSecEdges; j++) {
      float subsecEtaEdge = eta[i] + subsecWidth * j;
      // z at r = chosenR;
      float z = chosenR/tan(2 * atan(exp(-subsecEtaEdge)));
      // Calculate (r,z) at periphery of Tracker from two ends of beam spot.
      // Start by assuming exit through barrel.
      float rPeriphNeg = trkOutRad; 
      float rPeriphPos = trkOutRad; 
      float zPeriphNeg = -beamLen + (z + beamLen)*(rPeriphNeg/chosenR); 
      float zPeriphPos =  beamLen + (z - beamLen)*(rPeriphNeg/chosenR); 
      // Now check if actual exit through endcap.
      if (fabs(zPeriphNeg) > trkLength) {
	int whichEndcap = (zPeriphNeg + beamLen > 0)  ?  1  :  -1;
	zPeriphNeg = whichEndcap*trkLength;
	rPeriphNeg = chosenR*(zPeriphNeg + beamLen)/(z + beamLen);
      } 
      if (fabs(zPeriphPos) > trkLength) {
	int whichEndcap = (zPeriphPos - beamLen > 0)  ?  1  :  -1;
	zPeriphPos = whichEndcap*trkLength;
	rPeriphPos = chosenR*(zPeriphPos - beamLen)/(z - beamLen);
      } 
      subsecBoundary[i][j].SetNextPoint(-beamLen,0);
      subsecBoundary[i][j].SetNextPoint(zPeriphNeg, rPeriphNeg);
      subsecBoundary[i][j].SetNextPoint(zPeriphPos, rPeriphPos);
      subsecBoundary[i][j].SetNextPoint(beamLen,0);
      subsecBoundary[i][j].SetNextPoint(-beamLen,0);
      subsecBoundary[i][j].SetFillColor(kYellow);
      unsigned int iHash = 3405+k*10;
      subsecBoundary[i][j].SetFillStyle(iHash);
      subsecBoundary[i][j].Draw("f");
      d1.Update();
    }
  }
  */

  TPolyLine secBoundary[nSecEdges];

  // Draw sector boundaries
  for (unsigned int k = 0; k < nSecEdges; k++) {
    // z at r = chosenR;
    float z = chosenR/tan(2.0 * atan(exp(-eta[k])));
    // Calculate (r,z) at periphery of Tracker from two ends of beam spot.
    // Start by assuming exit through barrel.
    float rPeriphNeg = trkOutRad; 
    float rPeriphPos = trkOutRad; 
    float zPeriphNeg = -beamLen + (z + beamLen)*(rPeriphNeg/chosenR); 
    float zPeriphPos =  beamLen + (z - beamLen)*(rPeriphNeg/chosenR); 
    // Now check if actual exit through endcap.
    if (fabs(zPeriphNeg) > trkLength) {
      int whichEndcap = (zPeriphNeg + beamLen > 0)  ?  1  :  -1;
      zPeriphNeg = whichEndcap*trkLength;
      rPeriphNeg = chosenR*(zPeriphNeg + beamLen)/(z + beamLen);
    } 
    if (fabs(zPeriphPos) > trkLength) {
      int whichEndcap = (zPeriphPos - beamLen > 0)  ?  1  :  -1;
      zPeriphPos = whichEndcap*trkLength;
      rPeriphPos = chosenR*(zPeriphPos - beamLen)/(z - beamLen);
    } 
    secBoundary[k].SetNextPoint(-beamLen,0);
    secBoundary[k].SetNextPoint(zPeriphNeg, rPeriphNeg);
    secBoundary[k].SetNextPoint(zPeriphPos, rPeriphPos);
    secBoundary[k].SetNextPoint(beamLen,0);
    secBoundary[k].SetNextPoint(-beamLen,0);
    secBoundary[k].SetFillColor(kGreen-2);
    //cout<<k<<" eta "<<eta[k]<<" coords "<<zPeriphNeg<<" "<<rPeriphNeg<<" "<<zPeriphPos<<" "<<rPeriphPos<<endl;
    unsigned int iHash = 3405+k*9;
    secBoundary[k].SetFillStyle(iHash);
    secBoundary[k].Draw("f");
    d1.Update();
  }

  /*
  // Draw inner tracker radius.
  TLine trkIn(0, trkInRad, trkLength, trkInRad);
  trkIn.SetLineWidth(2);
  trkIn.Draw();
  */
  // Draw chosen radius.
  TLine chosen(0, chosenR, trkLength, chosenR);
  chosen.SetLineWidth(3);
  chosen.SetLineStyle(2);
  chosen.SetLineColor(kRed);
  chosen.Draw();
  // Draw beam spot.
  TLine beam(-beamLen, 0., beamLen, 0.);
  beam.SetLineWidth(5);
  beam.SetLineColor(kRed);
  beam.Draw();

  // Draw a digitized stub.
  /*
  if (drawStub) {
    float r = 58+103.0382*iDigi_RT/pow(2,10);
    float z = 640.*iDigi_Z/pow(2,12);
    cout<<"Stub at r = "<<r<<" z = "<<z<<endl;
    TMarker stub(z,r,30);
    stub.Draw();
    d1.Update();
  }
  */

  d1.Update();

  cout<<"Writing EtaSectors.png"<<endl;
  d1.Print("EtaSectors.png");

  cin.get();
}
