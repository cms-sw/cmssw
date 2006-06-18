//
// Macro to read in geomdet positions and draw them
//

int plotDets( TString cut )
{

  int showFlag = 3;

  return plotDets( cut, showFlag );
}


int plotDets( TString cut, int showFlag )
{

  TCanvas* c1 = new TCanvas("c1","Misalignment validation",10,10,800,420);
  c1->Divide(2,1);

  TFile* alignedFile = new TFile("aligned.root");
  TTree* tmpTree     = (TTree*)alignedFile->Get("theTree");
  TTree* alignedTree = (TTree*)tmpTree->CopyTree(cut);

  TFile* misalignedFile = new TFile("misaligned.root");
  tmpTree        = (TTree*)misalignedFile->Get("theTree");
  TTree* misalignedTree = (TTree*)tmpTree->CopyTree(cut);

  // X-Y projection
  float xmax = TMath::Nint(alignedTree->GetMaximum("x")/10+1)*10;
  float xmin = TMath::Nint(alignedTree->GetMinimum("x")/10-1)*10;
  int nx = TMath::Nint(xmax-xmin);
  TH2F* hRange1        = new TH2F("hRange1","X-Y Projection",nx,xmin,xmax,nx,xmin,xmax);
  TH2F* hAlignedXY    = new TH2F("hAlignedXY","X-Y Projection - Aligned",nx,xmin,xmax,nx,xmin,xmax);
  TH2F* hMisalignedXY = new TH2F("hMisalignedXY","X-Y Projection - Misaligned",
								 nx,xmin,xmax,nx,xmin,xmax);
  hAlignedXY->SetMarkerColor(4);
  hMisalignedXY->SetMarkerColor(2);

  alignedTree->Project("hAlignedXY","y:x");
  misalignedTree->Project("hMisalignedXY","y:x");

  c1->cd(1);
  hRange1->Draw();
  hRange1->SetXTitle("X");
  hRange1->SetYTitle("Y");
  if ( showFlag&1 ) hAlignedXY->Draw("same");
  if ( showFlag&(1<<1) ) hMisalignedXY->Draw("same");

  // R-Z projection
  xmin = 0;
  float zmax = TMath::Nint(alignedTree->GetMaximum("z")/10+1)*10;
  float zmin = TMath::Nint(alignedTree->GetMinimum("z")/10-1)*10;
  int nx = TMath::Nint(xmax-xmin);
  int nz = TMath::Nint(zmax-zmin);
  TH2F* hRange2       = new TH2F("hRange2","R-Z Projection",nz,zmin,zmax,nx,xmin,xmax);
  TH2F* hAlignedRZ    = new TH2F("hAlignedRZ","X-Y Projection - Aligned",nz,zmin,zmax,nx,xmin,xmax);
  TH2F* hMisalignedRZ = new TH2F("hMisalignedRZ","X-Y Projection - Misaligned",
								 nz,zmin,zmax,nx,xmin,xmax);

  hAlignedRZ->SetMarkerColor(4);
  hMisalignedRZ->SetMarkerColor(2);

  alignedTree->Project("hAlignedRZ","sqrt(x^2+y^2):z");
  misalignedTree->Project("hMisalignedRZ","sqrt(x^2+y^2):z");

  c1->cd(2);
  hRange2->Draw();
  hRange2->SetYTitle("R");
  hRange2->SetXTitle("Z");
  if ( showFlag&1 ) hAlignedRZ->Draw("same");
  if ( showFlag&(1<<1) ) hMisalignedRZ->Draw("same");

  return 0;

}
