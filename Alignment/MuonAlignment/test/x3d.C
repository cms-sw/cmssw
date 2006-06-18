//
// Macro to read in geomdet positions and draw them
//

int x3d( TString cut )
{

  int showFlag = 2;

  return x3d( cut, showFlag );

}


int x3d( TString cut, int showFlag )
{

  // Retrieve trees and apply cut
  TFile* alignedFile = new TFile("aligned.root");
  TTree* tmpTree     = (TTree*)alignedFile->Get("theTree");
  TTree* alignedTree = (TTree*)tmpTree->CopyTree(cut);

  TFile* misalignedFile = new TFile("misaligned.root");
  tmpTree        = (TTree*)misalignedFile->Get("theTree");
  TTree* misalignedTree = (TTree*)tmpTree->CopyTree(cut);

  // Set tree branches
  float x,y,z,phi,theta,length,thick,width;
  float mx,my,mz,mphi,mtheta,mlength,mthick,mwidth;
  TRotMatrix* rot;
  TRotMatrix* mrot;
  double rad2deg = 180./3.1415926;

  alignedTree->SetBranchAddress( "x",      &x      );
  alignedTree->SetBranchAddress( "y",      &y      );
  alignedTree->SetBranchAddress( "z",      &z      );
  alignedTree->SetBranchAddress( "phi",    &phi    );
  alignedTree->SetBranchAddress( "theta",  &theta  );
  alignedTree->SetBranchAddress( "length", &length );
  alignedTree->SetBranchAddress( "width",  &width  );
  alignedTree->SetBranchAddress( "thick",  &thick  );
  alignedTree->SetBranchAddress( "rot",    &rot    );

  misalignedTree->SetBranchAddress( "x",      &mx      );
  misalignedTree->SetBranchAddress( "y",      &my      );
  misalignedTree->SetBranchAddress( "z",      &mz      );
  misalignedTree->SetBranchAddress( "phi",    &mphi    );
  misalignedTree->SetBranchAddress( "theta",  &mtheta  );
  misalignedTree->SetBranchAddress( "length", &mlength );
  misalignedTree->SetBranchAddress( "width",  &mwidth  );
  misalignedTree->SetBranchAddress( "thick",  &mthick  );
  misalignedTree->SetBranchAddress( "rot",    &mrot    );

  // Create canvas
  TCanvas* c1 = new TCanvas("c1","Detector units", 200, 10, 700, 500);
  c1->cd();

  TBRIK* IP = new TBRIK("IP","IP","void",0.,0.,0.);
  TNode* rootNode = new TNode("Root","Root","IP",0.,0.,0.);
  rootNode->cd();

  int entry = 0;
  while ( alignedTree->GetEntry(entry) && misalignedTree->GetEntry(entry) )
	{
	  entry++;
	  std::ostringstream name;

	  // Aligned detector
	  name << "aBrik" << entry;
	  TBRIK* aBrik = new TBRIK(name.str().c_str(),"Aligned detector unit","void",
							   0.01,0.01,length);
	  aBrik->SetLineColor(4);

	  // Detector node (position and orientation)
	  name.str("aNode"); name << entry;
	  TNode* aNode = new TNode(name.str().c_str(),name.str().c_str(),aBrik,x,y,z);
	  // Misaligned detector
 	  name.str("mBrik");
 	  name << entry;
 	  TBRIK* mBrik = new TBRIK(name.str().c_str(),"Misaligned detector unit","void",
							   0.01,0.01,mlength);
 	  mBrik->SetLineColor(2);

	  // Detector node (position and orientation)
 	  name.str("mNode"); name << entry;
 	  TNode* mNode = new TNode(name.str().c_str(),name.str().c_str(),mBrik,mx,my,mz);

	  //if (entry>5) break;
	}

  rootNode->cd();
  rootNode->Draw();
  
  c1->GetViewer3D();

  return 0;

}
