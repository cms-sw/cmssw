init()
{
gSystem->Load("libFWCoreFWLite");
AutoLibraryLoader::enable();

}



setAlias()
{
Events->SetAlias("hits","recoTracksedmRefProdTorecoDeDxHitssAssociationVector_dedxHitsFromRefitter__tracks.obj.data_");
Events->SetAlias("harmonic4","recoTracksedmRefProdTofloatsAssociationVector_dedxHarmonic4__tracks.obj.data_");
Events->SetAlias("normCharge","hits.m_charge");
Events->SetAlias("hitLayer","hits.layer()");
Events->SetAlias("hitSubDet","hits.subDet()");
Events->SetAlias("hitSubDetSide","hits.subDetSide()");
Events->SetAlias("pathLen","hits.m_pathLength");
cout << "Aliases defined: " << endl;
cout << "  hits:  DeDxHit object, see LXR" << endl; 
cout << "  normCharge:  normalized charge (i.e. dE/dX)" << endl; 
cout << "  hitLayer:  layer of a given DeDxHit" << endl; 
cout << "  hitSubDet:  subDet of a given DeDxHit" << endl; 
cout << "  hitSubDetSide:  +Z or -Z side of a given DeDxHit" << endl; 
cout << "  pathLen:  path length used for normalization" << endl; 
}

drawAll()
{
 new TCanvas;
 Events->Draw("harmonic4");
  new TCanvas;
 Events->Draw("normCharge");
 new TCanvas ;
 Events->Draw("normCharge:pathLen","pathLen<1. && normCharge < 7","COLZ");
 new TCanvas;
 Events->Draw("normCharge:pathLen","pathLen<1. && normCharge < 7","profile");

}

doAll(const char * name = "test.root")
{
 init();
 TFile *_file0 = TFile::Open(name);
 setAlias();
 gStyle->SetPalette(1) ;
 drawAll();
 
}

