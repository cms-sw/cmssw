void drawEdm(){
  
  gStyle->SetOptStat("nemruo");

  TChain *ev = new TChain("Events");
  ev->Add("rfio:/castor/cern.ch/user/e/edwenger/MBSkimHIRECO_900GeV_2k_v3.root");  


  // vertices
  ev->SetAlias("hivtx","recoVertexs_hiSelectedVertex__HIRECOSKIM.obj");
  ev->SetAlias("ppvtx","recoVertexs_offlinePrimaryVertices__RECO.obj");

  // tracks
  ev->SetAlias("pptrk","recoTracks_generalTracks__RECO.obj");
  ev->SetAlias("hitrk","recoTracks_hiSelectedTracks__HIRECOSKIM.obj");  
  
  // jets
  ev->SetAlias("ppjet","recoCaloJets_iterativeCone5CaloJets__RECO.obj");
  ev->SetAlias("hijet","recoCaloJets_iterativeConePu5CaloJets__HIRECOSKIM.obj");

  // number of high purity tracks
  ev->SetAlias("nHPtracks","Sum$( (pptrk.qualityMask() & (1<<2) ) > 0 )");


  // draw example
  TCanvas *c1 = new TCanvas("c1","c1",600,500);
  ev->Draw("hivtx[0].z()-ppvtx[0].z()>>hvtx(100,-1,1)","!ppvtx.isFake()","goff");
  hvtx->SetTitle("hiSelectedVertex - offlinePrimaryVertices; z-vtx residual [cm]");
  hvtx->Draw();
  gPad->SetLogy();


  // scan example
  ev->Scan("ppjet.pt():hijet.pt():ppjet.eta():hijet.eta():ppjet.phi():hijet.phi()","hijet.pt()>7");

}
