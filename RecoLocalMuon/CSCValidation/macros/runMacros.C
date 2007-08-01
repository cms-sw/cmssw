{
  gROOT->Reset();
  gROOT->ProcessLine(".L myFunctions.C");

  std::string newReleasePath = "~/public/Validation/gametime/CMSSW_1_6_0_pre6/src/RecoLocalMuon/CSCValidation/test/";
  std::string refReleasePath = "~/public/Validation/gametime/CMSSW_1_5_2/src/RecoLocalMuon/CSCValidation/test/";
  int datatype = 2;              // 1 = data, 2 = mc

  TFile *f1;
  TFile *f2;

  f1 = OpenFiles(refReleasePath,datatype);
  f2 = OpenFiles(newReleasePath,datatype);

  //procuce wire and strip digi comparison plots
  Compare1DPlots2("Digis/hStripAll","Digis/hWireAll",f1,f2,"Strip Numbers Fired (All Chambers)","Wire Groups Fired (All Chambers)","digi_stripswires_all.gif");
  Compare1DPlots2("Digis/hStripCodeBroad","Digis/hWireCodeBroad",f1,f2,"hStripCodeBroad","hWireCodeBroad","digi_stripswires_hCodeBroad.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow1","Digis/hWireCodeNarrow1",f1,f2,"hStripCodeNarrow (Station 1)","hWireCodeNarrow (Station 1)","digi_stripswires_hCodeNarrow1.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow2","Digis/hWireCodeNarrow2",f1,f2,"hStripCodeNarrow (Station 2)","hWireCodeNarrow (Station 2)","digi_stripswires_hCodeNarrow2.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow3","Digis/hWireCodeNarrow3",f1,f2,"hStripCodeNarrow (Station 3)","hWireCodeNarrow (Station 3)","digi_stripswires_hCodeNarrow3.gif");
  Compare1DPlots2("Digis/hStripCodeNarrow4","Digis/hWireCodeNarrow4",f1,f2,"hStripCodeNarrow (Station 4)","hWireCodeNarrow (Station 4)","digi_stripswires_hCodeNarrow4.gif");
  Compare1DPlots2("Digis/hStripLayer1","Digis/hWireLayer1",f1,f2,"Strips Fired per Layer(Station 1)","Wires Fired per Layer(Station 1)","digi_stripswires_perlayer1.gif");
  Compare1DPlots2("Digis/hStripLayer2","Digis/hWireLayer2",f1,f2,"Strips Fired per Layer(Station 2)","Wires Fired per Layer(Station 2)","digi_stripswires_perlayer2.gif");
  Compare1DPlots2("Digis/hStripLayer3","Digis/hWireLayer3",f1,f2,"Strips Fired per Layer(Station 3)","Wires Fired per Layer(Station 3)","digi_stripswires_perlayer3.gif");
  Compare1DPlots2("Digis/hStripLayer4","Digis/hWireLayer4",f1,f2,"Strips Fired per Layer(Station 4)","Wires Fired per Layer(Station 4)","digi_stripswires_perlayer4.gif");
  Compare1DPlots2("Digis/hStripNFired","Digis/hWirenGroupsTotal",f1,f2,"Number of Fired Strips per Event","Number of Fired Wiregroups per Event","digi_stripswires_perevent.gif");
  Compare1DPlots2("Digis/hStripStrip1","Digis/hWireWire1",f1,f2,"Strip Numbers Fired (Station 1)","Wiregroup Numbers Fired (Station 1)","digi_stripswires_1.gif");
  Compare1DPlots2("Digis/hStripStrip2","Digis/hWireWire2",f1,f2,"Strip Numbers Fired (Station 2)","Wiregroup Numbers Fired (Station 2)","digi_stripswires_2.gif");
  Compare1DPlots2("Digis/hStripStrip3","Digis/hWireWire3",f1,f2,"Strip Numbers Fired (Station 3)","Wiregroup Numbers Fired (Station 3)","digi_stripswires_3.gif");
  Compare1DPlots2("Digis/hStripStrip4","Digis/hWireWire4",f1,f2,"Strip Numbers Fired (Station 4)","Wiregroup Numbers Fired (Station 4)","digi_stripswires_4.gif");
  Compare1DPlots1("Digis/hStripADCAll",f1,f2,"All ADC Values Above Cutoff","digi_stripadcs.gif");
  Compare1DPlots1("Digis/hWireTBinAll",f1,f2,"Signal Time Bin for All Wires","digi_wireTB.gif");


  //produce rechit comparison plots
  Compare1DPlots1("recHits/hRHCodeBroad",f1,f2,"hRHCodeBroad","rH_hRHCodeBroad.gif"); 
  Compare1DPlots2("recHits/hRHCodeNarrow1","recHits/hRHCodeNarrow2",f1,f2,"hRHCodeNarrow (Station 1)","hRHCodeNarrow (Station 2)","rH_hRHCodeNarrow_1a2.gif");
  Compare1DPlots2("recHits/hRHCodeNarrow3","recHits/hRHCodeNarrow4",f1,f2,"hRHCodeNarrow (Station 3)","hRHCodeNarrow (Station 4)","rH_hRHCodeNarrow_3a4.gif");
  Compare1DPlots2("recHits/hRHX1","recHits/hRHY1",f1,f2,"recHits, LocalX, Station 1","recHits, LocalY, Station 1","rH_local_pos_station1.gif");
  Compare1DPlots2("recHits/hRHX2","recHits/hRHY2",f1,f2,"recHits, LocalX, Station 2","recHits, LocalY, Station 2","rH_local_pos_station2.gif");
  Compare1DPlots2("recHits/hRHX3","recHits/hRHY3",f1,f2,"recHits, LocalX, Station 3","recHits, LocalY, Station 3","rH_local_pos_station3.gif");
  Compare1DPlots2("recHits/hRHX4","recHits/hRHY4",f1,f2,"recHits, LocalX, Station 4","recHits, LocalY, Station 4","rH_local_pos_station4.gif");
  Compare1DPlots2("recHits/hRHLayer1","recHits/hRHLayer2",f1,f2,"recHits in a Layer, Station 1","recHits in a Layer, Station 2","rH_per_layer_stations1and2.gif");
  Compare1DPlots2("recHits/hRHLayer3","recHits/hRHLayer4",f1,f2,"recHits in a Layer, Station 3","recHits in a Layer, Station 4","rH_per_layer_stations3and4.gif");


  //produce segment comparison plots
  Compare1DPlots2("Segments/hSCodeNarrow1","Segments/hSCodeNarrow2",f1,f2,"hSCodeNarrow (Station 1)","hSCodeNarrow (Station 2)","seg_hRHCodeNarrow_1a2.gif");
  Compare1DPlots2("Segments/hSCodeNarrow3","Segments/hSCodeNarrow4",f1,f2,"hSCodeNarrow (Station 3)","hSCodeNarrow (Station 4)","seg_hRHCodeNarrow_3a4.gif");
  Compare1DPlots2("Segments/hSGlobalPhi","Segments/hSGlobalTheta",f1,f2,"Segment Global Phi (all stations)","Segment Global Theta (all stations)","seg_globthetaphi.gif");
  Compare1DPlots2("Segments/hSTheta1","Segments/hSTheta2",f1,f2,"Segment Local Theta (Station 1)","Segment Local Theta (Station 2)","seg_localtheta_1a2.gif");
  Compare1DPlots2("Segments/hSTheta3","Segments/hSTheta4",f1,f2,"Segment Local Theta (Station 3)","Segment Local Theta (Station 4)","seg_localtheta_3a4.gif");
  Compare1DPlots2("Segments/hSnHits1","Segments/hSnHits2",f1,f2,"recHits per Segment (Station 1)","recHits per Segment (Station 2)","seg_nhits_1a2.gif");
  Compare1DPlots2("Segments/hSnHits3","Segments/hSnHits4",f1,f2,"recHits per Segment (Station 3)","recHits per Segment (Station 4)","seg_nhits_3a4.gif");
  Compare1DPlots2("Segments/hSnSegments","Segments/hSnhits",f1,f2,"Segments per Event","recHits per Segment (all stations)","seg_nhits_all.gif");


  //Make global position graphs from trees
  GlobalrHPosfromTree("Global recHit positions (Station 1)",f1,f2,1,"rH_global_pos_station1.gif");
  GlobalrHPosfromTree("Global recHit positions (Station 2)",f1,f2,2,"rH_global_pos_station2.gif");
  GlobalrHPosfromTree("Global recHit positions (Station 3)",f1,f2,3,"rH_global_pos_station3.gif");
  GlobalrHPosfromTree("Global recHit positions (Station 4)",f1,f2,4,"rH_global_pos_station4.gif");
  GlobalsegPosfromTree("Global Segment positions (Station 1)",f1,f2,1,"seg_global_pos_station1.gif");
  GlobalsegPosfromTree("Global Segment positions (Station 2)",f1,f2,2,"seg_global_pos_station2.gif");
  GlobalsegPosfromTree("Global Segment positions (Station 3)",f1,f2,3,"seg_global_pos_station3.gif");
  GlobalsegPosfromTree("Global Segment positions (Station 4)",f1,f2,4,"seg_global_pos_station4.gif");


}

