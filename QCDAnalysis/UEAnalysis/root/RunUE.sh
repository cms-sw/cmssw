// Compilation instructions:
//   make lib
//   make bin
//
//Examples for use the analysis macro
//List of parameter to be passed
// 1) file that contain th elist of root file to be analyzed
// 2) output file name
// 3) kind of Analysis possible option are UE, Jet, MPI that respectively means 
//    analysis of Underlying Event, study on charged jet properties, study on counting the MPI via MiniJets
// 4) trigger stream, possibel otion are MB, Jet20, Jet60, Jet120
// 5) luminosity scenario in pb
// 6) eta region 
// 7) Pt of Calo Jet for trigger selection
// 8) cuts on the minimu pt of the tracks in MeV
// 9) cuts on the minimum pt of MiniJet to be used only in case you are running the MPI scheme  

./UEAnalysis listMB_09.dat StreamMB_900_lumi1pb.root UE MB 1 2 0 900 0
