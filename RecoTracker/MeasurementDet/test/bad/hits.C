void hits() {
    TFile *_file0 = TFile::Open("tracks.root");
    Events->SetScanField(999999);
    Events->Scan("recoTracks_ctfWithMaterialTracks__TEST.obj.numberOfValidHits():recoTracks_ctfWithMaterialTracks__Rec.obj.numberOfValidHits():recoTracks_ctfWithMaterialTracks__Rec.obj.eta()","","",999999);
}
