{

  gSystem->Load("libFWCoreFWLite.so"); 
  AutoLibraryLoader::enable();
//  TFile* file = TFile::Open("dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/RelVal/2006/11/3/RelVal120pre4ZPrimeDijets700/D61917D6-2E6C-DB11-BBBA-0030487074C3.root");
  TFile* file = TFile::Open("SinglePiE50HCAL_cfi_GEN_SIM_DIGI_L1_DIGI2RAW_RAW2DIGI_RECO.root");

}
