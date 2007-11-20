{

  gSystem->Load("libFWCoreFWLite.so"); 
  AutoLibraryLoader::enable();
//  TFile* file = TFile::Open("dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/RelVal/2006/11/3/RelVal120pre4ZPrimeDijets700/D61917D6-2E6C-DB11-BBBA-0030487074C3.root");
  TFile* file = TFile::Open("dcap://cmsdca.fnal.gov:24136/pnfs/fnal.gov/usr/cms/WAX/11/store/RelVal/2007/9/3/RelVal-RelValZPrimeDijets700-1188844800/0002/0690E11F-BC5A-DC11-81A2-0019DB29C620.root");

}
