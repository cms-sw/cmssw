root -l -b << EOF
  TString makeshared(gSystem->GetMakeSharedLib());
  makeshared = makeshared.ReplaceAll("-W ", "");
  makeshared = makeshared.ReplaceAll("-Wshadow ", " -std=c++0x ");
  gSystem->SetMakeSharedLib(makeshared);
  gSystem->Load("libFWCoreFWLite");
  AutoLibraryLoader::enable();
  gSystem->Load("libDataFormatsFWLite.so");
  gSystem->Load("libDataFormatsCommon.so");
  gSystem->Load("libDataFormatsSiStripDetId.so");
 .x PlotMacro.C++();
EOF
