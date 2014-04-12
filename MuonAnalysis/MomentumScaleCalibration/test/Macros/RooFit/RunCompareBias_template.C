{
  gROOT->Reset();
  gROOT->GetInterpreter()->AddIncludePath("INCLUDEPATH");
  gROOT->LoadMacro("CompareBias.cc+");
  CompareBias();
}
