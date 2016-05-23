#!/bin/bash

executable=`echo $0 | sed 's/.sh/.C/'` #assume the .sh and .C file have the same name
arguments=''
if [ $# -ge 1 ]; then arguments=$arguments"($1" ;fi
if [ $# -ge 2 ]; then arguments=$arguments",$2" ;fi
if [ $# -ge 3 ]; then arguments=$arguments",$3" ;fi
if [ $# -ge 4 ]; then arguments=$arguments",$4" ;fi
if [ $# -ge 5 ]; then arguments=$arguments",$5" ;fi
if [ $# -ge 6 ]; then arguments=$arguments",$6" ;fi
if [ $# -ge 1 ]; then arguments=$arguments");" ;fi

# #AutoLibraryLoader::enable();
# #gSystem->Load("libDataFormatsCommon.so");
# #gSystem->Load("libDataFormatsHepMCCandidate.so");
# #gSystem->Load("libDataFormatsFWLite.so");


root -l -b << EOF
  TString makeshared(gSystem->GetMakeSharedLib());
  makeshared.ReplaceAll("-W ", "-Wno-deprecated-declarations -Wno-deprecated -Wno-unused-local-typedefs -Wno-attributes ");
  makeshared.ReplaceAll("-Woverloaded-virtual ", " ");
  makeshared.ReplaceAll("-Wshadow ", " -std=c++0x -D__USE_XOPEN2K8 ");
  cout << "Compilling with the following arguments: " << makeshared << endl;
  gSystem->SetMakeSharedLib(makeshared);
  gSystem->Load("libFWCoreFWLite");
  FWLiteEnabler::enable()
  gSystem->Load("libCommonToolsTrackerMap.so");
  .x $executable+$arguments
EOF

