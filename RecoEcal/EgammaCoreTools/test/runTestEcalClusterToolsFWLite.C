{
gSystem->Load("libFWCoreFWLite.so"); 
FWLiteEnabler::enable();
gSystem->Load("libDataFormatsFWLite.so");
gSystem->Load("libGeometryCaloTopology.so");
gSystem->Load("libRecoEcalEgammaCoreTools.so");
TFile *f=TFile::Open("rfio:///castor/cern.ch/cms/store/user/meridian/meridian/SingleGammaPt35_DSZS_V1/SingleGammaPt35_DSZS_V1/00b02d884670d693cb397a1e0af88088/SingleGammaPt35_cfi_py_GEN_SIM_DIGI_L1_DIGI2RAW_RAW2DIGI_RECO_1.root");
gROOT->ProcessLine(".x testEcalClusterToolsFWLite.C++");
}
