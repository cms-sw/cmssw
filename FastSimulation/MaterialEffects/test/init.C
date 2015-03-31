{

// initialize the core of the framework, and load the PFRootEvent 
// library, which contains the ROOT interface
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libFastSimulationMaterialEffects.so");
AutoLibraryLoader::enable();

}
