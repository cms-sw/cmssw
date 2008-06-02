void rootlogon()
{
	gSystem->Load("libFWCoreFWLite");
	gSystem->Load("libPhysicsToolsMVAComputer");
	gSystem->Load("libPhysicsToolsMVATrainer");
	AutoLibraryLoader::enable();
}
