
//SIMPLE SCRIPT for geometry plotting and debugging. 
void Geometry_plotter() {

 
  TGeoManager::Import("cmsSimGeom-Run4D500_validation.root");

  gGeoManager->GetListOfVolumes()->Print();
  TGeoVolume* tb2s = gGeoManager->FindVolumeFast("tracker:Tracker");

  if (!tb2s) {
    std::cerr << "ERROR: Volume 'tracker:OTDisc5R11EModule' not found." << std::endl;
    return;
  }

  // Assign a unique color to each material used inside tb2s
  std::map<std::string, Color_t> materialColors;
  int colorIndex = 2;

  // Loop over all nodes (components) inside the volume
  for (int i = 0; i < tb2s->GetNdaughters(); ++i) {
    TGeoNode* node = tb2s->GetNode(i);
    if (!node) continue;

    TGeoVolume* daughterVol = node->GetVolume();
    if (!daughterVol || !daughterVol->GetMaterial()) continue;

    std::string matName = daughterVol->GetMaterial()->GetName();
    //std::cout << "material is: " << matName << std::endl;

    if (materialColors.find(matName) == materialColors.end()) {
      materialColors[matName] = colorIndex++;
      //std::cout << "material is: " << matName << "  color is " << colorIndex << std::endl;

      if (colorIndex > 50) colorIndex = 2;

    }
    /*if(matName == "materials:Air") {
       daughterVol->SetTransparency(100);
    }*/

    //daughterVol->SetLineColor(materialColors[matName]);
    //daughterVol->SetFillColor(materialColors[matName]);
    //daughterVol->SetTransparency(10); // Optional transparency
  }

  tb2s->Draw("ogl");
}

