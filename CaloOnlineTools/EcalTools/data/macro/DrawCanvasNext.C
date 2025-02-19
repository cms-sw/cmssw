{
  if(canvasNum+1 >= canvasNames.size())
    return;
  
  canvasNum++;
  currentCanvas->Close();
  currentCanvas = (TCanvas*) gDirectory->Get((canvasNames->at(canvasNum)).c_str());
  currentCanvas->Draw();
  currentCanvas->SetWindowPosition(200,50);
  currentCanvas->SetWindowSize(900,900);
}
