#include "../interface/PrincipalFitGenerator.h"
#include "../interface/PatternFinder.h"

PrincipalFitGenerator::PrincipalFitGenerator(string f, SectorTree *s){
  inputDirectory = f;
  st = s;
}

TChain* PrincipalFitGenerator::createTChain(){

  cout<<"Loading files from "<<inputDirectory<<" (this may take several minutes)..."<<endl;

  TChain* TT = new TChain("L1TrackTrigger");

  TSystemDirectory dir(inputDirectory.c_str(), inputDirectory.c_str());
  TList *files = dir.GetListOfFiles();

  if (files) {
    TSystemFile *file;
    TString fname;
    TIter next(files);

    while((file=(TSystemFile*)next())){
      fname = file->GetName();
      if (!file->IsDirectory() && fname.EndsWith(".root")) {
	TT->Add((inputDirectory + fname.Data()).c_str());
      }
    }
  }

  p_m_stub_tp = &m_stub_tp; 
  p_m_stub_layer = &m_stub_layer; 
  p_m_stub_module = &m_stub_module;
  p_m_stub_ladder = &m_stub_ladder;
  p_m_stub_pxGEN = &m_stub_pxGEN;  
  p_m_stub_pyGEN = &m_stub_pyGEN;  
  p_m_stub_etaGEN = &m_stub_etaGEN;  
  p_m_stub_x = &m_stub_x;
  p_m_stub_y = &m_stub_y;
  p_m_stub_z = &m_stub_z;
  p_m_clus_zmc = &m_clus_zmc;
  p_m_stub_clust1 = &m_stub_clust1;
  p_m_stub_pdg = &m_stub_pdg;
  p_m_stub_x0 = &m_stub_x0;
  p_m_stub_y0 = &m_stub_y0;
  p_m_stub_z0 = &m_stub_z0;

  TT->SetBranchAddress("STUB_n",         &m_stub);
  TT->SetBranchAddress("STUB_tp",        &p_m_stub_tp);
  TT->SetBranchAddress("STUB_layer",     &p_m_stub_layer);
  TT->SetBranchAddress("STUB_module",    &p_m_stub_module);
  TT->SetBranchAddress("STUB_ladder",    &p_m_stub_ladder);
  TT->SetBranchAddress("STUB_pxGEN",     &p_m_stub_pxGEN);
  TT->SetBranchAddress("STUB_pyGEN",     &p_m_stub_pyGEN);
  TT->SetBranchAddress("STUB_etaGEN",    &p_m_stub_etaGEN);
  TT->SetBranchAddress("STUB_x",         &p_m_stub_x);
  TT->SetBranchAddress("STUB_y",         &p_m_stub_y);
  TT->SetBranchAddress("STUB_z",         &p_m_stub_z);
  TT->SetBranchAddress("STUB_pdgID",     &p_m_stub_pdg);
  TT->SetBranchAddress("STUB_clust1",    &p_m_stub_clust1);
  TT->SetBranchAddress("CLUS_zmc",       &p_m_clus_zmc);
  TT->SetBranchAddress("STUB_X0",    &p_m_stub_x0);
  TT->SetBranchAddress("STUB_Y0",    &p_m_stub_y0);
  TT->SetBranchAddress("STUB_Z0",    &p_m_stub_z0);

  int nb_entries = TT->GetEntries();
  cout<<nb_entries<<" events found."<<endl;

  return TT;
}

void PrincipalFitGenerator::generatePrincipal(map<int,pair<float,float> > eta_limits, float min_pt, float max_pt, float min_eta, float max_eta){

  if(st->getNbSectors()==0){
    cout<<"No sector found"<<endl;
    return;
  }

  map< int, vector<int> > detector_config = Sector::readConfig("detector.cfg");
  //Get the layers IDs
  vector<int> tracker_layers;
  Sector* first_sector = st->getAllSectors()[0];
  for(int i=0;i<first_sector->getNbLayers();i++){
    tracker_layers.push_back(first_sector->getLayerID(i));
  }
  int nb_ladders = -1;
  if(detector_config.find(first_sector->getLayerID(0))!=detector_config.end()){
    nb_ladders = detector_config[first_sector->getLayerID(0)][1];
  }
  else{
    cout<<"We do not know the number of ladders in layer "<<first_sector->getLayerID(0)<<endl;
    return;
  }

  TChain* TT = createTChain();
  int n_entries_TT = TT->GetEntries();

  int evtIndex=0;

  int layers[tracker_layers.size()];
  vector<int> ladder_per_layer(tracker_layers.size());
  vector<int> module_per_layer(tracker_layers.size());

  int max_tracks_number = 10000000;

  while(evtIndex<n_entries_TT){
    TT->GetEntry(evtIndex++);

    if(evtIndex%100000==0)
      cout<<"Event "<<evtIndex<<endl;

    if(!selectTrack(tracker_layers, layers, ladder_per_layer, module_per_layer, eta_limits, min_pt, max_pt, min_eta, max_eta))
      continue;

    Sector* sector = st->getSector(ladder_per_layer, module_per_layer);
    if(sector==NULL){
      //cout<<"No sector found"<<endl;
      continue;
    }

    double sec_phi = (2*M_PI/nb_ladders)*(sector->getLadderCode(tracker_layers[0],CMSPatternLayer::getLadderCode(tracker_layers[0],m_stub_ladder[layers[0]])));

    PrincipalTrackFitter* fitter = (PrincipalTrackFitter*)sector->getFitter();
    if(fitter==NULL){
      cout<<"No fitter associated to the sector, creating a default one!"<<endl;
      fitter = new PrincipalTrackFitter(sector->getNbLayers(), 1000);
      sector->setFitter(fitter);
    }
    fitter->setPhiRotation(sec_phi);

    double data_coord[3*tracker_layers.size()];
    int data_tracker[3*tracker_layers.size()];

    for(unsigned int i=0;i<tracker_layers.size();i++){
      int j = layers[i];
      
      //data_coord[i*3]=m_stub_x[j]*cos(sec_phi)+m_stub_y[j]*sin(sec_phi);
      //data_coord[i*3+1]=-m_stub_x[j]*sin(sec_phi)+m_stub_y[j]*cos(sec_phi);
      data_coord[i*3]=m_stub_x[j];
      data_coord[i*3+1]=m_stub_y[j];
      if(m_stub_layer[j]>7)
	data_coord[i*3+2]=m_stub_z[j];
      else
	data_coord[i*3+2]=m_clus_zmc[m_stub_clust1[j]];

      data_tracker[i*3]=sector->getLayerIndex(m_stub_layer[j]);
      data_tracker[i*3+1]=sector->getLadderCode(m_stub_layer[j],m_stub_ladder[j]);
      data_tracker[i*3+2] = sector->getModuleCode(m_stub_layer[j], CMSPatternLayer::getLadderCode(m_stub_layer[j],m_stub_ladder[j]), CMSPatternLayer::getModuleCode(m_stub_layer[j],m_stub_module[j]));
    }

    fitter->addTrackForPrincipal(data_tracker, data_coord);
    if(fitter->hasPrincipalParams())//The params are computed->we stop the process
      break;

    if(evtIndex>max_tracks_number){
      fitter->forcePrincipalParamsComputing();
    }

  }

  delete TT;
}

void PrincipalFitGenerator::generateMultiDim(map<int,pair<float,float> > eta_limits, float min_pt, float max_pt, float min_eta, float max_eta){

  if(st->getNbSectors()==0){
    cout<<"No sector found"<<endl;
    return;
  }

  map< int, vector<int> > detector_config = Sector::readConfig("detector.cfg");

  //Get the layers IDs
  vector<int> tracker_layers;
  Sector* first_sector = st->getAllSectors()[0];
  cout<<"on utilise les layers ";
  for(int i=0;i<first_sector->getNbLayers();i++){
    tracker_layers.push_back(first_sector->getLayerID(i));
    cout<<first_sector->getLayerID(i)<<",";
  }
  cout<<endl;

  /*
  int nb_ladders = -1;
  if(detector_config.find(first_sector->getLayerID(0))!=detector_config.end()){
    nb_ladders = detector_config[first_sector->getLayerID(0)][1];
  }
  else{
    cout<<"We do not know the number of ladders in layer "<<first_sector->getLayerID(0)<<endl;
    return;
  }
  */

  TChain* TT = createTChain();
  int n_entries_TT = TT->GetEntries();

  int evtIndex=0;

  int layers[tracker_layers.size()];
  vector<int> ladder_per_layer(tracker_layers.size());
  vector<int> module_per_layer(tracker_layers.size());

  int max_tracks_number = 10000000;

  while(evtIndex<n_entries_TT){
    TT->GetEntry(evtIndex++);

    if(evtIndex%100000==0)
      cout<<"Event "<<evtIndex<<endl;

    if(!selectTrack(tracker_layers, layers, ladder_per_layer, module_per_layer, eta_limits, min_pt, max_pt, min_eta, max_eta))
      continue;
    
    //cout<<"track is complete, searching for a sector"<<endl;

    Sector* sector = st->getSector(ladder_per_layer, module_per_layer);
    if(sector==NULL){
      //cout<<"No sector found"<<endl;
      continue;
    }

    PrincipalTrackFitter* fitter = (PrincipalTrackFitter*)sector->getFitter();
    if(fitter==NULL){
      cout<<"No fitter associated to the sector!"<<endl;
      break;
    }

    double data_coord[3*tracker_layers.size()];
    int data_tracker[3*tracker_layers.size()];

    for(unsigned int i=0;i<tracker_layers.size();i++){
      int j = layers[i];
      
      //data_coord[i*3]=m_stub_x[j]*cos(sec_phi)+m_stub_y[j]*sin(sec_phi);
      //data_coord[i*3+1]=-m_stub_x[j]*sin(sec_phi)+m_stub_y[j]*cos(sec_phi);
      data_coord[i*3]=m_stub_x[j];
      data_coord[i*3+1]=m_stub_y[j];
      if(m_stub_layer[j]>7)
	data_coord[i*3+2]=m_stub_z[j];
      else
	data_coord[i*3+2]=m_clus_zmc[m_stub_clust1[j]];

      data_tracker[i*3]=sector->getLayerIndex(m_stub_layer[j]);
      data_tracker[i*3+1]=sector->getLadderCode(m_stub_layer[j],CMSPatternLayer::getLadderCode(m_stub_layer[j],m_stub_ladder[j]));
      data_tracker[i*3+2] = sector->getModuleCode(m_stub_layer[j], CMSPatternLayer::getLadderCode(m_stub_layer[j],m_stub_ladder[j]), CMSPatternLayer::getModuleCode(m_stub_layer[j],m_stub_module[j]));
    }

    int charge = m_stub_pdg[layers[0]]/abs(m_stub_pdg[layers[0]]);
    double pt_GEN = sqrt(m_stub_pxGEN[layers[0]]*m_stub_pxGEN[layers[0]]+m_stub_pyGEN[layers[0]]*m_stub_pyGEN[layers[0]]);
    double phi0 = atan2(m_stub_pyGEN[layers[0]],m_stub_pxGEN[layers[0]]);
    double d0 = (m_stub_y0[layers[0]]-tan(phi0)*m_stub_x0[layers[0]])*cos(phi0);
    //double d0 = 10.0;

    double values[5];
    values[0] = charge*pt_GEN;//PT
    values[1] = phi0;
    values[2] = d0;
    values[3] = m_stub_etaGEN[layers[0]];
    values[4] = m_stub_z0[layers[0]];

    fitter->addTrackForMultiDimFit(data_tracker, data_coord, values);
    if(fitter->hasMultiDimFitParams())//The params are computed->we stop the process
      break;

    if(evtIndex>max_tracks_number){
      fitter->forceMultiDimFitParamsComputing();
    }

  }

  delete TT;
}

bool PrincipalFitGenerator::selectTrack(vector<int> &tracker_layers, int* layers, vector<int> &ladder_per_layer, vector<int> &module_per_layer,
					map<int,pair<float,float> > &eta_limits, float min_pt, float max_pt, float min_eta, float max_eta){

   //initialize arrays
    for(unsigned int j=0;j<tracker_layers.size();j++){
      layers[j]=-1;
    }
    for(unsigned int j=0;j<tracker_layers.size();j++){
      ladder_per_layer[j]=-1;
    }
    for(unsigned int j=0;j<tracker_layers.size();j++){
      module_per_layer[j]=-1;
    }

    float current_eta = -1;

    for(int j=0;j<m_stub;j++){

      //cout<<"layer "<<m_stub_layer[j]<<" ladder "<<m_stub_ladder[j]<<" module "<<m_stub_module[j]<<endl;

      if(m_stub_etaGEN[j]<min_eta){// eta of the generating particule is bellow the threshold -> we do not use it
      	return false;
      }
      if(m_stub_etaGEN[j]>max_eta){// eta of the generating particule is above the threshold -> we do not use it
      	return false;
      }
      float pt_GEN = sqrt(m_stub_pxGEN[j]*m_stub_pxGEN[j]+m_stub_pyGEN[j]*m_stub_pyGEN[j]);
      if(pt_GEN<min_pt){// The PT of the generating particule is below the minimum required -> we do not use it
	return false;
      }
      if(pt_GEN>max_pt){// The PT of the generating particule is above the maximum accepted -> we do not use it
	return false;
      }

      int layer = m_stub_layer[j];
      
      int layer_position=-1;
      for(unsigned int cpt=0;cpt<tracker_layers.size();cpt++){
	if(layer==tracker_layers[cpt]){
	  layer_position=(int)cpt;
	  break;
	}
      }
      
      if(layer_position!=-1){ // is this layer in the layer list?
	layers[layer_position]=j;
	ladder_per_layer[layer_position]=CMSPatternLayer::getLadderCode(layer,m_stub_ladder[j]);
	short module = -1;
	module = CMSPatternLayer::getModuleCode(layer, m_stub_module[j]);
	module_per_layer[layer_position]=module;
      }
      
      current_eta = m_stub_etaGEN[j];
    }

    /**************************************
    Selection on the stubs/layer
    We need at least one stub per layer
    **************************************/
    bool missing_stub = false;
    for(unsigned int j=0;j<tracker_layers.size();j++){
      if(layers[j]==-1){
	missing_stub=true;
	if(st->getNbSectors()==1){
	  if(eta_limits.find(tracker_layers[j])!=eta_limits.end()){//we have eta boundaries for this layer
	    pair<float,float> limits = eta_limits[tracker_layers[j]];
	    if(current_eta<limits.first || current_eta>limits.second){ // we are outside the eta limits for this layer
	      //cout<<"missing hit on layer "<<tracker_layers[j]<<" for track with eta="<<current_eta<<endl;
	      layers[j]=-2;
	      //we put a ladder and a module just to be inside the sector
	      ladder_per_layer[j]=st->getAllSectors()[0]->getLadders(j)[0];
	      module_per_layer[j]=st->getAllSectors()[0]->getModules(j,ladder_per_layer[j])[0];
	      //cout<<"Add stub for sector : "<<ladder_per_layer[j]<<" / "<<module_per_layer[j]<<endl;
	      //debug=true;
	      missing_stub=false;//we will create a fake superstrip, so the stub is not missing
	    }
	  }
	}
      }
      if(missing_stub)
	break;
    }

    if(missing_stub){
      return false;//no stub on each layer -> drop the event    
    }

    return true;
}

void PrincipalFitGenerator::generate(map<int,pair<float,float> > eta_limits, float min_pt, float max_pt, float min_eta, float max_eta){
  generatePrincipal(eta_limits, min_pt, max_pt, min_eta, max_eta);
  generateMultiDim(eta_limits, min_pt, max_pt, min_eta, max_eta);
}
