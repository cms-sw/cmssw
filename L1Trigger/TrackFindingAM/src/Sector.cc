#include "../interface/Sector.h"

map< int, vector<int> > Sector::readConfig(string name){
  string line;
  ifstream myfile (name.c_str());
  int lineNumber=0;
  map< int, vector<int> > detector_config;
  if (myfile.is_open()){
    cout<<"Using configuration found in detector.cfg"<<endl;
    while ( myfile.good() )
      {
	getline (myfile,line);
	bool error=false;
	vector<int> layer_config;
	if(line.length()>0 && line.find("#")!=0){
	  layer_config.clear();
	  size_t pos = -1;
	  pos = line.find(",");
	  bool go = true;
	  while(go){
	    if(pos==string::npos)
	      go=false;
	    string sub = line.substr(0,pos);
	    line=line.substr(pos+1);

	    int number;
	    std::istringstream ss( sub );
	    ss >> number;
	    if (ss.bad()){
	      cout<<"Syntax error line "<<lineNumber<<" of detector.cfg : line ignored."<<endl;
	      error = true;
	    }
	    else{
	      layer_config.push_back(number);
	    }

	    pos = line.find(",");
	  }
	}
	lineNumber++;
	if(!error){
	  if(layer_config.size()>0 && layer_config.size()<4)
	    cout<<"Syntax error line "<<lineNumber<<" of detector.cfg : line ignored."<<endl;
	  else if(layer_config.size()>0)
	    detector_config[layer_config[0]]=layer_config;
	}
	error=false;
      }
    myfile.close();
  }
  else{
    cout << "No detector.cfg file found : using default configuration."<<endl;
    vector<int> layers = CMSPatternLayer::getLayerIDs();
    for(unsigned int i=0;i<layers.size();i++){
      vector<int> layer_config;
      layer_config.push_back(layers[i]);//layer's ID
      layer_config.push_back(CMSPatternLayer::getNbLadders(layers[i]));//ladder number
      layer_config.push_back(CMSPatternLayer::getNbModules(layers[i],-1));//max module number
      layer_config.push_back(CMSPatternLayer::getNbStripsInSegment());//strip number in segment
      detector_config[layer_config[0]]=layer_config;
    }
  }
  return detector_config;
}

Sector::Sector(){
  patterns = new PatternTree();
  fitter = NULL;
  officialID=-1;
}

Sector::Sector(vector<int> layersID){
  for(unsigned int i=0;i<layersID.size();i++){
    map<int, vector<int> > l;
    vector<int> ladders;
    m_modules[layersID[i]]=l;
    m_ladders[layersID[i]]=ladders;
  }
  patterns = new PatternTree();
  fitter = NULL;
  officialID=-1;
}

Sector::Sector(const Sector& s){
  m_modules=s.m_modules;
  m_ladders=s.m_ladders;
  patterns = new PatternTree();
  // if(s.fitter!=NULL)
  //  fitter = s.fitter->clone();
  //else
  fitter=NULL;
  officialID=s.officialID;
}

Sector::~Sector(){
  delete patterns;
  if(fitter!=NULL)
    delete fitter;
}

Sector& Sector::operator=(Sector& s){
  m_modules = s.m_modules; 
  m_ladders = s.m_ladders;
  fitter=NULL;
  officialID=s.officialID;
  return *this;
}

bool Sector::operator==(Sector& s){
  return (m_modules==s.m_modules && m_ladders==s.m_ladders);
}

void Sector::addLayer(int layer){
  map<int, vector<int> > l;
  m_modules[layer]=l;
}

void Sector::addLadders(int layer, int firstLadder, int nbLadders){
  map<int, map<int, vector<int> > >::iterator it;
  it=m_modules.find(layer);
  
  if(it!=m_modules.end()){//the layer does exist
    if(it->second.size()==0){//we have no ladders yet->we add the ladders
      int maxLadderNumber = CMSPatternLayer::getNbLadders(layer);
      vector<int> ladderList;
      for(int i=0;i<nbLadders;i++){
	vector<int> v;
	int ladderID = (firstLadder+i)%maxLadderNumber;
	it->second[ladderID]=v;
	ladderList.push_back(ladderID);
      }
      m_ladders[layer]=ladderList;
    }
  }
}

void Sector::addModules(int layer, int ladder, int firstModule, int nbModules){
  map<int, map<int, vector<int> > >::iterator it;
  it=m_modules.find(layer);
  
  if(it!=m_modules.end()){//the layer does exist
    map<int, vector<int> >::iterator it_ladder;
    it_ladder=it->second.find(ladder);
    if(it_ladder!=it->second.end()){ // the ladder does exist
      if(it_ladder->second.size()==0){//we have no module yet->we add the modules
	int maxModuleNumber = CMSPatternLayer::getNbModules(layer, ladder);
	for(int i=0;i<nbModules;i++){
	  int moduleID = (firstModule+i)%maxModuleNumber;
	  it_ladder->second.push_back(moduleID);
	}
      }
    }
  }
}

int Sector::getLadderCode(int layer, int ladder){
  map<int, vector<int> >::const_iterator it = m_ladders.find(layer);

  if(it==m_ladders.end())
    return -1;//the layer does not exist

  for(unsigned int i=0;i<it->second.size();i++){
    if(it->second[i]==ladder)
      return i;
  }

  return -1;
  
}

int Sector::getModuleCode(int layer, int ladder, int module){
  map<int, map<int, vector<int> > >::const_iterator it = m_modules.find(layer);

  if(it==m_modules.end()){
    cout<<"layer non trouve"<<endl;
    return -1;//the layer does not exist
  }

  map<int, vector<int> >::const_iterator it_ladder = it->second.find(ladder);

  if(it_ladder==it->second.end())
    return -1;//the module does not exist

  for(unsigned int i=0;i<it_ladder->second.size();i++){
    if(it_ladder->second[i]==module)
      return i;
  }

  return -1;
      
}

int Sector::getNbLayers(){
  return m_modules.size();
}

bool Sector::contains(const Hit& h){
  map<int, map<int, vector<int> > >::iterator it = m_modules.find(h.getLayer());
  if(it==m_modules.end())
    return false;
  map<int, vector<int> >::iterator it2 = it->second.find(h.getLadder());
  if(it2==it->second.end())
    return false;
  for(unsigned int i=0;i<it2->second.size();i++){
    if(it2->second[i]==h.getModule())
      return true;
  }
  return false;
}

vector<int> Sector::getLadders(int l) const{
  vector<int> v;
  if(l<0 && l>=(int)m_modules.size())//no such layer
    return v;
  
  map<int, vector<int> >::const_iterator it=m_ladders.begin();
  int cpt=0;
  while(cpt<l){
    cpt++;
    it++;
  }

  for(unsigned int i=0;i<it->second.size();i++){
    v.push_back(it->second[i]);
  }
  return v;
}

vector<int> Sector::getModules(int lay, int l) const{
  vector<int> mods;
  if(lay<0 && lay>=(int)m_modules.size())//no such layer
    return mods;

  map<int, map<int, vector<int> > >::const_iterator it=m_modules.begin();
  int cpt=0;
  while(cpt<lay){
    cpt++;
    it++;
  }
  map<int, vector<int> >::const_iterator it_ladder = it->second.find(l);
  if(it_ladder==it->second.end())//no such ladder
    return mods;
  return it_ladder->second;
}

int Sector::getLayerID(int i) const{
  vector<int> l;
  for(map<int, map<int, vector<int> > >::const_iterator it=m_modules.begin();it!=m_modules.end();it++){
    l.push_back(it->first);
  }
  sort(l.begin(),l.end());
  if(i>-1 && i<(int)l.size())
    return l[i];
  else
    return -1;
}

int Sector::getLayerIndex(int i) const{
  vector<int> l;
  for(map<int, map<int, vector<int> > >::const_iterator it=m_modules.begin();it!=m_modules.end();it++){
    l.push_back(it->first);
  }
  sort(l.begin(),l.end());
  for(unsigned int j=0;j<l.size();j++){
    if(l[j]==i)
      return j;
  }
  return -1;
}

PatternTree* Sector::getPatternTree(){
  return patterns;
}

string Sector::getIDString(){
  ostringstream oss;
  vector<int> layer_list;
  for(map<int, map<int, vector<int> > >::const_iterator it_layer=m_modules.begin();it_layer!=m_modules.end();it_layer++){
    layer_list.push_back(it_layer->first);
  }
  sort(layer_list.begin(), layer_list.end());
  for(unsigned int i=0;i<layer_list.size();i++){
    vector<int> lad = m_ladders[layer_list[i]];
    for(unsigned int j=0;j<lad.size();j++){
      if(j!=0)
	oss<<"-";
      oss<<lad[j];
    }
    oss<<" ";
  }
  return oss.str();  
}

void Sector::setOfficialID(int id){
  if(id>-1)
    officialID=id;
  else
    officialID=-1;
}

int Sector::getOfficialID(){
  return officialID;
}

int Sector::getKey(){
  int k = 0;
  stringstream oss;
  int min_layer = 1000;
  if(m_modules.size()>0){
    for(map<int, map<int, vector<int> > >::const_iterator it=m_modules.begin();it!=m_modules.end();it++){
      if(it->first<min_layer)
	min_layer=it->first;
    }
    map<int, vector<int> > lad = m_modules[min_layer];
    vector<int> list;
    for(map<int, vector<int> >::const_iterator it_ladder=lad.begin();it_ladder!=lad.end();it_ladder++){
      list.push_back(it_ladder->first);
    }
    if(list.size()!=0){
      sort(list.begin(), list.end());
      for(unsigned int j=0;j<list.size();j++){
	oss<<list[j];
      }
      oss>>k;
    }
  }
  return k;
}

vector<string> Sector::getKeys(){
  vector<string> res;
  vector< vector<int> > copy;

  for(map<int, map<int, vector<int> > >::const_iterator it=m_modules.begin();it!=m_modules.end();it++){
    vector<int> v;
    for(map<int, vector<int> >::const_iterator it_ladder=it->second.begin();it_ladder!=it->second.end();it_ladder++){
      v.push_back(it_ladder->first);
    }
    copy.push_back(v);
  }

  getRecKeys(copy, 1, "", res);
  return res;
}

void Sector::getRecKeys(vector< vector<int> > &v, int level, string temp, vector<string> &res){
  if((int)v.size()==level){
    vector<int> list = v[level-1];
    for(unsigned int i=0;i<list.size();i++){
      ostringstream oss;
      oss<<std::setfill('0');
      oss<<setw(2)<<list[i];
      string k = temp+oss.str();
      res.push_back(k);
    }
  }
  else{
    int newLevel = level+1;
    for(unsigned int i=0;i<v[level-1].size();i++){
      ostringstream oss;
      oss<<std::setfill('0');
      oss<<setw(2)<<(v[level-1][i]);
      getRecKeys(v, newLevel, temp+oss.str(), res);
    }
  }
}

ostream& operator<<(ostream& out, const Sector& s){
  
  for(map<int, vector<int> >::const_iterator it=s.m_ladders.begin();it!=s.m_ladders.end();it++){
    out<<"Layer : "<<it->first<<endl;
    vector<int> ladders = it->second;
    map<int, vector<int> > ladders_modules = (s.m_modules.find(it->first))->second;
    for(unsigned int j=0;j<ladders.size();j++){
      out<<"\tLadder : "<<ladders[j]<<endl;
      out<<"\t\tModules : ";
      vector<int> modules = ladders_modules[ladders[j]];
      for(unsigned int i=0;i<modules.size();i++){
      	out<<modules[i]<<" ";
      }
      out<<endl;
    }
  }
  return out;
}

int Sector::getLDPatternNumber(){
  return patterns->getLDPatternNumber();
}

int Sector::getFDPatternNumber(){
  return patterns->getFDPatternNumber();
}

void Sector::computeAdaptativePatterns(short r){
    patterns->computeAdaptativePatterns(r);
}

void Sector::link(Detector& d){

  vector<vector<int> > ladders;
  vector<map<int, vector<int> > > modules;
  vector<int> layer_list;
  for(map<int, map<int, vector<int> > >::const_iterator it_layer=m_modules.begin();it_layer!=m_modules.end();it_layer++){
    layer_list.push_back(it_layer->first);
  }
  sort(layer_list.begin(), layer_list.end());

  for(unsigned int i=0;i<layer_list.size();i++){
    vector<int> lad = m_ladders[layer_list[i]];
    vector<int> ladder_list;
    for(unsigned int j=0;j<lad.size();j++){
      ladder_list.push_back(lad[j]);
    }
    ladders.push_back(ladder_list);
    modules.push_back(m_modules[layer_list[i]]);
  }

  for(unsigned int i=0;i<ladders.size();i++){
    for(unsigned int j=0;j<ladders[i].size();j++){
      cout<<ladders[i][j]<<" ";
    }
    cout<<endl;
  }

  patterns->link(d, ladders, modules);
}

vector<GradedPattern*> Sector::getActivePatterns(int active_threshold){
  vector<GradedPattern*> active_patterns;
  patterns->getActivePatterns(active_threshold, active_patterns);
  return active_patterns;
}

void Sector::setFitter(TrackFitter* f){
  if(fitter!=NULL)
    delete fitter;
  if(f!=NULL)
    fitter = f;
  else
    fitter = NULL;
}

TrackFitter* Sector::getFitter(){
  return fitter;
}

void Sector::updateFitterPhiRotation(){
  map< int, vector<int> > detector_config = readConfig("detector.cfg");
  //Get the layers IDs
  vector<int> tracker_layers;
  for(int i=0;i<getNbLayers();i++){
    tracker_layers.push_back(this->getLayerID(i));
  }
  int nb_ladders = -1;
  if(detector_config.find(getLayerID(0))!=detector_config.end()){
    nb_ladders = detector_config[getLayerID(0)][1];
  }
  else{
    cout<<"We do not know the number of ladders in layer "<<getLayerID(0)<<endl;
    return;
  }
  double sec_phi = (2*M_PI/nb_ladders)*(getLadders(0)[0]);
  if(fitter!=NULL){
    fitter->setPhiRotation(sec_phi);
  }
}
