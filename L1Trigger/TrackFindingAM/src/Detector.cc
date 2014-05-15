#include "../interface/Detector.h"

Detector::Detector(){
  dump=NULL;
}

void Detector::addLayer(int lNum, int nbLad, int nbMod, int segmentSize, int sstripSize){
  if(dump==NULL)
      dump = new SuperStrip(sstripSize);
  Layer* l = new Layer(nbLad, nbMod, segmentSize, sstripSize);
  layerNumber.push_back(lNum);
  layers.push_back(l);
}

Detector::~Detector(){
  if(dump!=NULL)
    delete dump;
  for(unsigned int i=0;i<layers.size();i++){
    delete layers[i];
  }
  layers.clear();
  layerNumber.clear();
}

int Detector::getLayerPosition(int pos){
  for(unsigned int i=0;i<layerNumber.size();i++){
    if(layerNumber[i]==pos)
      return i;
  }
  return -1;
}

Layer* Detector::getLayerFromAbsolutePosition(int pos){
  if(pos>-1 && pos<(int)layers.size())
    return layers[pos];
  return NULL;
}

Layer* Detector::getLayer(int pos){
  int localPosition = getLayerPosition(pos);
  if(localPosition!=-1)
    return layers[localPosition];
  return NULL;
}

void Detector::clear(){
  for(unsigned int i=0;i<layers.size();i++){
    layers[i]->clear();
  }
}

void Detector::receiveHit(const Hit& h){
  //  cout<<h<<endl;
  int l = getLayerPosition(h.getLayer());
  if(l!=-1){
    Layer* la = getLayerFromAbsolutePosition(l);
    if(la!=NULL){
      SuperStrip* s = la->getLadder(h.getLadder())->getModule(h.getModule())->getSegment(h.getSegment())->getSuperStrip(h.getStripNumber());
      if(s==NULL)
	cout<<"ERROR : Cannot find superStrip corresponding to the following hit : "<<h<<endl;
      else
	s->touch(&h);
    }
  }
  else{
    //int tmp_layer = (int)h.getLayer();
    //cout<<"no layer "<<tmp_layer<<endl;
  }
}

int Detector::getNbLayers(){
  return (int)layers.size();
}

SuperStrip* Detector::getDump(){
  return dump;
}
