#include "../interface/CMSPatternLayer.h"

CMSPatternLayer::CMSPatternLayer():PatternLayer(){

}

CMSPatternLayer* CMSPatternLayer::clone(){
  CMSPatternLayer* p = new CMSPatternLayer();
  p->bits=this->bits;
  memcpy(p->dc_bits,this->dc_bits, DC_BITS*sizeof(char));
  return p;
}

vector<SuperStrip*> CMSPatternLayer::getSuperStrip(int l, const vector<int>& ladd, const map<int, vector<int> >& modules, Detector& d){
  int nb_dc = getDCBitsNumber();
  int factor = (int)pow(2.0,nb_dc);
  vector<SuperStrip*> v;

  if(getPhi()==15){ // this is a fake superstrip! We link it to the dump superstrip
    vector<string> positions=getPositionsFromDC();
    for(unsigned int i=0;i<positions.size();i++){
      SuperStrip* patternStrip = d.getDump();
      v.push_back(patternStrip);
    }
    return v;
  }
  else{
    Layer* la = d.getLayerFromAbsolutePosition(l);
    if(la!=NULL){
      int ladderID = ladd[getPhi()];//getPhi() is the position in the sector;ladd[getPhi()] gives the ID of the ladder
      Ladder* patternLadder = la->getLadder(ladderID);
      if(patternLadder!=NULL){
	map<int, vector<int> >::const_iterator iterator = modules.find(ladderID); // get the vector of module IDs for this ladder
	int moduleID = iterator->second[getModule()];// getthe module ID from its position
	Module* patternModule = patternLadder->getModule(moduleID);
	if(patternModule!=NULL){
	  Segment* patternSegment = patternModule->getSegment(getSegment());
	  if(patternSegment!=NULL){
	    int base_index = getStrip()*factor;
	    vector<string> positions = getPositionsFromDC();
	    for(unsigned int i=0;i<positions.size();i++){
	      SuperStrip* patternStrip = patternSegment->getSuperStripFromIndex(base_index+PatternLayer::GRAY_POSITIONS[positions[i]]);
	      v.push_back(patternStrip);
	    }
	    return v;
	  }
	}
      }
    }
    cout<<"Error : can not link layer "<<l<<" ladder "<<ladd[getPhi()]<<" module "<<getModule()<<" segment "<<getSegment()<<" strip "<<getStrip()<<endl;
  }
  return v;
}

void CMSPatternLayer::setValues(short m, short phi, short strip, short seg){
  bits |= (m&MOD_MASK)<<MOD_START_BIT |
    (phi&PHI_MASK)<<PHI_START_BIT |
    (strip&STRIP_MASK)<<STRIP_START_BIT |
    (seg&SEG_MASK)<<SEG_START_BIT;
}

short CMSPatternLayer::getModule(){
  int val = bits.to_ulong();
  short r = (val>>MOD_START_BIT)&MOD_MASK;
  return r;
}

short CMSPatternLayer::getPhi(){
  int val = bits.to_ulong();
  short r = (val>>PHI_START_BIT)&PHI_MASK;
  return r;
}

short CMSPatternLayer::getStrip(){
  int val = bits.to_ulong();
  short r = (val>>STRIP_START_BIT)&STRIP_MASK;
  return r;
}

short CMSPatternLayer::getSegment(){
  int val = bits.to_ulong();
  short r = (val>>SEG_START_BIT)&SEG_MASK;
  return r;
}

string CMSPatternLayer::toString(){
  ostringstream oss;
  oss<<"Ladder "<<getPhi()<<" Module "<<getModule()<<" Segment "<<getSegment()<<" strip "<<getStrip();
  if(dc_bits[0]!=3){
    oss<<" (";
    for(int i=0;i<DC_BITS;i++){
      if(dc_bits[i]==2)
	oss<<"X";
      else if(dc_bits[i]!=3)
	oss<<(int)dc_bits[i];
    }
    oss<<")";
  }
  return oss.str();
}

string CMSPatternLayer::toStringBinary(){
  ostringstream oss;
  oss<<getIntValue();
  if(dc_bits[0]!=3){
    oss<<" (";
    for(int i=0;i<DC_BITS;i++){
      if(dc_bits[i]==2)
	oss<<"X";
      else if(dc_bits[i]!=3)
	oss<<(int)dc_bits[i];
    }
    oss<<")";
  }
  return oss.str();
}

vector<int> CMSPatternLayer::getLayerIDs(){
  vector<int> layers;

  //BARREL
  layers.push_back(5);
  layers.push_back(6);
  layers.push_back(7);
  layers.push_back(8);
  layers.push_back(9);
  layers.push_back(10);

  //ENDCAP 1
  layers.push_back(11);
  layers.push_back(12);
  layers.push_back(13);
  layers.push_back(14);
  layers.push_back(15);

  //ENDCAP 2
  layers.push_back(18);
  layers.push_back(19);
  layers.push_back(20);
  layers.push_back(21);
  layers.push_back(22);

  return layers;
}

int CMSPatternLayer::getNbStripsInSegment(){
  return 1024;
}

int CMSPatternLayer::getSegmentCode(int layerID, int ladderID, int segmentID){
  if(layerID>7 && layerID<11)
    return segmentID;
  if(layerID>=5 && layerID<=7)
    return segmentID/16;
  if(ladderID<=8)
    return segmentID/16;
  return segmentID;
}


int CMSPatternLayer::getModuleCode(int layerID, int moduleID){
  switch(layerID){
  case 5 : return (moduleID/2);
  case 6 : return (moduleID/2);
  case 7 : return (moduleID/2);
  case 8 : return moduleID;
  case 9 : return moduleID;
  case 10 : return moduleID;
  default : return moduleID;
  }
}

int CMSPatternLayer::getLadderCode(int layerID, int ladderID){
  return ladderID;
}

 int CMSPatternLayer::getNbLadders(int layerID){
   if(layerID<5 || layerID>24)
     return -1;
   switch(layerID){
   case 5 : return 16;
   case 6 : return 24;
   case 7 : return 34;
   case 8 : return 48;
   case 9 : return 62;
   case 10 : return 76;
   default : return 15;
   }
 }

int CMSPatternLayer::getNbModules(int layerID, int ladderID){
  if(layerID==5)
    return 64;
  if(layerID==6)
    return 56;
  if(layerID==7)
    return 54;
  if(layerID>=8 && layerID<=10)
    return 24;
  if(layerID>=11 && layerID<=24){
    switch(ladderID){
    case 0:return 20;
    case 1:return 24;
    case 2:return 28;
    case 3:return 28;
    case 4:return 32;
    case 5:return 36;
    case 6:return 36;
    case 7:return 40;
    case 8:return 40;
    case 9:return 48;
    case 10:return 56;
    case 11:return 64;
    case 12:return 68;
    case 13:return 72;
    case 14:return 80;
    default:return 80;
    }
  }
  return -1;
}

map<int, pair<float,float> > CMSPatternLayer::getLayerDefInEta(){
  map<int,pair<float,float> > eta;
  eta[5]=pair<float,float>(-2.2,2.2);
  eta[6]=pair<float,float>(-1.72,1.72);
  eta[7]=pair<float,float>(-1.4,1.4);
  eta[8]=pair<float,float>(-1.2,1.2);
  eta[9]=pair<float,float>(-1.1,1.1);
  eta[10]=pair<float,float>(-0.9,0.9);
  eta[11]=pair<float,float>(1.1,2.4);
  eta[12]=pair<float,float>(1.2,2.4);
  eta[13]=pair<float,float>(1.3,2.4);
  eta[14]=pair<float,float>(1.4,2.5);
  eta[15]=pair<float,float>(1.6,2.5);
  eta[18]=pair<float,float>(-2.4,-1.1);
  eta[19]=pair<float,float>(-2.4,-1.2);
  eta[20]=pair<float,float>(-2.4,-1.3);
  eta[21]=pair<float,float>(-2.5,-1.4);
  eta[22]=pair<float,float>(-2.5,-1.6);
  return eta;
}
