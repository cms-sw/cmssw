#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <iomanip>

class CCAmap {
public:
  void map(int rbx, int rm, int card, int cca, int ccaq,
	   int& ieta, int& iphi, int& depth, int& det,int& spigot, int& fiber, int& crate, int& fiber_chan, int& G_Dcc, int& H_slot, int& TB);
  //};

  CCAmap(const char* fname);

private:
  struct DataMap{
        
    int m_side; 
    int m_eta;
    int m_phi;
    int m_depth;
    int m_det;
    int m_rbx;
    int m_rm;
    int m_qie;
    int m_adc;
    int m_fi_ch;
    int m_crate;
    int m_htr;
    int m_fpga;
    int m_htr_fi;
    int m_spigo;
    int m_dcc;
    
   
  } v;
  
  std::vector<DataMap> vm;
  

  DataMap DataMapMaker(std::vector<std::string>& Words);

  void convert(const char* lineData, std::vector<std::string>& chunkWords);  


};
void CCAmap::map(int rbx, int rm, int card, int cca, int ccaq,
		 int& ieta, int& iphi, int& depth, int& det, int& spigot, int& fiber, int& crate, int& fiber_chan, int& G_Dcc, int& H_slot, int& TB) {
  
    
  
  std::vector<DataMap>::const_iterator i;
  int sad;

for (i=vm.begin();i!=vm.end();i++){
  
    if ((rbx==i->m_rbx)&&(rm==i->m_rm)&&(card==i->m_qie)&&(cca*2+ccaq==i->m_adc)){
      break;
    }
  }//end of i loop
 if (i!=vm.end()){
   sad=i->m_rbx;
   ieta=i->m_eta*i->m_side;
   iphi=i->m_phi;
   depth=i->m_depth;
   det=i->m_det;
   spigot=i->m_spigo;
   fiber=i->m_htr_fi;
   crate =i->m_crate;
   fiber_chan=i->m_fi_ch;
   G_Dcc=i->m_dcc;
   H_slot=i->m_htr;
   TB=i->m_fpga;
   

 }

}


CCAmap::DataMap CCAmap::DataMapMaker(std::vector<std::string>& Words){


  int k_side=atoi (Words[0].c_str());
  int k_eta=atoi (Words[1].c_str());
  int k_phi=atoi (Words[2].c_str());
  int k_depth=atoi (Words[3].c_str());
  int k_rm=atoi (Words[7].c_str());
  int k_qie=atoi (Words[9].c_str());
  int k_adc=atoi (Words[10].c_str());
  int k_fi_ch=atoi (Words[12].c_str());
  int k_crate=atoi (Words[13].c_str());
  int k_htr=atoi (Words[14].c_str());
  int k_htr_fi=atoi (Words[16].c_str());
  int k_spigo=atoi (Words[18].c_str());
  int k_dcc=atoi (Words[19].c_str());
  std::string k_det= Words[4];
  std::string k_rbx= Words[5];
  std::string k_fpga= Words[15];
  
   
  DataMap dM;

  dM.m_side=k_side;
    
  dM.m_eta=k_eta;
  dM.m_phi=k_phi;
  dM.m_depth=k_depth;
  if(k_det=="HB"){dM.m_det=1;}
  else if(k_det=="HE"){dM.m_det=2;}
  else if(k_det=="HF"){dM.m_det=4;}
  else if(k_det=="HO"){dM.m_det=3;}

  
  int rbxsign;  
  int z_zero;
  
  if (dM.m_det==3){
    if(k_side==1){ 
      rbxsign=1;
      z_zero=0;
    }
    else if(k_side==-1){ 
      rbxsign=0;
      z_zero=0;
    }
    else if(k_side==0){ //temporary until map of HO is done
      rbxsign=1;
      z_zero=12;
    }

  } else {//not HO
    z_zero=0;
    if(k_side==1){ rbxsign=1;}
    else if(k_side==-1){ rbxsign=0;}
  
}
  

  int det_num;
  std::string rbxnum = k_rbx.substr(3);  
  if(k_det=="HB"){det_num=1;}
  else if(k_det=="HE"){det_num=2;}
  else if(k_det=="HF"){det_num=3;}
  else if(k_det=="HO"){det_num=4;}
  
  dM.m_rbx=(det_num-1)*18+rbxsign*90+z_zero+(atoi (rbxnum.c_str()));
   
  dM.m_rm=k_rm;
  dM.m_qie=k_qie;//RM_card
  dM.m_adc=k_adc;//RM_chan
  dM.m_fi_ch=k_fi_ch;
  dM.m_crate=k_crate;
  dM.m_htr=k_htr;
  if (k_fpga=="top"){dM.m_fpga=1;}
  else if (k_fpga=="bot"){dM.m_fpga=0;}
  dM.m_htr_fi=k_htr_fi;//fiber
  dM.m_spigo=k_spigo;
  //FED id
  if ((k_crate==4)&&(k_dcc==1)){
    dM.m_dcc=0;}
  if ((k_crate==4)&&(k_dcc==2)){
    dM.m_dcc=1;}
  if ((k_crate==0)&&(k_dcc==1)){
    dM.m_dcc=2;}
  if ((k_crate==0)&&(k_dcc==2)){
    dM.m_dcc=3;}
  if ((k_crate==1)&&(k_dcc==1)){
    dM.m_dcc=4;}
  if ((k_crate==1)&&(k_dcc==2)){
    dM.m_dcc=5;}
  if ((k_crate==5)&&(k_dcc==1)){
    dM.m_dcc=6;}
  if ((k_crate==5)&&(k_dcc==2)){
    dM.m_dcc=7;}
  if ((k_crate==11)&&(k_dcc==1)){
    dM.m_dcc=8;}
  if ((k_crate==11)&&(k_dcc==2)){
    dM.m_dcc=9;}
  if ((k_crate==15)&&(k_dcc==1)){
    dM.m_dcc=10;}
  if ((k_crate==15)&&(k_dcc==2)){
    dM.m_dcc=11;}
  if ((k_crate==17)&&(k_dcc==1)){
    dM.m_dcc=12;}
  if ((k_crate==17)&&(k_dcc==2)){
    dM.m_dcc=13;}
  if ((k_crate==14)&&(k_dcc==1)){
    dM.m_dcc=14;}
  if ((k_crate==14)&&(k_dcc==2)){
    dM.m_dcc=15;}
  if ((k_crate==10)&&(k_dcc==1)){
    dM.m_dcc=16;}
  if ((k_crate==10)&&(k_dcc==2)){
    dM.m_dcc=17;}
  if ((k_crate==2)&&(k_dcc==1)){
    dM.m_dcc=18;}
  if ((k_crate==2)&&(k_dcc==2)){
    dM.m_dcc=19;}
  if ((k_crate==9)&&(k_dcc==1)){
    dM.m_dcc=20;}
  if ((k_crate==9)&&(k_dcc==2)){
    dM.m_dcc=21;}
  if ((k_crate==12)&&(k_dcc==1)){
    dM.m_dcc=22;}
  if ((k_crate==12)&&(k_dcc==2)){
    dM.m_dcc=23;}
  if ((k_crate==3)&&(k_dcc==1)){
    dM.m_dcc=24;}
  if ((k_crate==3)&&(k_dcc==2)){
    dM.m_dcc=25;}
  if ((k_crate==7)&&(k_dcc==1)){
    dM.m_dcc=26;}
  if ((k_crate==7)&&(k_dcc==2)){
    dM.m_dcc=27;}
  if ((k_crate==6)&&(k_dcc==1)){
    dM.m_dcc=28;}
  if ((k_crate==6)&&(k_dcc==2)){
    dM.m_dcc=29;}
  if ((k_crate==13)&&(k_dcc==1)){
    dM.m_dcc=30;}
  if ((k_crate==13)&&(k_dcc==2)){
    dM.m_dcc=31;}
  return dM;  

}

void CCAmap::convert(const char* lineData, std::vector<std::string>& chunkWords) {
  std::string chunk;
  chunkWords.clear();
  for (int i=0; lineData[i]!=0; ++i) {
    if (isspace(lineData[i])) {
      if (!chunk.empty()) {
	chunkWords.push_back(chunk);
	chunk.clear();
      }
    } else {
      chunk+=lineData[i];
    }
	
  }
  if (!chunk.empty()) chunkWords.push_back(chunk);
}//end of convert

CCAmap::CCAmap(const char* fname){

  
  FILE* fp;
  char instr[3024];
  fp=fopen(fname,"r");
  std::vector<std::string> WordChunks;


  if(fp==NULL){printf("file not found,%s\n",fname);exit(1);}
  while (!feof(fp) ){
    fgets(instr,3024,fp);
    if (instr[0]=='#')continue;
   
    convert(instr,WordChunks);    
    if (WordChunks.size()>20) {
      v=DataMapMaker(WordChunks);
      vm.push_back(v);
    }
    
    
  }
  if (fp!=NULL) fclose(fp);
}



class CCApatternmaker {
public:
  void makePattern(char* rbxchar,const char* fname);
private:
  
  std::vector<unsigned char> makeSequence(int rbx, int rm, int card, int cca, int ccaq);
  
  CCAmap* theMap;
};




void CCApatternmaker::makePattern(char* rbxchar,const char* fname){
  std::string rbxstr=rbxchar;  
  theMap=new CCAmap(fname);
  int rm_mx,card_mx;
  bool in_he = false;
  bool in_hb = false;
  bool in_hf = false;
  bool in_ho = false;
  time_t Now=time(0);
 
  std::string timeStamp=ctime(&Now);
  timeStamp[timeStamp.length()-1] = 0; // remove '\n'

  std::cout << "<?xml version='1.0'?>"<<std::endl;
  
  std::cout << "<CFGBrick>"<<std::endl;
  std::cout << " <Parameter name='RBX' type='string'>"<< rbxstr << "</Parameter>"<<std::endl;
  std::cout << "  <Parameter name='INFOTYPE' type='string'>PATTERNS</Parameter>"<<std::endl;
  std::cout << " <Parameter name='CREATIONTAG' type='string'>CableMappingPattern</Parameter>"<<std::endl;
  std::cout << "  <Parameter name='CREATIONSTAMP' type='string'>"<<timeStamp<<"</Parameter>"<<std::endl;
 
  
  int detnum,PorM,rbx;
  std::string det = rbxstr.substr(0,2);
  std::string PM = rbxstr.substr(2,1);
  std::string rbxnum = rbxstr.substr(3);  
  if(det=="HB"){detnum=1;rm_mx=4;card_mx=3;}
  else if(det=="HE"){detnum=2;rm_mx=4;card_mx=3;}
  else if(det=="HF"){detnum=3;rm_mx=3;card_mx=4;}
  else if (det=="HO"){detnum=4;rm_mx=4;card_mx=3;}//not verified
  PorM=(PM=="P")?(1):(0);
  rbx=(detnum-1)*18+PorM*90+(atoi (rbxnum.c_str()));//HO 0 not inclueded yet
  
 
  for (int rm=1; rm<=rm_mx; rm++) {
    //std::cout << "; rm " << rm << std::endl;
    for (int card=1; card<=card_mx; card++) {
      for (int cca=0; cca<3; cca++) {
	for (int ccaq=0; ccaq<2; ccaq++) {
	  
	  std::vector<unsigned char> ccadata=makeSequence(rbx,rm,card,cca,ccaq);
	  
	  std::cout<<"<Data elements='10' encoding='hex' rm='"<<rm<<"' card='"<<card<<"' qie='"<<(cca*2+ccaq)<<"' >";
	  
	  for (int i=0;i<10;i++){
	    std::cout<<" "<<std::setw(2)<<std::setfill('0')<<std::hex<<(int)ccadata[i];
	 }
	  std::cout <<"</Data>"<<std::endl;
	}
      }
    }
  }
  std::cout<<std::dec<<" </CFGBrick>"<<std::endl;
  //std::cout<<rbx<<std::endl;
}







std::vector<unsigned char> CCApatternmaker::makeSequence(int rbx, int rm, int card, int cca, int ccaq) {
  
  int ieta,iphi,depth,det,spigot, fiber, crate, fiber_chan, G_Dcc, H_slot, TB;
  theMap->map(rbx,rm,card,cca,ccaq,ieta,iphi,depth,det,spigot, fiber, crate, fiber_chan, G_Dcc, H_slot, TB);

  
std::vector<unsigned char> cc;
  const int header = 0x75;
  cc.push_back(((header&0x7F)<<0));
  cc.push_back(((abs(ieta)&0x3F)<<0)|((ieta>0)?(0x40):(0x0)));
  cc.push_back(((iphi&0x7F)<<0));
  cc.push_back(((depth&0x7)<<0)|((det&0xF)<<3));
  cc.push_back(((spigot&0xF)<<0)|(((fiber-1)&0x7)<<4));
  cc.push_back(((crate&0x1F)<<0)|((fiber_chan&0x3)<<5));
  cc.push_back(((G_Dcc&0x3F)<<0));
  cc.push_back(((H_slot&0x1F)<<0)|((TB&0x1)<<5)|(((rbx&0x80)>>7)<<6));
  cc.push_back(((rbx&0x7F)<<0));
  cc.push_back((((rm-1)&0x3)<<0)|(((card-1)&0x3)<<2)|(((cca*2+ccaq)&0x7)<<4));
  
  
  
  return cc;
}

int main(int argc, char* argv[]) {
  if (argc<3) {
    std::cout << "Usage: " << argv[0] << " [rbx] [filename]\n";
    return 1;
  }
  CCApatternmaker maker;
  maker.makePattern(argv[1], argv[2]);
  
  return 0;
}
