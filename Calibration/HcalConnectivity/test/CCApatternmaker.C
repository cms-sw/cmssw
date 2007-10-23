#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>

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
  else if(k_det=="HF"){dM.m_det=3;}
  else if(k_det=="HO"){dM.m_det=4;}

  
  int rbxsign;  
  int z_zero;
  
  if (dM.m_det==4){
    if(k_side==1){ 
      rbxsign=1;
      z_zero=0;
    }
    else if(k_side==-1){ 
      rbxsign=0;
      z_zero=0;
    }
    else if(k_side==0){ 
      rbxsign=1;
      z_zero=12;
    }

  } else {//not HO
    z_zero=0;
    if(k_side==1){ rbxsign=1;}
    else if(k_side==-1){ rbxsign=0;}
  
}
  
  std::string rbxnum;  
  if (k_rbx[4]!=0){rbxnum = k_rbx[3]+k_rbx[4];}
  else {rbxnum= k_rbx[3];}
  dM.m_rbx=(dM.m_det-1)*18+rbxsign*90+z_zero+(atoi (rbxnum.c_str()));
  //dM.m_rbx=1;
 
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
  dM.m_dcc=k_dcc;

  //i plan to delet the unused members
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
      chunk+=lineData[i];//find actual function i need!
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
  void makePattern(int rbx,const char* fname);
private:
  void packPair(unsigned char q0, unsigned char q1, unsigned char& b0, unsigned char& b1);
  std::vector<unsigned char> packCCA(int rbx, int rm, int card, int cca);
  std::vector<unsigned char> makeSequence(int rbx, int rm, int card, int cca, int ccaq);
  int rmCode(int rbx, int rm);
  CCAmap* theMap;
};


int CCApatternmaker::rmCode(int rbx, int rm) {
  // this is valid for HE!
  switch (rm) {
  case(1):return 32;
  case(2):return 16;
  case(3):return 2;
  case(4):return 1;
  default: return 0;
  }

}

void CCApatternmaker::makePattern(int rbx,const char* fname){
  theMap=new CCAmap(fname);
  for (int rm=1; rm<=4; rm++) {
    std::cout << "; rm " << rm << std::endl;
    for (int card=1; card<=3; card++) {
      for (int cca=0; cca<3; cca++) {
	std::vector<unsigned char> ccadata=packCCA(rbx,rm,card,cca);
	
		
	std::cout << "w 0 " << rmCode(rbx,rm) << " 10 ";
	int dev=cca;
	if (card==2) dev+=16;
	if (card==3) dev+=32;
	std::cout << dev << " 0x8";
	
	for (int i=0; i<10; i++)
	  std::cout << ' ' << (int)ccadata[i];
	std::cout << std::endl;
	
	std::cout << "w 0 " << rmCode(rbx,rm) << " 10 ";
	std::cout << dev << " 0x12";
	
	for (int i=10; i<20; i++)
	  std::cout << ' ' << (int)ccadata[i];
	std::cout << std::endl;
	
      }
    }
  }
}

std::vector<unsigned char> CCApatternmaker::packCCA(int rbx, int rm, int card, int cca) {
  std::vector<unsigned char> q0=makeSequence(rbx,rm,card,cca,0);
  std::vector<unsigned char> q1=makeSequence(rbx,rm,card,cca,1);
  std::vector<unsigned char> data;
  for (int i=0; i<10; i++) {
    unsigned char b0,b1;
    packPair(q0[i],q1[i],b0,b1);
    data.push_back(b0);
    data.push_back(b1);
 }
  return data;
}

void CCApatternmaker::packPair(unsigned char q0, unsigned char q1, unsigned char& b0, unsigned char& b1) {

  b0 = 0;
  if (q0&0x01){ b0|=0x40;}
  if (q0&0x02){ b0|=0x20;}
  if (q0&0x04){ b0|=0x10;}
  if (q0&0x08){ b0|=0x08;}
  if (q0&0x10){ b0|=0x04;}
  if (q0&0x20){ b0|=0x02;}
  if (q0&0x40){ b0|=0x01;}


 b1 = 0;
  if (q1&0x01){ b1|=0x80;}
  if (q1&0x02){ b1|=0x40;}
  if (q1&0x04){ b1|=0x20;}
  if (q1&0x08){ b1|=0x10;}
  if (q1&0x10){ b1|=0x08;}
  if (q1&0x20){ b1|=0x04;}
  if (q1&0x40){ b1|=0x02;}
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
  maker.makePattern(atoi(argv[1]), argv[2]);
  
  return 0;
}







