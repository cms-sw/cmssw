#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <string>

using namespace std;

int main(){
  float elem33,elem34,elem44,elem35,elem45,elem55,elem46,elem56,elem66,elem57,elem67,elem77;
  int index,flag,flag1;
  int nrlines=0;

  std::vector<int>   index_id;
  std::vector<float> Elem33;
  std::vector<float> Elem34;
  std::vector<float> Elem44;
  std::vector<float> Elem35;
  std::vector<float> Elem45;
  std::vector<float> Elem55;
  std::vector<float> Elem46;
  std::vector<float> Elem56;
  std::vector<float> Elem66;
  std::vector<float> Elem57;
  std::vector<float> Elem67;
  std::vector<float> Elem77;
  
  std::ifstream dbdata; 
  dbdata.open("/nfshome0/boeriu/cal_data/merged_data/matrixSummary2010_03_18_run131361.dat",std::ios::in); 
  if(!dbdata) {
    std::cerr <<"Error: matrixSummary2010_03_18_run131361.dat -> no such file!"<< std::endl;
    exit(1);
  }

  while (!dbdata.eof() ) { 
    dbdata >> index >>elem33>>elem34>>elem44>>elem35>>elem45>>elem55>>elem46>>elem56>>elem66>>elem57>>elem67>>elem77 >>flag >>flag1; 
    index_id.push_back(index);
    Elem33.push_back(elem33);
    Elem34.push_back(elem34);
    Elem44.push_back(elem44);
    Elem35.push_back(elem35);
    Elem45.push_back(elem45);
    Elem55.push_back(elem55);
    Elem46.push_back(elem46);
    Elem56.push_back(elem56);
    Elem66.push_back(elem66);
    Elem57.push_back(elem57);
    Elem67.push_back(elem67);
    Elem77.push_back(elem77);
    nrlines++;
  }
  dbdata.close();
  std::ofstream myMatrixFile("goodMatrix2010_03_18_run131361.dat",std::ios::out);
 
  for(int i=0; i<nrlines-1;++i){
    
    if (Elem33[i]<30.0 && Elem33[i]>0){
      if (Elem34[i]>-5.0 && Elem34[i]<15.0){
	if (Elem44[i]>0 && Elem44[i]<30.0){
	  if (Elem35[i]<25. && Elem35[i]>-5.0){ 
	    if (Elem45[i]<30.&& Elem45[i]>-5.0){ 
	      if (Elem55[i]<30.&& Elem55[i]>0){ 
		if (Elem46[i]>-5. && Elem46[i]<30.0){
		  if(Elem56[i]<25.&& Elem56[i]>-5.0){
		    if (Elem66[i]<25.&& Elem66[i]>0){
		      if (Elem57[i]>-5. && Elem57[i]<30.0){
			if (Elem67[i]<15.0 && Elem67[i]>-5.0) {
			  if (Elem77[i]<25. && Elem77[i]>0 ){
			      if(Elem34[i]*Elem34[i]<Elem33[i] && Elem34[i]*Elem34[i]<Elem44[i]){
				if(Elem35[i]*Elem35[i]<Elem33[i] && Elem35[i]*Elem35[i]<Elem55[i]){
				  if(Elem45[i]*Elem45[i]<Elem44[i] && Elem45[i]*Elem45[i]<Elem55[i]){
				    if(Elem46[i]*Elem46[i]<Elem44[i] && Elem46[i]*Elem46[i]<Elem66[i]){
				      if(Elem56[i]*Elem56[i]<Elem55[i] && Elem56[i]*Elem56[i]<Elem66[i]){
					if(Elem57[i]*Elem57[i]<Elem55[i] && Elem57[i]*Elem57[i]<Elem77[i]){
					  myMatrixFile<<index_id[i]<<"  "<<Elem33[i]<<"  "<<Elem34[i]<<"  "<<Elem44[i]<<"  "<<Elem35[i]<<"  "<<Elem45[i]<<"  "<<Elem55[i]<<"  "<<Elem46[i]<<"  "<<Elem56[i]<<"  "<<Elem66[i]<<"  "<<Elem57[i]<<"  "<<Elem67[i]<<"  "<<Elem77[i]<<std::endl;
					}
				      }
				    }
				  }
				}
			      }
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
    if (flag==1 || flag1!=1){
      std::cout<<"Flag not 0: "<<index_id[i]<<" " <<flag<<"  "<<flag1<<std::endl;
    }	    
  }

  /*
    if(Elem34[i]*Elem34[i]<Elem33[i] && Elem34[i]*Elem34[i]<Elem44[i]){
      if(Elem35[i]*Elem35[i]<Elem33[i] && Elem35[i]*Elem35[i]<Elem55[i]){
	if(Elem45[i]*Elem45[i]<Elem44[i] && Elem45[i]*Elem45[i]<Elem55[i]){
	  if(Elem46[i]*Elem46[i]<Elem44[i] && Elem46[i]*Elem46[i]<Elem66[i]){
	    if(Elem56[i]*Elem56[i]<Elem55[i] && Elem56[i]*Elem56[i]<Elem66[i]){
	      if(Elem57[i]*Elem57[i]<Elem55[i] && Elem57[i]*Elem57[i]<Elem77[i]){
		myMatrixFile<<index_id[i]<<"  "<<Elem33[i]<<"  "<<Elem34[i]<<"  "<<Elem44[i]<<"  "<<Elem35[i]<<"  "<<Elem45[i]<<"  "<<Elem55[i]<<"  "<<Elem46[i]<<"  "<<Elem56[i]<<"  "<<Elem66[i]<<"  "<<Elem57[i]<<"  "<<Elem67[i]<<"  "<<Elem77[i]<<std::endl;
	      }
	    }
	  }
	}
      }
    }
  }
	      */
}
