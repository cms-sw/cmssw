#include "FastSimulation/PPSFastObjects/interface/PPSRecoVertex.h"
void PPSRecoVertex::AddGolden(double x,double y, double z,int tF,int tB){
     int Id=this->Add(x,y,z,tF,tB);
     this->at(Id-1).Flag=1;
};
void PPSRecoVertex::SetGolden(int i){
     this->at(i).Flag=1;
}
PPSRecoVertex PPSRecoVertex::GetGoldenVertices() {
     PPSRecoVertex GoldenVertices;
     for(int i=0;i<(int)this->size();i++) {
        if (this->at(i).Flag==1) GoldenVertices.push_back(this->at(i));
     }
     return GoldenVertices;
}
