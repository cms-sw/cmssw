#include "FastSimulation/PPSFastObjects/interface/PPSBaseVertex.h"
PPSBaseVertex::PPSBaseVertex():std::vector<PPSVertex>(),index(-1){};
int PPSBaseVertex::Add(double x,double y, double z,int tF, int tB) {
               PPSVertex vtx;
               vtx.x=x;vtx.y=y;vtx.z=z;
               vtx.Id=this->size()+1;
               vtx.idxTrkF=tF;
               vtx.idxTrkB=tB;
               vtx.Flag=0;
               this->push_back(vtx);
               return vtx.Id;
}
PPSVertex& PPSBaseVertex::GetVertex() {
           index++;
           return GetVertex(index);
}
PPSVertex& PPSBaseVertex::GetVertexById(int id) {
           for (int i=0;i<(int)this->size();i++) {
               if (this->at(i).Id==id) return this->at(i);
           }
           PPSVertex* vtx = new PPSVertex();
           vtx->Id=-1; // set the vertex index to -1 to signal the end of the vertex collection
           return (*vtx);
}
PPSVertex& PPSBaseVertex::GetVertex(int& i) {
           if (i>=(int)this->size()) {
              PPSVertex* vtx = new PPSVertex();
              vtx->Id=-1; // set the vertex index to -1 to signal the end of the vertex collection
              i=-1;
              return (*vtx);
           }
           return this->at(i);
}
