#ifndef PPSBASEVERTEX_H
#define PPSBASEVERTEX_H
#include "FastSimulation/PPSFastObjects/interface/PPSVertex.h"
#include <vector>
#include "TObject.h"

class PPSBaseVertex:  public std::vector<PPSVertex> {
public:
       PPSBaseVertex();
       virtual ~PPSBaseVertex() {};

       void clear() {std::vector<PPSVertex>::clear();index=-1;};
       int Add(double x, double y, double z,int trkF,int trkB); // return the vertex ID;
       int Add(const PPSVertex& vtx) {this->push_back(vtx);return this->size()+1;};
       int Nvtx() {return (int)this->size()+1;};
       void SetFlag(int idx,int flag) {if (idx<0||idx>=(int)this->size()) return;
                                       this->at(idx).Flag=flag;
                                      };
       PPSVertex& GetVertex();
       PPSVertex& GetVertex(int&);
       PPSVertex& GetVertexById(int id);

ClassDef(PPSBaseVertex,1);

private:
       int index; // index to the next vertex to be retrieved (-1 before first call)
};
#endif
