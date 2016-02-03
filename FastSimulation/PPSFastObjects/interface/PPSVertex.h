#ifndef PPSVERTEX_H
#define PPSVERTEX_H
#include "TObject.h"

class PPSVertex: public TObject {
public:
       PPSVertex();
       ~PPSVertex(){};
       int Id;
       int idxTrkF; // index of forward track making up this vertex
       int idxTrkB; // index of backward track making up this vertex
       int Flag; // For GenVertex: 1 for signal, 0 for pileup; For RecoVertex: 1 for GoldenVertex, 0 for other vertices
       double x;
       double y;
       double z;
ClassDef(PPSVertex,1);
};
#endif
