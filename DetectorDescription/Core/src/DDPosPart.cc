
#include <utility>  
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

void DDpos(const DDLogicalPart & self,
           const DDLogicalPart & mother,
	   std::string copyno,
	   const DDTranslation & trans,
	   const DDRotation & rot,
	   const DDDivision * div)
{
  int cpno = atoi(copyno.c_str());
  DDpos(self,mother,cpno,trans,rot,div);
}

void DDpos(const DDLogicalPart & self,
           const DDLogicalPart & mother,
	   int copyno,
	   const DDTranslation & trans,
	   const DDRotation & rot,
	   const DDDivision * div)
{
  DDCompactView cpv(true); 
  graph_type & graph = cpv.writeableGraph();
  //DDTranslation * tt = new DDTranslation(trans);
  //DDPosData * pd = new DDPosData(*tt,rot,cpno);
  DDPosData * pd = new DDPosData(trans,rot,copyno,div);
  DCOUT('G', " DDPos: graph.addEdge: \n mother=" << mother << "\n child=" << self);
  graph.addEdge(mother,self,pd);
}
