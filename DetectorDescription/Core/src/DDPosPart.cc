
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <utility>  

void DDpos(const DDLogicalPart & self,
           const DDLogicalPart & mother,
	   std::string copyno,
	   const DDTranslation & trans,
	   const DDRotation & rot,
	   const DDDivision * div)
{
//   std::cout << "about to pos using string copy_no " << copyno << "  of  " << std::endl;
//   std::cout << self << " in mother " << std::endl << mother << std::endl;
//   std::cout << "Rotation matrix " << std::endl << *(rot.rotation()) << std::endl;
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
//   std::cout << "about to pos using int copy_no " << copyno << "  of  " << std::endl;
//   std::cout << self << " in mother " << std::endl<< mother << std::endl;
//   std::cout << "Rotation matrix " << std::endl << *(rot.rotation()) << std::endl;
  //DDTranslation * tt = new DDTranslation(trans);
  //DDPosData * pd = new DDPosData(*tt,rot,cpno);
  DDPosData * pd = new DDPosData(trans,rot,copyno,div);
  graph.addEdge(mother,self,pd);
}
