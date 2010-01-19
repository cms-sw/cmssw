
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

#include <utility>  

DDPositioner::DDPositioner( DDCompactView * cpv ) : cpv_(cpv) { }

DDPositioner::~DDPositioner() { }

void DDPositioner::operator() (const DDLogicalPart & self, 
			       const DDLogicalPart & parent,
			       std::string copyno,
			       const DDTranslation & trans,
			       const DDRotation & rot,
			       const DDDivision * div)
{
  int cpno = atoi(copyno.c_str());
  this->operator()(self,parent,cpno,trans,rot, div);
}

void DDPositioner::operator() (const DDLogicalPart & self,
			       const DDLogicalPart & parent,
			       int copyno,
			       const DDTranslation & trans,
			       const DDRotation & rot,
			       const DDDivision * div)
{
  cpv_->position( self, parent, copyno, trans, rot, div );
//   DDCompactView::graph_type & graph = cpv_->writeableGraph();
//   DDPosData * pd = new DDPosData(trans,rot,copyno,div);
//   graph.addEdge(parent,self,pd);
}

// void DDpos(const DDLogicalPart & self,
//            const DDLogicalPart & mother,
// 	   std::string copyno,
// 	   const DDTranslation & trans,
// 	   const DDRotation & rot,
// 	   const DDDivision * div)
// {
// //   std::cout << "about to pos using string copy_no " << copyno << "  of  " << std::endl;
// //   std::cout << self << " in mother " << std::endl << mother << std::endl;
// //   std::cout << "Rotation matrix " << std::endl << *(rot.rotation()) << std::endl;
//   int cpno = atoi(copyno.c_str());
//   DDpos(self,mother,cpno,trans,rot, div);
// }

// void DDpos(const DDLogicalPart & self,
//            const DDLogicalPart & mother,
// 	   int copyno,
// 	   const DDTranslation & trans,
// 	   const DDRotation & rot,
// 	   const DDDivision * div)
// {
//   DDCompactView cpv(true); 
//   DDCompactView::graph_type & graph = cpv.writeableGraph();
// //   std::cout << "about to pos using int copy_no " << copyno << "  of  " << std::endl;
// //   std::cout << self << " in mother " << std::endl<< mother << std::endl;
// //   std::cout << "Rotation matrix " << std::endl << *(rot.rotation()) << std::endl;
//   //DDTranslation * tt = new DDTranslation(trans);
//   //DDPosData * pd = new DDPosData(*tt,rot,cpno);
//   DDPosData * pd = new DDPosData(trans,rot,copyno,div);
//   graph.addEdge(mother,self,pd);
// }
