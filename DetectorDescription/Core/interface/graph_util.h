#ifndef x_graph_util_h
#define x_graph_util_h

#include "DetectorDescription/Core/interface/adjgraph.h"
#include "DetectorDescription/Core/interface/graphwalker.h"
#include <iostream>
#include <string>





template<class N, class E>
void output(const graph<N,E> & g, const N & root)
{
  graphwalker<N,E> w(g,root);
  bool go=true;
  while(go) {
    std::cout << w.current().first << ' ';
    go=w.next();
  }
  std::cout << std::endl;
}

template<class N, class E>
void graph_combine(const graph<N,E> & g1, const graph<N,E> & g2,
                   const N & n1, const N & n2, const N & root,
	            graph<N,E> & result)
{
  result = g1;
  result.replace(n1,n2);
  //output(result,n2);
  graphwalker<N,E> walker(g2,n2);
  while (walker.next()) {
    const N & parent = g2.nodeData((++walker.stack().rbegin())->first->first);
    /*
    N parent = g2.nodeData((++walker.stack().rbegin())->first->first);
    N child  = walker.current().first;
    E edge   = walker.current().second;
    */
    //std::cout << parent << ' ' << walker.current().first << ' ' << walker.current().second<< std::endl;
    result.addEdge(parent, walker.current().first, walker.current().second);
    //result.addEdge(parent,child,edge);
    //result.dump_graph();
    //output(result,n2);
  }
  result.replace(n2,root);  			
  //output(result,root);
}		


template<class N, class E>
void graph_tree_output(const graph<N,E> & g, const N & root, std::ostream & os)
{
   graphwalker<N,E> w(g,root);
   bool go=true;
   unsigned int depth=0;
   while (go) {
     std::string s(2*depth,' ');
     os << ' ' << s << w.current().first << '(' << w.current().second << ')' << std::endl;
     if (go=w.firstChild()) {
       ++depth;
     }
     else if(w.stack().size() >1 && w.nextSibling()) {
        go=true;
     }
     else {
       go=false;
       while(w.parent()) {
         --depth;
	  if (w.stack().size()>1 && w.nextSibling()) {
	     go=true;
	     break;
	  }
       }
     }
     
   }  
}
#endif
