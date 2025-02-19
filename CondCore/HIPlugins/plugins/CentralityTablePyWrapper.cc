
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"

#include <string>
#include <sstream>
#include <iostream>


namespace cond {

  template<>
  std::string
  PayLoadInspector<CentralityTable>::summary() const {

    CentralityTable const & table = object();

    std::stringstream ss;

    ss<<"Bin \t";
    ss<<"Lower Boundary\t";
    ss<<"Npart \t";
    ss<<"sigma \t";
    ss<<"Ncoll \t";
    ss<<"sigma \t";
    ss<<"B \t";
    ss<<"sigma \t"<<std::endl;
    ss<<"__________________________________________________"<<std::endl;

    for(unsigned int j=0; j<table.m_table.size(); j++){

      const CentralityTable::CBin& thisBin = table.m_table[j];
      ss<<j<<" \t";
      ss<<thisBin.bin_edge <<"\t";
      ss<<thisBin.n_part.mean<<" \t";
      ss<<thisBin.n_part.var<<" \t";
      ss<<thisBin.n_coll.mean<<" \t";
      ss<<thisBin.n_coll.var<<" \t";
      ss<<thisBin.b.mean<<" \t";
      ss<<thisBin.b.var<<" \t"<<std::endl;
      ss<<"__________________________________________________"<<std::endl;

    }

    return ss.str();
  }

}

PYTHON_WRAPPER(CentralityTable,CentralityTable);

