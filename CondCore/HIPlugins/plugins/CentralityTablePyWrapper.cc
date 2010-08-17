
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

    for(unsigned int j=0; j<table.m_table.size(); j++){

      const CentralityTable::CBin& thisBin = table.m_table[j];

      ss<<"HF Cut = "<<thisBin.bin_edge<<std::endl;
      ss<<"Npart = "<<thisBin.n_part.mean<<std::endl;
      ss<<"sigma = "<<thisBin.n_part.var<<std::endl;
      ss<<"Ncoll = "<<thisBin.n_coll.mean<<std::endl;
      ss<<"sigma = "<<thisBin.n_coll.var<<std::endl;
      ss<<"B     = "<<thisBin.b.mean<<std::endl;
      ss<<"sigma = "<<thisBin.b.var<<std::endl;
      ss<<"__________________________________________________"<<std::endl;

    }

    return ss.str();
  }

}

PYTHON_WRAPPER(CentralityTable,CentralityTable);

