#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include <string>
#include <iostream>
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

//#define DEBUG

MuonDDDConstants::MuonDDDConstants(){
  theMuonNamespace = "muonNumbering";
}

int MuonDDDConstants::getValue(const std::string constantName) {
  ExprEvalInterface & evaluator = ExprEval::instance();
#ifdef DEBUG
  std::cout << "MuonDDDConstants::GetValue "<<theMuonNamespace
       << " "<<constantName<<" "<<std::endl;
#endif
  double result=evaluator.eval(theMuonNamespace,constantName);
#ifdef DEBUG
  std::cout << "MuonDDDConstants::GetValue "<<constantName<<" "<<
    result<<std::endl;
#endif
  return int(result);
}

// string defns = "muon-numbering"; // default namespace
// string expr1 = "[levelTag]";
// string expr2 = "[arnos-constants:levelTag]"
// double result = evaluator.eval(defns,expr1); // sollte 1000 geben
// std::cout << result << std::endl;
// result = evaluator.eval(defns,expr2); // ebenso 1000
// std::cout << result << std::endl;
// 
// // sichere variante
// try {
//   result = evaluator.eval(defns,expr1);
// }
// catch(const DDException & e) // it's a DDException!
// {
//   std::cerr << "ups!" << std::endl << e << std::endl;
// }
