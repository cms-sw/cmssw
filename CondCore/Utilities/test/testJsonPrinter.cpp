#include<iostream>
#include<sstream>
#include "CondCore/Utilities/interface/JsonPrinter.h"

int main() {

  cond::utilities::JsonPrinter printer("PO","P1");
  for (int i=0;i<10;i++){
    std::stringstream ssx, ssy;
    ssx << i;
    ssy << i+1;
    printer.append(ssx.str(),ssy.str());
  }
  std::cout << printer.print() <<std::endl;
  return 0;

}
