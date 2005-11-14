/***************************************************************************
                          main.cpp  -  description
                             -------------------
    begin                : Wed Oct 24 17:36:15 PDT 2001
    copyright            : (C) 2001 by Michael Case
    email                : case@ucdhep.ucdavis.edu
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/


#include <iostream>
#include <stdlib.h>

#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Parser/interface/DDLConfiguration.h"
#include "DetectorDescription/Core/interface/DDMap.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/DDNumeric.h"
#include "DetectorDescription/Core/interface/DDString.h"
#include "DetectorDescription/Algorithm/src/AlgoInit.h"
#include "DetectorDescription/Core/src/DDCheck.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/CoreImpl/src/DDCheckMaterials.cc"
#include "DetectorDescription/Base/interface/DDException.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

int main(int argc, char *argv[])
{

  std::cout << "main:: initialize" << std::endl;

  AlgoInit();

  std::cout << "main::initialize DDL parser" << std::endl;
  DDLParser* myP = DDLParser::instance();

  DDLConfiguration dp;

  dp.readConfig("configuration.xml");

  std::cout << "main::about to start parsing" << std::endl;
 
  myP->parse(dp);

  std::cout << "main::completed Parser" << std::endl;
  
  std::cout << std::endl << std::endl << "main::Start checking!" << std::endl << std::endl;
  DDCheckMaterials(std::cout);

  std::cout << "======== ev - node - pospart test ======" << std::endl;
  DDName ddn("MUON","cms");
  std::cout << DDRootDef::instance().root().ddname() << std::endl;
  DDLogicalPart lp(ddn);
  std::cout << "lp = " << lp << std::endl;
  std::cout << "== about to get du cpv ==" << std::endl;
  DDRootDef::instance().set(ddn);
//   std::cout << DDRootDef::instance().root().ddname() << std::endl;
  DDCompactView cpvglobal;
  DDCompactView cpv(lp);
  cpv.writeableGraph() = cpvglobal.graph();
  std::cout << "== done got du cpv ==" << std::endl;
  DDExpandedView ev(cpv);
  std::cout << "== got the epv ==" << std::endl;
  ev.firstChild();
  ev.nextSibling();
  std::cout << ev.geoHistory() << std::endl;
  ev.nextSibling();
  std::cout << ev.geoHistory() << std::endl;
  ev.firstChild();
  std::cout << ev.geoHistory() << std::endl;
  ev.nextSibling();
  std::cout << ev.geoHistory() << std::endl;
  ev.nextSibling();
  std::cout << ev.geoHistory() << std::endl;
  ev.firstChild();
  std::cout << ev.geoHistory() << std::endl;
  ev.nextSibling();
  std::cout << ev.geoHistory() << std::endl;
  ev.firstChild();
  std::cout << ev.geoHistory() << std::endl;
  ev.nextSibling();
  std::cout << ev.geoHistory() << std::endl;
  ev.firstChild();
  std::cout << ev.geoHistory() << std::endl;
  ev.nextSibling();
  std::cout << ev.geoHistory() << std::endl;
  return EXIT_SUCCESS;

    cpvglobal.clear();
    std::cout << "cleared DDCompactView.  " << std::endl;
}
